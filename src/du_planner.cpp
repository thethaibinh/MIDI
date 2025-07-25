/*!
 * Copyright 2024 by Binh Nguyen <thethaibinh@gmail.com>
 *
 * This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
 * You may use, redistribute, and modify this code for non-commercial purposes only, provided that:
 * 1. You give appropriate credit to the original author
 * 2. You indicate if changes were made
 *
 * This code is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * For commercial use, please contact the author for licensing terms.
 * Full license text: https://creativecommons.org/licenses/by-nc/4.0/
 */

#include "depth_uncertainty_planner/base_planner.hpp"
#include <omp.h>

using namespace std::chrono;
using namespace common_math;
using namespace depth_uncertainty_planner;

DuPlanner::DuPlanner(const cv::Mat& depthImage, const cv::Mat& vx, const cv::Mat& vy, const cv::Mat& vz, const PinholeCamera& camera,
                     const CollisionCheckingMethod& collision_checking_method,
                     const double& checking_time_ratio,
                     const uint32_t& sampled_trajectories_threshold,
                     const uint32_t& checked_trajectories_threshold,
                     const bool& debug_num_trajectories,
                     const double& collision_probability_threshold,
                     const uint32_t& openmp_chunk_size)
  : _camera(camera),
    _checking_time_ratio(checking_time_ratio),
    _sampled_trajectories_threshold(sampled_trajectories_threshold),
    _checked_trajectories_threshold(checked_trajectories_threshold),
    _debug_num_trajectories(debug_num_trajectories),
    _collision_probability_threshold(collision_probability_threshold),
    _openmp_chunk_size(openmp_chunk_size),
    _minimum_allowed_thrust(0),  // By default don't allow negative thrust
    _maximum_allowed_thrust(30),  // By default limit maximum thrust to about 3g (30 m/s^2)
    _maximum_allowed_angular_velocity(20),  // By default limit maximum angular velocity to 20 rad/s
    _minimum_section_time_dynamic_feasibility(0.02),  // By default restrict the dynamic feasibility check to a minimum
                                                      // section duration of 20ms
    _max_pyramid_gen_time(1000),  // Don't limit pyramid generation time by default [seconds].
    _pyramid_gen_time_nanoseconds(0),
    _max_num_pyramids(std::numeric_limits<int>::max()),  // Don't limit the number of generated
                                                          // pyramids by default
    _allocated_computation_time(0),      // To be set when the planner is called
    _num_trajectories_sampled(0),
    _num_trajectories_generated(0),
    _num_collision_checked(0),
    _num_collision_free(0),
    _checking_time(0),
    _generating_time(0),
    _pyramid_search_pixel_buffer(2),  // A sample point must be more than 2 pixels away from the edge of a
                                      // pyramid to use that pyramid for collision checking
    steering_direction(0),
    max_collision_probability(0.0),
    collision_checking_method(collision_checking_method) {
  _depth_data = reinterpret_cast<const float*>(depthImage.data);
  _vx_data = reinterpret_cast<const float*>(vx.data);
  _vy_data = reinterpret_cast<const float*>(vy.data);
  _vz_data = reinterpret_cast<const float*>(vz.data);
}

bool DuPlanner::find_lowest_cost_trajectory(
  ruckig::InputParameter<3>& initial_state,
  ruckig::Trajectory<3>& opt_trajectory,
  RandomTrajectorySampler& trajectory_sampler,
  double allocated_computation_time,
  void* cost_function_definition_object,
  double (*cost_function_wrapper)(void* cost_function_definition_object,
                                Eigen::Vector3d& endpoint_position)) {
  // Start timing the planner
  start_time = high_resolution_clock::now();
  _allocated_computation_time = allocated_computation_time;
  ruckig::Ruckig<3> otg;
  ruckig::Trajectory<3> candidate_trajectory;
  ruckig::Result result;
  bool feasible_trajectory_found = false;
  double lowest_collision_cost = std::numeric_limits<double>::max();
  double best_traveling_cost = std::numeric_limits<double>::max();
  double best_mahalanobis_distance = std::numeric_limits<double>::max();
  double best_trajectory_collision_probability = 0.0;
  uint32_t time_elapsed_us = 0;
  const uint32_t allocated_planning_time_us = int(_allocated_computation_time * 1e6);
  const uint32_t allocated_collision_checking_time_us = int(allocated_planning_time_us * _checking_time_ratio);
  while (true) {
    time_elapsed_us = duration_cast<microseconds>(high_resolution_clock::now() - start_time).count();
    if (time_elapsed_us > allocated_planning_time_us) {
      break;
    }

    // Get the next candidate trajectory to evaluate using the provided trajectory generator
    Eigen::Vector3d sampled_endpoint_position = trajectory_sampler.get_next_time_optimal_trajectory(_depth_data, _camera);
    _num_trajectories_sampled++;
    // Compute the cost of the trajectory using the provided cost function
    double sampled_traveling_cost = (*cost_function_wrapper)(cost_function_definition_object, sampled_endpoint_position);
    if (sampled_traveling_cost > best_traveling_cost) {
      // Sample a new trajectory because this one is already worse than the collision-free trajectory found
      continue;
    }

    double trajectory_collision_probability = 0.0;
    double mahalanobis_distance = std::numeric_limits<double>::max();

    // The trajectory is a lower cost than lowest cost trajectory found so far
    // Check whether the trajectory collides with obstacles
    // First split trajectory into possible 7 segment as described in the
    // profile then check every one of them.
    initial_state.target_position = {sampled_endpoint_position.x(),
                                    sampled_endpoint_position.y(),
                                    sampled_endpoint_position.z()};

    auto start_generating_time = high_resolution_clock::now();
    // Calculate the trajectory in an offline manner (outside of the control loop)
    result = otg.calculate(initial_state, candidate_trajectory);
    if (result == ruckig::Result::ErrorInvalidInput) {
      ROS_WARN("Invalid input");
      continue;
    }
    if (result != ruckig::Result::Working) {
      ROS_WARN("Something else!");
      continue;
    }
    _num_trajectories_generated++;
    _generating_time += duration_cast<microseconds>(high_resolution_clock::now() - start_generating_time).count();
    auto start_checking_time = high_resolution_clock::now();

    // Skip trajectory if it is too short
    if (candidate_trajectory.get_duration() <= 1e-6) continue;

    std::vector<Segment*> segments = get_segments(candidate_trajectory);
    // Skip trajectory if it has no segments (empty trajectory)
    if (segments.empty()) continue;

    bool is_collision_free = true;
    for (uint8_t i = 0; i < segments.size(); i++) {
      if (const auto* second_order_segment = dynamic_cast<const common_math::SecondOrderSegment*>(segments[i])) {
        if (!second_order_segment->is_monotonically_increasing_depth()) {
          is_collision_free = false;
          break;
        }
        is_collision_free &= is_segment2_collision_free(initial_state, second_order_segment, trajectory_collision_probability, mahalanobis_distance);
      } else if (const auto* third_order_segment = dynamic_cast<const common_math::ThirdOrderSegment*>(segments[i])) {
        is_collision_free &= is_segment3_collision_free(third_order_segment);
      } else {
        is_collision_free = false;
        break;
      }
      if (!is_collision_free) break;
    }

    _checking_time += duration_cast<microseconds>(high_resolution_clock::now() - start_generating_time).count();
    // Increase as another trajectory has been checked
    _num_collision_checked++;
    if (is_collision_free) {
      _num_collision_free++;
      // Update segment_max_collision_probability
      feasible_trajectory_found = true;
      if (collision_checking_method == CollisionCheckingMethod::MIDI && time_elapsed_us > allocated_collision_checking_time_us) {
        if (trajectory_collision_probability < lowest_collision_cost) {
          lowest_collision_cost = trajectory_collision_probability;
          opt_trajectory = candidate_trajectory;
        }
      } else {
        best_traveling_cost = sampled_traveling_cost;
        opt_trajectory = candidate_trajectory;
      }
    }
  }
  if (_debug_num_trajectories) {
    ROS_WARN("sampled: %d, checked: %d, free: %d", _num_trajectories_sampled,
             _num_collision_checked, _num_collision_free);
  }
  if (feasible_trajectory_found &&
      (_num_collision_checked > _checked_trajectories_threshold ||
       _num_trajectories_sampled > _sampled_trajectories_threshold)) {
    return true;
  } else {
    steering_direction = scan_depth();
    return false;
  }
}

int DuPlanner::scan_depth() {
  // We don't look at any pixels closer than this distance (e.g. if the
  // propellers are in the field of view)
  float ignore_distance = _camera.get_true_vehicle_radius();
  const uint16_t height = _camera.get_height();
  const uint16_t width = _camera.get_width();
  const uint32_t num_pixels = height * width;

  // Initialize atomic variables for thread-safe updates
  std::atomic<float> min_depth_value{std::numeric_limits<float>::max()};
  std::atomic<uint16_t> min_depth_x{0};
  std::atomic<uint16_t> min_depth_y{0};

  // Parallel processing of depth data
  #pragma omp parallel
  {
    // Thread-local variables for better performance
    float local_min_depth = std::numeric_limits<float>::max();
    uint16_t local_min_x = 0;
    uint16_t local_min_y = 0;

    #pragma omp for nowait
    for (uint32_t depth_idx = 0; depth_idx < num_pixels; depth_idx++) {
      float pixel_depth = _depth_data[depth_idx];
      if (pixel_depth <= local_min_depth && pixel_depth > ignore_distance) {
        local_min_depth = pixel_depth;
        local_min_x = depth_idx % width;
        local_min_y = depth_idx / width;
      }
    }

    // Update global minimum using atomic operations
    #pragma omp critical
    {
      if (local_min_depth < min_depth_value) {
        min_depth_value = local_min_depth;
        min_depth_x = local_min_x;
        min_depth_y = local_min_y;
      }
    }
  }
  if (min_depth_x < _camera.get_cx())
    return -1;
  else
    return 1;
}

std::vector<Segment*> DuPlanner::get_segments(
  const ruckig::Trajectory<3> traj) {
  std::vector<Segment*> segments;

  auto profile_array = traj.get_profiles();
  // We only consider one-target-waypoint trajectory
  assert(profile_array.size() == 1);
  auto profiles = profile_array[0];
  // Confirming there are 3 DOFs in a trajectory
  uint8_t num_dof = profiles.size();
  assert(num_dof == 3);

  // Adding two possible pre-trajectories (brake/accel) if there is any
  double brake_duration = profiles[0].brake.duration;
  if (brake_duration > 1e-6) {
    Eigen::Vector3d j(profiles[0].brake.j[0], profiles[1].brake.j[0], profiles[2].brake.j[0]);
    Eigen::Vector3d a(profiles[0].brake.a[0], profiles[1].brake.a[0],
                      profiles[2].brake.a[0]);
    Eigen::Vector3d v0(profiles[0].brake.v[0], profiles[1].brake.v[0],
                       profiles[2].brake.v[0]);
    Eigen::Vector3d p0(profiles[0].brake.p[0], profiles[1].brake.p[0],
                       profiles[2].brake.p[0]);
    if (collision_checking_method == CollisionCheckingMethod::PYRAMID) {
      ThirdOrderSegment* seg = new ThirdOrderSegment({j / 6.0, a / 2.0, v0, p0}, 0.0, brake_duration);
      segments.push_back(seg);
    } else if (collision_checking_method == CollisionCheckingMethod::MIDI) {
      SecondOrderSegment* seg = new SecondOrderSegment({a / 2.0, v0, p0}, 0.0, brake_duration);
      segments.push_back(seg);
    }
  }
  double accel_duration = profiles[0].accel.duration;
  if (accel_duration > 1e-6) {
    Eigen::Vector3d j(profiles[0].accel.j[0], profiles[1].j[0], profiles[2].accel.j[0]);
    Eigen::Vector3d a(profiles[0].accel.a[0], profiles[1].accel.a[0],
                      profiles[2].accel.a[0]);
    Eigen::Vector3d v0(profiles[0].accel.v[0], profiles[1].accel.v[0],
                       profiles[2].accel.v[0]);
    Eigen::Vector3d p0(profiles[0].accel.p[0], profiles[1].accel.p[0],
                       profiles[2].accel.p[0]);
    if (collision_checking_method == CollisionCheckingMethod::PYRAMID) {
      ThirdOrderSegment* seg = new ThirdOrderSegment({j / 6.0, a / 2.0, v0, p0}, 0.0, accel_duration);
      segments.push_back(seg);
    } else if (collision_checking_method == CollisionCheckingMethod::MIDI) {
      SecondOrderSegment* seg = new SecondOrderSegment({a / 2.0, v0, p0}, 0.0, accel_duration);
      segments.push_back(seg);
    }
  }

  // Then adding possible 7 sections of the trajectory
  uint8_t num_sec = profiles[0].t.size();
  assert(num_sec == 7);
  for (uint8_t i = 0; i < num_sec; i++) {
    double end_time = profiles[0].t[i];
    if (fabs(end_time) < 1e-6) continue;
    Eigen::Vector3d j(profiles[0].j[i], profiles[1].j[i], profiles[2].j[i]);
    Eigen::Vector3d a(profiles[0].a[i], profiles[1].a[i], profiles[2].a[i]);
    Eigen::Vector3d v0(profiles[0].v[i], profiles[1].v[i], profiles[2].v[i]);
    Eigen::Vector3d p0(profiles[0].p[i], profiles[1].p[i], profiles[2].p[i]);
    if (collision_checking_method == CollisionCheckingMethod::PYRAMID) {
      ThirdOrderSegment* seg = new ThirdOrderSegment({j / 6.0, a / 2.0, v0, p0}, 0.0, end_time);
      segments.push_back(seg);
    } else if (collision_checking_method == CollisionCheckingMethod::MIDI) {
      SecondOrderSegment* seg = new SecondOrderSegment({a / 2.0, v0, p0}, 0.0, end_time);
      segments.push_back(seg);
    }
  }
  return segments;
}

bool DuPlanner::is_segment2_collision_free(const ruckig::InputParameter<3>& initial_state, const SecondOrderSegment* original_segment, double& trajectory_collision_probability, double& mahalanobis_distance) {
  PinholeCamera camera = _camera;
  // Skip if the segment is too short
  if (original_segment->get_duration() <= 1e-6) return true;
  // Skip if the segment is closer than the vehicle radius
  if (original_segment->get_end_point().z() <= camera.get_true_vehicle_radius()) return true;

  double start_checking_time = 0.0;
  // Split the part of the segment that is closer than the vehicle radius
  if (original_segment->get_start_point().z() < camera.get_true_vehicle_radius()) {
    uint8_t num_roots = original_segment->solve_first_time_at_depth(camera.get_true_vehicle_radius(), start_checking_time);
    // Missing root means the segment is somehow still inside the vehicle radius
    if (num_roots == 0) return true;
    if (start_checking_time >= original_segment->get_end_time()) return true;
  }
  const std::vector<Eigen::Vector3d> coeffs = original_segment->get_coeffs();
  SecondOrderSegment segment(coeffs, start_checking_time, original_segment->get_end_time());
  if (segment.get_duration() < 1e-6) return true;

  // Compute interest region for collision checking
  const std::vector<int16_t> boundary = segment.get_projection_boundary(camera);
  const int16_t left = boundary[0];
  const int16_t top = boundary[1];
  const int16_t right = boundary[2];
  const int16_t bottom = boundary[3];

  if (left < 0 || right > camera.get_width() || top < 0 ||
      bottom > camera.get_height()) {
    ROS_ERROR("Boundary out of frame");
    return false;
  }

  // Check for collision and compute probability for all pixels in bb_ltrb
  const Eigen::Vector3d endpoint = segment.get_end_point();
  const double checking_depth = endpoint.z() + camera.get_planning_vehicle_radius();
  const uint16_t img_width = camera.get_width();
  const double true_vehicle_radius = camera.get_true_vehicle_radius();
  const double min_clear_distance = camera.get_minimum_clear_distance();
  const double planning_vehicle_radius = camera.get_planning_vehicle_radius();
  // Evaluate the Euclidean distance of all depth pixels in bb_ltrb to check for collision
  double segment_collision_probability = 0.0;
  std::atomic<bool> collision_detected{false};  // Make collision_detected atomic

  #pragma omp parallel for collapse(2) reduction(max : segment_collision_probability) schedule(dynamic, _openmp_chunk_size)
  for (uint16_t y = top; y < bottom; y++) {
    for (uint16_t x = left; x < right; x++) {
      // Check if cancellation has been requested
      #pragma omp cancellation point for

      // Check collision_detected flag and cancel if true
      if (collision_detected.load(std::memory_order_relaxed)) {
        #pragma omp cancel for
      }

      const double spatial_z = _depth_data[y * img_width + x];
      if (std::isnan(spatial_z)) continue;

      // Skip pixels with depth smaller than the true vehicle radius
      if (spatial_z < true_vehicle_radius) continue;
      // Skip pixels with depth greater than checking_depth + 1 meter
      if (spatial_z > (checking_depth + 2)) continue;

      const Eigen::Vector3d depth_point = camera.deproject_pixel_to_point(x, y, spatial_z);

      // Skip collision checking for pixels with depth smaller than the minimum collision distance
      if (spatial_z < min_clear_distance) {
        if (segment.get_euclidean_distance(depth_point) < planning_vehicle_radius) {
          collision_detected.store(true, std::memory_order_relaxed);
        }
        continue;
      }

      // Get the velocity of the camera
      const double vx_cam = initial_state.current_velocity[0];
      const double vy_cam = initial_state.current_velocity[1];
      const double vz_cam = initial_state.current_velocity[2];
      // Camera motion induced flow
      // const double induced_flow_x = (camera.get_fx() * vx_cam - (x - camera.get_cx()) * vz_cam) / spatial_z;
      // const double induced_flow_y = (camera.get_fy() * vy_cam - (y - camera.get_cy()) * vz_cam) / spatial_z;
      const double vx = _vx_data[y * img_width + x] - vx_cam;
      const double vy = _vy_data[y * img_width + x] - vy_cam;
      const double vz = _vz_data[y * img_width + x] - vz_cam;
      // Collision checking for pixels with depth greater than checking_depth and less than checking_depth + 2
      if (spatial_z > checking_depth) {
        // Skip collision checking if this pixel is far away and moving away from the camera
        if (vz < -3.0) continue;
        // Otherwise, check if the dynamic pixel collides with the vehicle
        if (segment.get_dynamic_euclidean_distance(depth_point, vx, vy, vz) < planning_vehicle_radius) {
          collision_detected.store(true, std::memory_order_relaxed);
          continue;
        }
        // If it is collision free, compute the collision probability for the dynamic pixel
        double prob = segment.get_dynamic_collision_probability(depth_point, vx, vy, vz, camera, mahalanobis_distance);
        segment_collision_probability = std::max(segment_collision_probability, prob);
        if (prob > _collision_probability_threshold) {
          collision_detected.store(true, std::memory_order_relaxed);  // Use atomic store
          continue;
        }
        // We're done with this pixel
        continue;
      }

      // Check for collision and compute probability for all remaining pixels
      // (with minimum_clear_distance <= spatial_z <= checking_depth)
      // Check if the static pixel collides with the vehicle
      if (segment.get_euclidean_distance(depth_point) < planning_vehicle_radius) {
        collision_detected.store(true, std::memory_order_relaxed);
        continue;
      }
      // Skip collision checking for pixels moving away very fast from the camera
      if (vz < -5.0) continue;
      // Otherwise, check if the dynamic pixel collides with the vehicle
      if (segment.get_dynamic_euclidean_distance(depth_point, vx, vy, vz) < planning_vehicle_radius) {
        collision_detected.store(true, std::memory_order_relaxed);
        continue;
      }
      // If it is collision free, compute the collision probability for the dynamic pixel
      double prob = segment.get_dynamic_collision_probability(depth_point, vx, vy, vz, camera, mahalanobis_distance);
      segment_collision_probability = std::max(segment_collision_probability, prob);
      if (prob > _collision_probability_threshold) {
        collision_detected.store(true, std::memory_order_relaxed);  // Use atomic store
      }
      // We're done with this pixel
    }
  }
  if (collision_detected.load(std::memory_order_relaxed)) return false;

  trajectory_collision_probability = std::max(trajectory_collision_probability, segment_collision_probability);
  return true;
}
