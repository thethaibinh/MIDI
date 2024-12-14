#pragma once
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
// CV
#include <cv_bridge/cv_bridge.h>

#include <Eigen/Dense>

#include "common_math/pinhole_camera_model.hpp"
#include "common_math/frame_transforms.hpp"

// ROS base
#include <ros/console.h>
#include "ros/ros.h"
#include <geometry_msgs/TransformStamped.h>
// ROS TF2
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

using namespace frame_transform;

//! Class used to generate random candidate trajectories for the planner to
//! evaluate. All generated trajectories come to rest at the end of their
//! duration.
class RandomTrajectorySampler {
 public:
  //! Default constructor. Sample pixels uniformly in the entire image, and then
  //! deproject those pixels to 3D points with depths uniformly sampled
  //! between 1.5 and 3 meters. The duration of each trajectory is sampled
  //! uniformly between 2 and 3 seconds.
  /*!
   * @param planner Pointer to the DuPlanner object that will use the
   * generated trajectories. We include this pointer so that we can call the
   * deproject_pixel_to_point function defined by the DuPlanner object
   * which uses the associated camera intrinsics.
   */
  RandomTrajectorySampler(const common_math::PinholeCamera& camera,
                          const double& depth_upper_bound,
                          const double& depth_lower_bound,
                          const Eigen::Vector3d& exploration_vector,
                          const double& depth_sampling_margin,
                          const geometry_msgs::TransformStamped& body_to_world,
                          const geometry_msgs::Point& goal_in_world_frame,
                          const std::string& world_frame_name,
                          const bool& _3d_planning,
                          const double& _2d_z_margin,
                          const bool& is_spiral_sampling,
                          const uint8_t& spiral_sampling_step)
    : _gen(_rd()),
      _depth_upper_bound(depth_upper_bound),
      _depth_lower_bound(depth_lower_bound),
      _exploration_vector(exploration_vector),
      _depth_sampling_margin(depth_sampling_margin),
      _body_to_world(body_to_world),
      _goal_in_world_frame(goal_in_world_frame),
      _world_frame(world_frame_name),
      _3d_planning(_3d_planning),
      _2d_z_margin(_2d_z_margin),
      _is_spiral_sampling(is_spiral_sampling),
      _spiral_sampling_step(spiral_sampling_step) {
    _spiral_x = 0;
    _spiral_y = 0;
    _spiral_leg = 1;
    _spiral_direction = 0;
    _spiral_initialized = false;
    // Project the exploration vector onto the image plane
    _projected_goal = camera.project_point_to_pixel(exploration_vector);
    std::vector<uint16_t> frame_dims = camera.get_frame_dimensions_with_true_radius_margin();
    sampling_range = {
      frame_dims[0], frame_dims[1],
      frame_dims[2], frame_dims[3]};
    _pixelX = std::uniform_int_distribution<>(sampling_range[0], sampling_range[1]);
    _pixelY = std::uniform_int_distribution<>(sampling_range[2], sampling_range[3]);
  }

  std::vector<int> get_sampling_range() {
    return sampling_range;
  }

  //! Returns a random candidate trajectory to be evaluated by the planner.
  Eigen::Vector3d get_next_time_optimal_trajectory(
    const float* _depth_data, const common_math::PinholeCamera& camera) {
    // Start timing the sampling
    std::chrono::high_resolution_clock::time_point start_sampling_time =
      std::chrono::high_resolution_clock::now();
    double sampling_timeout = 0.01;
    double _scaled_sampled_depth = _depth_upper_bound;
    double sampled_depth = _scaled_sampled_depth;
    Eigen::Vector3d heading_unit_vector(0, 0, 1);
    Eigen::Vector2i gen_pixel;
    while (true) {
      if (std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start_sampling_time)
            .count() > int(sampling_timeout * 1e6)) {
        break;
      }
      // X and Y sampling
      if (_is_spiral_sampling)
        gen_pixel = spiral_search();
      else
        gen_pixel = camera.clamp_to_frame_with_margin(_pixelX(_gen), _pixelY(_gen));

      // Depth sampling
      double pixel_depth_margin =
        _depth_data[gen_pixel.y() * camera.get_width() + gen_pixel.x()] -
        camera.get_planning_vehicle_radius() - _depth_sampling_margin;

      // This one will collide, sample a new one
      if (pixel_depth_margin < _depth_lower_bound) continue;

      // Take max depth of this one
      if (pixel_depth_margin < _depth_upper_bound)
        sampled_depth = pixel_depth_margin;

      // Calculate heading direction factor using normalized 3D vectors
      Eigen::Vector3d sample_unit_vector = camera.deproject_pixel_to_point(gen_pixel.x(), gen_pixel.y(), sampled_depth).normalized();
      double heading_direction_factor = heading_unit_vector.dot(sample_unit_vector);
      double goal_direction_factor = _exploration_vector.normalized().dot(sample_unit_vector);
      double _scaled_sampled_depth =
        heading_direction_factor *
        (sampled_depth -
         (2 - (goal_direction_factor + 1)) *
           (sampled_depth - _depth_lower_bound) / 2);

      if (_3d_planning) break;

      Eigen::Vector3d sampled_point = camera.deproject_pixel_to_point(gen_pixel.x(), gen_pixel.y(), _scaled_sampled_depth);
      if (!valid_2d_z(sampled_point)) continue;
      break;
    }
    // Create 3D vector with sampled_depth as Z
    return camera.deproject_pixel_to_point(gen_pixel.x(), gen_pixel.y(), _scaled_sampled_depth);
  }

  bool valid_2d_z(const Eigen::Vector3d& sampled_point) {
    geometry_msgs::PointStamped sampled_point_in_body_frame;
    sampled_point_in_body_frame.header.frame_id = _world_frame;
    transform_camera_to_body(sampled_point, sampled_point_in_body_frame.point);

    geometry_msgs::PointStamped sampled_point_in_world_frame;
    try {
      tf2::doTransform(sampled_point_in_body_frame, sampled_point_in_world_frame, _body_to_world);
    } catch (tf2::TransformException& ex) {
      ROS_WARN("Failure %s\n", ex.what());  // Print exception which was caught
    }
    return abs(sampled_point_in_world_frame.point.z - _goal_in_world_frame.z) <= _2d_z_margin;
  }

 private:
  Eigen::Vector2i spiral_search() {
    // X and Y sampling
    Eigen::Vector2i gen_pixel;
    // Initialize spiral parameters if not already done
    if (!_spiral_initialized) {
      _spiral_center = _projected_goal;
      _spiral_initialized = true;
    }
    // Generate next point on rectangular spiral
    gen_pixel = {
      _spiral_center.x() + _spiral_x,
      _spiral_center.y() + _spiral_y
    };
    // Move to next position on spiral
    switch (_spiral_direction) {
      case 0:  // Right
        _spiral_x += _spiral_sampling_step;
        if (_spiral_x >= _spiral_leg) {
          _spiral_direction = 1;
        }
        break;
      case 1:  // Down
        _spiral_y += _spiral_sampling_step;
        if (_spiral_y >= _spiral_leg) {
          _spiral_direction = 2;
          _spiral_leg += _spiral_sampling_step;
        }
        break;
      case 2:  // Left
        _spiral_x -= _spiral_sampling_step;
        if (-_spiral_x >= _spiral_leg) {
          _spiral_direction = 3;
        }
        break;
      case 3:  // Up
        _spiral_y -= _spiral_sampling_step;
        if (-_spiral_y >= _spiral_leg) {
          _spiral_direction = 0;
          _spiral_leg += _spiral_sampling_step;
        }
        break;
    }
    // Clamp coordinates to sampling range
    gen_pixel = {
      std::clamp(gen_pixel.x(), sampling_range[0], sampling_range[1]),
      std::clamp(gen_pixel.y(), sampling_range[2], sampling_range[3])
    };
    // Reset spiral if we've hit all boundaries
    if (gen_pixel.x() == sampling_range[0] || gen_pixel.x() == sampling_range[1]) {
      if (gen_pixel.y() == sampling_range[2] || gen_pixel.y() == sampling_range[3]) {
        _spiral_x = 0;
        _spiral_y = 0;
        _spiral_leg = 1;
        _spiral_direction = 0;
      }
    }
    return gen_pixel;
  }
  std::uniform_int_distribution<> _pixelX;
  std::uniform_int_distribution<> _pixelY;
  std::random_device _rd;
  std::mt19937 _gen;
  double _depth_upper_bound, _depth_lower_bound;
  std::vector<int> sampling_range;
  Eigen::Vector3d _exploration_vector;
  double _depth_sampling_margin, _2d_z_margin;
  bool _3d_planning;
  std::string _world_frame;
  geometry_msgs::TransformStamped _body_to_world;
  geometry_msgs::Point _goal_in_world_frame;
  int _spiral_x;           // Current x offset from center
  int _spiral_y;           // Current y offset from center
  int _spiral_leg;         // Current leg length
  uint8_t _spiral_sampling_step;        // Step size in pixels
  int _spiral_direction;   // Current direction (0-3)
  Eigen::Vector2i _spiral_center; // Center point of spiral
  bool _is_spiral_sampling;  // Whether spiral sampling is enabled
  bool _spiral_initialized;  // Whether spiral parameters have been initialized
  Eigen::Vector2i _projected_goal;
};
