#include "planner_node.hpp"

using namespace quadrotor_common;

cv::Mat PlannerNode::preprocess_depth_image(const sm::Image::SharedPtr depth_msg) {
  cv_bridge::CvImageConstPtr cv_img_ptr = cv_bridge::toCvShare(depth_msg, depth_msg->encoding);
  cv::Mat depth_mat;
  cv_img_ptr->image.convertTo(depth_mat, CV_32FC1, _depth_scale);
  return depth_mat;
}

pointcloud_type* PlannerNode::create_point_cloud(const sm::Image::SharedPtr depth_msg)
{
  cv::Mat depth_mat = preprocess_depth_image(depth_msg);
  
  // Use camera intrinsics from config (unified for both modes)
  double cx = _real_cx;
  double cy = _real_cy;
  double fx = _real_focal_length;
  double fy = _real_focal_length;

  pointcloud_type* cloud (new pointcloud_type());
  cloud->header.stamp     = this->now().nanoseconds() / 1000;
  cloud->header.frame_id  = _vehicle_frame;
  cloud->is_dense         = false;
  cloud->height = depth_mat.rows;
  cloud->width = depth_mat.cols;
  cloud->points.resize(cloud->height * cloud->width);

  const float* depth_data = reinterpret_cast<const float*>(depth_mat.data);
  const int rows = depth_mat.rows;
  const int cols = depth_mat.cols;

  #pragma omp parallel for collapse(2) schedule(static)
  for (int y = 0; y < rows; y++) {
    for (int x = 0; x < cols; x++) {
      uint32_t depth_idx = y * cols + x;
      const float Z = depth_data[depth_idx];
      if (std::isnan(Z)) continue;
      if (Z < _true_vehicle_radius) continue;
      point_type& pt = cloud->points[depth_idx];
      pt.y = -(x - cx) * Z / fx;
      pt.z = -(y - cy) * Z / fy;
      pt.x = Z;
    }
  }
    
  return cloud;
}

void PlannerNode::img_callback(const sm::Image::SharedPtr depth_msg) {
  if (_planner_state != PlanningStates::TRAJECTORY_CONTROL &&
      _planner_state != PlanningStates::WAITING_FOR_OPUS)
    return;
  
  rclcpp::Time time_now = this->now();

  double depth_age = time_now.seconds() - rclcpp::Time(depth_msg->header.stamp).seconds();
  
  if (depth_age > _depth_age_threshold) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                         "Depth image too old (%.3f s > %.3f s threshold), rejecting",
                         depth_age, _depth_age_threshold);
    return;
  }
  
  geometry_msgs::msg::TransformStamped world_to_body, body_to_world;
  geometry_msgs::msg::Point position_world_frame;
  geometry_msgs::msg::Vector3 velocity_world_frame;
  geometry_msgs::msg::Vector3 velocity_body_frame;
  double state_timestamp;

  // Snapshot _state under the lock, then release. TF2 has its own internal
  // lock, so holding state_mutex_ across TF2 / doTransform calls would only
  // serialise odom updates against planning for no benefit.
  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    state_timestamp = _state.t;
    position_world_frame = _state.pose.position;
    if (_runtime_mode == RuntimeModes::OMNIDRONES) {
      velocity_world_frame = _state.velocity.linear;
    } else if (_runtime_mode == RuntimeModes::MAVROS) {
      velocity_body_frame = _state.velocity.linear;
    }
  }

  double state_age = time_now.seconds() - state_timestamp;
  if (state_age > _state_age_threshold) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
          "State data too old (%.3f s > %.3f s threshold), rejecting",
          state_age, _state_age_threshold);
    return;
  }

  try {
    body_to_world = to_world_buffer->lookupTransform(
      _world_frame, _vehicle_frame, tf2::TimePointZero);
    world_to_body = to_vehicle_buffer->lookupTransform(
      _vehicle_frame, _world_frame, tf2::TimePointZero);
  } catch (tf2::TransformException& ex) {
    RCLCPP_WARN(this->get_logger(), "%s", ex.what());
    return;
  }

  rclcpp::Time body_to_world_time = rclcpp::Time(body_to_world.header.stamp);
  rclcpp::Time world_to_body_time = rclcpp::Time(world_to_body.header.stamp);
  double b2w_age = (time_now - body_to_world_time).seconds();
  double w2b_age = (time_now - world_to_body_time).seconds();
  if (b2w_age > _transform_age_threshold || w2b_age > _transform_age_threshold) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
          "Transform too old (b2w: %.3f s, w2b: %.3f s > %.3f s threshold), rejecting",
          b2w_age, w2b_age, _transform_age_threshold);
    return;
  }

  if (_runtime_mode == RuntimeModes::OMNIDRONES) {
    tf2::doTransform(velocity_world_frame, velocity_body_frame, world_to_body);
  }

  if (std::fabs(velocity_body_frame.y) > _vel_planning_threshold || std::fabs(velocity_body_frame.z) > _vel_planning_threshold)
    return;

  geometry_msgs::msg::Vector3 velocity_camera_frame, acceleration_camera_frame;
  // Transform from body frame (FLU) to camera frame (RDF)
  frame_transform::transform_body_to_camera(velocity_body_frame, velocity_camera_frame);

  ruckig::InputParameter<3> initial_state_camera_frame;
  double forward_time = _planning_cycle_time + (time_now.seconds() - state_timestamp);
  initial_state_camera_frame.current_position = {
    velocity_camera_frame.x * forward_time,
    velocity_camera_frame.y * forward_time,
    velocity_camera_frame.z * forward_time
  };
  initial_state_camera_frame.current_velocity = {velocity_camera_frame.x, velocity_camera_frame.y, velocity_camera_frame.z};
  // Clamp near-zero negative forward velocity to prevent spurious monotonic
  // rejection: noise-level backward velocity (between 0 and -1e-3 m/s) falls
  // in the deadband between zero and the allow_non_monotonic threshold in
  // du_planner, causing 100% trajectory rejection while hovering.
  if (initial_state_camera_frame.current_velocity[2] > -1e-3 &&
      initial_state_camera_frame.current_velocity[2] < 0.0) {
    initial_state_camera_frame.current_velocity[2] = 0.0;
    initial_state_camera_frame.current_position[2] = 0.0;
  }
  initial_state_camera_frame.target_velocity = {0.0, 0.0, 0.0};
  initial_state_camera_frame.max_velocity = {_max_velocity_x, _max_velocity_y, _max_velocity_z};
  initial_state_camera_frame.max_acceleration = {_max_acceleration_x, _max_acceleration_y, _max_acceleration_z};
  if (_collision_checking_method == CollisionCheckingMethod::PYRAMID) {
    initial_state_camera_frame.current_acceleration = {acceleration_camera_frame.x, acceleration_camera_frame.y, acceleration_camera_frame.z};
    initial_state_camera_frame.target_acceleration = {0.0, 0.0, 0.0};
    initial_state_camera_frame.max_jerk = {15.0, 15.0, 10.0};
  }

  geometry_msgs::msg::PointStamped goal_in_camera_frame, goal_in_world_frame, goal_in_body_frame;
  goal_in_world_frame.header.frame_id = _world_frame;
  goal_in_world_frame.point = _goal_in_world_frame;
  try {
    tf2::doTransform(goal_in_world_frame, goal_in_body_frame, world_to_body);
  } catch (tf2::TransformException& ex) {
    RCLCPP_WARN(this->get_logger(), "Transform failure: %s", ex.what());
  }
  // Transform goal from body (FLU) to camera frame (RDF)
  frame_transform::transform_body_to_camera(goal_in_body_frame.point, goal_in_camera_frame.point);
  
  // Debug: Log all frame transforms
  // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
  //   "Goal transforms - world=[%.2f,%.2f,%.2f] body=[%.2f,%.2f,%.2f] camera=[%.2f,%.2f,%.2f]",
  //   goal_in_world_frame.point.x, goal_in_world_frame.point.y, goal_in_world_frame.point.z,
  //   goal_in_body_frame.point.x, goal_in_body_frame.point.y, goal_in_body_frame.point.z,
  //   goal_in_camera_frame.point.x, goal_in_camera_frame.point.y, goal_in_camera_frame.point.z);
  
  Eigen::Vector3d goal_vector_camera_frame(goal_in_camera_frame.point.x,
                                     goal_in_camera_frame.point.y,
                                     goal_in_camera_frame.point.z);

  cv::Mat depth_mat = preprocess_depth_image(depth_msg);
  
  // Use camera intrinsics from config (unified for both modes)
  double cx = _real_cx;
  double cy = _real_cy;
  double fy = _real_focal_length;

  // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
  //   "Camera params - fy: %.2f, cx: %.2f, cy: %.2f, cols: %d, rows: %d",
  //   fy, cx, cy, depth_mat.cols, depth_mat.rows);
  
  PinholeCamera camera(fy, cx, cy, depth_mat.cols, depth_mat.rows,
                       _depth_uncertainty_coeffs, _true_vehicle_radius,
                       _planning_vehicle_radius, _minimum_clear_distance);

  RandomTrajectorySampler trajectory_sampler(
    camera, _depth_upper_bound, _depth_lower_bound, goal_vector_camera_frame,
    _depth_sampling_margin, body_to_world, _goal_in_world_frame, _world_frame,
    _3d_planning, _2d_z_margin, _is_spiral_sampling, _spiral_sampling_step);

  DuPlanner planner(depth_mat, camera, _collision_checking_method,
                    _checking_time_ratio, _sampled_trajectories_threshold,
                    _checked_trajectories_threshold, _debug_num_trajectories,
                    _collision_probability_threshold, _openmp_chunk_size);
  ruckig::Trajectory<3> opt_traj;

  ExplorationCost exploration_cost(goal_vector_camera_frame, _traveling_cost);

  // OPUS coordination: manage submission protocol (non-blocking).
  // We always plan locally (below) and only engage the OPUS protocol
  // when opus_submission_needed_ is set by update_reference_trajectory().
  // Gate by planner state: drones in GO_TO_GOAL / FINISHED / BRAKE / LAND
  // must not request new locks — their CANCEL already removed them from
  // the coordinator's trajectory database.
  bool opus_ready_to_submit = false;
  const auto state_now = _planner_state.load();
  const bool opus_state_active =
      (state_now == PlanningStates::TRAJECTORY_CONTROL ||
       state_now == PlanningStates::WAITING_FOR_OPUS ||
       state_now == PlanningStates::ALIGNING_HEADING ||
       state_now == PlanningStates::HOLDING_WAYPOINT);
  if (opus_enabled_ && opus_state_active) {
    const std::lock_guard<std::mutex> lock(opus_mutex_);

    // Ack timeout: if we submitted but haven't received an ack within
    // the timeout, the message was likely lost (zenoh drop). Clear
    // check_pending and re-arm submission so the next cycle re-requests.
    if (opus_check_pending_ &&
        opus_grant_time_ != std::chrono::steady_clock::time_point{}) {
      const double elapsed = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - opus_grant_time_).count();
      if (elapsed > opus_ack_timeout_) {
        RCLCPP_WARN(this->get_logger(),
            "OPUS: Ack timeout after %.1fs — clearing and re-requesting",
            elapsed);
        opus_check_pending_ = false;
        opus_granted_ = false;
        opus_lock_pending_ = false;
        opus_submission_needed_ = true;
        opus_grant_time_ = std::chrono::steady_clock::time_point{};
        opus_pre_queue_.reset();
      }
    }

    if (!opus_submission_needed_) {
      // Not in submission mode — plan locally below
    } else if (opus_check_pending_) {
      // Still waiting for ack — plan locally below (don't return)
    } else if (opus_granted_) {
      opus_ready_to_submit = true;
    } else if (!opus_lock_pending_) {
      opus_send_lock_request();
      // Plan locally below while waiting for grant via /opus/status
    }
    // Lock pending (request in flight) — plan locally below
  }

  // OPUS fast-path: if granted, try submitting the latest pre-queued
  // trajectory directly (no fresh planning needed — GCS does collision check).
  if (opus_ready_to_submit) {
    OpusPreQueueEntry fast_entry;
    bool have_entry = false;
    {
      std::lock_guard<std::mutex> olock(opus_mutex_);
      // Re-verify grant still held (could have been revoked by status callback)
      if (!opus_granted_) {
        opus_ready_to_submit = false;
      } else if (opus_pre_queue_.has_value()) {
        fast_entry = opus_pre_queue_.value();
        have_entry = true;
      }
    }

    if (!opus_ready_to_submit) {
      // Grant was revoked between initial check and here — fall through to local planning
    } else if (have_entry) {
      RCLCPP_INFO(this->get_logger(),
        "OPUS: Submitting pre-queued trajectory (GCS collision check)");
      const rclcpp::Time submit_time = this->now();
      opus_submit_trajectory(fast_entry.trajectory,
                             fast_entry.body_to_world, fast_entry.world_position);
      {
        std::lock_guard<std::mutex> olock(opus_mutex_);
        opus_pending_trajectory_ = fast_entry.trajectory;
        opus_submission_time_ = submit_time;
        opus_check_pending_ = true;
        opus_submission_needed_ = false;
        opus_pre_queue_.reset();
      }
      return;  // Submitted — wait for ack on /opus/trajectory_ack
    }
    // Pre-queue empty — plan fresh below, then submit
  }

  if (!planner.find_lowest_cost_trajectory(
        initial_state_camera_frame, opt_traj, trajectory_sampler,
        _planning_cycle_time, &exploration_cost,
        &ExplorationCost::get_cost_wrapper)) {
    // RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
    //   "find_lowest_cost_trajectory FAILED - no valid trajectory found");
    if ((this->now() - _reference_trajectory_start_time).seconds() > 2.5 &&
        _planner_state == PlanningStates::TRAJECTORY_CONTROL && !_steered) {
      const std::lock_guard<std::mutex> lock(trajectory_mutex_);
      steering_value = planner.get_steering() / 8;
      _steered = true;
    }
    // Keep the OPUS lock while retrying local replanning. The coordinator
    // should only release the global lock on submit, reset, or timeout.
    if (opus_enabled_ && opus_submission_needed_ && opus_ready_to_submit &&
        opus_should_abort_replanning()) {
      opus_abort_planning(
        "No feasible trajectory found within local OPUS replanning timeout");
      // Re-arm so the drone retries on the next planning cycle.
      // Unlike terminal aborts (Goal reached, Brake, Land), a replan
      // timeout is transient — the situation may change next frame.
      const std::lock_guard<std::mutex> olock(opus_mutex_);
      opus_submission_needed_ = true;
    }
    return;
  }
  // New traj generated — assign transforms (opt_traj is local, no lock needed)
  opt_traj.assign_body_to_world_transform(body_to_world);
  opt_traj.assign_world_to_body_transform(world_to_body);

  if (opus_enabled_) {
    // Re-verify grant under lock before submitting freshly planned trajectory
    if (opus_ready_to_submit) {
      std::lock_guard<std::mutex> olock(opus_mutex_);
      if (!opus_granted_) {
        opus_ready_to_submit = false;
      }
    }
    if (opus_ready_to_submit) {
      // Submit directly — GCS performs collision check
      const rclcpp::Time submit_time = this->now();
      opus_submit_trajectory(opt_traj, body_to_world, position_world_frame);
      {
        const std::lock_guard<std::mutex> olock(opus_mutex_);
        opus_pending_trajectory_ = opt_traj;
        opus_submission_time_ = submit_time;
        opus_check_pending_ = true;
        opus_submission_needed_ = false;
        opus_pre_queue_.reset();
      }
      return;  // Wait for service response before pushing to queue
    }

    // Not in submission mode — overwrite pre-queue with latest trajectory
    {
      const std::lock_guard<std::mutex> olock(opus_mutex_);
      opus_pre_queue_ = OpusPreQueueEntry{opt_traj, body_to_world, position_world_frame};
    }
    return;
  }

  // Non-OPUS mode: push trajectory directly to execution queue
  {
    const std::lock_guard<std::mutex> tlock(trajectory_mutex_);
    steering_value = 0.0f;
    _steered = false;
    trajectory_queue_.push_back(opt_traj);
  }
}
