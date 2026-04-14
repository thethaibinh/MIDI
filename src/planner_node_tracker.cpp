#include "planner_node.hpp"

using namespace quadrotor_common;

void PlannerNode::update_reference_trajectory() {
  // trajectory_queue_ is mutated by img_callback (default group) and by the
  // OPUS service response (reentrant group); the control timer now runs on
  // its own group, so all three can race without this lock.
  const std::lock_guard<std::mutex> lock(trajectory_mutex_);

  if (trajectory_queue_.empty()) {
    // OPUS mode: if no approved trajectory and not already requesting,
    // trigger OPUS submission so the next img_callback can submit.
    if (opus_enabled_ && had_reference_trajectory && !opus_submission_needed_) {
      rclcpp::Time wall_time_now = this->now();
      rclcpp::Duration trajectory_point_time = wall_time_now - _reference_trajectory_start_time;
      double point_time = trajectory_point_time.seconds();
      if (point_time > (reference_trajectory_.get_duration() / _replan_factor)) {
        const std::lock_guard<std::mutex> olock(opus_mutex_);
        if (!opus_check_pending_) {
          opus_submission_needed_ = true;
          RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "OPUS: Replan trigger — requesting swarm coordination");
        }
      }
    }
    return;
  }

  while (trajectory_queue_.size() > 1) {
    trajectory_queue_.pop_front();
  }

  rclcpp::Time wall_time_now = this->now();
  rclcpp::Duration trajectory_point_time = wall_time_now - _reference_trajectory_start_time;
  double point_time = trajectory_point_time.seconds();
  if (trajectory_queue_.size() > 0) {
    // Only track when there is a valid trajectory
    if (!had_reference_trajectory) {
      _steered = false;
      steering_value = 0.0f;
      reference_trajectory_ = trajectory_queue_.front();
      _reference_trajectory_start_time = wall_time_now;
      had_reference_trajectory = true;
    }
    if (point_time > (reference_trajectory_.get_duration() / _replan_factor)) {
      _steered = false;
      steering_value = 0.0f;
      reference_trajectory_ = trajectory_queue_.front();
      _reference_trajectory_start_time = wall_time_now;
    }
    trajectory_queue_.pop_front();
  }
}

void PlannerNode::control_loop() {
  update_reference_trajectory();
  // Hold state_mutex_ across the control-loop helpers: they dereference _state
  // fields directly in multiple places, and odom/pose/twist callbacks now run
  // on a dedicated thread. The critical section is short (no I/O, no TF
  // lookups), so odom updates block at most for microseconds.
  const std::lock_guard<std::mutex> lock(state_mutex_);
  update_planner_state();
  track_trajectory();
}

void PlannerNode::track_trajectory() {
  // Don't track trajectory in non-flight states
  if (_planner_state == PlanningStates::LAND ||
      _planner_state == PlanningStates::OFF ||
      !_goal_set)
    return;

  // ALIGNING_HEADING, HOLDING_WAYPOINT, WAITING_FOR_OPUS: hold position, rotate toward goal
  if (_planner_state == PlanningStates::ALIGNING_HEADING ||
      _planner_state == PlanningStates::HOLDING_WAYPOINT ||
      _planner_state == PlanningStates::WAITING_FOR_OPUS) {
    if (_runtime_mode == RuntimeModes::MAVROS) {
      TrajectoryPoint hold_point;
      hold_point.position = geometryToEigen(_state.pose.position);
      hold_point.velocity = Eigen::Vector3d(0.0, 0.0, 0.0);
      hold_point.acceleration = Eigen::Vector3d(0.0, 0.0, 0.0);
      hold_point.heading = _goal_heading;
      public_ref_pos(hold_point);
    } else if (_runtime_mode == RuntimeModes::OMNIDRONES) {
      TrajectoryPoint hold_point;
      hold_point.position = geometryToEigen(_state.pose.position);
      hold_point.velocity = Eigen::Vector3d(0.0, 0.0, 0.0);
      hold_point.acceleration = Eigen::Vector3d(0.0, 0.0, 0.0);
      hold_point.heading = _goal_heading;
      public_ref_pos(hold_point);
    }
    return;
  }
  
  // For TRAJECTORY_CONTROL state, we need a reference trajectory
  // But for TAKING_OFF and GO_TO_GOAL states, we can publish setpoints without trajectory
  if (!had_reference_trajectory && _runtime_mode == RuntimeModes::MAVROS)
    return;

  double control_command_delay = 0.0;
  rclcpp::Time wall_time_now = this->now();
  rclcpp::Time command_execution_time = wall_time_now + rclcpp::Duration::from_seconds(control_command_delay);

  // Initialize reference_point with current state to avoid uninitialized values
  TrajectoryPoint reference_point;
  reference_point.position = geometryToEigen(_state.pose.position);
  reference_point.velocity = Eigen::Vector3d(0.0, 0.0, 0.0);
  reference_point.acceleration = Eigen::Vector3d(0.0, 0.0, 0.0);
  reference_point.heading = 0.0;

  if (_planner_state == PlanningStates::TAKING_OFF || (_planner_state == PlanningStates::TRAJECTORY_CONTROL && !had_reference_trajectory)) {
    _reference_trajectory_start_time = command_execution_time;
    steering_value = 0.0f;
    if (_runtime_mode == RuntimeModes::MAVROS)
      return;
    if (_runtime_mode == RuntimeModes::OMNIDRONES) {
      // Takeoff to goal altitude at current XY position
      reference_point.position = Eigen::Vector3d(_home_in_world_frame.x, _home_in_world_frame.y, _goal_in_world_frame.z);
    }
  } else if ((_planner_state == PlanningStates::TRAJECTORY_CONTROL && had_reference_trajectory) || _planner_state == PlanningStates::GO_TO_GOAL) {
    rclcpp::Duration trajectory_point_time = command_execution_time - _reference_trajectory_start_time;
    double point_time = trajectory_point_time.seconds();
    get_reference_point_at_time(reference_trajectory_, point_time, reference_point);
  }

  // Publish position/velocity setpoint (kinematic control mode)
  if (_runtime_mode == RuntimeModes::MAVROS) {
    public_ref_pos(reference_point);
  } else if (_runtime_mode == RuntimeModes::OMNIDRONES) {
    // OmniDrones supports both position and velocity control
    // Use PositionTarget for full state control (position + velocity + acceleration)
    public_ref_pos(reference_point);
  }
}

void PlannerNode::public_ref_pos(const TrajectoryPoint& reference_point) {
  // Debug: Log current state, goal, and setpoint for coordinate debugging
  // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
  //   "DEBUG COORDS - State: (%.2f, %.2f, %.2f) | Goal: (%.2f, %.2f, %.2f) | Setpoint: (%.2f, %.2f, %.2f)",
  //   _state.pose.position.x, _state.pose.position.y, _state.pose.position.z,
  //   _goal_in_world_frame.x, _goal_in_world_frame.y, _goal_in_world_frame.z,
  //   reference_point.position(0), reference_point.position(1), reference_point.position(2));
  
  // Check fence limits - reject setpoints outside safe bounds (in ENU)
  const double x = reference_point.position(0);
  const double y = reference_point.position(1);
  const double z = reference_point.position(2);
  
  mavros_msgs::msg::PositionTarget msg;
  msg.header.stamp = this->now();
  msg.coordinate_frame = 1;  // FRAME_LOCAL_NED (but MAVROS actually accepts ENU here - legacy behavior)
  msg.type_mask = 0;
  
  bool outside_fence = (x < _fence_min_x || x > _fence_max_x ||
                        y < _fence_min_y || y > _fence_max_y ||
                        z < _fence_min_z || z > _fence_max_z);
  
  if (outside_fence) {
    // Fence breach: publish last valid position with zero velocity/acceleration to stop the drone
    if (!_has_valid_setpoint) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
        "Setpoint (%.2f, %.2f, %.2f) ENU outside fence limits, no valid setpoint to fallback to", x, y, z);
      return;
    }
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
      "Setpoint (%.2f, %.2f, %.2f) ENU outside fence limits, stopping at last valid pos (%.2f, %.2f, %.2f)",
      x, y, z, _last_valid_position(0), _last_valid_position(1), _last_valid_position(2));
    
    // Use last valid position with zero velocity/acceleration (position-only mode to stop)
    msg.position.x = _last_valid_position(0);
    msg.position.y = _last_valid_position(1);
    msg.position.z = _last_valid_position(2);
    msg.velocity.x = 0.0;
    msg.velocity.y = 0.0;
    msg.velocity.z = 0.0;
    msg.acceleration_or_force.x = 0.0;
    msg.acceleration_or_force.y = 0.0;
    msg.acceleration_or_force.z = 0.0;
    msg.yaw = _last_valid_heading;
    // Force position-only type_mask to ensure drone stops
    msg.type_mask = 8 + 16 + 32 + 64 + 128 + 256 + 2048;  // = 2552 (ignore vel, accel, yaw_rate)
  } else {
    // Valid setpoint: store it and publish normally
    _last_valid_position = reference_point.position;
    _last_valid_heading = reference_point.heading;
    _has_valid_setpoint = true;
    
    // MAVROS setpoint_raw/local accepts ENU directly despite coordinate_frame=NED (legacy quirk)
    msg.position.x = reference_point.position(0);
    msg.position.y = reference_point.position(1);
    msg.position.z = reference_point.position(2);
    if (_setpoint_type == SetpointTypes::POSITION_ONLY) {
      // type_mask: ignore velocity (8+16+32), acceleration (64+128+256), yaw_rate (2048)
      // This tells ArduPilot to only use position + yaw
      msg.type_mask = 8 + 16 + 32 + 64 + 128 + 256 + 2048;  // = 2552
      msg.velocity.x = 0.0;
      msg.velocity.y = 0.0;
      msg.velocity.z = 0.0;
      msg.acceleration_or_force.x = 0.0;
      msg.acceleration_or_force.y = 0.0;
      msg.acceleration_or_force.z = 0.0;
    } else if (_setpoint_type == SetpointTypes::FULL_STATE) {
      // Use all fields - position, velocity, acceleration, yaw
      // msg.type_mask = 2048;  // Only ignore yaw_rate
      msg.velocity.x = reference_point.velocity(0);
      msg.velocity.y = reference_point.velocity(1);
      msg.velocity.z = reference_point.velocity(2);
      msg.acceleration_or_force.x = reference_point.acceleration(0);
      msg.acceleration_or_force.y = reference_point.acceleration(1);
      msg.acceleration_or_force.z = reference_point.acceleration(2);
    }
    // Yaw in ENU: atan2(North, East) gives 0° = East, 90° = North (CCW positive)
    // This matches MAVROS ENU convention - no offset needed
    msg.yaw = reference_point.heading;
  }
  
  // Debug: Log the actual setpoint being published
  // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
  //   "PUBLISHING to setpoint_raw/local: pos=(%.2f,%.2f,%.2f) yaw=%.1f deg | vel=(%.2f,%.2f,%.2f)",
  //   msg.position.x, msg.position.y, msg.position.z, msg.yaw * 180.0 / M_PI,
  //   msg.velocity.x, msg.velocity.y, msg.velocity.z);
  
  raw_ref_pos_pub->publish(msg);
}

void PlannerNode::get_reference_point_at_time(
  const ruckig::Trajectory<3>& reference_trajectory, const double& _point_time,
  TrajectoryPoint& reference_point) {

  const double point_time = std::clamp(_point_time, 0.0, reference_trajectory.get_duration());

  geometry_msgs::msg::TransformStamped body_to_world =
    reference_trajectory.get_transform_to_world();

  // Debug: Check if transform is valid
  // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
  //   "get_reference_point_at_time: body_to_world translation=[%.4f, %.4f, %.4f]",
  //   body_to_world.transform.translation.x,
  //   body_to_world.transform.translation.y,
  //   body_to_world.transform.translation.z);

  std::array<double, 3> position_in_camera_frame, velocity_in_camera_frame,
    acceleration_in_camera_frame, jerk_in_camera_frame;
  size_t num_section;
  reference_trajectory.at_time(
    point_time, position_in_camera_frame, velocity_in_camera_frame,
    acceleration_in_camera_frame, jerk_in_camera_frame, num_section);

  // Debug: Log position in camera frame
  // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
  //   "  pos_camera_frame=[%.4f, %.4f, %.4f]",
  //   position_in_camera_frame[0], position_in_camera_frame[1], position_in_camera_frame[2]);

  geometry_msgs::msg::Point position_in_body_frame, position_in_world_frame;
  geometry_msgs::msg::Vector3 velocity_in_body_frame, velocity_in_world_frame,
    acceleration_in_body_frame, acceleration_in_world_frame, jerk_in_body_frame, jerk_in_world_frame;

  // Transform from camera frame (RDF) to body frame (FLU)
  // Camera RDF: X=Right, Y=Down, Z=Forward
  // Body FLU: X=Forward, Y=Left, Z=Up
  frame_transform::transform_camera_to_body(position_in_camera_frame, position_in_body_frame);
  frame_transform::transform_camera_to_body(velocity_in_camera_frame, velocity_in_body_frame);
  frame_transform::transform_camera_to_body(acceleration_in_camera_frame, acceleration_in_body_frame);
  frame_transform::transform_camera_to_body(jerk_in_camera_frame, jerk_in_body_frame);

  try {
    tf2::doTransform(position_in_body_frame, position_in_world_frame, body_to_world);
    tf2::doTransform(velocity_in_body_frame, velocity_in_world_frame, body_to_world);
    tf2::doTransform(acceleration_in_body_frame, acceleration_in_world_frame, body_to_world);
    tf2::doTransform(jerk_in_body_frame, jerk_in_world_frame, body_to_world);
  } catch (tf2::TransformException& ex) {
    RCLCPP_WARN(this->get_logger(), "Transform failure: %s", ex.what());
  }

  // Debug: Log frame transformations
  // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
  //   "DEBUG TRAJ - cam:(%.2f,%.2f,%.2f) -> body:(%.2f,%.2f,%.2f) -> world:(%.2f,%.2f,%.2f) | b2w_t:(%.2f,%.2f,%.2f)",
  //   position_in_camera_frame[0], position_in_camera_frame[1], position_in_camera_frame[2],
  //   position_in_body_frame.x, position_in_body_frame.y, position_in_body_frame.z,
  //   position_in_world_frame.x, position_in_world_frame.y, position_in_world_frame.z,
  //   body_to_world.transform.translation.x, body_to_world.transform.translation.y, body_to_world.transform.translation.z);

  // Eigen::Vector3d trajectory_vector =
  //   geometryToEigen(reference_trajectory.get_terminal_position_in_world_frame()) -
  //   geometryToEigen(reference_trajectory.get_initial_position_in_world_frame());
  // double terminal_heading = atan2f(trajectory_vector[1], trajectory_vector[0]);
  // reference_point.heading = terminal_heading;
  // Use the goal heading computed once when goal was set (direction from start to goal)
  reference_point.heading = _goal_heading;  
  Eigen::Vector3d current_euler_angles = quaternionToEulerAnglesZYX(geometryToEigen(_state.pose.orientation));

  // Comment to turn off steering
  // if (fabs(steering_value) > 1e-6)
  //   reference_point.heading = current_euler_angles(2) + steering_value;

  reference_point.position = geometryToEigen(position_in_world_frame);
  reference_point.velocity = geometryToEigen(velocity_in_world_frame);
  reference_point.acceleration = geometryToEigen(acceleration_in_world_frame);
  reference_point.jerk = geometryToEigen(jerk_in_world_frame);
}
