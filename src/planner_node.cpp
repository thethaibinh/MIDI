#include "planner_node.hpp"

using namespace quadrotor_common;

PlannerNode::PlannerNode()
  : Node("midi_planner"),
    steering_value(0.0),
    _steered(false),
    trajectory_queue_(),
    _planner_state(PlanningStates::OFF),
    had_reference_trajectory(false),
    _goal_set(false) {

  // Initialize TF2 buffers and broadcaster
  to_world_buffer = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  to_vehicle_buffer = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  to_world_tf2 = std::make_shared<tf2_ros::TransformListener>(*to_world_buffer);
  to_vehicle_tf2 = std::make_shared<tf2_ros::TransformListener>(*to_vehicle_buffer);
  tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

  // Load parameters
  if (!loadParameters()) {
    RCLCPP_ERROR(this->get_logger(), "Could not load parameters.");
    rclcpp::shutdown();
    return;
  }

  // Publishers
  point_cloud_pub = this->create_publisher<sm::PointCloud2>("/cloud_out", 10);
  visual_pub = this->create_publisher<visualization_msgs::msg::Marker>("/visualization", 10);

  // Publishers based on runtime mode
  if (_runtime_mode == RuntimeModes::MAVROS || _runtime_mode == RuntimeModes::OMNIDRONES) {
    // Position/velocity/acceleration setpoints (PositionTarget)
    raw_ref_pos_pub = this->create_publisher<mavros_msgs::msg::PositionTarget>("mavros/setpoint_raw/local", 10);
  }
  if (_runtime_mode == RuntimeModes::OMNIDRONES) {
    // Velocity-only setpoints (TwistStamped) - alternative control mode
    vel_cmd_pub = this->create_publisher<geometry_msgs::msg::TwistStamped>("mavros/setpoint_velocity/cmd_vel", 10);
  }

  // Subscribers
  image_sub = this->create_subscription<sm::Image>(
    _depth_topic, 10,
    std::bind(&PlannerNode::img_callback, this, std::placeholders::_1));

  if (_visualise) {
    visual_sub = this->create_subscription<sm::Image>(
      _depth_topic, 10,
      std::bind(&PlannerNode::visualise, this, std::placeholders::_1));
  }

  // Mission command subscriber (unified for sim and real)
  mission_sub = this->create_subscription<ground_system_msgs::msg::StartSwarmMission>(
    "/start_swarm_mission", 10,
    std::bind(&PlannerNode::mission_callback, this, std::placeholders::_1));

  reset_sub = this->create_subscription<std_msgs::msg::Empty>(
    "/reset_planner", 10,
    std::bind(&PlannerNode::reset_callback, this, std::placeholders::_1));

  // Subscribe to odometry - use relative topic so namespace remapping works
  // When running in /Drone1 namespace, this becomes /Drone1/odometry
  odom_sub = this->create_subscription<nav_msgs::msg::Odometry>(
    "odometry", 10,
    std::bind(&PlannerNode::odometry_callback, this, std::placeholders::_1));

  mav_state_sub = this->create_subscription<mavros_msgs::msg::State>(
    "mavros/state", 10,
    std::bind(&PlannerNode::ardupilot_status_callback, this, std::placeholders::_1));

  mav_pose_sub = this->create_subscription<geometry_msgs::msg::PoseStamped>(
    "mavros/local_position/pose", 10,
    std::bind(&PlannerNode::mav_pose_callback, this, std::placeholders::_1));

  mav_twist_sub = this->create_subscription<geometry_msgs::msg::TwistStamped>(
    "mavros/local_position/velocity_body", 10,
    std::bind(&PlannerNode::mav_twist_callback, this, std::placeholders::_1));

  mav_accel_sub = this->create_subscription<sensor_msgs::msg::Imu>(
    "mavros/imu/data_raw", 10,
    std::bind(&PlannerNode::mav_accel_callback, this, std::placeholders::_1));

  // MAVROS service clients (for real FC)
  arming_srv = this->create_client<mavros_msgs::srv::CommandBool>("/mavros/cmd/arming");
  takeoff_srv = this->create_client<mavros_msgs::srv::CommandTOL>("/mavros/cmd/takeoff");
  land_srv = this->create_client<mavros_msgs::srv::CommandTOL>("/mavros/cmd/land");
  mode_srv = this->create_client<mavros_msgs::srv::SetMode>("mavros/set_mode");

  // Timer
  control_loop_timer_ = this->create_wall_timer(
    std::chrono::duration<double>(_trajectory_discretisation_cycle),
    std::bind(&PlannerNode::control_loop, this));

  RCLCPP_INFO(this->get_logger(), "MIDI Planner initialized");
}

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
  cloud->header.stamp     = rclcpp::Time(depth_msg->header.stamp).nanoseconds() / 1000;
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

void PlannerNode::mission_callback(const ground_system_msgs::msg::StartSwarmMission::SharedPtr msg) {
  RCLCPP_INFO(this->get_logger(), "Received mission command: %s", msg->mission_name.c_str());
  
  if (mission_received_) {
    RCLCPP_WARN(this->get_logger(), "Mission already received, ignoring duplicate");
    return;
  }
  
  // Set goal coordinates based on coordinate frame convention
  if (_runtime_mode == RuntimeModes::OMNIDRONES) {
    // OmniDrones uses NWU (North-West-Up): X=North, Y=West, Z=Up
    _goal_in_world_frame.x = _state.pose.position.x + _goal_north_coordinate;
    _goal_in_world_frame.y = _state.pose.position.y + _goal_west_coordinate;
    _goal_in_world_frame.z = _goal_up_coordinate;
  } else {
    // MAVROS uses ENU (East-North-Up): X=East, Y=North, Z=Up
    _goal_in_world_frame.x = _state.pose.position.x - _goal_west_coordinate;
    _goal_in_world_frame.y = _state.pose.position.y + _goal_north_coordinate;
    _goal_in_world_frame.z = _goal_up_coordinate;
  }
  RCLCPP_INFO(this->get_logger(), "Setting goal to (%.2f, %.2f, %.2f)",
              _goal_in_world_frame.x, _goal_in_world_frame.y, _goal_in_world_frame.z);
  _goal_set = true;
  mission_received_ = true;
  
  if (_runtime_mode == RuntimeModes::OMNIDRONES) {
    // Simulation mode: directly start trajectory control
    RCLCPP_WARN(this->get_logger(), "[SIM] Starting navigation!");
    set_auto_pilot_state_forced(PlanningStates::TAKING_OFF);
    steering_value = 0.0f;
    _steered = false;
    trajectory_queue_.clear();
    reference_trajectory_ = ruckig::Trajectory<3>();
    had_reference_trajectory = false;
  } else if (_runtime_mode == RuntimeModes::MAVROS) {
    // Real FC mode: will initiate GUIDED->ARM->TAKEOFF sequence in update_planner_state()
    RCLCPP_WARN(this->get_logger(), "[MAVROS] Mission received, initiating flight sequence...");
  }
}

void PlannerNode::reset_callback(const std_msgs::msg::Empty::SharedPtr msg) {
  (void)msg;
  RCLCPP_WARN(this->get_logger(), "Planner: Reset quadrotor!");
  set_auto_pilot_state_forced(PlanningStates::OFF);
  steering_value = 0.0f;
  _steered = false;
  _goal_set = false;
  mission_received_ = false;
  mode_switch_pending_ = false;
  arming_pending_ = false;
  takeoff_pending_ = false;
  land_pending_ = false;
  trajectory_queue_.clear();
  reference_trajectory_ = ruckig::Trajectory<3>();
  had_reference_trajectory = false;
}

void PlannerNode::ardupilot_status_callback(const mavros_msgs::msg::State::SharedPtr msg) {
  flight_controller_status = *msg;
}

void PlannerNode::mav_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
  _latest_pose_stamp = rclcpp::Time(msg->header.stamp);

  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    _state.pose = msg->pose;
    auto min_stamp = std::min({_latest_pose_stamp, _latest_twist_stamp, _latest_accel_stamp});
    _state.t = min_stamp.seconds();
  }
}

void PlannerNode::mav_twist_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg) {
  _latest_twist_stamp = rclcpp::Time(msg->header.stamp);

  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    _state.velocity.linear = msg->twist.linear;
    _state.velocity.angular = msg->twist.angular;
    _state.t = std::min(_latest_pose_stamp, _latest_twist_stamp).seconds();
  }
}

void PlannerNode::mav_accel_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
  // Acceleration callback - only used for PYRAMID collision checking which is removed
  // For MIDI collision checking (CPU-only), we don't need acceleration
  (void)msg;  // Suppress unused parameter warning
  return;
}

void PlannerNode::odometry_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
  const std::lock_guard<std::mutex> lock(state_mutex_);

  _state.t = rclcpp::Time(msg->header.stamp).seconds();
  _state.pose = msg->pose.pose;
  _state.velocity = msg->twist.twist;
  // Note: acceleration can be computed from velocity if needed, but not used in MIDI method

  // Debug: Log odometry with yaw periodically
  // Convert quaternion to yaw (rotation around Z axis)
  double qw = msg->pose.pose.orientation.w;
  double qx = msg->pose.pose.orientation.x;
  double qy = msg->pose.pose.orientation.y;
  double qz = msg->pose.pose.orientation.z;
  double yaw_rad = atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
  double yaw_deg = yaw_rad * 180.0 / M_PI;
  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
    "Odometry: pos=(%.2f, %.2f, %.2f), yaw=%.1f deg (%.2f rad)",
    msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z,
    yaw_deg, yaw_rad);

  // Publish TF transform (world_frame -> vehicle_frame) from odometry
  // This allows MIDI to use TF internally without relying on external TF publishers
  geometry_msgs::msg::TransformStamped tf_msg;
  tf_msg.header.stamp = msg->header.stamp;
  tf_msg.header.frame_id = _world_frame;
  tf_msg.child_frame_id = _vehicle_frame;
  tf_msg.transform.translation.x = msg->pose.pose.position.x;
  tf_msg.transform.translation.y = msg->pose.pose.position.y;
  tf_msg.transform.translation.z = msg->pose.pose.position.z;
  tf_msg.transform.rotation = msg->pose.pose.orientation;
  tf_broadcaster_->sendTransform(tf_msg);
}

void PlannerNode::update_reference_trajectory() {
  if (trajectory_queue_.empty()) return;

  while (trajectory_queue_.size() > 1) {
    trajectory_queue_.pop_front();
  }

  rclcpp::Time wall_time_now = this->now();
  rclcpp::Duration trajectory_point_time = wall_time_now - _reference_trajectory_start_time;
  double point_time = trajectory_point_time.seconds();
  if (trajectory_queue_.size()) {
    if (!had_reference_trajectory) {
      asign_reference_trajectory(wall_time_now);
      had_reference_trajectory = true;
    }
    if (point_time > (reference_trajectory_.get_duration() / _replan_factor)) {
      asign_reference_trajectory(wall_time_now);
    }
    trajectory_queue_.pop_front();
  }
}

void PlannerNode::asign_reference_trajectory(rclcpp::Time wall_time_now) {
  const std::lock_guard<std::mutex> lock(trajectory_mutex_);
  _steered = false;
  steering_value = 0.0f;
  reference_trajectory_ = trajectory_queue_.front();
  _reference_trajectory_start_time = wall_time_now;
}

void PlannerNode::control_loop() {
  update_reference_trajectory();
  update_planner_state();
  track_trajectory();
}

void PlannerNode::update_planner_state() {
  // For MAVROS mode: Handle FC startup sequence (GUIDED -> ARM -> TAKEOFF)
  if (_runtime_mode == RuntimeModes::MAVROS) {
    // Step 1: Switch to GUIDED mode if not already
    if (flight_controller_status.mode != "GUIDED" && !mode_switch_pending_) {
      if (mode_srv->service_is_ready()) {
        auto request = std::make_shared<mavros_msgs::srv::SetMode::Request>();
        request->custom_mode = "GUIDED";
        mode_switch_pending_ = true;
        
        mode_srv->async_send_request(request,
          [this](rclcpp::Client<mavros_msgs::srv::SetMode>::SharedFuture future) {
            mode_switch_pending_ = false;
            try {
              auto response = future.get();
              if (response->mode_sent) {
                RCLCPP_INFO(this->get_logger(), "GUIDED mode request sent");
              } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to send GUIDED mode request");
              }
            } catch (const std::exception& e) {
              RCLCPP_ERROR(this->get_logger(), "Mode switch service failed: %s", e.what());
            }
          });
      }
      return;  // Wait for mode switch
    }
    
    // Step 2: Arm if in GUIDED but not armed
    if (flight_controller_status.mode == "GUIDED" && !flight_controller_status.armed && !arming_pending_) {
      if (arming_srv->service_is_ready()) {
        auto request = std::make_shared<mavros_msgs::srv::CommandBool::Request>();
        request->value = true;
        arming_pending_ = true;
        
        arming_srv->async_send_request(request,
          [this](rclcpp::Client<mavros_msgs::srv::CommandBool>::SharedFuture future) {
            arming_pending_ = false;
            try {
              auto response = future.get();
              if (response->success) {
                RCLCPP_INFO(this->get_logger(), "Arming command accepted");
              } else {
                RCLCPP_ERROR(this->get_logger(), "Arming command rejected");
              }
            } catch (const std::exception& e) {
              RCLCPP_ERROR(this->get_logger(), "Arming service failed: %s", e.what());
            }
          });
      }
      return;  // Wait for arming
    }
    
    // Step 3: Takeoff if armed
    if (flight_controller_status.mode == "GUIDED" && flight_controller_status.armed && !takeoff_pending_) {
      if (takeoff_srv->service_is_ready()) {
        auto request = std::make_shared<mavros_msgs::srv::CommandTOL::Request>();
        request->altitude = _goal_in_world_frame.z;
        takeoff_pending_ = true;
        
        takeoff_srv->async_send_request(request,
          [this](rclcpp::Client<mavros_msgs::srv::CommandTOL>::SharedFuture future) {
            takeoff_pending_ = false;
            try {
              auto response = future.get();
              if (response->success) {
                RCLCPP_WARN(this->get_logger(), "Takeoff command accepted!");
                set_auto_pilot_state_forced(PlanningStates::TAKING_OFF);
              } else {
                RCLCPP_ERROR(this->get_logger(), "Takeoff command rejected by FCU");
              }
            } catch (const std::exception& e) {
              RCLCPP_ERROR(this->get_logger(), "Takeoff service failed: %s", e.what());
            }
          });
      }
    }
    return;  // Don't proceed with other state logic while waiting for FC
  }

  // Handle disarm detection (for real FC)
  if (_runtime_mode == RuntimeModes::MAVROS && 
      _planner_state != PlanningStates::OFF &&
      !flight_controller_status.armed) {
    RCLCPP_WARN(this->get_logger(), "Vehicle disarmed, resetting planner");
    reset_callback(nullptr);
    return;
  }

  if (!_goal_set) return;

  geometry_msgs::msg::Point goal_in_world_frame = _goal_in_world_frame;
  goal_in_world_frame.z = _state.pose.position.z;
  double distance_to_goal = (geometryToEigen(_state.pose.position) - geometryToEigen(goal_in_world_frame)).norm();

  // Transition from START to TRAJECTORY_CONTROL when altitude reached
  // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Current altitude: %.2f, Goal altitude: %.2f", _state.pose.position.z, _goal_in_world_frame.z);
  // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Current planner state: %d", static_cast<int>(_planner_state));
  if (_state.pose.position.z >= (_goal_in_world_frame.z - 0.1) && _planner_state == PlanningStates::TAKING_OFF) {
    // RCLCPP_INFO(this->get_logger(), "New state!");
    set_auto_pilot_state_forced(PlanningStates::TRAJECTORY_CONTROL);
  }
  // Transition to GO_TO_GOAL when near goal
  else if (_planner_state == PlanningStates::TRAJECTORY_CONTROL &&
           (distance_to_goal < _go_to_goal_threshold ||
            ((_state.pose.position.y + _go_to_goal_threshold / 10) > _goal_in_world_frame.y &&
             _runtime_mode == RuntimeModes::MAVROS))) {
    set_auto_pilot_state_forced(PlanningStates::GO_TO_GOAL);
    _stop_planning_point_in_world_frame = _state.pose.position;
  }
  // Land when at goal (MAVROS only)
  else if (_runtime_mode == RuntimeModes::MAVROS &&
           _planner_state == PlanningStates::GO_TO_GOAL &&
           distance_to_goal < _go_to_goal_threshold * 0.2 &&
           !land_pending_) {
    if (land_srv->service_is_ready()) {
      auto request = std::make_shared<mavros_msgs::srv::CommandTOL::Request>();
      land_pending_ = true;
      
      land_srv->async_send_request(request,
        [this](rclcpp::Client<mavros_msgs::srv::CommandTOL>::SharedFuture future) {
          land_pending_ = false;
          try {
            auto response = future.get();
            if (response->success) {
              set_auto_pilot_state_forced(PlanningStates::LAND);
              RCLCPP_WARN(this->get_logger(), "Land command accepted!");
            } else {
              RCLCPP_ERROR(this->get_logger(), "Land command rejected by FCU");
            }
          } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Land service failed: %s", e.what());
          }
        });
    }
  }
  // Reset after landing complete
  else if (_runtime_mode == RuntimeModes::MAVROS &&
           _planner_state == PlanningStates::LAND && !flight_controller_status.armed) {
    reset_callback(nullptr);
  }
}

void PlannerNode::track_trajectory() {
  // Don't track trajectory in non-flight states
  if (_planner_state == PlanningStates::LAND ||
      _planner_state == PlanningStates::OFF ||
      !_goal_set)
    return;
  if (!had_reference_trajectory && _runtime_mode == RuntimeModes::MAVROS)
    return;

  double control_command_delay = 0.0;
  rclcpp::Time wall_time_now = this->now();
  rclcpp::Time command_execution_time = wall_time_now + rclcpp::Duration::from_seconds(control_command_delay);

  TrajectoryPoint reference_point;
  if (_planner_state == PlanningStates::TAKING_OFF) {
    _reference_trajectory_start_time = command_execution_time;
    steering_value = 0.0f;
    if (_runtime_mode == RuntimeModes::MAVROS)
      return;
    if (_runtime_mode == RuntimeModes::OMNIDRONES) {
      reference_point.acceleration = Eigen::Vector3d(0.0, 0.0, 0.0);
      reference_point.velocity = Eigen::Vector3d(0.0, 0.0, 0.0);
      reference_point.position = Eigen::Vector3d(0.0, 0.0, _goal_in_world_frame.z);
    }
  } else if (_planner_state == PlanningStates::TRAJECTORY_CONTROL) {
    rclcpp::Duration trajectory_point_time = command_execution_time - _reference_trajectory_start_time;
    double point_time = trajectory_point_time.seconds();
    get_reference_point_at_time(reference_trajectory_, point_time, reference_point);
  } else if (_planner_state == PlanningStates::GO_TO_GOAL) {
    _reference_trajectory_start_time = command_execution_time;
    steering_value = 0.0f;
    reference_point.position = geometryToEigen(_goal_in_world_frame);
    reference_point.velocity = Eigen::Vector3d(0.0, 0.0, 0.0);
    reference_point.acceleration = Eigen::Vector3d(0.0, 0.0, 0.0);
    Eigen::Vector3d current_euler_angles = quaternionToEulerAnglesZYX(geometryToEigen(_state.pose.orientation));
    reference_point.heading = current_euler_angles(2);
  }

  if (_state.t - this->now().seconds() > 0.2) {
    RCLCPP_WARN(this->get_logger(), "State is too old, skipping control command");
    return;
  }

  if (_runtime_mode == RuntimeModes::MAVROS && _mavros_control_mode == MavrosControlModes::KINEMATIC) {
    public_ref_pos(reference_point);
  } else if (_runtime_mode == RuntimeModes::OMNIDRONES) {
    // OmniDrones supports both position and velocity control
    // Use PositionTarget for full state control (position + velocity + acceleration)
    public_ref_pos(reference_point);
    // Also publish velocity command for velocity-only control mode
    publish_velocity_command(reference_point);
  }
}

void PlannerNode::public_ref_pos(const TrajectoryPoint& reference_point) {
  mavros_msgs::msg::PositionTarget msg;
  msg.header.stamp = this->now();
  msg.coordinate_frame = 1;  // FRAME_LOCAL_NED
  msg.type_mask = 0;
  msg.position.x = reference_point.position(0);
  msg.position.y = reference_point.position(1);
  msg.position.z = reference_point.position(2);
  msg.velocity.x = reference_point.velocity(0);
  msg.velocity.y = reference_point.velocity(1);
  msg.velocity.z = reference_point.velocity(2);
  msg.acceleration_or_force.x = reference_point.acceleration(0);
  msg.acceleration_or_force.y = reference_point.acceleration(1);
  msg.acceleration_or_force.z = reference_point.acceleration(2);
  msg.yaw = M_PI_2;
  raw_ref_pos_pub->publish(msg);
}

void PlannerNode::publish_velocity_command(const TrajectoryPoint& reference_point) {
  geometry_msgs::msg::TwistStamped vel_cmd;
  vel_cmd.header.stamp = this->now();
  vel_cmd.header.frame_id = "map";

  // Velocity in world frame (ENU)
  vel_cmd.twist.linear.x = reference_point.velocity(0);
  vel_cmd.twist.linear.y = reference_point.velocity(1);
  vel_cmd.twist.linear.z = reference_point.velocity(2);

  // No angular velocity for simple navigation
  vel_cmd.twist.angular.x = 0.0;
  vel_cmd.twist.angular.y = 0.0;
  vel_cmd.twist.angular.z = 0.0;

  vel_cmd_pub->publish(vel_cmd);
}

void PlannerNode::get_reference_point_at_time(
  const ruckig::Trajectory<3>& reference_trajectory, const double& _point_time,
  TrajectoryPoint& reference_point) {

  const double point_time = std::clamp(_point_time, 0.0, reference_trajectory.get_duration());

  geometry_msgs::msg::TransformStamped body_to_world =
    reference_trajectory.get_transform_to_world();

  std::array<double, 3> position_in_camera_frame, velocity_in_camera_frame,
    acceleration_in_camera_frame, jerk_in_camera_frame;
  size_t num_section;
  reference_trajectory.at_time(
    point_time, position_in_camera_frame, velocity_in_camera_frame,
    acceleration_in_camera_frame, jerk_in_camera_frame, num_section);

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

  Eigen::Vector3d trajectory_vector =
    geometryToEigen(reference_trajectory.get_terminal_position_in_world_frame()) -
    geometryToEigen(reference_trajectory.get_initial_position_in_world_frame());
  double terminal_heading = atan2f(trajectory_vector[1], trajectory_vector[0]);
  reference_point.heading = terminal_heading;
  Eigen::Vector3d current_euler_angles = quaternionToEulerAnglesZYX(geometryToEigen(_state.pose.orientation));

  if (fabs(steering_value) > 1e-6)
    reference_point.heading = current_euler_angles(2) + steering_value;

  reference_point.position = geometryToEigen(position_in_world_frame);
  reference_point.velocity = geometryToEigen(velocity_in_world_frame);
  reference_point.acceleration = geometryToEigen(acceleration_in_world_frame);
  reference_point.jerk = geometryToEigen(jerk_in_world_frame);
}

void PlannerNode::set_auto_pilot_state_forced(const PlanningStates& new_state) {
  const rclcpp::Time time_now = this->now();

  if (new_state != PlanningStates::TRAJECTORY_CONTROL && !trajectory_queue_.empty()) {
    trajectory_queue_.clear();
  }
  time_of_switch_to_current_state_ = time_now;
  _planner_state = new_state;

  std::string state_name;
  switch (_planner_state) {
    case PlanningStates::OFF:
      state_name = "OFF";
      break;
    case PlanningStates::TAKING_OFF:
      state_name = "TAKING_OFF";
      break;
    case PlanningStates::TRAJECTORY_CONTROL:
      state_name = "TRAJECTORY_CONTROL";
      break;
    case PlanningStates::GO_TO_GOAL:
      state_name = "GO_TO_GOAL";
      break;
    case PlanningStates::LAND:
      state_name = "LAND";
      break;
  }
  RCLCPP_WARN(this->get_logger(), "Switched to %s state", state_name.c_str());
}

bool PlannerNode::check_valid_trajectory(
  const geometry_msgs::msg::Point& current_position,
  const ruckig::Trajectory<3>& trajectory) {
  if (trajectory.get_duration() < 1e-6) {
    RCLCPP_WARN(this->get_logger(), "The received trajectory is empty, rejecting it!");
    return false;
  }
  double pos_diff = (geometryToEigen(current_position) -
                     geometryToEigen(trajectory.get_initial_position_in_world_frame())).norm();
  if (pos_diff > kPositionJumpTolerance_) {
    RCLCPP_WARN(this->get_logger(),
      "The received trajectory does not start at current position, rejecting it!");
    return false;
  }
  return true;
}

void PlannerNode::img_callback(const sm::Image::SharedPtr depth_msg) {
  if (_planner_state != PlanningStates::TRAJECTORY_CONTROL)
    return;

  geometry_msgs::msg::TransformStamped world_to_body, body_to_world;
  geometry_msgs::msg::Point position_world_frame;
  geometry_msgs::msg::Vector3 velocity_world_frame;
  geometry_msgs::msg::Vector3 acceleration_world_frame, test_acceleration_world_frame;
  geometry_msgs::msg::Vector3 velocity_body_frame;
  geometry_msgs::msg::Vector3 acceleration_body_frame, test_acceleration_body_frame;
  test_acceleration_body_frame.x = 0.0;
  test_acceleration_body_frame.y = 0.0;
  test_acceleration_body_frame.z = 1.0;

  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    try {
      body_to_world = to_world_buffer->lookupTransform(
        _world_frame, _vehicle_frame, tf2::TimePointZero);
      world_to_body = to_vehicle_buffer->lookupTransform(
        _vehicle_frame, _world_frame, tf2::TimePointZero);
    } catch (tf2::TransformException& ex) {
      RCLCPP_WARN(this->get_logger(), "%s", ex.what());
    }
    position_world_frame = _state.pose.position;
    if (_runtime_mode == RuntimeModes::OMNIDRONES) {
      // OmniDrones: velocity and acceleration are in world frame (ENU)
      velocity_world_frame = _state.velocity.linear;
      acceleration_world_frame = _state.acceleration.linear;
      tf2::doTransform(velocity_world_frame, velocity_body_frame, world_to_body);
      tf2::doTransform(acceleration_world_frame, acceleration_body_frame, world_to_body);
    } else if (_runtime_mode == RuntimeModes::MAVROS) {
      velocity_body_frame = _state.velocity.linear;
      acceleration_body_frame = _state.acceleration.linear;
    }
    tf2::doTransform(test_acceleration_body_frame, test_acceleration_world_frame, body_to_world);
  }

  if (test_acceleration_world_frame.x > _acc_planning_threshold || test_acceleration_world_frame.y > _acc_planning_threshold)
    return;

  geometry_msgs::msg::Vector3 velocity_camera_frame, acceleration_camera_frame;
  // Transform from body frame (FLU) to camera frame (RDF)
  frame_transform::transform_body_to_camera(velocity_body_frame, velocity_camera_frame);
  if (_collision_checking_method == CollisionCheckingMethod::PYRAMID) {
    frame_transform::transform_body_to_camera(acceleration_body_frame, acceleration_camera_frame);
  }

  ruckig::InputParameter<3> initial_state_camera_frame;
  initial_state_camera_frame.current_position = {0.0, 0.0, 0.0};
  initial_state_camera_frame.current_velocity = {velocity_camera_frame.x, velocity_camera_frame.y, velocity_camera_frame.z};
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
  Eigen::Vector3d exploration_vector(goal_in_camera_frame.point.x,
                                     goal_in_camera_frame.point.y,
                                     goal_in_camera_frame.point.z);

  cv::Mat depth_mat = preprocess_depth_image(depth_msg);
  
  // Use camera intrinsics from config (unified for both modes)
  double cx = _real_cx;
  double cy = _real_cy;
  double fy = _real_focal_length;

  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
    "Camera params - fy: %.2f, cx: %.2f, cy: %.2f, cols: %d, rows: %d",
    fy, cx, cy, depth_mat.cols, depth_mat.rows);
  
  PinholeCamera camera(fy, cx, cy, depth_mat.cols, depth_mat.rows,
                       _depth_uncertainty_coeffs, _true_vehicle_radius,
                       _planning_vehicle_radius, _minimum_clear_distance);

  RandomTrajectorySampler trajectory_sampler(
    camera, _depth_upper_bound, _depth_lower_bound, exploration_vector,
    _depth_sampling_margin, body_to_world, _goal_in_world_frame, _world_frame,
    _3d_planning, _2d_z_margin, _is_spiral_sampling, _spiral_sampling_step);

  DuPlanner planner(depth_mat, camera, _collision_checking_method,
                    _checking_time_ratio, _sampled_trajectories_threshold,
                    _checked_trajectories_threshold, _debug_num_trajectories,
                    _collision_probability_threshold, _openmp_chunk_size);
  ruckig::Trajectory<3> opt_traj;

  ExplorationCost exploration_cost(exploration_vector, _traveling_cost);
  if (!planner.find_lowest_cost_trajectory(
        initial_state_camera_frame, opt_traj, trajectory_sampler,
        _planning_cycle_time, &exploration_cost,
        &ExplorationCost::get_cost_wrapper)) {
    if ((this->now() - _reference_trajectory_start_time).seconds() > 2.5 &&
        _planner_state == PlanningStates::TRAJECTORY_CONTROL && !_steered) {
      const std::lock_guard<std::mutex> lock(trajectory_mutex_);
      steering_value = planner.get_steering() / 8;
      _steered = true;
    }
    return;
  }

  if (!check_valid_trajectory(position_world_frame, opt_traj)) return;

  {
    const std::lock_guard<std::mutex> lock(trajectory_mutex_);
    steering_value = 0.0f;
    _steered = false;
    opt_traj.assign_body_to_world_transform(body_to_world);
    opt_traj.assign_world_to_body_transform(world_to_body);
    trajectory_queue_.push_back(opt_traj);
  }
}

void PlannerNode::visualise(const sm::Image::SharedPtr depth_msg) {
  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
    "Visualise callback: depth image %dx%d, encoding=%s",
    depth_msg->width, depth_msg->height, depth_msg->encoding.c_str());

  pointcloud_type* cloud = create_point_cloud(depth_msg);
  
  // Count valid points
  size_t valid_points = 0;
  for (const auto& pt : cloud->points) {
    if (!std::isnan(pt.x) && !std::isnan(pt.y) && !std::isnan(pt.z)) {
      valid_points++;
    }
  }
  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
    "Point cloud: %zu total points, %zu valid points, frame_id=%s",
    cloud->points.size(), valid_points, cloud->header.frame_id.c_str());

  sm::PointCloud2 cloudMessage;
  pcl::toROSMsg(*cloud, cloudMessage);
  point_cloud_pub->publish(cloudMessage);
  
  // Clean up memory
  delete cloud;

  visualization_msgs::msg::Marker goal_marker;
  goal_marker.header.frame_id = _world_frame;
  goal_marker.header.stamp = this->now();
  goal_marker.ns = "visualization";
  goal_marker.action = visualization_msgs::msg::Marker::ADD;
  goal_marker.pose.orientation.w = 1.0;
  goal_marker.id = 4;
  goal_marker.type = visualization_msgs::msg::Marker::CUBE;
  goal_marker.scale.x = 0.2;
  goal_marker.scale.y = 0.2;
  goal_marker.scale.z = 0.2;
  goal_marker.color.g = 1.0;
  goal_marker.color.a = 1.0;

  if (_goal_set) {
    goal_marker.pose.position.x = _goal_in_world_frame.x;
    goal_marker.pose.position.y = _goal_in_world_frame.y;
    goal_marker.pose.position.z = _goal_in_world_frame.z;
    goal_marker.pose.orientation.w = 1.0;
    visual_pub->publish(goal_marker);
  }
}
