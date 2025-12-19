#include "planner_node.hpp"

PlannerNode::PlannerNode()
  : rclcpp::Node("planner_node"),
    steering_value(0.0),
    _steered(false),
    trajectory_queue_(),
    _planner_state(PlanningStates::OFF),
    had_reference_trajectory(false),
    _goal_set(false),
    _trajectory_discretisation_cycle(0.01) {
  // Note: Most initialization is done in init() method because 
  // shared_from_this() cannot be called in constructor
}

bool PlannerNode::init() {
  // Initialize TF2 buffers, listeners, and broadcaster
  to_world_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  to_vehicle_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  to_world_tf2_ = std::make_shared<tf2_ros::TransformListener>(*to_world_buffer_);
  to_vehicle_tf2_ = std::make_shared<tf2_ros::TransformListener>(*to_vehicle_buffer_);
  tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

  // Load parameters
  // Since topics are stored in the config file,
  // we need to load parameters from the config file before subscribing to topics
  if (!loadParameters()) {
    RCLCPP_ERROR(this->get_logger(), "[%s] Could not load parameters.", this->get_name());
    return false;
  }

  // QoS settings
  auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
  auto sensor_qos = rclcpp::SensorDataQoS();

  // Publishers
  point_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_out", 1);
  visual_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/visualization", 1);
  debug_vectors_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/debug_vectors", 1);

  // Publishers based on runtime mode
  if (_runtime_mode == RuntimeModes::MAVROS || _runtime_mode == RuntimeModes::OMNIDRONES) {
    // Position/velocity/acceleration setpoints (PositionTarget)
    raw_ref_pos_pub_ = this->create_publisher<mavros_msgs::msg::PositionTarget>(
      "mavros/setpoint_raw/local", 10);
  }
  if (_runtime_mode == RuntimeModes::MAVROS) {
    att_ctrl_pub_ = this->create_publisher<mavros_msgs::msg::AttitudeTarget>(
      "/mavros/setpoint_raw/attitude", 1);
  }
  if (_runtime_mode == RuntimeModes::OMNIDRONES) {
    // Velocity-only setpoints (TwistStamped) - alternative control mode
    vel_cmd_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>(
      "mavros/setpoint_velocity/cmd_vel", 10);
  }

  // Subscribers
  image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    _depth_topic, sensor_qos,
    std::bind(&PlannerNode::img_callback, this, std::placeholders::_1));

  if (_visualise) {
    visual_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      _depth_topic, sensor_qos,
      std::bind(&PlannerNode::visualise, this, std::placeholders::_1));
  }

  reset_sub_ = this->create_subscription<std_msgs::msg::Empty>(
    "/kingfisher/dodgeros_pilot/reset_sim", 1,
    std::bind(&PlannerNode::reset_callback, this, std::placeholders::_1));

  // Mission command subscriber (unified for OmniDrones and MAVROS)
  mission_sub_ = this->create_subscription<ground_system_msgs::msg::StartSwarmMission>(
    "/start_swarm_mission", 10,
    std::bind(&PlannerNode::mission_callback, this, std::placeholders::_1));

  // Subscribe to odometry - use relative topic so namespace remapping works
  // When running in /Drone1 namespace, this becomes /Drone1/odometry
  odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
    "odometry", rclcpp::QoS(1).best_effort(),
    std::bind(&PlannerNode::odometry_callback, this, std::placeholders::_1));

  mav_state_sub_ = this->create_subscription<mavros_msgs::msg::State>(
    "mavros/state", rclcpp::QoS(1).best_effort(),
    std::bind(&PlannerNode::ardupilot_status_callback, this, std::placeholders::_1));

  mav_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
    "mavros/local_position/pose", rclcpp::QoS(1).best_effort(),
    std::bind(&PlannerNode::mav_pose_callback, this, std::placeholders::_1));

  mav_twist_sub_ = this->create_subscription<geometry_msgs::msg::TwistStamped>(
    "mavros/local_position/velocity_body", rclcpp::QoS(1).best_effort(),
    std::bind(&PlannerNode::mav_twist_callback, this, std::placeholders::_1));

  mav_accel_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
    "mavros/imu/data_raw", rclcpp::QoS(1).best_effort(),
    std::bind(&PlannerNode::mav_accel_callback, this, std::placeholders::_1));

  // MAVROS service clients (for real FC)
  arming_srv_ = this->create_client<mavros_msgs::srv::CommandBool>("/mavros/cmd/arming");
  takeoff_srv_ = this->create_client<mavros_msgs::srv::CommandTOL>("/mavros/cmd/takeoff");
  land_srv_ = this->create_client<mavros_msgs::srv::CommandTOL>("/mavros/cmd/land");
  mode_srv_ = this->create_client<mavros_msgs::srv::SetMode>("mavros/set_mode");

  // Timer for control loop
  control_loop_timer_ = this->create_wall_timer(
    std::chrono::duration<double>(_trajectory_discretisation_cycle),
    std::bind(&PlannerNode::control_loop, this));

  RCLCPP_INFO(this->get_logger(), "[%s] Node initialized successfully.", this->get_name());
  return true;
}

cv::Mat PlannerNode::preprocess_depth_image(const sensor_msgs::msg::Image::SharedPtr depth_msg) {
  cv_bridge::CvImageConstPtr cv_img_ptr = cv_bridge::toCvShare(depth_msg, depth_msg->encoding);
  cv::Mat depth_mat;
  cv_img_ptr->image.convertTo(depth_mat, CV_32FC1, _depth_scale);
  return depth_mat;
}

pointcloud_type* PlannerNode::create_point_cloud(const sensor_msgs::msg::Image::SharedPtr depth_msg) {
  cv::Mat depth_mat = preprocess_depth_image(depth_msg);
  double fy, fx, cx, cy;
  if (_runtime_mode == RuntimeModes::FLIGHTMARE) {
    cx = depth_mat.cols / 2.0f;
    cy = depth_mat.rows / 2.0f;
    fy = (depth_mat.rows / 2) / std::tan(M_PI * _flightmare_fov / 180.0 / 2.0);
    fx = fy;
  } else if (_runtime_mode == RuntimeModes::MAVROS || _runtime_mode == RuntimeModes::OMNIDRONES) {
    // Use camera intrinsics from config (unified for both modes)
    cx = _real_cx;
    cy = _real_cy;
    fx = fy = _real_focal_length;
  }

  pointcloud_type* cloud(new pointcloud_type());
  cloud->header.stamp = rclcpp::Time(depth_msg->header.stamp).nanoseconds() / 1000;
  cloud->header.frame_id = _vehicle_frame;
  cloud->is_dense = false;  // single point of view, 2d rasterized
  cloud->height = depth_mat.rows;
  cloud->width = depth_mat.cols;
  cloud->points.resize(cloud->height * cloud->width);

  const float* depth_data = reinterpret_cast<const float*>(depth_mat.data);
  const int rows = depth_mat.rows;
  const int cols = depth_mat.cols;

  // Use static scheduling since each iteration takes similar time
  #pragma omp parallel for collapse(2) schedule(static)
  for (int y = 0; y < rows; y++) {
    for (int x = 0; x < cols; x++) {
      uint32_t depth_idx = y * cols + x;
      const float Z = depth_data[depth_idx];
      // Check for invalid measurements
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
    // For OmniDrones NWU, goal is relative to current position
    // NWU: X=North, Y=West, Z=Up
    _goal_in_world_frame.x = _state.pose.position.x + _goal_north_coordinate;
    _goal_in_world_frame.y = _state.pose.position.y + _goal_west_coordinate;
    _goal_in_world_frame.z = _goal_up_coordinate;  // Z = Up (absolute)
  } else if (_runtime_mode == RuntimeModes::MAVROS) {
    // For MAVROS ENU, record goal coordinates relative to takeoff position
    _goal_in_world_frame.x = _state.pose.position.x - _goal_west_coordinate;
    _goal_in_world_frame.y = _state.pose.position.y + _goal_north_coordinate;
    _goal_in_world_frame.z = _goal_up_coordinate;
  }
  
  RCLCPP_INFO(this->get_logger(), "Goal set to ENU: (%.2f, %.2f, %.2f)",
              _goal_in_world_frame.x, _goal_in_world_frame.y, _goal_in_world_frame.z);
  
  _goal_set = true;
  mission_received_ = true;
  
  // For OmniDrones, immediately transition to TAKING_OFF state
  if (_runtime_mode == RuntimeModes::OMNIDRONES) {
    // Save home position for takeoff
    _home_in_world_frame = _state.pose.position;
    // Reset trajectory state
    steering_value = 0.0f;
    _steered = false;
    trajectory_queue_.clear();
    reference_trajectory_ = ruckig::Trajectory<3>();
    had_reference_trajectory = false;
    set_auto_pilot_state_forced(PlanningStates::TAKING_OFF);
  }
}

void PlannerNode::reset_callback(const std_msgs::msg::Empty::SharedPtr msg) {
  (void)msg;
  RCLCPP_WARN(this->get_logger(), "[%s] Planner: Reset quadrotor simulator!", this->get_name());
  set_auto_pilot_state_forced(PlanningStates::OFF);
  steering_value = 0.0f;
  _steered = false;
  if (_runtime_mode == RuntimeModes::MAVROS || _runtime_mode == RuntimeModes::OMNIDRONES) {
    _goal_set = false;
    mission_received_ = false;
    // Reset async service pending flags
    mode_switch_pending_ = false;
    arming_pending_ = false;
    takeoff_pending_ = false;
    land_pending_ = false;
  }
  trajectory_queue_.clear();
  reference_trajectory_ = ruckig::Trajectory<3>();
  had_reference_trajectory = false;
}

void PlannerNode::ardupilot_status_callback(const mavros_msgs::msg::State::SharedPtr msg) {
  flight_controller_status = *msg;
}

// World: "map" - ENU
void PlannerNode::mav_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
  _latest_pose_stamp = rclcpp::Time(msg->header.stamp);

  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    _state.pose = msg->pose;
    _state.t = std::min(_latest_pose_stamp, _latest_twist_stamp).seconds();
  }
}

// Body: "base_link" - FLU
void PlannerNode::mav_twist_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg) {
  _latest_twist_stamp = rclcpp::Time(msg->header.stamp);

  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    _state.velocity.linear = msg->twist.linear;
    _state.velocity.angular = msg->twist.angular;
    _state.t = std::min(_latest_pose_stamp, _latest_twist_stamp).seconds();
  }
}

// Body: "base_link" - FLU
void PlannerNode::mav_accel_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
  // Acceleration callback - only used for PYRAMID collision checking which requires CUDA
  // For MIDI collision checking (CPU-only), we don't need acceleration
  (void)msg;  // Suppress unused parameter warning
  return;
}

void PlannerNode::odometry_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    _state.t = rclcpp::Time(msg->header.stamp).seconds();
    _state.pose = msg->pose.pose;
    _state.velocity = msg->twist.twist;
    // Note: acceleration not provided in Odometry, leave as zero
  }

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
  // checking if new trajectory planned
  if (trajectory_queue_.empty()) return;

  // only consider the latest
  while (trajectory_queue_.size() > 1) {
    trajectory_queue_.pop_front();
  }

  // update reference trajectory
  rclcpp::Time wall_time_now = this->now();
  rclcpp::Duration trajectory_point_time = wall_time_now - _reference_trajectory_start_time;
  double point_time = trajectory_point_time.seconds();
  if (trajectory_queue_.size() > 0) {
    // Only track when there is a valid trajectory
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
  if (_runtime_mode == RuntimeModes::MAVROS && _planner_state == PlanningStates::OFF && _goal_set) {
    // Step 1: Switch to GUIDED mode if not already
    if (flight_controller_status.mode != "GUIDED" && !mode_switch_pending_) {
      if (mode_srv_->service_is_ready()) {
        auto request = std::make_shared<mavros_msgs::srv::SetMode::Request>();
        request->custom_mode = "GUIDED";
        mode_switch_pending_ = true;
        
        mode_srv_->async_send_request(request,
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
      if (arming_srv_->service_is_ready()) {
        auto request = std::make_shared<mavros_msgs::srv::CommandBool::Request>();
        request->value = true;
        arming_pending_ = true;
        
        arming_srv_->async_send_request(request,
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
      if (takeoff_srv_->service_is_ready()) {
        auto request = std::make_shared<mavros_msgs::srv::CommandTOL::Request>();
        request->altitude = _goal_in_world_frame.z;
        takeoff_pending_ = true;
        
        takeoff_srv_->async_send_request(request,
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

  // Skip updating planner state if goal is not set
  if (!_goal_set) return;

  // Trajectory control switching
  geometry_msgs::msg::Point goal_in_world_frame = _goal_in_world_frame;
  // only check distance to goal horizontally for switching control
  goal_in_world_frame.z = _state.pose.position.z;
  double distance_to_goal =
    (geometryToEigen(_state.pose.position) - geometryToEigen(goal_in_world_frame)).norm();
  
  // TAKING_OFF -> TRAJECTORY_CONTROL when altitude is reached
  if (_state.pose.position.z >= (_goal_in_world_frame.z - 0.1) &&
      _planner_state == PlanningStates::TAKING_OFF) {
    set_auto_pilot_state_forced(PlanningStates::TRAJECTORY_CONTROL);
  }

  // Stop planning when the goal is close - TRAJECTORY_CONTROL -> GO_TO_GOAL
  else if (_planner_state == PlanningStates::TRAJECTORY_CONTROL &&
           (distance_to_goal < _go_to_goal_threshold ||
            // NWU (Flightmare and OmniDrones)
            ((_state.pose.position.x + _go_to_goal_threshold / 3) > _goal_in_world_frame.x &&
             (_runtime_mode == RuntimeModes::FLIGHTMARE || _runtime_mode == RuntimeModes::OMNIDRONES)) ||
            // ENU (MAVROS)
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
    if (land_srv_->service_is_ready()) {
      auto request = std::make_shared<mavros_msgs::srv::CommandTOL::Request>();
      land_pending_ = true;
      
      land_srv_->async_send_request(request,
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
  if (_planner_state == PlanningStates::LAND ||
      _planner_state == PlanningStates::OFF || !_goal_set)
    return;
  
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
  if (_planner_state == PlanningStates::TAKING_OFF) {
    _reference_trajectory_start_time = command_execution_time;
    steering_value = 0.0f;
    if (_runtime_mode == RuntimeModes::MAVROS)
      return;
    if (_runtime_mode == RuntimeModes::OMNIDRONES) {
      // Takeoff to goal altitude at home XY position
      reference_point.position = Eigen::Vector3d(
        _home_in_world_frame.x, 
        _home_in_world_frame.y, 
        _goal_in_world_frame.z);
    }
  } else if (_planner_state == PlanningStates::TRAJECTORY_CONTROL && had_reference_trajectory) {
    rclcpp::Duration trajectory_point_time = command_execution_time - _reference_trajectory_start_time;
    double point_time = trajectory_point_time.seconds();
    get_reference_point_at_time(reference_trajectory_, point_time, reference_point);
  } else if (_planner_state == PlanningStates::GO_TO_GOAL) {
    _reference_trajectory_start_time = command_execution_time;
    steering_value = 0.0f;
    reference_point.position = geometryToEigen(_goal_in_world_frame);
    reference_point.velocity = Eigen::Vector3d(0.0, 0.0, 0.0);
    reference_point.acceleration = Eigen::Vector3d(0.0, 0.0, 0.0);
    Eigen::Vector3d current_euler_angles =
      quaternionToEulerAnglesZYX(geometryToEigen(_state.pose.orientation));
    reference_point.heading = current_euler_angles(2);
  }

  // Validate state freshness
  if (_state.t - this->now().seconds() > 0.2) {
    RCLCPP_WARN(this->get_logger(), "[%s] State is too old, skipping control command", this->get_name());
    return;
  }
  
  // Publish position/velocity setpoint (kinematic control mode)
  // For MAVROS and OmniDrones, we only support KINEMATIC mode
  if (_runtime_mode == RuntimeModes::MAVROS || _runtime_mode == RuntimeModes::OMNIDRONES) {
    public_ref_pos(reference_point);
    return;
  }
  
  // Note: Flightmare with position_controller support has been removed
  // The ros2-humble-cuda branch focuses on MAVROS and OmniDrones simulation
  RCLCPP_WARN_ONCE(this->get_logger(), "[%s] Flightmare runtime mode not supported in this build", this->get_name());
}

void PlannerNode::public_ref_pos(const quadrotor_common::TrajectoryPoint& reference_point) {
  mavros_msgs::msg::PositionTarget msg;
  msg.header.stamp = this->now();
  // FRAME_LOCAL_NED
  msg.coordinate_frame = 1;
  msg.type_mask = 0;
  // reference_point in ENU for MAVROS and NWU for OmniDrones/Flightmare
  msg.position.x = reference_point.position(0);
  msg.position.y = reference_point.position(1);
  msg.position.z = reference_point.position(2);
  msg.velocity.x = reference_point.velocity(0);
  msg.velocity.y = reference_point.velocity(1);
  msg.velocity.z = reference_point.velocity(2);
  msg.acceleration_or_force.x = reference_point.acceleration(0);
  msg.acceleration_or_force.y = reference_point.acceleration(1);
  msg.acceleration_or_force.z = reference_point.acceleration(2);
  msg.yaw = 0.0;
  raw_ref_pos_pub_->publish(msg);
}

void PlannerNode::get_reference_point_at_time(
  const ruckig::Trajectory<3>& reference_trajectory, const double& _point_time,
  TrajectoryPoint& reference_point) {

  const double point_time = std::clamp(_point_time, 0.0, reference_trajectory.get_duration());

  // Get the corresponding transform attached to the trajectory
  geometry_msgs::msg::TransformStamped body_to_world =
    reference_trajectory.get_transform_to_world();

  // Get pvaj in the camera frame at a given time from the reference trajectory
  std::array<double, 3> position_in_camera_frame, velocity_in_camera_frame,
    acceleration_in_camera_frame, jerk_in_camera_frame;
  size_t num_section;
  reference_trajectory.at_time(
    point_time, position_in_camera_frame, velocity_in_camera_frame,
    acceleration_in_camera_frame, jerk_in_camera_frame, num_section);

  // Transform pvaj in the camera frame to the world frame
  geometry_msgs::msg::Point position_in_body_frame, position_in_world_frame;
  geometry_msgs::msg::Vector3 velocity_in_body_frame, velocity_in_world_frame,
    acceleration_in_body_frame, acceleration_in_world_frame, jerk_in_body_frame,
    jerk_in_world_frame;

  // Camera (RDF) to body (FLU)
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
    RCLCPP_WARN(this->get_logger(), "Failure %s", ex.what());
  }

  // Asigning heading
  Eigen::Vector3d trajectory_vector =
    geometryToEigen(reference_trajectory.get_terminal_position_in_world_frame()) -
    geometryToEigen(reference_trajectory.get_initial_position_in_world_frame());
  double terminal_heading = atan2f(trajectory_vector[1], trajectory_vector[0]);
  reference_point.heading = terminal_heading;
  Eigen::Vector3d current_euler_angles =
    quaternionToEulerAnglesZYX(geometryToEigen(_state.pose.orientation));
  // Overriding steering heading if needed to
  if (fabs(steering_value) > 1e-6)
    reference_point.heading = current_euler_angles(2) + steering_value;

  // Asigning to trajectory reference point
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
  RCLCPP_WARN(this->get_logger(), "[%s] Switched to %s state", this->get_name(), state_name.c_str());
}

bool PlannerNode::check_valid_trajectory(
  const geometry_msgs::msg::Point& current_position,
  const ruckig::Trajectory<3>& trajectory) {
  if (trajectory.get_duration() < 1e-6) {
    RCLCPP_WARN(this->get_logger(),
                "[%s] The received trajectory is empty, rejecting it!", this->get_name());
    return false;
  }
  // Check if the trajectory starts at the current position
  double pos_diff = (geometryToEigen(current_position) -
                     geometryToEigen(trajectory.get_initial_position_in_world_frame()))
                      .norm();
  if (pos_diff > kPositionJumpTolerance_) {
    RCLCPP_WARN(this->get_logger(),
      "[%s] The received trajectory does not start at current "
      "position, rejecting it!",
                this->get_name());
    return false;
  }
  return true;
}

// Callback for planning when a new depth image comes
void PlannerNode::img_callback(const sensor_msgs::msg::Image::SharedPtr depth_msg) {
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
    // Lookup for transforms in the TF2 transforming tree
    try {
      body_to_world = to_world_buffer_->lookupTransform(_world_frame, _vehicle_frame, tf2::TimePointZero);
      world_to_body = to_vehicle_buffer_->lookupTransform(_vehicle_frame, _world_frame, tf2::TimePointZero);
    } catch (tf2::TransformException& ex) {
      RCLCPP_WARN(this->get_logger(), "%s", ex.what());
    }
    position_world_frame = _state.pose.position;
    if (_runtime_mode == RuntimeModes::FLIGHTMARE || _runtime_mode == RuntimeModes::OMNIDRONES) {
      // in Flightmare/OmniDrones, raw velocity and acceleration are in NWU (world frame)
      velocity_world_frame = _state.velocity.linear;
      acceleration_world_frame = _state.acceleration.linear;
      // world to body: NWU to FLU
      tf2::doTransform(velocity_world_frame, velocity_body_frame, world_to_body);
      tf2::doTransform(acceleration_world_frame, acceleration_body_frame, world_to_body);
    } else if (_runtime_mode == RuntimeModes::MAVROS) {
      // in MAVROS, raw velocity and acceleration are in FLU (body frame)
      velocity_body_frame = _state.velocity.linear;
      acceleration_body_frame = _state.acceleration.linear;
    }
    tf2::doTransform(test_acceleration_body_frame, test_acceleration_world_frame, body_to_world);
  }
  // If the acceleration is too high, skip planning
  if (test_acceleration_world_frame.x > _acc_planning_threshold ||
      test_acceleration_world_frame.y > _acc_planning_threshold)
    return;

  // This is for body (FLU) to camera (RDF)
  geometry_msgs::msg::Vector3 velocity_camera_frame, acceleration_camera_frame;
  frame_transform::transform_body_to_camera(velocity_body_frame, velocity_camera_frame);
  if (_collision_checking_method == CollisionCheckingMethod::PYRAMID) {
    frame_transform::transform_body_to_camera(acceleration_body_frame, acceleration_camera_frame);
  }

  // Initial state expressed the local inertial RDF frame with its origin is at
  // the camera position
  ruckig::InputParameter<3> initial_state_camera_frame;
  initial_state_camera_frame.current_position = {0.0, 0.0, 0.0};
  initial_state_camera_frame.current_velocity = {velocity_camera_frame.x, velocity_camera_frame.y,
                                                  velocity_camera_frame.z};
  // initial_state_camera_frame.current_velocity = {0.0, 0.0, 0.0};
  initial_state_camera_frame.target_velocity = {0.0, 0.0, 0.0};
  initial_state_camera_frame.max_velocity = {_max_velocity_x, _max_velocity_y, _max_velocity_z};
  initial_state_camera_frame.max_acceleration = {_max_acceleration_x, _max_acceleration_y,
                                                  _max_acceleration_z};
  if (_collision_checking_method == CollisionCheckingMethod::PYRAMID) {
    initial_state_camera_frame.current_acceleration = {acceleration_camera_frame.x,
                                                        acceleration_camera_frame.y,
                                                        acceleration_camera_frame.z};
    initial_state_camera_frame.target_acceleration = {0.0, 0.0, 0.0};
    initial_state_camera_frame.max_jerk = {15.0, 15.0, 10.0};
  }

  // Transform the coordinate of goal_in_world_frame to
  // the coordinate of goal_in_camera_frame
  geometry_msgs::msg::PointStamped goal_in_camera_frame, goal_in_world_frame_stamped,
    goal_in_body_frame;
  goal_in_world_frame_stamped.header.frame_id = _world_frame;
  goal_in_world_frame_stamped.point = _goal_in_world_frame;
  try {
    tf2::doTransform(goal_in_world_frame_stamped, goal_in_body_frame, world_to_body);
  } catch (tf2::TransformException& ex) {
    RCLCPP_WARN(this->get_logger(), "Failure %s", ex.what());
  }
  frame_transform::transform_body_to_camera(goal_in_body_frame.point, goal_in_camera_frame.point);
  // Build exploration_vector from the coordinate of goal_in_camera_frame
  Eigen::Vector3d exploration_vector(goal_in_camera_frame.point.x,
                                     goal_in_camera_frame.point.y,
                                     goal_in_camera_frame.point.z);

  cv::Mat depth_mat = preprocess_depth_image(depth_msg);
  double cx, cy, fy;
  // Camera model initialization
  if (_runtime_mode == RuntimeModes::FLIGHTMARE) {
    cx = depth_mat.cols / 2.0f;
    cy = depth_mat.rows / 2.0f;
    fy = (depth_mat.rows / 2) / std::tan(M_PI * _flightmare_fov / 180.0 / 2.0);
  } else if (_runtime_mode == RuntimeModes::MAVROS || _runtime_mode == RuntimeModes::OMNIDRONES) {
    // Use camera intrinsics from config (unified for both modes)
    cx = _real_cx;
    cy = _real_cy;
    fy = _real_focal_length;
  }

  PinholeCamera camera(fy, cx, cy, depth_mat.cols, depth_mat.rows,
                       _depth_uncertainty_coeffs, _true_vehicle_radius,
                       _planning_vehicle_radius, _minimum_clear_distance);

  // Pass the projected_goal to the RandomTrajectorySampler
  RandomTrajectorySampler trajectory_sampler(
    camera, _depth_upper_bound, _depth_lower_bound, exploration_vector,
    _depth_sampling_margin, body_to_world, _goal_in_world_frame, _world_frame,
    _3d_planning, _2d_z_margin, _is_spiral_sampling, _spiral_sampling_step);

  DuPlanner planner(depth_mat, camera, _collision_checking_method,
                    _checking_time_ratio, _sampled_trajectories_threshold,
                    _checked_trajectories_threshold, _debug_num_trajectories,
                    _collision_probability_threshold, _openmp_chunk_size);
  ruckig::Trajectory<3> opt_traj;

  // Find the fastest trajectory candidate
  ExplorationCost exploration_cost(exploration_vector, _traveling_cost);
  if (!planner.find_lowest_cost_trajectory(
        initial_state_camera_frame, opt_traj, trajectory_sampler,
        _planning_cycle_time, &exploration_cost,
        &ExplorationCost::get_cost_wrapper)) {
    // We only sent steering commands when we could not find
    // any feasible trajectory for 1 second in a row.
    if ((this->now() - _reference_trajectory_start_time).seconds() > 2.5 &&
        _planner_state == PlanningStates::TRAJECTORY_CONTROL && !_steered) {
      const std::lock_guard<std::mutex> lock(trajectory_mutex_);
      steering_value = planner.get_steering() / 8;
      _steered = true;
    }
    return;
  }
  // New traj generated
  {
    const std::lock_guard<std::mutex> lock(trajectory_mutex_);
    // Assign transforms to the optimized trajectory right before checking validity
    // otherwise the validity check will fail because of uninitialized transforms
    opt_traj.assign_body_to_world_transform(body_to_world);
    opt_traj.assign_world_to_body_transform(world_to_body);    
    steering_value = 0.0f;
    _steered = false;
    // Only here we push the new trajectory to the queue
    trajectory_queue_.push_back(opt_traj);
  }
}

void PlannerNode::visualise(const sensor_msgs::msg::Image::SharedPtr depth_msg) {
  if (!rclcpp::ok()) {
    return;
  }
  
  // Debug: log that we received a depth image
  // static int depth_count = 0;
  // if (++depth_count % 100 == 1) {
  //   RCLCPP_INFO(this->get_logger(), "Received depth image #%d (%dx%d, encoding: %s)", 
  //               depth_count, depth_msg->width, depth_msg->height, depth_msg->encoding.c_str());
  // }
  
  // convert depth image to point cloud
  pointcloud_type* cloud = create_point_cloud(depth_msg);
  sensor_msgs::msg::PointCloud2 cloudMessage;
  pcl::toROSMsg(*cloud, cloudMessage);
  // Set header for RViz visualization (pcl::toROSMsg should copy this, but ensure it's set)
  cloudMessage.header.stamp = depth_msg->header.stamp;
  cloudMessage.header.frame_id = _vehicle_frame;
  point_cloud_pub_->publish(cloudMessage);
  delete cloud;  // Free memory

  // ============================================================================
  // DEBUG VISUALIZATION: Velocity, Goal Vector, Exploration Vector as arrows
  // ============================================================================
  visualization_msgs::msg::MarkerArray debug_markers;
  
  // Get current transforms for coordinate conversions
  geometry_msgs::msg::TransformStamped world_to_body, body_to_world;
  try {
    body_to_world = to_world_buffer_->lookupTransform(_world_frame, _vehicle_frame, tf2::TimePointZero);
    world_to_body = to_vehicle_buffer_->lookupTransform(_vehicle_frame, _world_frame, tf2::TimePointZero);
  } catch (tf2::TransformException& ex) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "TF lookup failed: %s", ex.what());
  }
  
  // Helper lambda to create an arrow marker
  auto create_arrow_marker = [&](int id, const std::string& ns, 
                                  const geometry_msgs::msg::Point& start,
                                  const geometry_msgs::msg::Point& end,
                                  float r, float g, float b, float a,
                                  const std::string& frame_id) {
    visualization_msgs::msg::Marker arrow;
    arrow.header.frame_id = frame_id;
    arrow.header.stamp = this->now();
    arrow.ns = ns;
    arrow.id = id;
    arrow.type = visualization_msgs::msg::Marker::ARROW;
    arrow.action = visualization_msgs::msg::Marker::ADD;
    arrow.points.push_back(start);
    arrow.points.push_back(end);
    arrow.scale.x = 0.05;  // shaft diameter
    arrow.scale.y = 0.1;   // head diameter
    arrow.scale.z = 0.1;   // head length
    arrow.color.r = r;
    arrow.color.g = g;
    arrow.color.b = b;
    arrow.color.a = a;
    arrow.lifetime = rclcpp::Duration::from_seconds(0.5);
    return arrow;
  };
  
  // 1. VELOCITY FEEDBACK (CYAN) - in world frame, origin at drone position
  // This shows the raw velocity from OmniDrones odometry
  {
    geometry_msgs::msg::Point start, end;
    start.x = _state.pose.position.x;
    start.y = _state.pose.position.y;
    start.z = _state.pose.position.z;
    
    // Scale velocity for visibility (1 m/s = 1 meter arrow)
    double vel_scale = 1.0;
    end.x = start.x + _state.velocity.linear.x * vel_scale;
    end.y = start.y + _state.velocity.linear.y * vel_scale;
    end.z = start.z + _state.velocity.linear.z * vel_scale;
    
    auto vel_arrow = create_arrow_marker(0, "velocity_world", start, end, 0.0, 1.0, 1.0, 1.0, _world_frame);
    // debug_markers.markers.push_back(vel_arrow);
    
    // Log velocity periodically
    // double vel_mag = std::sqrt(_state.velocity.linear.x * _state.velocity.linear.x +
    //                            _state.velocity.linear.y * _state.velocity.linear.y +
    //                            _state.velocity.linear.z * _state.velocity.linear.z);
    // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
    //   "VELOCITY (world): [%.2f, %.2f, %.2f] mag=%.2f m/s",
    //   _state.velocity.linear.x, _state.velocity.linear.y, _state.velocity.linear.z, vel_mag);
  }
  
  // 2. VELOCITY IN BODY FRAME (MAGENTA) - transformed to body frame, shown from origin in base_link
  {
    geometry_msgs::msg::Vector3 velocity_world_frame, velocity_body_frame;
    velocity_world_frame = _state.velocity.linear;
    tf2::doTransform(velocity_world_frame, velocity_body_frame, world_to_body);
    
    geometry_msgs::msg::Point start, end;
    start.x = start.y = start.z = 0.0;
    
    double vel_scale = 1.0;
    end.x = velocity_body_frame.x * vel_scale;
    end.y = velocity_body_frame.y * vel_scale;
    end.z = velocity_body_frame.z * vel_scale;
    
    auto vel_body_arrow = create_arrow_marker(1, "velocity_body", start, end, 1.0, 0.0, 1.0, 1.0, _vehicle_frame);
    // debug_markers.markers.push_back(vel_body_arrow);
    
    // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
    //   "VELOCITY (body FLU): [%.2f, %.2f, %.2f]",
    //   velocity_body_frame.x, velocity_body_frame.y, velocity_body_frame.z);
  }
  
  // 3. GOAL VECTOR (GREEN) - direction from drone to goal in world frame
  if (_goal_set) {
    geometry_msgs::msg::Point start, end;
    start.x = _state.pose.position.x;
    start.y = _state.pose.position.y;
    start.z = _state.pose.position.z;
    
    // Normalize and scale for visibility
    double dx = _goal_in_world_frame.x - start.x;
    double dy = _goal_in_world_frame.y - start.y;
    double dz = _goal_in_world_frame.z - start.z;
    double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
    double arrow_len = std::min(dist, 3.0);  // Cap at 3m for visibility
    
    if (dist > 0.1) {
      end.x = start.x + (dx / dist) * arrow_len;
      end.y = start.y + (dy / dist) * arrow_len;
      end.z = start.z + (dz / dist) * arrow_len;
      
      auto goal_arrow = create_arrow_marker(2, "goal_vector", start, end, 0.0, 1.0, 0.0, 1.0, _world_frame);
      // debug_markers.markers.push_back(goal_arrow);
      
      // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
      //   "GOAL VECTOR (world): [%.2f, %.2f, %.2f] dist=%.2f",
      //   dx, dy, dz, dist);
    }
  }
  
  // 4. GOAL IN BODY FRAME (YELLOW) - goal direction in body frame (FLU)
  if (_goal_set) {
    geometry_msgs::msg::PointStamped goal_world, goal_body;
    goal_world.header.frame_id = _world_frame;
    goal_world.point = _goal_in_world_frame;
    
    try {
      tf2::doTransform(goal_world, goal_body, world_to_body);
      
      geometry_msgs::msg::Point start, end;
      start.x = start.y = start.z = 0.0;
      
      double dist = std::sqrt(goal_body.point.x * goal_body.point.x +
                              goal_body.point.y * goal_body.point.y +
                              goal_body.point.z * goal_body.point.z);
      double arrow_len = std::min(dist, 3.0);
      
      if (dist > 0.1) {
        end.x = (goal_body.point.x / dist) * arrow_len;
        end.y = (goal_body.point.y / dist) * arrow_len;
        end.z = (goal_body.point.z / dist) * arrow_len;
        
        auto goal_body_arrow = create_arrow_marker(3, "goal_body", start, end, 1.0, 1.0, 0.0, 1.0, _vehicle_frame);
        // debug_markers.markers.push_back(goal_body_arrow);
        
        // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
        //   "GOAL (body FLU): [%.2f, %.2f, %.2f]",
        //   goal_body.point.x, goal_body.point.y, goal_body.point.z);
      }
    } catch (tf2::TransformException& ex) {
      // Ignore transform errors
    }
  }
  
  // 5. EXPLORATION VECTOR / GOAL IN CAMERA FRAME (RED) - this is what the planner uses
  if (_goal_set) {
    geometry_msgs::msg::PointStamped goal_world, goal_body;
    goal_world.header.frame_id = _world_frame;
    goal_world.point = _goal_in_world_frame;
    
    try {
      tf2::doTransform(goal_world, goal_body, world_to_body);
      
      // Transform from body (FLU) to camera (RDF)
      geometry_msgs::msg::Point goal_camera;
      frame_transform::transform_body_to_camera(goal_body.point, goal_camera);
      
      // The exploration vector in camera frame (RDF: X=Right, Y=Down, Z=Forward)
      // Visualize in vehicle frame by transforming back for display
      // Camera RDF to Body FLU: x_body = z_cam, y_body = -x_cam, z_body = -y_cam
      geometry_msgs::msg::Point start, end;
      start.x = start.y = start.z = 0.0;
      
      double dist = std::sqrt(goal_camera.x * goal_camera.x +
                              goal_camera.y * goal_camera.y +
                              goal_camera.z * goal_camera.z);
      double arrow_len = std::min(dist, 3.0);
      
      // Compute exploration cost values for debugging
      Eigen::Vector3d exploration_vector(goal_camera.x, goal_camera.y, goal_camera.z);
      Eigen::Vector3d exploration_unit = exploration_vector.normalized();
      
      // Simulate a sample endpoint (e.g., 1m forward in camera frame) to show cost calculation
      Eigen::Vector3d sample_endpoint(0.0, 0.0, 1.0);  // 1m forward in camera RDF
      double direction_cost = -exploration_unit.dot(sample_endpoint.normalized());
      double distance_cost = -sample_endpoint.dot(exploration_unit);
      
      if (dist > 0.1) {
        // Display the camera-frame vector transformed back to body frame for visualization
        // Camera RDF -> Body FLU: Forward=Z_cam, Left=-X_cam, Up=-Y_cam
        end.x = (goal_camera.z / dist) * arrow_len;  // Forward (body X) = Camera Z
        end.y = (-goal_camera.x / dist) * arrow_len; // Left (body Y) = -Camera X
        end.z = (-goal_camera.y / dist) * arrow_len; // Up (body Z) = -Camera Y
        
        auto explore_arrow = create_arrow_marker(4, "exploration_camera", start, end, 1.0, 0.0, 0.0, 1.0, _vehicle_frame);
        debug_markers.markers.push_back(explore_arrow);
        
        // Get traveling cost type as string
        // std::string cost_type_str = (_traveling_cost == TravelingCost::DIRECTION) ? "DIRECTION" : "DISTANCE";
        // double active_cost = (_traveling_cost == TravelingCost::DIRECTION) ? direction_cost : distance_cost;
        
        // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
        //   "EXPLORATION (camera RDF): [%.2f, %.2f, %.2f] | Cost type: %s | "
        //   "Direction cost: %.3f | Distance cost: %.3f | Active cost: %.3f",
        //   goal_camera.x, goal_camera.y, goal_camera.z,
        //   cost_type_str.c_str(), direction_cost, distance_cost, active_cost);
      }
    } catch (tf2::TransformException& ex) {
      // Ignore transform errors
    }
  }
  
  // 6. DRONE FORWARD DIRECTION (WHITE) - shows drone heading
  {
    geometry_msgs::msg::Point start, end;
    start.x = start.y = start.z = 0.0;
    end.x = 1.0;  // 1m forward in body frame
    end.y = 0.0;
    end.z = 0.0;
    
    auto forward_arrow = create_arrow_marker(5, "drone_forward", start, end, 1.0, 1.0, 1.0, 0.8, _vehicle_frame);
    debug_markers.markers.push_back(forward_arrow);
  }
  
  // Publish all debug markers
  debug_vectors_pub_->publish(debug_markers);
  
  // ============================================================================
  // Original visualization code
  // ============================================================================
  visualization_msgs::msg::Marker pyramids_bases, pyramids_edges, polynomial_trajectory, goal_marker;
  pyramids_bases.header.frame_id = pyramids_edges.header.frame_id =
    polynomial_trajectory.header.frame_id = goal_marker.header.frame_id = _world_frame;
  pyramids_bases.header.stamp = pyramids_edges.header.stamp =
    polynomial_trajectory.header.stamp = goal_marker.header.stamp = this->now();
  pyramids_bases.ns = pyramids_edges.ns = polynomial_trajectory.ns = goal_marker.ns =
    "visualization";
  pyramids_bases.action = pyramids_edges.action = polynomial_trajectory.action = goal_marker.action =
    visualization_msgs::msg::Marker::ADD;
  goal_marker.action = visualization_msgs::msg::Marker::MODIFY;
  pyramids_bases.pose.orientation.w = pyramids_edges.pose.orientation.w =
    polynomial_trajectory.pose.orientation.w = goal_marker.pose.orientation.w = 1.0;
  pyramids_bases.id = 1;
  pyramids_edges.id = 2;
  polynomial_trajectory.id = 3;
  goal_marker.id = 4;
  pyramids_bases.type = visualization_msgs::msg::Marker::LINE_LIST;
  pyramids_edges.type = visualization_msgs::msg::Marker::LINE_LIST;
  polynomial_trajectory.type = visualization_msgs::msg::Marker::LINE_STRIP;
  goal_marker.type = visualization_msgs::msg::Marker::CUBE;
  // LINE_STRIP markers use only the x component of scale, for the line width
  pyramids_bases.scale.x = pyramids_edges.scale.x = polynomial_trajectory.scale.x = 0.02;
  goal_marker.scale.x = 0.2;
  goal_marker.scale.y = 0.2;
  goal_marker.scale.z = 0.2;
  // Line strip is blue
  pyramids_bases.color.g = pyramids_edges.color.g = goal_marker.color.g = 1.0;
  polynomial_trajectory.color.r = 0.0;
  polynomial_trajectory.color.b = 1.0;
  pyramids_bases.color.a = pyramids_edges.color.a =
    polynomial_trajectory.color.a = goal_marker.color.a = 1.0;
  geometry_msgs::msg::Point p;

  if (_goal_set) {
    goal_marker.pose.position.x = _goal_in_world_frame.x;
    goal_marker.pose.position.y = _goal_in_world_frame.y;
    goal_marker.pose.position.z = _goal_in_world_frame.z;
    goal_marker.pose.orientation.w = 1.0;
    // Publish goal marker
    visual_pub_->publish(goal_marker);
  }

  // Publish polynomial trajectory
  if (!had_reference_trajectory) {
    return;
  }
  
  double trajectory_duration = reference_trajectory_.get_duration();
  if (trajectory_duration < 0.01) {
    polynomial_trajectory.points.clear();
    visual_pub_->publish(polynomial_trajectory);
    return;
  }
  for (int i = 0; i <= 100; i++) {
    geometry_msgs::msg::Point position =
      reference_trajectory_.get_position_in_world_frame(trajectory_duration * i / 100);
    p.x = position.x;
    p.y = position.y;
    p.z = position.z;
    polynomial_trajectory.points.push_back(p);
  }

  pyramids_bases.points.clear();
  if (_planner_state == PlanningStates::GO_TO_GOAL) {
    pyramids_bases.points.push_back(_stop_planning_point_in_world_frame);
    pyramids_bases.points.push_back(_goal_in_world_frame);
  polynomial_trajectory.color.b = 0.0;
    polynomial_trajectory.color.r = 1.0;
  }
  if (_planner_state == PlanningStates::TAKING_OFF) {
    polynomial_trajectory.points.clear();
  }
  visual_pub_->publish(pyramids_bases);
  visual_pub_->publish(polynomial_trajectory);
}
