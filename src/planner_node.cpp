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

  // Benchmark status publisher (for automated testing)
  benchmark_status_pub = this->create_publisher<ground_system_msgs::msg::BenchmarkStatus>(
    "/benchmark/planner_status", 10);

  // Publishers based on runtime mode
  if (_runtime_mode == RuntimeModes::MAVROS || _runtime_mode == RuntimeModes::OMNIDRONES) {
    // Position/velocity/acceleration setpoints (PositionTarget)
    // Use absolute path since MAVROS is at root namespace, not Drone1 namespace
    raw_ref_pos_pub = this->create_publisher<mavros_msgs::msg::PositionTarget>("/mavros/setpoint_raw/local", 10);
  }
  if (_runtime_mode == RuntimeModes::OMNIDRONES) {
    // Velocity-only setpoints (TwistStamped) - alternative control mode
    vel_cmd_pub = this->create_publisher<geometry_msgs::msg::TwistStamped>("mavros/setpoint_velocity/cmd_vel", 10);
  }
  if (_runtime_mode == RuntimeModes::MAVROS) {
    // Throttled odom for zenoh bridge (100Hz -> 10Hz)
    odom_throttled_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
      "/mavros/local_position/odom_throttled", 5);
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

  // FBV (Fly-by-Voice) goal subscriber - receives VLM-extracted goals
  fbv_goal_sub = this->create_subscription<ground_system_msgs::msg::FBVGoal>(
    "/fbv_goal", 10,
    std::bind(&PlannerNode::fbv_goal_callback, this, std::placeholders::_1));

  // Takeoff command subscriber - triggers takeoff only (no goal)
  takeoff_sub = this->create_subscription<ground_system_msgs::msg::Takeoff>(
    "/takeoff", 10,
    std::bind(&PlannerNode::takeoff_callback, this, std::placeholders::_1));

  // FlyTo command subscriber - auto takeoff and fly to specified goal
  fly_to_sub = this->create_subscription<ground_system_msgs::msg::FlyTo>(
    "/fly_to", 10,
    std::bind(&PlannerNode::fly_to_callback, this, std::placeholders::_1));

  reset_sub = this->create_subscription<std_msgs::msg::Empty>(
    "/reset_planner", 10,
    std::bind(&PlannerNode::reset_callback, this, std::placeholders::_1));

  // Subscribe to odometry - use relative topic so namespace remapping works
  // When running in /Drone1 namespace, this becomes /Drone1/odometry
  odom_sub = this->create_subscription<nav_msgs::msg::Odometry>(
    "odometry", 5,
    std::bind(&PlannerNode::odometry_callback, this, std::placeholders::_1));

  // For MAVROS mode: also subscribe to raw odom to throttle it for zenoh
  if (_runtime_mode == RuntimeModes::MAVROS) {
    // MAVROS publishes with BEST_EFFORT QoS - must match for subscription to work
    // Use absolute paths since MAVROS is at root namespace, not Drone1 namespace
    rclcpp::QoS mavros_qos(5);
    mavros_qos.best_effort();
    
    mavros_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/mavros/local_position/odom", mavros_qos,
      [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
        // Throttle from 100Hz to 10Hz
        rclcpp::Time now = this->now();
        if ((now - last_odom_throttle_time_).seconds() >= kOdomThrottleInterval_) {
          odom_throttled_pub_->publish(*msg);
          last_odom_throttle_time_ = now;
        }
      });

    // Pose/twist/accel also use BEST_EFFORT QoS from MAVROS
  mav_pose_sub = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/mavros/local_position/pose", mavros_qos,
    std::bind(&PlannerNode::mav_pose_callback, this, std::placeholders::_1));

  mav_twist_sub = this->create_subscription<geometry_msgs::msg::TwistStamped>(
      "/mavros/local_position/velocity_body", mavros_qos,
    std::bind(&PlannerNode::mav_twist_callback, this, std::placeholders::_1));

    // mav_accel_sub = this->create_subscription<sensor_msgs::msg::Imu>(
    //   "/mavros/imu/data_raw", mavros_qos,
    //   std::bind(&PlannerNode::mav_accel_callback, this, std::placeholders::_1));

    mav_state_sub = this->create_subscription<mavros_msgs::msg::State>(
      "/mavros/state", 10,
      std::bind(&PlannerNode::ardupilot_status_callback, this, std::placeholders::_1));

    // MAVROS service clients (for real FC) - use absolute paths since services are at root namespace
    arming_srv = this->create_client<mavros_msgs::srv::CommandBool>("/mavros/cmd/arming");
    takeoff_srv = this->create_client<mavros_msgs::srv::CommandTOL>("/mavros/cmd/takeoff");
    land_srv = this->create_client<mavros_msgs::srv::CommandTOL>("/mavros/cmd/land");
    mode_srv = this->create_client<mavros_msgs::srv::SetMode>("/mavros/set_mode");
  }
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
  // Use node's current time (respects use_sim_time) instead of depth_msg timestamp
  // This ensures TF lookup works when mixing real camera (wall time) with SITL (sim time)
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

void PlannerNode::mission_callback(const ground_system_msgs::msg::StartSwarmMission::SharedPtr msg) {
  RCLCPP_INFO(this->get_logger(), "Received mission command: %s", msg->mission_name.c_str());
  
  if (mission_received_) {
    RCLCPP_WARN(this->get_logger(), "Mission already received, ignoring duplicate");
    return;
  }
  
  // Extract trial ID from mission name if it's a benchmark trial
  // Format: "benchmark_trial_N" where N is the trial number
  if (msg->mission_name.find("benchmark_trial_") == 0) {
    try {
      _current_trial_id = std::stoi(msg->mission_name.substr(16));
      RCLCPP_INFO(this->get_logger(), "Benchmark trial %d started", _current_trial_id);
    } catch (...) {
      _current_trial_id++;
    }
  } else {
    _current_trial_id++;
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
    _home_in_world_frame = _state.pose.position;
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

void PlannerNode::fbv_goal_callback(const ground_system_msgs::msg::FBVGoal::SharedPtr msg) {
  RCLCPP_INFO(this->get_logger(), "Received FBV goal (body FLU): '%s' -> Forward=%.2f, Left=%.2f, Up=%.2f [confidence: %.2f]",
              msg->target_label.c_str(), msg->goal_x, msg->goal_y, msg->goal_z, msg->confidence);
  
  if (mission_received_) {
    RCLCPP_WARN(this->get_logger(), "Mission already in progress, ignoring FBV goal");
    return;
  }
  
  // FBV goal is in body frame (FLU: Forward-Left-Up)
  // Convert to world frame by adding current position (like mission_callback)
  // Note: This assumes yaw=0 alignment (NWU world = FLU body at start)
  // For proper rotation, would need to apply yaw rotation
  if (_runtime_mode == RuntimeModes::OMNIDRONES) {
    // OmniDrones: NWU world frame, goal_x=Forward=North, goal_y=Left=West
    _goal_in_world_frame.x = _state.pose.position.x + msg->goal_x;
    _goal_in_world_frame.y = _state.pose.position.y + msg->goal_y;
    _goal_in_world_frame.z = msg->goal_z;
  } else {
    // MAVROS: ENU world frame
    _goal_in_world_frame.x = _state.pose.position.x - msg->goal_y;  // East = -Left
    _goal_in_world_frame.y = _state.pose.position.y + msg->goal_x;  // North = Forward
    _goal_in_world_frame.z = msg->goal_z;
  }
  
  RCLCPP_INFO(this->get_logger(), "Setting FBV goal to world (%.2f, %.2f, %.2f) - target: %s",
              _goal_in_world_frame.x, _goal_in_world_frame.y, _goal_in_world_frame.z,
              msg->target_label.c_str());
  _goal_set = true;
  mission_received_ = true;
  
  if (_runtime_mode == RuntimeModes::OMNIDRONES) {
    // Simulation mode: directly start trajectory control
    RCLCPP_WARN(this->get_logger(), "[SIM/FBV] Starting navigation to '%s'!", msg->target_label.c_str());
    set_auto_pilot_state_forced(PlanningStates::TAKING_OFF);
    _home_in_world_frame = _state.pose.position;
    steering_value = 0.0f;
    _steered = false;
    trajectory_queue_.clear();
    reference_trajectory_ = ruckig::Trajectory<3>();
    had_reference_trajectory = false;
  } else if (_runtime_mode == RuntimeModes::MAVROS) {
    // Real FC mode: will initiate GUIDED->ARM->TAKEOFF sequence in update_planner_state()
    RCLCPP_WARN(this->get_logger(), "[MAVROS/FBV] Goal received, initiating flight sequence to '%s'...",
                msg->target_label.c_str());
  }
}

void PlannerNode::takeoff_callback(const ground_system_msgs::msg::Takeoff::SharedPtr msg) {
  RCLCPP_INFO(this->get_logger(), "Received takeoff command: altitude = %.2f m", msg->altitude);
  
  if (_planner_state != PlanningStates::OFF) {
    RCLCPP_WARN(this->get_logger(), "Planner not in OFF state, ignoring takeoff command");
    return;
  }
  
  // Set takeoff altitude in goal_up_coordinate (used by update_planner_state for takeoff)
  _goal_up_coordinate = msg->altitude;
  
  // NOTE: For takeoff-only, we do NOT set _goal_set or mission_received_
  // This allows the planner to just takeoff and hover
  
  if (_runtime_mode == RuntimeModes::OMNIDRONES) {
    // Simulation mode: directly start takeoff
    RCLCPP_WARN(this->get_logger(), "[SIM] Starting takeoff to %.2f m!", msg->altitude);
    set_auto_pilot_state_forced(PlanningStates::TAKING_OFF);
    _home_in_world_frame = _state.pose.position;
    steering_value = 0.0f;
    _steered = false;
    trajectory_queue_.clear();
    reference_trajectory_ = ruckig::Trajectory<3>();
    had_reference_trajectory = false;
  } else if (_runtime_mode == RuntimeModes::MAVROS) {
    // Real FC mode: set flag to trigger GUIDED->ARM->TAKEOFF sequence in update_planner_state()
    // Do NOT immediately change state - let the FC sequence complete first
    RCLCPP_WARN(this->get_logger(), "[MAVROS] Takeoff command received, initiating takeoff to %.2f m...",
                msg->altitude);
    takeoff_requested_ = true;
  }
}

void PlannerNode::fly_to_callback(const ground_system_msgs::msg::FlyTo::SharedPtr msg) {
  RCLCPP_INFO(this->get_logger(), "Received fly_to command: (%.2f, %.2f, %.2f) NWU", 
              msg->x, msg->y, msg->z);
  RCLCPP_INFO(this->get_logger(), "Current position (from _state): (%.2f, %.2f, %.2f)",
              _state.pose.position.x, _state.pose.position.y, _state.pose.position.z);
  
  if (mission_received_) {
    RCLCPP_WARN(this->get_logger(), "Mission already in progress, ignoring fly_to command");
    return;
  }
  
  // FlyTo coordinates are in NWU world frame (absolute position)
  // Convert based on runtime mode's internal coordinate convention
  if (_runtime_mode == RuntimeModes::OMNIDRONES) {
    // OmniDrones uses NWU internally, no conversion needed
    _goal_in_world_frame.x = _state.pose.position.x + msg->x;  // North
    _goal_in_world_frame.y = _state.pose.position.y + msg->y;  // West
    _goal_in_world_frame.z = msg->z;  // Up
  } else {
    // MAVROS uses ENU internally
    // NWU -> ENU: X_enu = -Y_nwu (East = -West), Y_enu = X_nwu (North), Z same
    _goal_in_world_frame.x = _state.pose.position.x - msg->y;  // East = -West
    _goal_in_world_frame.y = _state.pose.position.y + msg->x;   // North
    _goal_in_world_frame.z = msg->z;   // Up
  }
  
  RCLCPP_INFO(this->get_logger(), "Setting fly_to goal to world frame (NWU for OmniDrones, ENU for MAVROS): (%.2f, %.2f, %.2f)",
              _goal_in_world_frame.x, _goal_in_world_frame.y, _goal_in_world_frame.z);
  _goal_set = true;
  mission_received_ = true;
  
  if (_runtime_mode == RuntimeModes::OMNIDRONES) {
    // Simulation mode: directly start trajectory control
    RCLCPP_WARN(this->get_logger(), "[SIM/FLY_TO] Starting navigation!");
    set_auto_pilot_state_forced(PlanningStates::TAKING_OFF);
    _home_in_world_frame = _state.pose.position;
    steering_value = 0.0f;
    _steered = false;
    trajectory_queue_.clear();
    reference_trajectory_ = ruckig::Trajectory<3>();
    had_reference_trajectory = false;
  } else if (_runtime_mode == RuntimeModes::MAVROS) {
    // Real FC mode: will initiate GUIDED->ARM->TAKEOFF sequence in update_planner_state()
    RCLCPP_WARN(this->get_logger(), "[MAVROS/FLY_TO] Goal received, initiating flight sequence...");
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
  takeoff_requested_ = false;
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
    auto min_stamp = std::min({_latest_pose_stamp, _latest_twist_stamp});
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
  // double qw = msg->pose.pose.orientation.w;
  // double qx = msg->pose.pose.orientation.x;
  // double qy = msg->pose.pose.orientation.y;
  // double qz = msg->pose.pose.orientation.z;
  // double yaw_rad = atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
  // double yaw_deg = yaw_rad * 180.0 / M_PI;
  // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
  //   "Odometry: pos=(%.2f, %.2f, %.2f), yaw=%.1f deg (%.2f rad)",
  //   msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z,
  //   yaw_deg, yaw_rad);

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
  // Trigger on either _goal_set (mission/fly_to) or takeoff_requested_ (takeoff-only)
  if (_runtime_mode == RuntimeModes::MAVROS && _planner_state == PlanningStates::OFF && 
      (_goal_set || takeoff_requested_)) {
    // Step 1: Switch to GUIDED mode if not already
    if (flight_controller_status.mode != "GUIDED" && !mode_switch_pending_) {
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
        "Attempting GUIDED mode switch... current mode: '%s', service ready: %d",
        flight_controller_status.mode.c_str(), mode_srv->service_is_ready());
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
        // Use goal altitude if set, otherwise use _goal_up_coordinate (from takeoff command)
        request->altitude = _goal_set ? _goal_in_world_frame.z : _goal_up_coordinate;
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
  // Only reset if we were actually flying (TRAJECTORY_CONTROL or later), not during startup
  if (_runtime_mode == RuntimeModes::MAVROS && 
      _planner_state != PlanningStates::OFF &&
      _planner_state != PlanningStates::TAKING_OFF &&
      !flight_controller_status.armed) {
    RCLCPP_WARN(this->get_logger(), "Vehicle disarmed, resetting planner");
    reset_callback(nullptr);
    return;
  }

  if (!_goal_set) return;

  geometry_msgs::msg::Point goal_in_world_frame = _goal_in_world_frame;
  goal_in_world_frame.z = _state.pose.position.z;
  double distance_to_goal = (geometryToEigen(_state.pose.position) - geometryToEigen(goal_in_world_frame)).norm();

  // Debug: print state transition values
  // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
  //     "State check: current_z=%.2f, goal_z=%.2f, threshold=%.2f, state=%d",
  //     _state.pose.position.z, _goal_in_world_frame.z, _goal_in_world_frame.z - 0.1, (int)_planner_state);

  // Transition from START to TRAJECTORY_CONTROL when altitude reached
  if (_state.pose.position.z >= (_goal_in_world_frame.z - 0.1) && _planner_state == PlanningStates::TAKING_OFF) {
    set_auto_pilot_state_forced(PlanningStates::TRAJECTORY_CONTROL);
  }
  // Transition to GO_TO_GOAL when near goal
  else if (_planner_state == PlanningStates::TRAJECTORY_CONTROL &&
           (distance_to_goal < _go_to_goal_threshold ||
            ((_state.pose.position.y + _go_to_goal_threshold / 10) > _goal_in_world_frame.y &&
             _runtime_mode == RuntimeModes::MAVROS))) {
    set_auto_pilot_state_forced(PlanningStates::GO_TO_GOAL);
  }
  // Land when at goal (MAVROS only)
  // else if (_runtime_mode == RuntimeModes::MAVROS &&
  //          _planner_state == PlanningStates::GO_TO_GOAL &&
  //          distance_to_goal < _go_to_goal_threshold * 0.2 &&
  //          !land_pending_) {
  //   if (land_srv->service_is_ready()) {
  //     auto request = std::make_shared<mavros_msgs::srv::CommandTOL::Request>();
  //     land_pending_ = true;
      
  //     land_srv->async_send_request(request,
  //       [this](rclcpp::Client<mavros_msgs::srv::CommandTOL>::SharedFuture future) {
  //         land_pending_ = false;
  //         try {
  //           auto response = future.get();
  //           if (response->success) {
  //             set_auto_pilot_state_forced(PlanningStates::LAND);
  //             RCLCPP_WARN(this->get_logger(), "Land command accepted!");
  //           } else {
  //             RCLCPP_ERROR(this->get_logger(), "Land command rejected by FCU");
  //           }
  //         } catch (const std::exception& e) {
  //           RCLCPP_ERROR(this->get_logger(), "Land service failed: %s", e.what());
  //         }
  //       });
  //   }
  // }
  // Reset after landing complete
  // else if (_runtime_mode == RuntimeModes::MAVROS &&
  //          _planner_state == PlanningStates::LAND && !flight_controller_status.armed) {
  //   reset_callback(nullptr);
  // }
}

void PlannerNode::track_trajectory() {
  // Don't track trajectory in non-flight states
  if (_planner_state == PlanningStates::LAND ||
      _planner_state == PlanningStates::OFF ||
      !_goal_set)
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
    // Also publish velocity command for velocity-only control mode
    // publish_velocity_command(reference_point);
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
  
  if (x < _fence_min_x || x > _fence_max_x ||
      y < _fence_min_y || y > _fence_max_y ||
      z < _fence_min_z || z > _fence_max_z) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
      "Setpoint (%.2f, %.2f, %.2f) ENU outside fence limits, not publishing", x, y, z);
    return;
  }
  
  mavros_msgs::msg::PositionTarget msg;
  msg.header.stamp = this->now();
  msg.coordinate_frame = 1;  // FRAME_LOCAL_NED (but MAVROS actually accepts ENU here - legacy behavior)
  msg.type_mask = 0;
  
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
  
  // Debug: Log the actual setpoint being published
  // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
  //   "PUBLISHING to setpoint_raw/local: pos=(%.2f,%.2f,%.2f) yaw=%.1f deg | vel=(%.2f,%.2f,%.2f)",
  //   msg.position.x, msg.position.y, msg.position.z, msg.yaw * 180.0 / M_PI,
  //   msg.velocity.x, msg.velocity.y, msg.velocity.z);
  
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

  Eigen::Vector3d trajectory_vector =
    geometryToEigen(reference_trajectory.get_terminal_position_in_world_frame()) -
    geometryToEigen(reference_trajectory.get_initial_position_in_world_frame());
  double terminal_heading = atan2f(trajectory_vector[1], trajectory_vector[0]);
  reference_point.heading = terminal_heading;
  Eigen::Vector3d current_euler_angles = quaternionToEulerAnglesZYX(geometryToEigen(_state.pose.orientation));

  // Comment to turn off steering
  // if (fabs(steering_value) > 1e-6)
  //   reference_point.heading = current_euler_angles(2) + steering_value;

  reference_point.position = geometryToEigen(position_in_world_frame);
  reference_point.velocity = geometryToEigen(velocity_in_world_frame);
  reference_point.acceleration = geometryToEigen(acceleration_in_world_frame);
  reference_point.jerk = geometryToEigen(jerk_in_world_frame);
}

void PlannerNode::publish_benchmark_status(uint8_t status) {
  auto msg = ground_system_msgs::msg::BenchmarkStatus();
  msg.header.stamp = this->now();
  msg.trial_id = _current_trial_id;
  msg.drone_id = 1;  // TODO: Get from namespace if multi-drone
  msg.status = status;
  
  msg.position.x = _state.pose.position.x;
  msg.position.y = _state.pose.position.y;
  msg.position.z = _state.pose.position.z;
  
  msg.goal.x = _goal_in_world_frame.x;
  msg.goal.y = _goal_in_world_frame.y;
  msg.goal.z = _goal_in_world_frame.z;
  
  Eigen::Vector3d pos(_state.pose.position.x, _state.pose.position.y, _state.pose.position.z);
  Eigen::Vector3d goal(_goal_in_world_frame.x, _goal_in_world_frame.y, _goal_in_world_frame.z);
  msg.distance_to_goal = (pos - goal).norm();
  
  if (_trial_started) {
    msg.elapsed_time = (this->now() - _trial_start_time).seconds();
  } else {
    msg.elapsed_time = 0.0;
  }
  
  benchmark_status_pub->publish(msg);
}

void PlannerNode::set_auto_pilot_state_forced(const PlanningStates& new_state) {
  const rclcpp::Time time_now = this->now();

  if (new_state != PlanningStates::TRAJECTORY_CONTROL && !trajectory_queue_.empty()) {
    trajectory_queue_.clear();
  }
  time_of_switch_to_current_state_ = time_now;
  _planner_state = new_state;

  std::string state_name;
  uint8_t benchmark_status = 0;  // IN_PROGRESS
  switch (_planner_state) {
    case PlanningStates::OFF:
      state_name = "OFF";
      break;
    case PlanningStates::TAKING_OFF:
      state_name = "TAKING_OFF";
      _trial_start_time = time_now;  // Start benchmark timer
      _trial_started = true;
      break;
    case PlanningStates::TRAJECTORY_CONTROL:
      state_name = "TRAJECTORY_CONTROL";
      break;
    case PlanningStates::GO_TO_GOAL:
      state_name = "GO_TO_GOAL";
      benchmark_status = 1;  // GOAL_REACHED
      break;
    case PlanningStates::LAND:
      state_name = "LAND";
      benchmark_status = 1;  // GOAL_REACHED (landing is success)
      break;
  }
  RCLCPP_WARN(this->get_logger(), "Switched to %s state", state_name.c_str());
  
  // Publish benchmark status on state transitions
  publish_benchmark_status(benchmark_status);
}

void PlannerNode::img_callback(const sm::Image::SharedPtr depth_msg) {
  if (_planner_state != PlanningStates::TRAJECTORY_CONTROL)
    return;
  
  // Two time references needed:
  // 1. Wall clock - for real sensors (depth camera) that stamp with wall time
  // 2. Node time (this->now()) - for ROS messages that use sim time when use_sim_time=true
  auto wall_now = std::chrono::system_clock::now();
  double wall_now_sec = std::chrono::duration<double>(wall_now.time_since_epoch()).count();
  rclcpp::Time time_now = this->now();  // For state/transform checks (sim time domain)
  
  // Check if depth image is too old
  // Use wall clock because real camera stamps with wall time, not sim time
  rclcpp::Time depth_time = rclcpp::Time(depth_msg->header.stamp);
  double depth_age = wall_now_sec - depth_time.seconds();
  
  if (depth_age > _depth_age_threshold) {
    // RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
    //                      "Depth image too old (%.3f s > %.3f s threshold), rejecting",
    //                      depth_age, _depth_age_threshold);
    return;
  }
  
  geometry_msgs::msg::TransformStamped world_to_body, body_to_world;
  geometry_msgs::msg::Point position_world_frame;
  geometry_msgs::msg::Vector3 velocity_world_frame;
  geometry_msgs::msg::Vector3 acceleration_world_frame;
  geometry_msgs::msg::Vector3 velocity_body_frame;
  geometry_msgs::msg::Vector3 acceleration_body_frame;
  double state_timestamp;  // Store state timestamp for staleness check

  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    // Check if state data is too old before using for planning
    state_timestamp = _state.t;
    double state_age = time_now.seconds() - state_timestamp;
    if (state_age > _state_age_threshold) {
      // RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
      //       "State data too old (%.3f s > %.3f s threshold), rejecting",
      //       state_age, _state_age_threshold);
      return;
    }
    
    // Lookup for transforms in the TF2 transforming tree

    try {
      body_to_world = to_world_buffer->lookupTransform(
        _world_frame, _vehicle_frame, tf2::TimePointZero);
      world_to_body = to_vehicle_buffer->lookupTransform(
        _vehicle_frame, _world_frame, tf2::TimePointZero);
    } catch (tf2::TransformException& ex) {
      RCLCPP_WARN(this->get_logger(), "%s", ex.what());
      return;
    }
    
    // Check if transforms are too old
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
    
    position_world_frame = _state.pose.position;
    if (_runtime_mode == RuntimeModes::OMNIDRONES) {
      // OmniDrones: velocity and acceleration are in world frame (ENU)
      velocity_world_frame = _state.velocity.linear;
      acceleration_world_frame = _state.acceleration.linear;
      tf2::doTransform(velocity_world_frame, velocity_body_frame, world_to_body);
      tf2::doTransform(acceleration_world_frame, acceleration_body_frame, world_to_body);
    } else if (_runtime_mode == RuntimeModes::MAVROS) {
      // in MAVROS, raw velocity and acceleration are in FLU (body frame)
      velocity_body_frame = _state.velocity.linear;
      acceleration_body_frame = _state.acceleration.linear;
    }
    tf2::doTransform(acceleration_body_frame, acceleration_world_frame, body_to_world);
  }

  if (acceleration_world_frame.x > _acc_planning_threshold || acceleration_world_frame.y > _acc_planning_threshold)
    return;
  
  if (velocity_body_frame.y > _vel_planning_threshold || velocity_body_frame.z > _vel_planning_threshold)
    return;

  geometry_msgs::msg::Vector3 velocity_camera_frame, acceleration_camera_frame;
  // Transform from body frame (FLU) to camera frame (RDF)
  frame_transform::transform_body_to_camera(velocity_body_frame, velocity_camera_frame);
  if (_collision_checking_method == CollisionCheckingMethod::PYRAMID) {
    frame_transform::transform_body_to_camera(acceleration_body_frame, acceleration_camera_frame);
  }

  ruckig::InputParameter<3> initial_state_camera_frame;
  double forward_time = _planning_cycle_time + (time_now.seconds() - state_timestamp);
  initial_state_camera_frame.current_position = {
    velocity_camera_frame.x * forward_time,
    velocity_camera_frame.y * forward_time,
    velocity_camera_frame.z * forward_time
  };
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
  
  // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
  //   "img_callback: goal_vector_camera_frame=[%.2f, %.2f, %.2f], depth_mat size=%dx%d",
  //   goal_vector_camera_frame[0], goal_vector_camera_frame[1], goal_vector_camera_frame[2],
  //   depth_mat.cols, depth_mat.rows);

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
  // Use node's current time (respects use_sim_time) for TF compatibility
  // when mixing real camera (wall time) with SITL (sim time)
  cloudMessage.header.stamp = this->now();
  cloudMessage.header.frame_id = _vehicle_frame;
  point_cloud_pub->publish(cloudMessage);
  delete cloud;  // Free memory

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
  // {
  //   geometry_msgs::msg::Point start, end;
  //   start.x = _state.pose.position.x;
  //   start.y = _state.pose.position.y;
  //   start.z = _state.pose.position.z;
    
  //   // Scale velocity for visibility (1 m/s = 1 meter arrow)
  //   double vel_scale = 1.0;
  //   end.x = start.x + _state.velocity.linear.x * vel_scale;
  //   end.y = start.y + _state.velocity.linear.y * vel_scale;
  //   end.z = start.z + _state.velocity.linear.z * vel_scale;
    
  //   auto vel_arrow = create_arrow_marker(0, "velocity_world", start, end, 0.0, 1.0, 1.0, 1.0, _world_frame);
  //   debug_markers.markers.push_back(vel_arrow);
    
  //   Log velocity periodically
  //   double vel_mag = std::sqrt(_state.velocity.linear.x * _state.velocity.linear.x +
  //                              _state.velocity.linear.y * _state.velocity.linear.y +
  //                              _state.velocity.linear.z * _state.velocity.linear.z);
  //   RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
  //     "VELOCITY (world): [%.2f, %.2f, %.2f] mag=%.2f m/s",
  //     _state.velocity.linear.x, _state.velocity.linear.y, _state.velocity.linear.z, vel_mag);
  // }
  
  // 2. VELOCITY IN BODY FRAME (MAGENTA) - transformed to body frame, shown from origin in base_link
  // {
  //   geometry_msgs::msg::Vector3 velocity_world_frame, velocity_body_frame;
  //   velocity_world_frame = _state.velocity.linear;
  //   tf2::doTransform(velocity_world_frame, velocity_body_frame, world_to_body);
    
  //   geometry_msgs::msg::Point start, end;
  //   start.x = start.y = start.z = 0.0;
    
  //   double vel_scale = 1.0;
  //   end.x = velocity_body_frame.x * vel_scale;
  //   end.y = velocity_body_frame.y * vel_scale;
  //   end.z = velocity_body_frame.z * vel_scale;
    
  //   auto vel_body_arrow = create_arrow_marker(1, "velocity_body", start, end, 1.0, 0.0, 1.0, 1.0, _vehicle_frame);
  //   debug_markers.markers.push_back(vel_body_arrow);
    
  //   RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
  //     "VELOCITY (body FLU): [%.2f, %.2f, %.2f]",
  //     velocity_body_frame.x, velocity_body_frame.y, velocity_body_frame.z);
  // }
  
  // 3. GOAL VECTOR (GREEN) - direction from drone to goal in world frame
  // if (_goal_set) {
  //   geometry_msgs::msg::Point start, end;
  //   start.x = _state.pose.position.x;
  //   start.y = _state.pose.position.y;
  //   start.z = _state.pose.position.z;
    
  //   // Normalize and scale for visibility
  //   double dx = _goal_in_world_frame.x - start.x;
  //   double dy = _goal_in_world_frame.y - start.y;
  //   double dz = _goal_in_world_frame.z - start.z;
  //   double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
  //   double arrow_len = std::min(dist, 3.0);  // Cap at 3m for visibility
    
  //   if (dist > 0.1) {
  //     end.x = start.x + (dx / dist) * arrow_len;
  //     end.y = start.y + (dy / dist) * arrow_len;
  //     end.z = start.z + (dz / dist) * arrow_len;
      
  //     auto goal_arrow = create_arrow_marker(2, "goal_vector", start, end, 0.0, 1.0, 0.0, 1.0, _world_frame);
  //     debug_markers.markers.push_back(goal_arrow);
      
  //     RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
  //       "GOAL VECTOR (world): [%.2f, %.2f, %.2f] dist=%.2f",
  //       dx, dy, dz, dist);
  //   }
  // }
  // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
  //   "Point cloud: %zu total points, %zu valid points, frame_id=%s",
  //   cloud->points.size(), valid_points, cloud->header.frame_id.c_str());

  
  // 4. GOAL IN BODY FRAME (YELLOW) - goal direction in body frame (FLU)
  // if (_goal_set) {
  //   geometry_msgs::msg::PointStamped goal_world, goal_body;
  //   goal_world.header.frame_id = _world_frame;
  //   goal_world.point = _goal_in_world_frame;
    
  //   try {
  //     tf2::doTransform(goal_world, goal_body, world_to_body);
      
  //     geometry_msgs::msg::Point start, end;
  //     start.x = start.y = start.z = 0.0;
      
  //     double dist = std::sqrt(goal_body.point.x * goal_body.point.x +
  //                             goal_body.point.y * goal_body.point.y +
  //                             goal_body.point.z * goal_body.point.z);
  //     double arrow_len = std::min(dist, 3.0);
      
  //     if (dist > 0.1) {
  //       end.x = (goal_body.point.x / dist) * arrow_len;
  //       end.y = (goal_body.point.y / dist) * arrow_len;
  //       end.z = (goal_body.point.z / dist) * arrow_len;
        
  //       auto goal_body_arrow = create_arrow_marker(3, "goal_body", start, end, 1.0, 1.0, 0.0, 1.0, _vehicle_frame);
  //       debug_markers.markers.push_back(goal_body_arrow);
        
  //       RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
  //         "GOAL (body FLU): [%.2f, %.2f, %.2f]",
  //         goal_body.point.x, goal_body.point.y, goal_body.point.z);
  //     }
  //   } catch (tf2::TransformException& ex) {
  //     // Ignore transform errors
  //   }
  // }
  
  // 5. EXPLORATION VECTOR / GOAL IN CAMERA FRAME (RED) - this is what the planner uses
  // if (_goal_set) {
  //   geometry_msgs::msg::PointStamped goal_world, goal_body;
  //   goal_world.header.frame_id = _world_frame;
  //   goal_world.point = _goal_in_world_frame;
    
  //   try {
  //     tf2::doTransform(goal_world, goal_body, world_to_body);
      
  //     // Transform from body (FLU) to camera (RDF)
  //     geometry_msgs::msg::Point goal_camera;
  //     frame_transform::transform_body_to_camera(goal_body.point, goal_camera);
      
  //     // The exploration vector in camera frame (RDF: X=Right, Y=Down, Z=Forward)
  //     // Visualize in vehicle frame by transforming back for display
  //     // Camera RDF to Body FLU: x_body = z_cam, y_body = -x_cam, z_body = -y_cam
  //     geometry_msgs::msg::Point start, end;
  //     start.x = start.y = start.z = 0.0;
      
  //     double dist = std::sqrt(goal_camera.x * goal_camera.x +
  //                             goal_camera.y * goal_camera.y +
  //                             goal_camera.z * goal_camera.z);
  //     double arrow_len = std::min(dist, 3.0);
      
  //     // Compute exploration cost values for debugging
  //     Eigen::Vector3d goal_vector_camera_frame(goal_camera.x, goal_camera.y, goal_camera.z);
  //     Eigen::Vector3d exploration_unit = goal_vector_camera_frame.normalized();
      
  //     // Simulate a sample endpoint (e.g., 1m forward in camera frame) to show cost calculation
  //     Eigen::Vector3d sample_endpoint(0.0, 0.0, 1.0);  // 1m forward in camera RDF
  //     double direction_cost = -exploration_unit.dot(sample_endpoint.normalized());
  //     double distance_cost = -sample_endpoint.dot(exploration_unit);
      
  //     if (dist > 0.1) {
  //       // Display the camera-frame vector transformed back to body frame for visualization
  //       // Camera RDF -> Body FLU: Forward=Z_cam, Left=-X_cam, Up=-Y_cam
  //       end.x = (goal_camera.z / dist) * arrow_len;  // Forward (body X) = Camera Z
  //       end.y = (-goal_camera.x / dist) * arrow_len; // Left (body Y) = -Camera X
  //       end.z = (-goal_camera.y / dist) * arrow_len; // Up (body Z) = -Camera Y
        
  //       auto explore_arrow = create_arrow_marker(4, "exploration_camera", start, end, 1.0, 0.0, 0.0, 1.0, _vehicle_frame);
  //       debug_markers.markers.push_back(explore_arrow);
        
  //       // Get traveling cost type as string
  //       std::string cost_type_str = (_traveling_cost == TravelingCost::DIRECTION) ? "DIRECTION" : "DISTANCE";
  //       double active_cost = (_traveling_cost == TravelingCost::DIRECTION) ? direction_cost : distance_cost;
        
  //       RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
  //         "EXPLORATION (camera RDF): [%.2f, %.2f, %.2f] | Cost type: %s | "
  //         "Direction cost: %.3f | Distance cost: %.3f | Active cost: %.3f",
  //         goal_camera.x, goal_camera.y, goal_camera.z,
  //         cost_type_str.c_str(), direction_cost, distance_cost, active_cost);
  //     }
  //   } catch (tf2::TransformException& ex) {
  //     // Ignore transform errors
  //   }
  // }
  
  // 6. DRONE FORWARD DIRECTION (WHITE) - shows drone heading
  // {
  //   geometry_msgs::msg::Point start, end;
  //   start.x = start.y = start.z = 0.0;
  //   end.x = 1.0;  // 1m forward in body frame
  //   end.y = 0.0;
  //   end.z = 0.0;
    
  //   auto forward_arrow = create_arrow_marker(5, "drone_forward", start, end, 1.0, 1.0, 1.0, 0.8, _vehicle_frame);
  //   debug_markers.markers.push_back(forward_arrow);
  // }
  
  // // Publish all debug markers
  // debug_vectors_pub_->publish(debug_markers);
  
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
  polynomial_trajectory.type = visualization_msgs::msg::Marker::LINE_STRIP;
  goal_marker.type = visualization_msgs::msg::Marker::CUBE;
  // LINE_STRIP markers use only the x component of scale, for the line width
  polynomial_trajectory.scale.x = 0.05;
  goal_marker.scale.x = 0.2;
  goal_marker.scale.y = 0.2;
  goal_marker.scale.z = 0.2;
  // Trajectory is blue, goal is green
  polynomial_trajectory.color.b = 1.0;
  polynomial_trajectory.color.a = 1.0;
  goal_marker.color.g = 1.0;
  goal_marker.color.a = 1.0;

  // Publish goal marker
  if (_goal_set) {
    goal_marker.pose.position.x = _goal_in_world_frame.x;
    goal_marker.pose.position.y = _goal_in_world_frame.y;
    goal_marker.pose.position.z = _goal_in_world_frame.z;
    goal_marker.pose.orientation.w = 1.0;
    visual_pub->publish(goal_marker);
  }

  // Publish polynomial trajectory
  if (!had_reference_trajectory) {
    return;
  }
  
  double trajectory_duration = reference_trajectory_.get_duration();
  if (trajectory_duration < 0.01) {
    polynomial_trajectory.points.clear();
    visual_pub->publish(polynomial_trajectory);
    return;
  }
  geometry_msgs::msg::Point p;
  for (int i = 0; i <= 100; i++) {
    geometry_msgs::msg::Point position =
      reference_trajectory_.get_position_in_world_frame(trajectory_duration * i / 100);
    p.x = position.x;
    p.y = position.y;
    p.z = position.z;
    polynomial_trajectory.points.push_back(p);
  }

    // Change color to red when in GO_TO_GOAL state
    if (_planner_state == PlanningStates::GO_TO_GOAL) {
      polynomial_trajectory.color.b = 0.0;
      polynomial_trajectory.color.r = 1.0;
    }
    if (_planner_state == PlanningStates::TAKING_OFF) {
      polynomial_trajectory.points.clear();
    }
    visual_pub->publish(polynomial_trajectory);
}
