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

  // Mission command subscriber (start trigger — after upload + ACK)
  mission_sub = this->create_subscription<ground_system_msgs::msg::StartSwarmMission>(
    "/start_swarm_mission", 10,
    std::bind(&PlannerNode::mission_callback, this, std::placeholders::_1));

  // Mission upload subscriber (waypoints — Phase 1, before start)
  mission_upload_sub = this->create_subscription<ground_system_msgs::msg::SwarmMissionUpload>(
    "/swarm_mission_upload", 10,
    std::bind(&PlannerNode::mission_upload_callback, this, std::placeholders::_1));

  // Mission ACK publisher
  mission_ack_pub_ = this->create_publisher<ground_system_msgs::msg::SwarmMissionAck>(
    "/swarm_mission_ack", 10);

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

  // OPUS coordination setup
  // Parse drone_id from namespace (e.g. /Drone1 -> 1)
  std::string ns = this->get_namespace();
  if (ns.find("Drone") != std::string::npos) {
    try {
      opus_drone_id_ = static_cast<uint8_t>(std::stoi(ns.substr(ns.find("Drone") + 5)));
    } catch (...) {
      opus_drone_id_ = 1;
    }
  } else {
    // Fallback: try DRONE_ID environment variable
    const char* env_id = std::getenv("DRONE_ID");
    opus_drone_id_ = env_id ? static_cast<uint8_t>(std::stoi(env_id)) : 1;
  }

  // Enable OPUS if OPUS_ENABLED env var is set (default: true for multi-drone)
  const char* opus_env = std::getenv("OPUS_ENABLED");
  opus_enabled_ = (opus_env == nullptr) || (std::string(opus_env) != "false");

  if (opus_enabled_) {
    // Use RELIABLE QoS matching the coordinator for OPUS coordination messages
    rclcpp::QoS opus_qos(10);
    opus_qos.reliable();

    opus_plan_request_pub_ = this->create_publisher<ground_system_msgs::msg::OpusPlanRequest>(
      "/opus/plan_request", opus_qos);
    opus_plan_abort_pub_ = this->create_publisher<ground_system_msgs::msg::OpusPlanAbort>(
      "/opus/plan_abort", opus_qos);
    opus_trajectory_submit_pub_ = this->create_publisher<ground_system_msgs::msg::OpusTrajectorySubmit>(
      "/opus/trajectory_submit", opus_qos);
    opus_plan_grant_sub_ = this->create_subscription<ground_system_msgs::msg::OpusPlanGrant>(
      "/opus/plan_grant", opus_qos,
      std::bind(&PlannerNode::opus_plan_grant_callback, this, std::placeholders::_1));
    opus_trajectory_ack_sub_ = this->create_subscription<ground_system_msgs::msg::OpusTrajectoryAck>(
      "/opus/trajectory_ack", opus_qos,
      std::bind(&PlannerNode::opus_trajectory_ack_callback, this, std::placeholders::_1));
    RCLCPP_INFO(this->get_logger(), "OPUS coordination enabled (drone_id=%d)", opus_drone_id_);
  } else {
    RCLCPP_INFO(this->get_logger(), "OPUS coordination disabled (solo mode)");
  }

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
  // Use node's current time for consistent TF lookup timing
  // All nodes use wall clock (no use_sim_time in OmniDrones/real hardware)
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

// Phase 1: Upload waypoints, convert FLU→world, log, ACK back to GCS
void PlannerNode::mission_upload_callback(const ground_system_msgs::msg::SwarmMissionUpload::SharedPtr msg) {
  RCLCPP_INFO(this->get_logger(), "Received mission upload: '%s' (%zu waypoints, loop_count=%u)",
              msg->mission_name.c_str(), msg->waypoints.size(), msg->loop_count);

  // Build ACK message
  auto ack = ground_system_msgs::msg::SwarmMissionAck();
  ack.drone_id = opus_drone_id_;
  ack.mission_name = msg->mission_name;

  if (mission_received_) {
    RCLCPP_WARN(this->get_logger(), "Mission already in flight, rejecting upload");
    ack.success = false;
    ack.message = "Mission already in flight";
    ack.num_waypoints = 0;
    mission_ack_pub_->publish(ack);
    return;
  }

  if (msg->waypoints.empty()) {
    RCLCPP_WARN(this->get_logger(), "Mission '%s' has no waypoints, rejecting", msg->mission_name.c_str());
    ack.success = false;
    ack.message = "No waypoints provided";
    ack.num_waypoints = 0;
    mission_ack_pub_->publish(ack);
    return;
  }

  // Extract trial ID from mission name if it's a benchmark trial
  if (msg->mission_name.find("benchmark_trial_") == 0) {
    try {
      _current_trial_id = std::stoi(msg->mission_name.substr(16));
      RCLCPP_INFO(this->get_logger(), "Benchmark trial %d", _current_trial_id);
    } catch (...) {
      _current_trial_id++;
    }
  } else {
    _current_trial_id++;
  }

  _mission_name = msg->mission_name;

  // Store home position before computing goal (relative waypoints use this)
  _home_in_world_frame = _state.pose.position;

  // Get initial yaw for FLU→world conversion
  double qw = _state.pose.orientation.w;
  double qx = _state.pose.orientation.x;
  double qy = _state.pose.orientation.y;
  double qz = _state.pose.orientation.z;
  double initial_yaw = std::atan2(2.0 * (qw * qz + qx * qy),
                                   1.0 - 2.0 * (qy * qy + qz * qz));

  _waypoint_list.assign(msg->waypoints.begin(), msg->waypoints.end());
  _current_waypoint_index = 0;
  _remaining_loops = msg->loop_count > 0 ? msg->loop_count - 1 : 0;
  _waypoint_mission_active = true;

  // Derive takeoff altitude from first waypoint (can be overridden by prior takeoff command)
  if (_goal_up_coordinate <= 0.0) {
    _goal_up_coordinate = msg->waypoints[0].up;
  }

  // Convert FLU waypoints to world frame ONCE using initial yaw.
  convert_waypoints_to_world(initial_yaw);

  RCLCPP_INFO(this->get_logger(), "Mission '%s' uploaded: %zu waypoints, %u total passes, initial_yaw=%.1f deg",
              msg->mission_name.c_str(), _waypoint_list.size(),
              msg->loop_count > 0 ? msg->loop_count : 1,
              initial_yaw * 180.0 / M_PI);

  set_goal_from_waypoint();
  _goal_heading = compute_heading_to_goal();
  _goal_set = true;
  mission_uploaded_ = true;
  log_mission_to_yaml(msg);

  // Publish ACK
  ack.success = true;
  ack.num_waypoints = static_cast<uint32_t>(msg->waypoints.size());
  ack.message = "OK";
  mission_ack_pub_->publish(ack);
  RCLCPP_INFO(this->get_logger(), "Mission ACK sent (success, %u waypoints)", ack.num_waypoints);
}

// Phase 2: Start swarming (triggered by GCS after all agents ACK)
void PlannerNode::mission_callback(const ground_system_msgs::msg::StartSwarmMission::SharedPtr msg) {
  RCLCPP_INFO(this->get_logger(), "Received START command: '%s'", msg->mission_name.c_str());

  if (mission_received_) {
    RCLCPP_WARN(this->get_logger(), "Mission already started, ignoring duplicate start");
    return;
  }

  if (!mission_uploaded_) {
    RCLCPP_ERROR(this->get_logger(), "No mission uploaded! Upload waypoints before starting.");
    return;
  }

  if (msg->mission_name != _mission_name) {
    RCLCPP_ERROR(this->get_logger(), "Mission name mismatch: uploaded='%s', start='%s'",
                 _mission_name.c_str(), msg->mission_name.c_str());
    return;
  }

  mission_received_ = true;

  RCLCPP_INFO(this->get_logger(), "Starting mission '%s': goal=(%.2f, %.2f, %.2f), heading=%.1f deg",
              _mission_name.c_str(),
              _goal_in_world_frame.x, _goal_in_world_frame.y, _goal_in_world_frame.z,
              _goal_heading * 180.0 / M_PI);

  if (_runtime_mode == RuntimeModes::OMNIDRONES) {
    RCLCPP_WARN(this->get_logger(), "[SIM] Starting navigation!");
    set_auto_pilot_state_forced(PlanningStates::TAKING_OFF);
    steering_value = 0.0f;
    _steered = false;
    trajectory_queue_.clear();
    reference_trajectory_ = ruckig::Trajectory<3>();
    had_reference_trajectory = false;
  } else if (_runtime_mode == RuntimeModes::MAVROS) {
    RCLCPP_WARN(this->get_logger(), "[MAVROS] Initiating flight sequence...");
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
  // We need to rotate by current yaw to convert to world frame
  
  // Get current yaw from quaternion
  double qw = _state.pose.orientation.w;
  double qx = _state.pose.orientation.x;
  double qy = _state.pose.orientation.y;
  double qz = _state.pose.orientation.z;
  double yaw = atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
  
  // Store current position for goal computation (use same state for both goal and heading)
  double current_x = _state.pose.position.x;
  double current_y = _state.pose.position.y;
  
  RCLCPP_INFO(this->get_logger(), "Current pos: (%.2f, %.2f, %.2f), yaw: %.1f deg (%.2f rad)",
              current_x, current_y, _state.pose.position.z, yaw * 180.0 / M_PI, yaw);
  
  // Body FLU: X=Forward, Y=Left, Z=Up
  double body_forward = msg->goal_x;
  double body_left = msg->goal_y;
  double world_offset_x, world_offset_y;  // Offset in world frame
  
  if (_runtime_mode == RuntimeModes::OMNIDRONES) {
    // OmniDrones: NWU world frame (X=North, Y=West, Z=Up)
    // Rotate body FLU by yaw to get world NWU
    // At yaw=0: body Forward=North, body Left=West
    double world_north = body_forward * cos(yaw) - body_left * sin(yaw);
    double world_west = body_forward * sin(yaw) + body_left * cos(yaw);
    world_offset_x = world_north;
    world_offset_y = world_west;
    _goal_in_world_frame.x = current_x + world_north;
    _goal_in_world_frame.y = current_y + world_west;
    _goal_in_world_frame.z = msg->goal_z;
  } else {
    // MAVROS: ENU world frame (X=East, Y=North, Z=Up)
    // Rotate body FLU by yaw to get world ENU
    // In ENU: yaw=0 means facing East, yaw increases CCW (toward North)
    // Body FLU at yaw=0: Forward=East, Left=North
    // Body FLU at yaw=90°: Forward=North, Left=West=-East
    double world_east = body_forward * cos(yaw) - body_left * sin(yaw);
    double world_north = body_forward * sin(yaw) + body_left * cos(yaw);
    world_offset_x = world_east;
    world_offset_y = world_north;
    _goal_in_world_frame.x = current_x + world_east;
    _goal_in_world_frame.y = current_y + world_north;
    _goal_in_world_frame.z = msg->goal_z;
  }
  
  // Compute goal heading NOW using the offset (not later with potentially different state)
  // This ensures heading matches the goal direction we just computed
  _goal_heading = atan2(world_offset_y, world_offset_x);
  
  // If _goal_up_coordinate wasn't set by a prior takeoff command, use the FBV goal altitude
  // This ensures takeoff completion check works even without explicit takeoff command
  if (_goal_up_coordinate <= 0.0) {
    _goal_up_coordinate = msg->goal_z;
  }
  
  RCLCPP_INFO(this->get_logger(), "World offset: (%.2f, %.2f), heading: %.1f deg",
              world_offset_x, world_offset_y, _goal_heading * 180.0 / M_PI);
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
  RCLCPP_INFO(this->get_logger(), "Received fly_to command (body FLU): Forward=%.2f, Left=%.2f, Up=%.2f", 
              msg->x, msg->y, msg->z);
  RCLCPP_INFO(this->get_logger(), "Current position (from _state): (%.2f, %.2f, %.2f)",
              _state.pose.position.x, _state.pose.position.y, _state.pose.position.z);
  
  if (mission_received_) {
    RCLCPP_WARN(this->get_logger(), "Mission already in progress, ignoring fly_to command");
    return;
  }
  
  // FlyTo coordinates are in body frame (FLU: Forward-Left-Up)
  // We need to rotate by current yaw to convert to world frame
  
  // Get current yaw from quaternion
  double qw = _state.pose.orientation.w;
  double qx = _state.pose.orientation.x;
  double qy = _state.pose.orientation.y;
  double qz = _state.pose.orientation.z;
  double yaw = atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
  
  // Store current position for goal computation
  double current_x = _state.pose.position.x;
  double current_y = _state.pose.position.y;
  
  RCLCPP_INFO(this->get_logger(), "Current yaw: %.1f deg (%.2f rad)", yaw * 180.0 / M_PI, yaw);
  
  // Body FLU: X=Forward, Y=Left, Z=Up
  double body_forward = msg->x;
  double body_left = msg->y;
  double world_offset_x, world_offset_y;  // Offset in world frame
  
  if (_runtime_mode == RuntimeModes::OMNIDRONES) {
    // OmniDrones: NWU world frame (X=North, Y=West, Z=Up)
    // Rotate body FLU by yaw to get world NWU
    double world_north = body_forward * cos(yaw) - body_left * sin(yaw);
    double world_west = body_forward * sin(yaw) + body_left * cos(yaw);
    world_offset_x = world_north;
    world_offset_y = world_west;
    _goal_in_world_frame.x = current_x + world_north;
    _goal_in_world_frame.y = current_y + world_west;
    _goal_in_world_frame.z = msg->z;
  } else {
    // MAVROS: ENU world frame (X=East, Y=North, Z=Up)
    // Rotate body FLU by yaw to get world ENU
    double world_east = body_forward * cos(yaw) - body_left * sin(yaw);
    double world_north = body_forward * sin(yaw) + body_left * cos(yaw);
    world_offset_x = world_east;
    world_offset_y = world_north;
    _goal_in_world_frame.x = current_x + world_east;
    _goal_in_world_frame.y = current_y + world_north;
    _goal_in_world_frame.z = msg->z;
  }
  
  // Compute goal heading NOW using the offset (not later with potentially different state)
  _goal_heading = atan2(world_offset_y, world_offset_x);
  
  // If _goal_up_coordinate wasn't set by a prior takeoff command, use the fly_to altitude
  // This ensures takeoff completion check works even without explicit takeoff command
  if (_goal_up_coordinate <= 0.0) {
    _goal_up_coordinate = msg->z;
  }
  
  RCLCPP_INFO(this->get_logger(), "World offset: (%.2f, %.2f), heading: %.1f deg",
              world_offset_x, world_offset_y, _goal_heading * 180.0 / M_PI);
  RCLCPP_INFO(this->get_logger(), "Setting fly_to goal to world frame: (%.2f, %.2f, %.2f)",
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
  mission_uploaded_ = false;
  mode_switch_pending_ = false;
  arming_pending_ = false;
  takeoff_pending_ = false;
  land_pending_ = false;
  takeoff_requested_ = false;
  trajectory_queue_.clear();
  reference_trajectory_ = ruckig::Trajectory<3>();
  had_reference_trajectory = false;
  // Reset fence breach recovery state
  _has_valid_setpoint = false;
  _last_valid_position = Eigen::Vector3d(0.0, 0.0, 0.0);
  _last_valid_heading = 0.0;
  // Reset takeoff altitude
  _goal_up_coordinate = 0.0;
  // Reset waypoint mission state
  _waypoint_list.clear();
  _world_waypoints.clear();
  _waypoint_headings.clear();
  _waypoint_hold_times.clear();
  _current_waypoint_index = 0;
  _remaining_loops = 0;
  _waypoint_mission_active = false;
  // Reset OPUS coordination state
  if (opus_enabled_) {
    const std::lock_guard<std::mutex> olock(opus_mutex_);
    opus_granted_ = false;
    opus_ack_pending_ = false;
    opus_request_pending_ = false;
    opus_queued_ = false;
    opus_grant_time_ = std::chrono::steady_clock::time_point{};
    opus_active_trajectories_.clear();
  }
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
  // Only for OmniDrones mode - MAVROS handles TF publishing via tf.send: true
  if (_runtime_mode == RuntimeModes::OMNIDRONES) {
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
    
    // Note: _goal_heading is now computed in the goal callbacks (fbv_goal_callback, fly_to_callback)
    // at the same time as goal position, using the same state snapshot.
    // For takeoff-only (no goal), use default heading of 0.
    if (!_goal_set) {
      _goal_heading = 0.0;  // Default heading for takeoff-only
    }
    // Log the heading that will be used
    RCLCPP_INFO_ONCE(this->get_logger(), "Using goal heading: %.1f deg (%.2f rad)",
                     _goal_heading * 180.0 / M_PI, _goal_heading);
    
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

  // Transition from TAKING_OFF to ALIGNING_HEADING when takeoff altitude reached
  // Use _goal_up_coordinate (from takeoff command) for transition, not _goal_in_world_frame.z
  // This allows FBV/fly_to goals with different altitudes to work properly
  double takeoff_complete_altitude = _goal_up_coordinate - 0.1;
  if (_state.pose.position.z >= takeoff_complete_altitude && _planner_state == PlanningStates::TAKING_OFF) {
    RCLCPP_INFO(this->get_logger(), "Takeoff complete at z=%.2f (threshold=%.2f), aligning heading to goal",
                _state.pose.position.z, takeoff_complete_altitude);
    if (_waypoint_mission_active && _current_waypoint_index < _waypoint_headings.size()) {
      _goal_heading = _waypoint_headings[_current_waypoint_index];
    } else {
      _goal_heading = compute_heading_to_goal();
    }
    set_auto_pilot_state_forced(PlanningStates::ALIGNING_HEADING);
  }
  // ALIGNING_HEADING → TRAJECTORY_CONTROL when heading is within threshold
  else if (_planner_state == PlanningStates::ALIGNING_HEADING) {
    // Get current yaw from quaternion
    double qw = _state.pose.orientation.w;
    double qx = _state.pose.orientation.x;
    double qy = _state.pose.orientation.y;
    double qz = _state.pose.orientation.z;
    double current_yaw = std::atan2(2.0 * (qw * qz + qx * qy),
                                     1.0 - 2.0 * (qy * qy + qz * qz));
    double yaw_error = _goal_heading - current_yaw;
    // Normalize to [-pi, pi]
    while (yaw_error > M_PI) yaw_error -= 2.0 * M_PI;
    while (yaw_error < -M_PI) yaw_error += 2.0 * M_PI;

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
      "Aligning heading: current=%.1f deg, target=%.1f deg, error=%.1f deg",
      current_yaw * 180.0 / M_PI, _goal_heading * 180.0 / M_PI,
      yaw_error * 180.0 / M_PI);

    if (std::abs(yaw_error) < kHeadingAlignThreshold_) {
      RCLCPP_INFO(this->get_logger(), "Heading aligned (error=%.1f deg), starting trajectory control",
                  yaw_error * 180.0 / M_PI);
      // Clear trajectory state so planner starts fresh
      had_reference_trajectory = false;
      trajectory_queue_.clear();
      set_auto_pilot_state_forced(PlanningStates::TRAJECTORY_CONTROL);
    }
  }
  // Transition to GO_TO_GOAL / HOLDING_WAYPOINT when near goal
  else if (_planner_state == PlanningStates::TRAJECTORY_CONTROL &&
           (distance_to_goal < _go_to_goal_threshold ||
            ((_state.pose.position.y + _go_to_goal_threshold / 10) > _goal_in_world_frame.y &&
             _runtime_mode == RuntimeModes::MAVROS))) {
    if (_waypoint_mission_active) {
      // Abort any OPUS planning state since we're done with this segment
      opus_abort_planning("Waypoint reached, advancing");
      set_auto_pilot_state_forced(PlanningStates::HOLDING_WAYPOINT);
    } else {
      opus_abort_planning("Goal reached");
      set_auto_pilot_state_forced(PlanningStates::GO_TO_GOAL);
    }
  }
  // HOLDING_WAYPOINT: hold position for hold_time, then advance
  else if (_planner_state == PlanningStates::HOLDING_WAYPOINT) {
    double time_in_state = (this->now() - time_of_switch_to_current_state_).seconds();
    double hold_time = 1.0;  // default
    if (_current_waypoint_index < _waypoint_hold_times.size()) {
      hold_time = _waypoint_hold_times[_current_waypoint_index];
    }
    if (time_in_state >= hold_time) {
      if (advance_waypoint()) {
        // More waypoints → align heading then plan to next
        set_auto_pilot_state_forced(PlanningStates::ALIGNING_HEADING);
      } else {
        // Mission complete → go to goal (final position hold)
        set_auto_pilot_state_forced(PlanningStates::GO_TO_GOAL);
      }
    }
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

  // ALIGNING_HEADING and HOLDING_WAYPOINT: hold position, rotate toward goal
  if (_planner_state == PlanningStates::ALIGNING_HEADING ||
      _planner_state == PlanningStates::HOLDING_WAYPOINT) {
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

void PlannerNode::publish_benchmark_status(uint8_t status) {
  auto msg = ground_system_msgs::msg::BenchmarkStatus();
  msg.header.stamp = this->now();
  msg.trial_id = _current_trial_id;
  msg.drone_id = opus_drone_id_;
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

// ============================================================================
// Waypoint Mission Helpers
// ============================================================================

void PlannerNode::set_goal_from_waypoint() {
  if (_current_waypoint_index >= _world_waypoints.size()) return;

  const auto& wp_world = _world_waypoints[_current_waypoint_index];
  _goal_in_world_frame.x = wp_world.x();
  _goal_in_world_frame.y = wp_world.y();
  _goal_in_world_frame.z = wp_world.z();

  // Use pre-computed heading (NaN was resolved during convert_waypoints_to_world)
  _goal_heading = _waypoint_headings[_current_waypoint_index];

  RCLCPP_INFO(this->get_logger(), "Waypoint %zu/%zu: world=(%.2f,%.2f,%.2f) heading=%.1f deg",
              _current_waypoint_index + 1, _world_waypoints.size(),
              _goal_in_world_frame.x, _goal_in_world_frame.y, _goal_in_world_frame.z,
              _goal_heading * 180.0 / M_PI);
}

bool PlannerNode::advance_waypoint() {
  _current_waypoint_index++;

  if (_current_waypoint_index >= _world_waypoints.size()) {
    if (_remaining_loops > 0) {
      _remaining_loops--;
      _current_waypoint_index = 0;
      RCLCPP_INFO(this->get_logger(), "Waypoint loop restart (%u loops remaining)",
                  _remaining_loops);
    } else {
      RCLCPP_INFO(this->get_logger(), "Waypoint mission complete");
      _waypoint_mission_active = false;
      return false;
    }
  }

  set_goal_from_waypoint();
  return true;
}

void PlannerNode::convert_waypoints_to_world(double initial_yaw) {
  _world_waypoints.clear();
  _waypoint_headings.clear();
  _waypoint_hold_times.clear();

  double cos_yaw = std::cos(initial_yaw);
  double sin_yaw = std::sin(initial_yaw);
  double home_x = _home_in_world_frame.x;
  double home_y = _home_in_world_frame.y;

  for (size_t i = 0; i < _waypoint_list.size(); i++) {
    const auto& wp = _waypoint_list[i];

    // Rotate FLU (Forward/Left/Up) by initial yaw to get world-frame offset
    // Body FLU: X=Forward, Y=Left, Z=Up
    // World ENU: X=East, Y=North, Z=Up (MAVROS)
    // World NWU: X=North, Y=West, Z=Up (OmniDrones)
    // The rotation is the same in both conventions — yaw rotates the horizontal plane
    double world_dx = cos_yaw * wp.forward - sin_yaw * wp.left;
    double world_dy = sin_yaw * wp.forward + cos_yaw * wp.left;

    Eigen::Vector3d wp_world(home_x + world_dx, home_y + world_dy, wp.up);
    _world_waypoints.push_back(wp_world);
    _waypoint_hold_times.push_back(wp.hold_time);

    // Pre-compute heading: if NaN, compute from home (or previous WP) toward this WP
    if (std::isnan(wp.heading)) {
      Eigen::Vector3d from;
      if (i == 0) {
        from = Eigen::Vector3d(home_x, home_y, wp.up);
      } else {
        from = _world_waypoints[i - 1];
      }
      double dx = wp_world.x() - from.x();
      double dy = wp_world.y() - from.y();
      double auto_heading = std::atan2(dy, dx);
      _waypoint_headings.push_back(auto_heading);
    } else {
      // Absolute heading from message (already in world frame)
      _waypoint_headings.push_back(wp.heading);
    }

    RCLCPP_INFO(this->get_logger(),
      "  WP%zu FLU(fwd=%.2f, left=%.2f, up=%.2f) -> world(%.2f, %.2f, %.2f) heading=%.1f deg",
      i + 1, wp.forward, wp.left, wp.up,
      wp_world.x(), wp_world.y(), wp_world.z(),
      _waypoint_headings.back() * 180.0 / M_PI);
  }
}

double PlannerNode::compute_heading_to_goal() const {
  double dx = _goal_in_world_frame.x - _state.pose.position.x;
  double dy = _goal_in_world_frame.y - _state.pose.position.y;
  // ENU: yaw = atan2(dy, dx) → 0 = East, π/2 = North
  // NWU: yaw = atan2(dy, dx) → 0 = North, π/2 = West
  // Both use the same atan2(y_diff, x_diff) in their respective frames
  return std::atan2(dy, dx);
}

void PlannerNode::log_mission_to_yaml(const ground_system_msgs::msg::SwarmMissionUpload::SharedPtr& msg) {
  // Build YAML mission log
  YAML::Emitter out;
  out << YAML::BeginMap;
  out << YAML::Key << "mission_name" << YAML::Value << msg->mission_name;
  out << YAML::Key << "drone_id" << YAML::Value << static_cast<int>(opus_drone_id_);
  out << YAML::Key << "timestamp" << YAML::Value << this->now().seconds();

  // Initial position and heading at mission receive time
  out << YAML::Key << "initial_position" << YAML::Value << YAML::BeginMap;
  out << YAML::Key << "x" << YAML::Value << _home_in_world_frame.x;
  out << YAML::Key << "y" << YAML::Value << _home_in_world_frame.y;
  out << YAML::Key << "z" << YAML::Value << _home_in_world_frame.z;
  out << YAML::EndMap;

  double qw = _state.pose.orientation.w;
  double qx = _state.pose.orientation.x;
  double qy = _state.pose.orientation.y;
  double qz = _state.pose.orientation.z;
  double initial_yaw = std::atan2(2.0 * (qw * qz + qx * qy),
                                   1.0 - 2.0 * (qy * qy + qz * qz));
  out << YAML::Key << "initial_yaw_rad" << YAML::Value << initial_yaw;
  out << YAML::Key << "initial_yaw_deg" << YAML::Value << initial_yaw * 180.0 / M_PI;

  out << YAML::Key << "loop_count" << YAML::Value << msg->loop_count;

  // Original FLU waypoints (as received from GCS)
  out << YAML::Key << "waypoints_flu" << YAML::Value << YAML::BeginSeq;
  for (const auto& wp : msg->waypoints) {
    out << YAML::BeginMap;
    out << YAML::Key << "forward" << YAML::Value << wp.forward;
    out << YAML::Key << "left" << YAML::Value << wp.left;
    out << YAML::Key << "up" << YAML::Value << wp.up;
    out << YAML::Key << "heading" << YAML::Value << wp.heading;
    out << YAML::Key << "hold_time" << YAML::Value << wp.hold_time;
    out << YAML::EndMap;
  }
  out << YAML::EndSeq;

  // Computed world-frame waypoints (what the agent will actually navigate to)
  out << YAML::Key << "waypoints_world" << YAML::Value << YAML::BeginSeq;
  for (size_t i = 0; i < _world_waypoints.size(); i++) {
    out << YAML::BeginMap;
    out << YAML::Key << "x" << YAML::Value << _world_waypoints[i].x();
    out << YAML::Key << "y" << YAML::Value << _world_waypoints[i].y();
    out << YAML::Key << "z" << YAML::Value << _world_waypoints[i].z();
    out << YAML::Key << "heading_rad" << YAML::Value << _waypoint_headings[i];
    out << YAML::Key << "hold_time" << YAML::Value << _waypoint_hold_times[i];
    out << YAML::EndMap;
  }
  out << YAML::EndSeq;
  out << YAML::EndMap;

  // Write to /tmp/mission_log_<drone_id>_<timestamp>.yaml
  auto time_now = std::chrono::system_clock::now();
  auto epoch = std::chrono::duration_cast<std::chrono::seconds>(
    time_now.time_since_epoch()).count();
  std::string filename = "/tmp/mission_log_drone" +
    std::to_string(opus_drone_id_) + "_" + std::to_string(epoch) + ".yaml";

  std::ofstream file(filename);
  if (file.is_open()) {
    file << out.c_str();
    file.close();
    RCLCPP_INFO(this->get_logger(), "Mission logged to %s", filename.c_str());
  } else {
    RCLCPP_WARN(this->get_logger(), "Failed to write mission log to %s", filename.c_str());
  }
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
    case PlanningStates::ALIGNING_HEADING:
      state_name = "ALIGNING_HEADING";
      break;
    case PlanningStates::TRAJECTORY_CONTROL:
      state_name = "TRAJECTORY_CONTROL";
      break;
    case PlanningStates::GO_TO_GOAL:
      state_name = "GO_TO_GOAL";
      benchmark_status = 1;  // GOAL_REACHED
      break;
    case PlanningStates::HOLDING_WAYPOINT:
      state_name = "HOLDING_WAYPOINT";
      break;
    case PlanningStates::LAND:
      state_name = "LAND";
      benchmark_status = 1;  // GOAL_REACHED (landing is success)
      break;
    case PlanningStates::WAITING_FOR_OPUS:
      state_name = "WAITING_FOR_OPUS";
      break;
    case PlanningStates::FINISHED:
      state_name = "FINISHED";
      benchmark_status = 1;
      break;
  }
  RCLCPP_WARN(this->get_logger(), "Switched to %s state", state_name.c_str());
  
  // Publish benchmark status on state transitions
  publish_benchmark_status(benchmark_status);
}

void PlannerNode::img_callback(const sm::Image::SharedPtr depth_msg) {
  if (_planner_state != PlanningStates::TRAJECTORY_CONTROL)
    return;
  
  rclcpp::Time time_now = this->now();  // Wall clock (no use_sim_time)
  
  // Check if depth image is too old
  // Both this->now() and depth_msg->header.stamp use wall clock
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
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "State data too old (%.3f s > %.3f s threshold), rejecting",
            state_age, _state_age_threshold);
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

  // OPUS coordination: manage planning lock and ACK-gated execution
  if (opus_enabled_) {
    const std::lock_guard<std::mutex> lock(opus_mutex_);

    // If waiting for ACK (trajectory submitted, awaiting execution permission)
    if (opus_ack_pending_) {
      auto elapsed = std::chrono::steady_clock::now() - opus_submit_time_;
      if (std::chrono::duration<double>(elapsed).count() > kOpusAckTimeout_) {
        RCLCPP_WARN(this->get_logger(),
          "OPUS: ACK timeout (%.1fs), releasing lock and re-requesting",
          kOpusAckTimeout_);
        opus_ack_pending_ = false;
        opus_granted_ = false;
        opus_request_pending_ = false;
      }
      return;  // Wait for ACK callback to push trajectory
    }

    // If waiting for grant
    if (!opus_granted_) {
      if (opus_queued_) {
        // Queued at the coordinator — but guard against a lost grant message.
        // If we've been queued longer than kOpusQueueTimeout_, clear the
        // queued state and re-request so the coordinator can re-issue a grant.
        auto q_elapsed = std::chrono::steady_clock::now() - opus_queue_time_;
        if (std::chrono::duration<double>(q_elapsed).count() > kOpusQueueTimeout_) {
          RCLCPP_WARN(this->get_logger(),
            "OPUS: Queue timeout (%.1fs), re-requesting planning lock",
            kOpusQueueTimeout_);
          opus_queued_ = false;
          opus_request_pending_ = false;
        }
        return;  // Already queued at the coordinator, wait for the grant
      }

      if (!opus_request_pending_) {
        opus_request_planning_lock();
        opus_request_pending_ = true;
        opus_request_time_ = std::chrono::steady_clock::now();
      } else {
        auto elapsed = std::chrono::steady_clock::now() - opus_request_time_;
        if (std::chrono::duration<double>(elapsed).count() > kOpusGrantTimeout_) {
          RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "OPUS: Grant timeout (%.1fs), re-requesting", kOpusGrantTimeout_);
          opus_request_pending_ = false;  // Will re-request next cycle
        }
      }
      return;  // Wait for grant before planning
    }
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
    if (opus_should_abort_replanning()) {
      opus_abort_planning(
        "No feasible trajectory found within local OPUS replanning timeout");
    }
    return;
  }
  // New traj generated — assign transforms (opt_traj is local, no lock needed)
  opt_traj.assign_body_to_world_transform(body_to_world);
  opt_traj.assign_world_to_body_transform(world_to_body);

  // OPUS: check against swarm trajectories and submit to GCS
  if (opus_enabled_) {
    std::vector<ground_system_msgs::msg::RuckigTrajectory> active_trajs;
    {
      const std::lock_guard<std::mutex> olock(opus_mutex_);
      active_trajs = opus_active_trajectories_;
    }

    // Note: position_world_frame was captured earlier from odometry (ENU coordinates)
    // Both trajectories are compared in world frame (ENU) for meaningful distances
    if (!is_trajectory_safe_against_swarm(opt_traj, initial_state_camera_frame,
          this->now(), body_to_world, position_world_frame, active_trajs)) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
        "Trajectory rejected: spatio-temporal collision with swarm");
      if (opus_should_abort_replanning()) {
        opus_abort_planning(
          "No swarm-safe trajectory found within local OPUS replanning timeout");
      }
      return;
    }

    // Submit trajectory to GCS; ACK callback will push to trajectory_queue_
    opus_submit_trajectory(opt_traj, initial_state_camera_frame, body_to_world, position_world_frame);
    {
      const std::lock_guard<std::mutex> olock(opus_mutex_);
      opus_pending_trajectory_ = opt_traj;
      opus_ack_pending_ = true;
      opus_submit_time_ = std::chrono::steady_clock::now();
    }
    return;  // Wait for ACK (execution permission) before pushing to queue
  }

  // Non-OPUS mode: push trajectory directly
  {
    const std::lock_guard<std::mutex> tlock(trajectory_mutex_);
    steering_value = 0.0f;
    _steered = false;
    trajectory_queue_.push_back(opt_traj);
  }
}

// ============================================================================
// OPUS Coordination Functions
// ============================================================================

void PlannerNode::opus_plan_grant_callback(const ground_system_msgs::msg::OpusPlanGrant::SharedPtr msg) {
  if (msg->drone_id != opus_drone_id_) return;

  const std::lock_guard<std::mutex> lock(opus_mutex_);
  if (msg->permitted) {
    opus_granted_ = true;
    opus_ack_pending_ = false;
    opus_queued_ = false;
    opus_request_pending_ = false;
    opus_grant_time_ = std::chrono::steady_clock::now();
    opus_active_trajectories_.assign(
      msg->active_trajectories.begin(), msg->active_trajectories.end());
    RCLCPP_INFO(this->get_logger(), "OPUS: Planning lock GRANTED (%zu active trajectories)",
                opus_active_trajectories_.size());
  } else {
    opus_granted_ = false;
    opus_queued_ = true;
    opus_request_pending_ = true;
    opus_grant_time_ = std::chrono::steady_clock::time_point{};
    opus_queue_time_ = std::chrono::steady_clock::now();
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
      "OPUS: Queued at position %d", msg->queue_position);
  }
}

void PlannerNode::opus_trajectory_ack_callback(const ground_system_msgs::msg::OpusTrajectoryAck::SharedPtr msg) {
  if (msg->drone_id != opus_drone_id_) return;

  const std::lock_guard<std::mutex> olock(opus_mutex_);
  if (!opus_ack_pending_) return;  // Stale/unexpected ACK

  opus_ack_pending_ = false;
  opus_granted_ = false;
  opus_queued_ = false;
  opus_request_pending_ = false;
  opus_grant_time_ = std::chrono::steady_clock::time_point{};

  if (msg->accepted) {
    RCLCPP_INFO(this->get_logger(), "OPUS: Trajectory ACCEPTED — executing");
    // ACK = execution permission: push pending trajectory to queue
    const std::lock_guard<std::mutex> tlock(trajectory_mutex_);
    steering_value = 0.0f;
    _steered = false;
    trajectory_queue_.push_back(opus_pending_trajectory_);
  } else {
    RCLCPP_WARN(this->get_logger(), "OPUS: Trajectory REJECTED by GCS: %s — will re-plan",
                msg->reason.c_str());
  }
}

void PlannerNode::opus_request_planning_lock() {
  auto msg = ground_system_msgs::msg::OpusPlanRequest();
  msg.header.stamp = this->now();
  msg.drone_id = opus_drone_id_;
  opus_plan_request_pub_->publish(msg);
  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
    "OPUS: Requesting planning lock (drone_id=%d)", opus_drone_id_);
}

bool PlannerNode::opus_should_abort_replanning(double* elapsed_sec) {
  const std::lock_guard<std::mutex> lock(opus_mutex_);
  if (!opus_enabled_ || !opus_granted_ || opus_ack_pending_ || opus_queued_ ||
      opus_local_replan_timeout_ <= 0.0) {
    return false;
  }

  const double elapsed = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - opus_grant_time_).count();
  if (elapsed_sec != nullptr) {
    *elapsed_sec = elapsed;
  }

  return elapsed >= opus_local_replan_timeout_;
}

void PlannerNode::opus_abort_planning(const std::string& reason) {
  double elapsed = 0.0;
  {
    const std::lock_guard<std::mutex> lock(opus_mutex_);
    if (!opus_enabled_ || !(opus_granted_ || opus_queued_ || opus_request_pending_)) {
      return;
    }

    if (opus_granted_) {
      elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - opus_grant_time_).count();
    }

    opus_granted_ = false;
    opus_ack_pending_ = false;
    opus_request_pending_ = false;
    opus_queued_ = false;
    opus_grant_time_ = std::chrono::steady_clock::time_point{};
    opus_active_trajectories_.clear();
  }

  auto msg = ground_system_msgs::msg::OpusPlanAbort();
  msg.header.stamp = this->now();
  msg.drone_id = opus_drone_id_;
  msg.reason = reason;
  opus_plan_abort_pub_->publish(msg);

  if (elapsed > 0.0) {
    RCLCPP_WARN(this->get_logger(),
      "OPUS: Aborting planning after %.2fs: %s",
      elapsed, reason.c_str());
  } else {
    RCLCPP_WARN(this->get_logger(),
      "OPUS: Aborting planning: %s",
      reason.c_str());
  }
}

Eigen::Vector3d PlannerNode::transform_camera_to_world(
    const Eigen::Vector3d& camera_vec,
    const geometry_msgs::msg::TransformStamped& body_to_world,
    bool is_position,
    const geometry_msgs::msg::Point& world_position) {
  // Camera (RDF: Right-Down-Forward) to Body (FLU: Forward-Left-Up) rotation:
  // cam_x(right) -> body_-y(left, negated)
  // cam_y(down) -> body_-z(up, negated)
  // cam_z(forward) -> body_x(forward)
  Eigen::Vector3d body_vec;
  body_vec.x() =  camera_vec.z();  // forward
  body_vec.y() = -camera_vec.x();  // left (negated right)
  body_vec.z() = -camera_vec.y();  // up (negated down)

  // Body (FLU) to World (ENU) rotation via quaternion from TF
  tf2::Quaternion q(
    body_to_world.transform.rotation.x,
    body_to_world.transform.rotation.y,
    body_to_world.transform.rotation.z,
    body_to_world.transform.rotation.w);
  tf2::Vector3 body_tf(body_vec.x(), body_vec.y(), body_vec.z());
  tf2::Vector3 world_tf = tf2::quatRotate(q, body_tf);

  Eigen::Vector3d result(world_tf.x(), world_tf.y(), world_tf.z());

  // For positions, add current world position offset
  // (camera frame positions are relative to current drone position)
  if (is_position) {
    result.x() += world_position.x;
    result.y() += world_position.y;
    result.z() += world_position.z;
  }

  return result;
}

void PlannerNode::opus_submit_trajectory(
    const ruckig::Trajectory<3>& traj,
    const ruckig::InputParameter<3>& input,
    const geometry_msgs::msg::TransformStamped& body_to_world,
    const geometry_msgs::msg::Point& world_position) {
  // Transform camera-frame trajectory to world-frame (ENU) for GCS collision checking
  // All drones submit trajectories in the shared world frame so inter-drone
  // distances are meaningful.
  //
  // Frame convention:
  //   - Camera: RDF (Right-Down-Forward) - depth image frame
  //   - Body: FLU (Forward-Left-Up) - MAVROS/OmniDrones body frame
  //   - World: ENU (East-North-Up) - shared local origin (VICON or sim)

  ground_system_msgs::msg::RuckigTrajectory msg;
  msg.header.stamp = this->now();
  msg.drone_id = opus_drone_id_;

  // Current state vectors in camera frame
  Eigen::Vector3d curr_pos_cam(input.current_position[0], input.current_position[1], input.current_position[2]);
  Eigen::Vector3d curr_vel_cam(input.current_velocity[0], input.current_velocity[1], input.current_velocity[2]);
  Eigen::Vector3d curr_acc_cam(input.current_acceleration[0], input.current_acceleration[1], input.current_acceleration[2]);

  // Target state vectors in camera frame
  Eigen::Vector3d tgt_pos_cam(input.target_position[0], input.target_position[1], input.target_position[2]);
  Eigen::Vector3d tgt_vel_cam(input.target_velocity[0], input.target_velocity[1], input.target_velocity[2]);
  Eigen::Vector3d tgt_acc_cam(input.target_acceleration[0], input.target_acceleration[1], input.target_acceleration[2]);

  // Transform to world frame (ENU)
  Eigen::Vector3d curr_pos_world = transform_camera_to_world(curr_pos_cam, body_to_world, true, world_position);
  Eigen::Vector3d curr_vel_world = transform_camera_to_world(curr_vel_cam, body_to_world, false, world_position);
  Eigen::Vector3d curr_acc_world = transform_camera_to_world(curr_acc_cam, body_to_world, false, world_position);

  Eigen::Vector3d tgt_pos_world = transform_camera_to_world(tgt_pos_cam, body_to_world, true, world_position);
  Eigen::Vector3d tgt_vel_world = transform_camera_to_world(tgt_vel_cam, body_to_world, false, world_position);
  Eigen::Vector3d tgt_acc_world = transform_camera_to_world(tgt_acc_cam, body_to_world, false, world_position);

  // Pack into message (world frame - ENU coordinates)
  msg.current_position[0] = curr_pos_world.x();
  msg.current_position[1] = curr_pos_world.y();
  msg.current_position[2] = curr_pos_world.z();
  msg.current_velocity[0] = curr_vel_world.x();
  msg.current_velocity[1] = curr_vel_world.y();
  msg.current_velocity[2] = curr_vel_world.z();
  msg.current_acceleration[0] = curr_acc_world.x();
  msg.current_acceleration[1] = curr_acc_world.y();
  msg.current_acceleration[2] = curr_acc_world.z();

  msg.target_position[0] = tgt_pos_world.x();
  msg.target_position[1] = tgt_pos_world.y();
  msg.target_position[2] = tgt_pos_world.z();
  msg.target_velocity[0] = tgt_vel_world.x();
  msg.target_velocity[1] = tgt_vel_world.y();
  msg.target_velocity[2] = tgt_vel_world.z();
  msg.target_acceleration[0] = tgt_acc_world.x();
  msg.target_acceleration[1] = tgt_acc_world.y();
  msg.target_acceleration[2] = tgt_acc_world.z();

  // Max constraints remain the same (frame-independent magnitudes)
  for (int i = 0; i < 3; i++) {
    msg.max_velocity[i] = input.max_velocity[i];
    msg.max_acceleration[i] = input.max_acceleration[i];
    msg.max_jerk[i] = input.max_jerk[i];
  }

  msg.start_time = this->now().seconds();
  msg.duration = traj.get_duration();

  auto submit_msg = ground_system_msgs::msg::OpusTrajectorySubmit();
  submit_msg.header.stamp = this->now();
  submit_msg.trajectory = msg;
  opus_trajectory_submit_pub_->publish(submit_msg);

  RCLCPP_INFO(this->get_logger(),
    "OPUS: Submitted trajectory (world: [%.2f,%.2f,%.2f] -> [%.2f,%.2f,%.2f], dur=%.2fs)",
    curr_pos_world.x(), curr_pos_world.y(), curr_pos_world.z(),
    tgt_pos_world.x(), tgt_pos_world.y(), tgt_pos_world.z(),
    traj.get_duration());
}

ground_system_msgs::msg::RuckigTrajectory PlannerNode::ruckig_input_to_msg(
    const ruckig::InputParameter<3>& input, double start_time, double duration) {
  ground_system_msgs::msg::RuckigTrajectory msg;
  msg.header.stamp = this->now();
  msg.drone_id = opus_drone_id_;

  for (int i = 0; i < 3; i++) {
    msg.current_position[i] = input.current_position[i];
    msg.current_velocity[i] = input.current_velocity[i];
    msg.current_acceleration[i] = input.current_acceleration[i];
    msg.target_position[i] = input.target_position[i];
    msg.target_velocity[i] = input.target_velocity[i];
    msg.target_acceleration[i] = input.target_acceleration[i];
    msg.max_velocity[i] = input.max_velocity[i];
    msg.max_acceleration[i] = input.max_acceleration[i];
    msg.max_jerk[i] = input.max_jerk[i];
  }

  msg.start_time = start_time;
  msg.duration = duration;
  return msg;
}

bool PlannerNode::is_trajectory_safe_against_swarm(
    const ruckig::Trajectory<3>& planned_traj,
    const ruckig::InputParameter<3>& input_camera_frame,
    const rclcpp::Time& planned_start_time,
    const geometry_msgs::msg::TransformStamped& body_to_world,
    const geometry_msgs::msg::Point& world_position,
    const std::vector<ground_system_msgs::msg::RuckigTrajectory>& active_trajectories) {
  // Both planned trajectory and active_trajectories must be compared in the same frame.
  // active_trajectories are already in world frame (ENU) from the GCS.
  // We need to transform the planned trajectory from camera frame to world frame.

  double safety_distance = 2.0 * _planning_vehicle_radius;
  double planned_start_sec = planned_start_time.seconds();
  double planned_duration = planned_traj.get_duration();
  double planned_end_sec = planned_start_sec + planned_duration;

  // Build world-frame input for the planned trajectory
  Eigen::Vector3d curr_pos_cam(input_camera_frame.current_position[0],
                                input_camera_frame.current_position[1],
                                input_camera_frame.current_position[2]);
  Eigen::Vector3d curr_vel_cam(input_camera_frame.current_velocity[0],
                                input_camera_frame.current_velocity[1],
                                input_camera_frame.current_velocity[2]);
  Eigen::Vector3d curr_acc_cam(input_camera_frame.current_acceleration[0],
                                input_camera_frame.current_acceleration[1],
                                input_camera_frame.current_acceleration[2]);
  Eigen::Vector3d tgt_pos_cam(input_camera_frame.target_position[0],
                               input_camera_frame.target_position[1],
                               input_camera_frame.target_position[2]);
  Eigen::Vector3d tgt_vel_cam(input_camera_frame.target_velocity[0],
                               input_camera_frame.target_velocity[1],
                               input_camera_frame.target_velocity[2]);
  Eigen::Vector3d tgt_acc_cam(input_camera_frame.target_acceleration[0],
                               input_camera_frame.target_acceleration[1],
                               input_camera_frame.target_acceleration[2]);

  // Transform to world frame
  Eigen::Vector3d curr_pos_world = transform_camera_to_world(curr_pos_cam, body_to_world, true, world_position);
  Eigen::Vector3d curr_vel_world = transform_camera_to_world(curr_vel_cam, body_to_world, false, world_position);
  Eigen::Vector3d curr_acc_world = transform_camera_to_world(curr_acc_cam, body_to_world, false, world_position);
  Eigen::Vector3d tgt_pos_world = transform_camera_to_world(tgt_pos_cam, body_to_world, true, world_position);
  Eigen::Vector3d tgt_vel_world = transform_camera_to_world(tgt_vel_cam, body_to_world, false, world_position);
  Eigen::Vector3d tgt_acc_world = transform_camera_to_world(tgt_acc_cam, body_to_world, false, world_position);

  // Build world-frame ruckig input for planned trajectory
  ruckig::InputParameter<3> planned_input_world;
  planned_input_world.current_position = {curr_pos_world.x(), curr_pos_world.y(), curr_pos_world.z()};
  planned_input_world.current_velocity = {curr_vel_world.x(), curr_vel_world.y(), curr_vel_world.z()};
  planned_input_world.current_acceleration = {curr_acc_world.x(), curr_acc_world.y(), curr_acc_world.z()};
  planned_input_world.target_position = {tgt_pos_world.x(), tgt_pos_world.y(), tgt_pos_world.z()};
  planned_input_world.target_velocity = {tgt_vel_world.x(), tgt_vel_world.y(), tgt_vel_world.z()};
  planned_input_world.target_acceleration = {tgt_acc_world.x(), tgt_acc_world.y(), tgt_acc_world.z()};
  // Copy kinematic limits (frame-independent)
  for (int i = 0; i < 3; i++) {
    planned_input_world.max_velocity[i] = input_camera_frame.max_velocity[i];
    planned_input_world.max_acceleration[i] = input_camera_frame.max_acceleration[i];
    planned_input_world.max_jerk[i] = input_camera_frame.max_jerk[i];
  }

  // Regenerate trajectory in world frame
  ruckig::Ruckig<3> otg_planned;
  ruckig::Trajectory<3> planned_traj_world;
  auto planned_result = otg_planned.calculate(planned_input_world, planned_traj_world);
  if (planned_result < 0) {
    RCLCPP_WARN(this->get_logger(), "OPUS: Failed to reconstruct planned trajectory in world frame");
    return false;  // Can't verify, reject for safety
  }

  for (const auto& existing : active_trajectories) {
    if (existing.drone_id == opus_drone_id_) continue;  // Skip own trajectory

    // Find overlapping time interval
    double ex_start = existing.start_time;
    double ex_end = ex_start + existing.duration;
    double overlap_start = std::max(planned_start_sec, ex_start);
    double overlap_end = std::min(planned_end_sec, ex_end);

    if (overlap_start >= overlap_end) continue;  // No temporal overlap

    // Reconstruct existing trajectory (already in world frame from GCS)
    ruckig::InputParameter<3> ex_input;
    for (int i = 0; i < 3; i++) {
      ex_input.current_position[i] = existing.current_position[i];
      ex_input.current_velocity[i] = existing.current_velocity[i];
      ex_input.current_acceleration[i] = existing.current_acceleration[i];
      ex_input.target_position[i] = existing.target_position[i];
      ex_input.target_velocity[i] = existing.target_velocity[i];
      ex_input.target_acceleration[i] = existing.target_acceleration[i];
      ex_input.max_velocity[i] = existing.max_velocity[i];
      ex_input.max_acceleration[i] = existing.max_acceleration[i];
      ex_input.max_jerk[i] = existing.max_jerk[i];
    }

    ruckig::Ruckig<3> otg;
    ruckig::Trajectory<3> ex_traj;
    auto result = otg.calculate(ex_input, ex_traj);
    if (result < 0) {
      RCLCPP_WARN(this->get_logger(),
        "OPUS: Cannot reconstruct trajectory for Drone %d (ruckig=%d), assuming collision",
        existing.drone_id, static_cast<int>(result));
      return false;  // Can't verify safety, reject for safety
    }

    // Sample both trajectories over the overlap at 10ms intervals
    double dt = 0.01;
    for (double t = overlap_start; t <= overlap_end; t += dt) {
      double t_local_planned = t - planned_start_sec;
      double t_local_existing = t - ex_start;

      std::array<double, 3> pos_p, vel_p, acc_p;
      std::array<double, 3> pos_e, vel_e, acc_e;
      planned_traj_world.at_time(t_local_planned, pos_p, vel_p, acc_p);
      ex_traj.at_time(t_local_existing, pos_e, vel_e, acc_e);

      double dx = pos_p[0] - pos_e[0];
      double dy = pos_p[1] - pos_e[1];
      double dz = pos_p[2] - pos_e[2];
      double dist = std::sqrt(dx*dx + dy*dy + dz*dz);

      if (dist < safety_distance) {
        RCLCPP_WARN(this->get_logger(),
          "OPUS: Collision with Drone %d at t=%.2f (dist=%.3f < %.3f)",
          existing.drone_id, t, dist, safety_distance);
        return false;
      }
    }
  }
  return true;
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
  // Use node's current time for consistent TF lookups
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
