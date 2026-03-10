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
    if (_runtime_mode == RuntimeModes::MAVROS) {
      // MAVROS: absolute path since MAVROS node is at root namespace
      raw_ref_pos_pub = this->create_publisher<mavros_msgs::msg::PositionTarget>("/mavros/setpoint_raw/local", 10);
    } else {
      // OmniDrones: relative path so namespace is applied (e.g. /Drone1/mavros/setpoint_raw/local)
      raw_ref_pos_pub = this->create_publisher<mavros_msgs::msg::PositionTarget>("mavros/setpoint_raw/local", 10);
    }
  }
  if (_runtime_mode == RuntimeModes::OMNIDRONES) {
    // Velocity-only setpoints (TwistStamped) - relative path for namespace
    vel_cmd_pub = this->create_publisher<geometry_msgs::msg::TwistStamped>("mavros/setpoint_velocity/cmd_vel", 10);
  }
  if (_runtime_mode == RuntimeModes::MAVROS) {
    // Throttled odom for zenoh bridge (100Hz -> 10Hz)
    odom_throttled_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
      "/mavros/local_position/odom_throttled", 5);
  }

  // Subscribers
  // image_sub = this->create_subscription<sm::Image>(
  //   _depth_topic, 10,
  //   std::bind(&PlannerNode::img_callback, this, std::placeholders::_1));

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

  // ===== Frontier-Led Swarming Setup =====
  // Derive drone_id from ROS namespace (e.g. /Drone1 -> 1)
  std::string ns = this->get_namespace();
  try {
    // Extract trailing digits from namespace like "/Drone1"
    std::string digits;
    for (auto it = ns.rbegin(); it != ns.rend() && std::isdigit(*it); ++it)
      digits.insert(digits.begin(), *it);
    if (!digits.empty()) _drone_id = std::stoi(digits);
  } catch (...) { _drone_id = 1; }
  RCLCPP_INFO(this->get_logger(), "Drone ID: %d (namespace: %s)", _drone_id, ns.c_str());

  // Swarm params subscriber (from ground GUI, runtime-tunable)
  swarm_params_sub = this->create_subscription<ground_system_msgs::msg::SwarmParams>(
    "/swarm_params", 10,
    std::bind(&PlannerNode::swarm_params_callback, this, std::placeholders::_1));

  // Subscribe to neighbor odometry topics
  for (int i = 1; i <= _num_drones; ++i) {
    if (i == _drone_id) continue;  // Skip self
    std::string topic = "/Drone" + std::to_string(i) + "/odometry";
    auto sub = this->create_subscription<nav_msgs::msg::Odometry>(
      topic, 5,
      [this, i](const nav_msgs::msg::Odometry::SharedPtr msg) {
        this->neighbor_odom_callback(msg, i);
      });
    neighbor_odom_subs.push_back(sub);
    RCLCPP_INFO(this->get_logger(), "Subscribed to neighbor: %s", topic.c_str());
  }

  // Swarm status publisher (for ground GUI)
  swarm_status_pub = this->create_publisher<ground_system_msgs::msg::SwarmExplorationStatus>(
    "/swarm_exploration_status", 10);

  // Occupancy grid publisher (for ground GUI)
  occupancy_grid_pub = this->create_publisher<ground_system_msgs::msg::OccupancyGrid2D>(
    "/occupancy_grid", 5);

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

  // ===== Frontier-Led Swarm Exploration =====
  if (msg->mission_name == "swarm_explore") {
    RCLCPP_WARN(this->get_logger(), "[SWARM] Starting frontier-led exploration (drone %d of %d)",
                _drone_id, _num_drones);
    _swarm_mode = true;
    mission_received_ = true;

    // Initialize occupancy grid
    _grid_cols = static_cast<int>(_grid_width / _grid_cell_size);
    _grid_rows = static_cast<int>(_grid_height / _grid_cell_size);
    // Center grid on current position
    _grid_origin_x = _state.pose.position.x - _grid_width / 2.0;
    _grid_origin_y = _state.pose.position.y - _grid_height / 2.0;
    {
      const std::lock_guard<std::mutex> lock(_grid_mutex);
      _occupancy_grid.assign(_grid_cols * _grid_rows, 0);  // All unknown
      _cells_explored = 0;
    }

    _home_in_world_frame = _state.pose.position;

    // Set initial altitude goal for takeoff
    _goal_up_coordinate = _swarm_altitude;
    _goal_in_world_frame.x = _state.pose.position.x;
    _goal_in_world_frame.y = _state.pose.position.y;
    _goal_in_world_frame.z = _swarm_altitude;
    _goal_heading = 0.0;
    _goal_set = true;

    if (_runtime_mode == RuntimeModes::OMNIDRONES) {
      set_auto_pilot_state_forced(PlanningStates::TAKING_OFF);
      steering_value = 0.0f;
      _steered = false;
      trajectory_queue_.clear();
      reference_trajectory_ = ruckig::Trajectory<3>();
      had_reference_trajectory = false;
    } else if (_runtime_mode == RuntimeModes::MAVROS) {
      RCLCPP_WARN(this->get_logger(), "[MAVROS/SWARM] Mission received, initiating flight sequence...");
    }

    // Start swarm exploration timer (10 Hz) - will begin after takeoff
    swarm_exploration_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&PlannerNode::swarm_exploration_loop, this));

    // Status publisher timer (2 Hz)
    swarm_status_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(500),
      std::bind(&PlannerNode::publish_swarm_status, this));

    // Occupancy grid publisher timer (1 Hz)
    occupancy_pub_timer_ = this->create_wall_timer(
      std::chrono::seconds(1),
      std::bind(&PlannerNode::publish_occupancy_grid, this));

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
  // Reset swarm state
  _swarm_mode = false;
  _has_frontier = false;
  _frontiers_remaining = 0;
  if (swarm_exploration_timer_) { swarm_exploration_timer_->cancel(); swarm_exploration_timer_.reset(); }
  if (swarm_status_timer_) { swarm_status_timer_->cancel(); swarm_status_timer_.reset(); }
  if (occupancy_pub_timer_) { occupancy_pub_timer_->cancel(); occupancy_pub_timer_.reset(); }
  {
    const std::lock_guard<std::mutex> lock(_grid_mutex);
    _occupancy_grid.clear();
    _cells_explored = 0;
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

  // In swarm mode, state transitions are simple: just takeoff then stay in TRAJECTORY_CONTROL
  // No fixed goal to approach, so skip distance-based transitions
  if (_swarm_mode) {
    double takeoff_complete_altitude = _goal_up_coordinate - 0.1;
    if (_state.pose.position.z >= takeoff_complete_altitude && _planner_state == PlanningStates::TAKING_OFF) {
      RCLCPP_INFO(this->get_logger(), "[SWARM] Takeoff complete at z=%.2f, transitioning to TRAJECTORY_CONTROL",
                  _state.pose.position.z);
      set_auto_pilot_state_forced(PlanningStates::TRAJECTORY_CONTROL);
    }
    return;  // Swarm loop handles everything else
  }

  geometry_msgs::msg::Point goal_in_world_frame = _goal_in_world_frame;
  goal_in_world_frame.z = _state.pose.position.z;
  double distance_to_goal = (geometryToEigen(_state.pose.position) - geometryToEigen(goal_in_world_frame)).norm();

  // Debug: print state transition values
  // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
  //     "State check: current_z=%.2f, goal_z=%.2f, threshold=%.2f, state=%d",
  //     _state.pose.position.z, _goal_in_world_frame.z, _goal_in_world_frame.z - 0.1, (int)_planner_state);

  // Transition from TAKING_OFF to TRAJECTORY_CONTROL when takeoff altitude reached
  // Use _goal_up_coordinate (from takeoff command) for transition, not _goal_in_world_frame.z
  // This allows FBV/fly_to goals with different altitudes to work properly
  double takeoff_complete_altitude = _goal_up_coordinate - 0.1;
  if (_state.pose.position.z >= takeoff_complete_altitude && _planner_state == PlanningStates::TAKING_OFF) {
    RCLCPP_INFO(this->get_logger(), "Takeoff complete at z=%.2f (threshold=%.2f), transitioning to TRAJECTORY_CONTROL",
                _state.pose.position.z, takeoff_complete_altitude);
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

  // In swarm mode, the swarm_exploration_loop() directly publishes setpoints
  // via public_ref_pos(). Don't also publish from the depth trajectory pipeline.
  if (_swarm_mode && _planner_state != PlanningStates::TAKING_OFF)
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

// =============================================================================
// Frontier-Led Swarming Implementation
// =============================================================================

void PlannerNode::swarm_params_callback(const ground_system_msgs::msg::SwarmParams::SharedPtr msg) {
  RCLCPP_INFO(this->get_logger(), "[SWARM] Received updated params: coh=%.2f sep=%.2f ali=%.2f fro=%.2f obs=%.2f",
              msg->w_cohesion, msg->w_separation, msg->w_alignment, msg->w_frontier, msg->w_obstacle);
  _w_cohesion = msg->w_cohesion;
  _w_separation = msg->w_separation;
  _w_alignment = msg->w_alignment;
  _w_frontier = msg->w_frontier;
  _w_obstacle = msg->w_obstacle;
  _separation_radius = msg->separation_radius;
  _neighbor_radius = msg->neighbor_radius;
  _max_swarm_speed = msg->max_swarm_speed;
  _swarm_altitude = msg->altitude;

  if (msg->cell_size > 0.01 && msg->map_width > 0.1 && msg->map_height > 0.1) {
    // Only rebuild grid if dimensions actually changed
    if (std::abs(msg->cell_size - _grid_cell_size) > 0.001 ||
        std::abs(msg->map_width - _grid_width) > 0.1 ||
        std::abs(msg->map_height - _grid_height) > 0.1) {
      RCLCPP_INFO(this->get_logger(), "[SWARM] Grid config changed, rebuilding grid");
      _grid_cell_size = msg->cell_size;
      _grid_width = msg->map_width;
      _grid_height = msg->map_height;
      _grid_cols = static_cast<int>(_grid_width / _grid_cell_size);
      _grid_rows = static_cast<int>(_grid_height / _grid_cell_size);
      const std::lock_guard<std::mutex> lock(_grid_mutex);
      _occupancy_grid.assign(_grid_cols * _grid_rows, 0);
      _cells_explored = 0;
    }
  }
}

void PlannerNode::neighbor_odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg, int neighbor_id) {
  const std::lock_guard<std::mutex> lock(_neighbor_mutex);
  auto& ns = _neighbor_states[neighbor_id];
  ns.position = Eigen::Vector3d(msg->pose.pose.position.x,
                                 msg->pose.pose.position.y,
                                 msg->pose.pose.position.z);
  ns.velocity = Eigen::Vector3d(msg->twist.twist.linear.x,
                                 msg->twist.twist.linear.y,
                                 msg->twist.twist.linear.z);
  // Extract yaw from quaternion
  double qw = msg->pose.pose.orientation.w;
  double qx = msg->pose.pose.orientation.x;
  double qy = msg->pose.pose.orientation.y;
  double qz = msg->pose.pose.orientation.z;
  ns.yaw = atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
  ns.last_update = rclcpp::Time(msg->header.stamp);
  ns.valid = true;
}

Eigen::Vector2d PlannerNode::world_to_grid(double wx, double wy) const {
  return Eigen::Vector2d(
    (wx - _grid_origin_x) / _grid_cell_size,
    (wy - _grid_origin_y) / _grid_cell_size);
}

Eigen::Vector2d PlannerNode::grid_to_world(int gx, int gy) const {
  return Eigen::Vector2d(
    _grid_origin_x + (gx + 0.5) * _grid_cell_size,
    _grid_origin_y + (gy + 0.5) * _grid_cell_size);
}

bool PlannerNode::is_in_grid(int gx, int gy) const {
  return gx >= 0 && gx < _grid_cols && gy >= 0 && gy < _grid_rows;
}

void PlannerNode::update_occupancy_grid() {
  const std::lock_guard<std::mutex> lock(_grid_mutex);

  // Mark cells around current position as free (sensor footprint)
  double sensor_range = _depth_upper_bound;  // Use planner's max depth as sensing range
  Eigen::Vector3d pos;
  double yaw;
  {
    const std::lock_guard<std::mutex> slock(state_mutex_);
    pos = Eigen::Vector3d(_state.pose.position.x, _state.pose.position.y, _state.pose.position.z);
    double qw = _state.pose.orientation.w, qx = _state.pose.orientation.x;
    double qy = _state.pose.orientation.y, qz = _state.pose.orientation.z;
    yaw = atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
  }

  // Sweep a cone in the camera's FOV and mark cells as free
  // Camera FOV ≈ 127° horizontal -> ±63.5° from heading
  double half_fov = 63.5 * M_PI / 180.0;
  int steps_angle = 64;
  int steps_range = static_cast<int>(sensor_range / _grid_cell_size);

  for (int a = 0; a <= steps_angle; ++a) {
    double angle = yaw - half_fov + (2.0 * half_fov * a / steps_angle);
    for (int r = 1; r <= steps_range; ++r) {
      double dist = r * _grid_cell_size;
      double wx = pos.x() + dist * cos(angle);
      double wy = pos.y() + dist * sin(angle);
      Eigen::Vector2d gc = world_to_grid(wx, wy);
      int gx = static_cast<int>(gc.x());
      int gy = static_cast<int>(gc.y());
      if (!is_in_grid(gx, gy)) continue;
      int idx = gy * _grid_cols + gx;
      if (_occupancy_grid[idx] == 0) {
        _occupancy_grid[idx] = 1;  // free
        _cells_explored++;
      }
    }
  }

  // Mark the cell immediately at position as free too
  Eigen::Vector2d my_gc = world_to_grid(pos.x(), pos.y());
  int mx = static_cast<int>(my_gc.x()), my = static_cast<int>(my_gc.y());
  if (is_in_grid(mx, my) && _occupancy_grid[my * _grid_cols + mx] == 0) {
    _occupancy_grid[my * _grid_cols + mx] = 1;
    _cells_explored++;
  }
}

std::vector<Eigen::Vector2d> PlannerNode::detect_frontiers() {
  const std::lock_guard<std::mutex> lock(_grid_mutex);
  std::vector<Eigen::Vector2d> frontiers;

  // A frontier cell is a free cell (1) adjacent to at least one unknown cell (0)
  static const int dx[] = {-1, 1, 0, 0, -1, -1, 1, 1};
  static const int dy[] = {0, 0, -1, 1, -1, 1, -1, 1};

  for (int gy = 0; gy < _grid_rows; ++gy) {
    for (int gx = 0; gx < _grid_cols; ++gx) {
      int idx = gy * _grid_cols + gx;
      if (_occupancy_grid[idx] != 1) continue;  // Only free cells
      bool is_frontier = false;
      for (int d = 0; d < 8; ++d) {
        int nx = gx + dx[d], ny = gy + dy[d];
        if (!is_in_grid(nx, ny)) continue;
        if (_occupancy_grid[ny * _grid_cols + nx] == 0) {
          is_frontier = true;
          break;
        }
      }
      if (is_frontier) {
        Eigen::Vector2d wc = grid_to_world(gx, gy);
        frontiers.push_back(wc);
      }
    }
  }
  return frontiers;
}

Eigen::Vector2d PlannerNode::select_frontier(const std::vector<Eigen::Vector2d>& frontiers) {
  if (frontiers.empty()) return Eigen::Vector2d(0, 0);

  Eigen::Vector3d my_pos;
  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    my_pos = Eigen::Vector3d(_state.pose.position.x, _state.pose.position.y, _state.pose.position.z);
  }
  Eigen::Vector2d my_pos_2d(my_pos.x(), my_pos.y());

  // Gather neighbor positions for frontier deconfliction
  std::vector<Eigen::Vector2d> neighbor_positions;
  {
    const std::lock_guard<std::mutex> lock(_neighbor_mutex);
    for (auto& [id, ns] : _neighbor_states) {
      if (!ns.valid) continue;
      double age = (this->now() - ns.last_update).seconds();
      if (age > 2.0) continue;  // Stale neighbor
      neighbor_positions.emplace_back(ns.position.x(), ns.position.y());
    }
  }

  // Score each frontier: prefer close to self, far from neighbors
  double best_score = std::numeric_limits<double>::max();
  Eigen::Vector2d best_frontier = frontiers[0];

  for (const auto& f : frontiers) {
    double dist_self = (f - my_pos_2d).norm();

    // Penalty for frontiers that are closer to another drone
    double neighbor_penalty = 0.0;
    for (const auto& np : neighbor_positions) {
      double dist_neighbor = (f - np).norm();
      if (dist_neighbor < dist_self) {
        // Another drone is closer — add penalty proportional to how much closer
        neighbor_penalty += (dist_self - dist_neighbor);
      }
    }

    // Simple cost: distance + neighbor penalty (encourages spatial distribution)
    double score = dist_self + 2.0 * neighbor_penalty;
    if (score < best_score) {
      best_score = score;
      best_frontier = f;
    }
  }
  return best_frontier;
}

Eigen::Vector3d PlannerNode::compute_cohesion() {
  // Reynolds rule 1: Steer toward average position of neighbors
  Eigen::Vector3d centroid(0, 0, 0);
  int count = 0;
  Eigen::Vector3d my_pos;
  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    my_pos = Eigen::Vector3d(_state.pose.position.x, _state.pose.position.y, _state.pose.position.z);
  }
  {
    const std::lock_guard<std::mutex> lock(_neighbor_mutex);
    for (auto& [id, ns] : _neighbor_states) {
      if (!ns.valid) continue;
      double age = (this->now() - ns.last_update).seconds();
      if (age > 2.0) continue;
      double dist = (ns.position - my_pos).norm();
      if (dist > _neighbor_radius) continue;
      centroid += ns.position;
      count++;
    }
  }
  if (count == 0) return Eigen::Vector3d(0, 0, 0);
  centroid /= count;
  Eigen::Vector3d steer = centroid - my_pos;
  double mag = steer.norm();
  if (mag > 0.01) steer = steer / mag;  // Normalize
  return steer;
}

Eigen::Vector3d PlannerNode::compute_separation() {
  // Reynolds rule 2: Steer away from nearby neighbors
  Eigen::Vector3d repulsion(0, 0, 0);
  Eigen::Vector3d my_pos;
  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    my_pos = Eigen::Vector3d(_state.pose.position.x, _state.pose.position.y, _state.pose.position.z);
  }
  {
    const std::lock_guard<std::mutex> lock(_neighbor_mutex);
    for (auto& [id, ns] : _neighbor_states) {
      if (!ns.valid) continue;
      double age = (this->now() - ns.last_update).seconds();
      if (age > 2.0) continue;
      Eigen::Vector3d diff = my_pos - ns.position;
      double dist = diff.norm();
      if (dist < 0.01 || dist > _separation_radius) continue;
      // Inverse-square repulsion
      repulsion += diff / (dist * dist);
    }
  }
  double mag = repulsion.norm();
  if (mag > 0.01) repulsion = repulsion / mag;
  return repulsion;
}

Eigen::Vector3d PlannerNode::compute_alignment() {
  // Reynolds rule 3: Match average heading of neighbors
  Eigen::Vector3d avg_vel(0, 0, 0);
  int count = 0;
  Eigen::Vector3d my_pos;
  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    my_pos = Eigen::Vector3d(_state.pose.position.x, _state.pose.position.y, _state.pose.position.z);
  }
  {
    const std::lock_guard<std::mutex> lock(_neighbor_mutex);
    for (auto& [id, ns] : _neighbor_states) {
      if (!ns.valid) continue;
      double age = (this->now() - ns.last_update).seconds();
      if (age > 2.0) continue;
      double dist = (ns.position - my_pos).norm();
      if (dist > _neighbor_radius) continue;
      avg_vel += ns.velocity;
      count++;
    }
  }
  if (count == 0) return Eigen::Vector3d(0, 0, 0);
  avg_vel /= count;
  double mag = avg_vel.norm();
  if (mag > 0.01) avg_vel = avg_vel / mag;
  return avg_vel;
}

Eigen::Vector3d PlannerNode::compute_frontier_attraction(const Eigen::Vector2d& target_frontier) {
  Eigen::Vector3d my_pos;
  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    my_pos = Eigen::Vector3d(_state.pose.position.x, _state.pose.position.y, _state.pose.position.z);
  }
  Eigen::Vector3d frontier_3d(target_frontier.x(), target_frontier.y(), _swarm_altitude);
  Eigen::Vector3d steer = frontier_3d - my_pos;
  double mag = steer.norm();
  if (mag > 0.01) steer = steer / mag;
  return steer;
}

void PlannerNode::swarm_exploration_loop() {
  if (!_swarm_mode) return;
  if (_planner_state == PlanningStates::OFF || _planner_state == PlanningStates::TAKING_OFF) return;

  // 1. Update occupancy grid from depth observations
  update_occupancy_grid();

  // 2. Detect frontiers
  auto frontiers = detect_frontiers();
  _frontiers_remaining = frontiers.size();

  // 3. Select best frontier for this drone
  if (!frontiers.empty()) {
    _assigned_frontier = select_frontier(frontiers);
    _has_frontier = true;
  } else {
    _has_frontier = false;
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
      "[SWARM] No frontiers remaining — exploration complete!");
    return;
  }

  // 4. Compute Reynolds flocking forces + frontier attraction
  Eigen::Vector3d cohesion_force = compute_cohesion();
  Eigen::Vector3d separation_force = compute_separation();
  Eigen::Vector3d alignment_force = compute_alignment();
  Eigen::Vector3d frontier_force = compute_frontier_attraction(_assigned_frontier);

  // 5. Blend forces
  Eigen::Vector3d combined_velocity =
    _w_cohesion * cohesion_force +
    _w_separation * separation_force +
    _w_alignment * alignment_force +
    _w_frontier * frontier_force;

  // Enforce altitude
  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    combined_velocity.z() = (_swarm_altitude - _state.pose.position.z) * 2.0;  // P-controller for altitude
  }

  // Clamp speed
  double speed = combined_velocity.head<2>().norm();
  if (speed > _max_swarm_speed) {
    combined_velocity.head<2>() *= _max_swarm_speed / speed;
  }

  // 6. Set goal = current position + velocity * lookahead
  double lookahead = 2.0;  // seconds
  Eigen::Vector3d my_pos;
  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    my_pos = Eigen::Vector3d(_state.pose.position.x, _state.pose.position.y, _state.pose.position.z);
  }

  Eigen::Vector3d target_pos = my_pos + combined_velocity * lookahead;

  // Clamp to fence
  target_pos.x() = std::clamp(target_pos.x(), _fence_min_x, _fence_max_x);
  target_pos.y() = std::clamp(target_pos.y(), _fence_min_y, _fence_max_y);
  target_pos.z() = std::clamp(target_pos.z(), _fence_min_z, _fence_max_z);

  // 7. Compute heading from velocity direction
  _goal_heading = atan2(combined_velocity.y(), combined_velocity.x());

  // 8. Directly publish setpoint to OmniDrones
  //    The depth planner may override this if it generates an obstacle-avoidance trajectory,
  //    but this ensures the drone moves even when no obstacles/trajectories are present.
  TrajectoryPoint swarm_ref;
  swarm_ref.position = target_pos;
  swarm_ref.velocity = combined_velocity;
  swarm_ref.acceleration = Eigen::Vector3d(0.0, 0.0, 0.0);
  swarm_ref.heading = _goal_heading;
  public_ref_pos(swarm_ref);

  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
    "[SWARM] drone=%d frontiers=%zu goal=(%.1f,%.1f,%.1f) vel=(%.2f,%.2f,%.2f) coh=(%.2f,%.2f) sep=(%.2f,%.2f) fro=(%.2f,%.2f)",
    _drone_id, frontiers.size(),
    _goal_in_world_frame.x, _goal_in_world_frame.y, _goal_in_world_frame.z,
    combined_velocity.x(), combined_velocity.y(), combined_velocity.z(),
    cohesion_force.x(), cohesion_force.y(),
    separation_force.x(), separation_force.y(),
    frontier_force.x(), frontier_force.y());
}

void PlannerNode::publish_swarm_status() {
  if (!_swarm_mode) return;

  auto msg = ground_system_msgs::msg::SwarmExplorationStatus();
  msg.header.stamp = this->now();
  msg.drone_id = _drone_id;

  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    msg.position.x = _state.pose.position.x;
    msg.position.y = _state.pose.position.y;
    msg.position.z = _state.pose.position.z;
    double qw = _state.pose.orientation.w, qx = _state.pose.orientation.x;
    double qy = _state.pose.orientation.y, qz = _state.pose.orientation.z;
    msg.yaw_rad = atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
    msg.velocity.x = _state.velocity.linear.x;
    msg.velocity.y = _state.velocity.linear.y;
    msg.velocity.z = _state.velocity.linear.z;
  }

  if (_planner_state == PlanningStates::TRAJECTORY_CONTROL) {
    msg.state = ground_system_msgs::msg::SwarmExplorationStatus::STATE_EXPLORING;
  } else {
    msg.state = ground_system_msgs::msg::SwarmExplorationStatus::STATE_IDLE;
  }

  if (_has_frontier) {
    msg.assigned_frontier.x = _assigned_frontier.x();
    msg.assigned_frontier.y = _assigned_frontier.y();
    msg.assigned_frontier.z = _swarm_altitude;
    Eigen::Vector3d my_pos(_state.pose.position.x, _state.pose.position.y, _state.pose.position.z);
    msg.distance_to_frontier = (Eigen::Vector2d(_assigned_frontier.x(), _assigned_frontier.y()) -
                                 Eigen::Vector2d(my_pos.x(), my_pos.y())).norm();
  }
  msg.frontiers_remaining = _frontiers_remaining;

  uint32_t total = _grid_cols * _grid_rows;
  msg.cells_explored = _cells_explored;
  msg.total_cells = total;
  msg.coverage_percent = (total > 0) ? (100.0f * _cells_explored / total) : 0.0f;

  // Neighbor info
  int num_neighbors = 0;
  double nearest_dist = std::numeric_limits<double>::max();
  Eigen::Vector3d my_pos;
  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    my_pos = Eigen::Vector3d(_state.pose.position.x, _state.pose.position.y, _state.pose.position.z);
  }
  {
    const std::lock_guard<std::mutex> lock(_neighbor_mutex);
    for (auto& [id, ns] : _neighbor_states) {
      if (!ns.valid) continue;
      double age = (this->now() - ns.last_update).seconds();
      if (age > 2.0) continue;
      double dist = (ns.position - my_pos).norm();
      if (dist <= _neighbor_radius) {
        num_neighbors++;
        nearest_dist = std::min(nearest_dist, dist);
      }
    }
  }
  msg.num_neighbors = num_neighbors;
  msg.nearest_neighbor_dist = (nearest_dist < 1e6) ? nearest_dist : 0.0;

  swarm_status_pub->publish(msg);
}

void PlannerNode::publish_occupancy_grid() {
  if (!_swarm_mode) return;

  auto msg = ground_system_msgs::msg::OccupancyGrid2D();
  msg.header.stamp = this->now();
  msg.drone_id = _drone_id;
  msg.cell_size = _grid_cell_size;
  msg.origin_x = _grid_origin_x;
  msg.origin_y = _grid_origin_y;
  msg.width = _grid_cols;
  msg.height = _grid_rows;

  {
    const std::lock_guard<std::mutex> lock(_grid_mutex);
    msg.data = _occupancy_grid;
  }

  occupancy_grid_pub->publish(msg);
}
