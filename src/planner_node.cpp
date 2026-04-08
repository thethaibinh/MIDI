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
    // Grid origin is a fixed config value shared by all drones
    // (loaded from YAML map_origin_x / map_origin_y, default 0,0)
    {
      const std::lock_guard<std::mutex> lock(_grid_mutex);
      _occupancy_grid.assign(_grid_cols * _grid_rows, 0);  // All unknown
      _visit_counts.assign(_grid_cols * _grid_rows, 0);    // No visits
      _last_visit_time.assign(_grid_cols * _grid_rows, 0.0);
      _cells_explored = 0;
    }

    // Initialize task allocation state
    {
      const std::lock_guard<std::mutex> lock(_task_mutex);
      _known_tasks.clear();
      _next_task_id = 1;
      _task_state = 0;  // SWARMING
      _current_task_id = 0;
      _task_exec_start_time = 0.0;
    }
    _task_rng.seed(static_cast<uint32_t>(_drone_id * 1000 + std::chrono::steady_clock::now().time_since_epoch().count()));

    // Initialize metrics
    _metrics_total_timesteps = 0;
    _metrics_agent_collision_count = 0;
    _metrics_wall_collision_count = 0;
    _was_in_agent_collision = false;
    _was_in_wall_collision = false;

    _home_in_world_frame = _state.pose.position;

    // Set initial altitude goal for takeoff
    _goal_up_coordinate = _swarm_altitude;
    _goal_in_world_frame.x = _state.pose.position.x;
    _goal_in_world_frame.y = _state.pose.position.y;
    _goal_in_world_frame.z = _swarm_altitude;
    _goal_heading = 0.0;
    _goal_set = true;

    // Initialize heading-rate model from current odometry yaw
    {
      const std::lock_guard<std::mutex> lock(state_mutex_);
      double qw = _state.pose.orientation.w;
      double qx = _state.pose.orientation.x;
      double qy = _state.pose.orientation.y;
      double qz = _state.pose.orientation.z;
      _current_azimuth = atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
      if (_current_azimuth < 0) _current_azimuth += 2.0 * M_PI;
    }
    _current_elevation = 0.0;

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

    // Occupancy grid publisher timer (5 Hz — MATLAB rebuilds global_map every timestep;
    // higher frequency reduces staleness of neighbor grids for frontier computation)
    occupancy_pub_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(200),
      std::bind(&PlannerNode::publish_occupancy_grid, this));

    // Metrics publisher timer (0.5 Hz)
    swarm_metrics_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(2000),
      std::bind(&PlannerNode::publish_swarm_metrics, this));

    // Task topic pub/sub
    swarm_task_pub = this->create_publisher<ground_system_msgs::msg::SwarmTask>("/swarm_tasks", 10);
    swarm_task_sub = this->create_subscription<ground_system_msgs::msg::SwarmTask>(
      "/swarm_tasks", 10,
      std::bind(&PlannerNode::task_callback, this, std::placeholders::_1));

    // Subscribe to neighbor occupancy grids for map merging (MATLAB: mergeMaps.m)
    // All drones publish to /occupancy_grid at 1 Hz; we merge from neighbors
    // within communication range, matching MATLAB's every-50-timestep merge.
    neighbor_grid_sub = this->create_subscription<ground_system_msgs::msg::OccupancyGrid2D>(
      "/occupancy_grid", 10,
      std::bind(&PlannerNode::neighbor_grid_callback, this, std::placeholders::_1));

    // Metrics publisher
    swarm_metrics_pub = this->create_publisher<ground_system_msgs::msg::SwarmMetrics>("/swarm_metrics", 10);

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
  _task_state = 0;
  _current_task_id = 0;
  if (swarm_exploration_timer_) { swarm_exploration_timer_->cancel(); swarm_exploration_timer_.reset(); }
  if (swarm_status_timer_) { swarm_status_timer_->cancel(); swarm_status_timer_.reset(); }
  if (occupancy_pub_timer_) { occupancy_pub_timer_->cancel(); occupancy_pub_timer_.reset(); }
  if (swarm_metrics_timer_) { swarm_metrics_timer_->cancel(); swarm_metrics_timer_.reset(); }
  {
    const std::lock_guard<std::mutex> lock(_grid_mutex);
    _occupancy_grid.clear();
    _visit_counts.clear();
    _last_visit_time.clear();
    _cells_explored = 0;
  }
  {
    const std::lock_guard<std::mutex> lock(_task_mutex);
    _known_tasks.clear();
  }
  _metrics_total_timesteps = 0;
  _metrics_agent_collision_count = 0;
  _metrics_wall_collision_count = 0;
  _was_in_agent_collision = false;
  _was_in_wall_collision = false;
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
    if (_planner_state == PlanningStates::TAKING_OFF) {
      // OmniDrones: drones spawn already at altitude, skip takeoff if z is valid
      if (_runtime_mode == RuntimeModes::OMNIDRONES && _state.pose.position.z > 0.3) {
        RCLCPP_INFO(this->get_logger(), "[SWARM] OmniDrones takeoff skipped (already at z=%.2f), transitioning to TRAJECTORY_CONTROL",
                    _state.pose.position.z);
        set_auto_pilot_state_forced(PlanningStates::TRAJECTORY_CONTROL);
      } else if (_state.pose.position.z >= takeoff_complete_altitude) {
        RCLCPP_INFO(this->get_logger(), "[SWARM] Takeoff complete at z=%.2f, transitioning to TRAJECTORY_CONTROL",
                    _state.pose.position.z);
        set_auto_pilot_state_forced(PlanningStates::TRAJECTORY_CONTROL);
      }
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
// Dynamic Frontier-Led Swarming with Task Allocation & Metrics
// =============================================================================

void PlannerNode::swarm_params_callback(const ground_system_msgs::msg::SwarmParams::SharedPtr msg) {
  RCLCPP_INFO(this->get_logger(), "[SWARM] Received updated params: coh=%.2f sep=%.2f ali=%.2f fro=%.2f obs=%.2f",
              msg->w_cohesion, msg->w_separation, msg->w_alignment, msg->w_frontier, msg->w_obstacle);
  _w_cohesion = msg->w_cohesion;
  _w_separation = msg->w_separation;
  _w_alignment = msg->w_alignment;
  _w_frontier = msg->w_frontier;
  _w_obstacle = msg->w_obstacle;
  // Per-rule radii (paper: R_c, R_a, R_s, R_critical)
  if (msg->r_cohesion > 0.0) _r_cohesion = msg->r_cohesion;
  if (msg->r_alignment > 0.0) _r_alignment = msg->r_alignment;
  if (msg->r_separation > 0.0) _r_separation = msg->r_separation;
  if (msg->r_critical > 0.0) _r_critical = msg->r_critical;
  if (msg->r_comm > 0.0) _r_comm = msg->r_comm;
  if (msg->wall_buffer > 0.0) _wall_buffer = msg->wall_buffer;
  _max_swarm_speed = msg->max_swarm_speed;
  _swarm_altitude = msg->altitude;

  // Frontier utility weights
  if (msg->psi_distance > 0.0) _psi_distance = msg->psi_distance;
  if (msg->psi_size > 0.0) _psi_size = msg->psi_size;

  // Task allocation params
  _enable_task_allocation = msg->enable_task_allocation;
  if (msg->task_spawn_probability > 0.0) _task_spawn_probability = msg->task_spawn_probability;
  if (msg->task_proximity_threshold > 0.0) _task_proximity_threshold = msg->task_proximity_threshold;

  // Motion mode (0=2D, 1=3D)
  uint8_t old_mode = _motion_mode;
  _motion_mode = msg->motion_mode;
  if (_motion_mode != old_mode) {
    RCLCPP_INFO(this->get_logger(), "[SWARM] Motion mode changed to %s",
                _motion_mode == 1 ? "3D" : "2D");
  }

  if (msg->cell_size > 0.01 && msg->map_width > 0.1 && msg->map_height > 0.1) {
    if (std::abs(msg->cell_size - _grid_cell_size) > 0.001 ||
        std::abs(msg->map_width - _grid_width) > 0.1 ||
        std::abs(msg->map_height - _grid_height) > 0.1 ||
        std::abs(msg->map_origin_x - _grid_origin_x) > 0.1 ||
        std::abs(msg->map_origin_y - _grid_origin_y) > 0.1) {
      RCLCPP_INFO(this->get_logger(), "[SWARM] Grid config changed, rebuilding grid");
      _grid_cell_size = msg->cell_size;
      _grid_width = msg->map_width;
      _grid_height = msg->map_height;
      _grid_origin_x = msg->map_origin_x;
      _grid_origin_y = msg->map_origin_y;
      _grid_cols = static_cast<int>(_grid_width / _grid_cell_size);
      _grid_rows = static_cast<int>(_grid_height / _grid_cell_size);
      // Grid origin updated from GUI message
      const std::lock_guard<std::mutex> lock(_grid_mutex);
      _occupancy_grid.assign(_grid_cols * _grid_rows, 0);
      _visit_counts.assign(_grid_cols * _grid_rows, 0);
      _last_visit_time.assign(_grid_cols * _grid_rows, 0.0);
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

  Eigen::Vector3d pos;
  double yaw;
  {
    const std::lock_guard<std::mutex> slock(state_mutex_);
    pos = Eigen::Vector3d(_state.pose.position.x, _state.pose.position.y, _state.pose.position.z);
    double qw = _state.pose.orientation.w, qx = _state.pose.orientation.x;
    double qy = _state.pose.orientation.y, qz = _state.pose.orientation.z;
    yaw = atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
  }

  double current_time = this->now().seconds();
  // Single-cell observation model (matches MATLAB updateRobotLocalMap.m exactly):
  // Mark the one grid cell containing the drone's ground-plane position.
  int gx = static_cast<int>((pos.x() - _grid_origin_x) / _grid_cell_size);
  int gy = static_cast<int>((pos.y() - _grid_origin_y) / _grid_cell_size);
  gx = std::clamp(gx, 0, _grid_cols - 1);
  gy = std::clamp(gy, 0, _grid_rows - 1);
  int idx = gy * _grid_cols + gx;
  if (_occupancy_grid[idx] == 0) {
    _occupancy_grid[idx] = 1;  // free/explored
    _cells_explored++;
  }
  _visit_counts[idx]++;
  _last_visit_time[idx] = current_time;
}

// =============================================================================
// Dynamic Frontier Region Grouping (Paper Section 3.1.3, Eq. 20-21)
// =============================================================================
// Paper definition: A cell g_m in G_known is a FRONTIER CELL if it has at least
// one 4-connected neighbor in G_unk. Frontier cells are clustered into connected
// regions {R_1, ..., R_NF}. Centroid is computed over frontier cells only.
//
// This differs from the MATLAB implementation which BFS-floods through all
// unexplored cells (producing one giant region with centroid near map center).
// The paper approach gives thin frontier strips whose centroids track the
// exploration boundary and move outward as drones explore.

std::vector<PlannerNode::FrontierRegion> PlannerNode::group_frontier_regions() {
  const std::lock_guard<std::mutex> lock(_grid_mutex);

  // Build global map from own grid + all cached neighbor grids
  size_t total_cells = _occupancy_grid.size();
  std::vector<uint8_t> global_grid(_occupancy_grid);
  for (const auto& [drone_id, ngrid] : _neighbor_grids) {
    if (ngrid.size() != total_cells) continue;
    for (size_t i = 0; i < total_cells; ++i) {
      if (ngrid[i] > global_grid[i]) global_grid[i] = ngrid[i];
    }
  }

  // 4-connected for frontier check (paper: "at least one 4-connected neighbor")
  static const int dx4[] = {-1, 1, 0, 0};
  static const int dy4[] = {0, 0, -1, 1};
  // 8-connected for BFS grouping of frontier cells into regions
  static const int dx8[] = {-1, 1, 0, 0, -1, -1, 1, 1};
  static const int dy8[] = {0, 0, -1, 1, -1, 1, -1, 1};

  // Step 1: Mark all frontier cells (explored cells with >=1 unexplored 4-neighbor)
  std::vector<bool> is_frontier(_grid_cols * _grid_rows, false);
  for (int gy = 0; gy < _grid_rows; ++gy) {
    for (int gx = 0; gx < _grid_cols; ++gx) {
      int idx = gy * _grid_cols + gx;
      if (global_grid[idx] != 1) continue;  // Must be explored

      for (int d = 0; d < 4; ++d) {
        int nx = gx + dx4[d], ny = gy + dy4[d];
        if (!is_in_grid(nx, ny)) continue;
        if (global_grid[ny * _grid_cols + nx] == 0) {
          is_frontier[idx] = true;
          break;
        }
      }
    }
  }

  // Step 2: BFS to group connected frontier cells into regions (8-connected)
  std::vector<bool> visited(_grid_cols * _grid_rows, false);
  std::vector<FrontierRegion> regions;

  for (int gy = 0; gy < _grid_rows; ++gy) {
    for (int gx = 0; gx < _grid_cols; ++gx) {
      int idx = gy * _grid_cols + gx;
      if (!is_frontier[idx] || visited[idx]) continue;

      FrontierRegion region;
      region.size = 0;
      region.avg_last_visit = 0.0;
      double sum_x = 0.0, sum_y = 0.0;
      double visit_time_sum = 0.0;

      std::queue<std::pair<int, int>> bfs_queue;
      bfs_queue.push({gx, gy});
      visited[idx] = true;

      while (!bfs_queue.empty()) {
        auto [cx, cy] = bfs_queue.front();
        bfs_queue.pop();

        int cidx = cy * _grid_cols + cx;
        Eigen::Vector2d wc = grid_to_world(cx, cy);
        sum_x += wc.x();
        sum_y += wc.y();
        visit_time_sum += _last_visit_time[cidx];
        region.size++;

        // Expand to 8-connected frontier neighbors only
        for (int d = 0; d < 8; ++d) {
          int nnx = cx + dx8[d], nny = cy + dy8[d];
          if (!is_in_grid(nnx, nny)) continue;
          int nidx = nny * _grid_cols + nnx;
          if (is_frontier[nidx] && !visited[nidx]) {
            visited[nidx] = true;
            bfs_queue.push({nnx, nny});
          }
        }
      }

      if (region.size > 0) {
        region.centroid = Eigen::Vector2d(sum_x / region.size, sum_y / region.size);
        // Paper Eq. 22: mean time since last visit over frontier cells in region
        region.avg_last_visit = visit_time_sum / region.size;
        regions.push_back(region);
      }
    }
  }
  return regions;
}

// =============================================================================
// Utility-Based Frontier Selection
// =============================================================================

PlannerNode::FrontierRegion PlannerNode::select_frontier_region(
    const std::vector<FrontierRegion>& regions) {
  if (regions.empty()) return FrontierRegion{};

  Eigen::Vector3d my_pos;
  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    my_pos = Eigen::Vector3d(_state.pose.position.x, _state.pose.position.y, _state.pose.position.z);
  }
  Eigen::Vector2d my_pos_2d(my_pos.x(), my_pos.y());

  double current_time = this->now().seconds();

  // Gather neighbor positions for deconfliction
  std::vector<Eigen::Vector2d> neighbor_positions;
  {
    const std::lock_guard<std::mutex> lock(_neighbor_mutex);
    for (auto& [id, ns] : _neighbor_states) {
      if (!ns.valid) continue;
      double age = (this->now() - ns.last_update).seconds();
      if (age > 2.0) continue;
      neighbor_positions.emplace_back(ns.position.x(), ns.position.y());
    }
  }

  // Score each region: utility = psi_distance * distance - psi_size * size - time_since_visit
  // Lower utility = better frontier
  double best_utility = std::numeric_limits<double>::max();
  size_t best_idx = 0;

  for (size_t i = 0; i < regions.size(); ++i) {
    const auto& r = regions[i];
    double dist = (r.centroid - my_pos_2d).norm();
    double time_since = current_time - r.avg_last_visit;

    double utility = _psi_distance * dist - _psi_size * r.size - time_since;

    // Neighbor penalty: penalize regions closer to other drones
    for (const auto& np : neighbor_positions) {
      double dist_neighbor = (r.centroid - np).norm();
      if (dist_neighbor < dist) {
        utility += 2.0 * (dist - dist_neighbor);
      }
    }

    if (utility < best_utility) {
      best_utility = utility;
      best_idx = i;
    }
  }
  return regions[best_idx];
}

Eigen::Vector3d PlannerNode::compute_cohesion() {
  // Paper: vc = normalized(avg_neighbor_pos - current_pos)
  // Neighbors within R_cohesion
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
      if (dist > _r_cohesion) continue;
      centroid += ns.position;
      count++;
    }
  }
  if (count == 0) return Eigen::Vector3d(0, 0, 0);
  centroid /= count;
  Eigen::Vector3d steer = centroid - my_pos;
  double mag = steer.norm();
  if (mag > 0.01) steer = steer / mag;
  return steer;
}

Eigen::Vector3d PlannerNode::compute_separation() {
  // Paper: for each neighbor within R_separation:
  //   repulsion_strength = (R_separation - distance) / R_separation
  //   sum += (current_pos - neighbor_pos) / distance * repulsion_strength
  // Result normalized to unit vector.
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
      if (dist < 0.01 || dist > _r_separation) continue;
      double strength = (_r_separation - dist) / _r_separation;  // Linear falloff [0,1]
      repulsion += (diff / dist) * strength;
    }
  }
  double mag = repulsion.norm();
  if (mag > 0.01) repulsion = repulsion / mag;
  return repulsion;
}

Eigen::Vector3d PlannerNode::compute_alignment() {
  // Paper: va = normalized(average_neighbor_velocities)
  // Neighbors within R_alignment
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
      if (dist > _r_alignment) continue;
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

Eigen::Vector3d PlannerNode::compute_boundary_repulsion() {
  // Paper (calculateWallAvoidance.m):
  //  For each dimension: if pos < min + wall_buffer → repel = (min+buffer-pos)/buffer
  //                      if pos > max - wall_buffer → repel = -(pos-(max-buffer))/buffer
  //  Result normalized to unit vector.
  Eigen::Vector3d my_pos;
  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    my_pos = Eigen::Vector3d(_state.pose.position.x, _state.pose.position.y, _state.pose.position.z);
  }

  double min_x = _grid_origin_x;
  double max_x = _grid_origin_x + _grid_width;
  double min_y = _grid_origin_y;
  double max_y = _grid_origin_y + _grid_height;

  Eigen::Vector3d vw(0.0, 0.0, 0.0);

  // X dimension
  if (my_pos.x() < min_x + _wall_buffer)
    vw.x() = (min_x + _wall_buffer - my_pos.x()) / _wall_buffer;
  else if (my_pos.x() > max_x - _wall_buffer)
    vw.x() = -(my_pos.x() - (max_x - _wall_buffer)) / _wall_buffer;

  // Y dimension
  if (my_pos.y() < min_y + _wall_buffer)
    vw.y() = (min_y + _wall_buffer - my_pos.y()) / _wall_buffer;
  else if (my_pos.y() > max_y - _wall_buffer)
    vw.y() = -(my_pos.y() - (max_y - _wall_buffer)) / _wall_buffer;

  // Z dimension (3D mode only — repel from ceiling/floor geofence)
  if (_motion_mode == 1) {
    double min_z = _fence_min_z;
    double max_z = _fence_max_z;
    if (my_pos.z() < min_z + _wall_buffer)
      vw.z() = (min_z + _wall_buffer - my_pos.z()) / _wall_buffer;
    else if (my_pos.z() > max_z - _wall_buffer)
      vw.z() = -(my_pos.z() - (max_z - _wall_buffer)) / _wall_buffer;
  }

  // Normalize to unit vector (paper does this)
  double mag = vw.norm();
  if (mag > 0.01) vw = vw / mag;
  return vw;
}

// =============================================================================
// Task Allocation
// =============================================================================

void PlannerNode::task_callback(const ground_system_msgs::msg::SwarmTask::SharedPtr msg) {
  bool should_auction = false;
  {
    const std::lock_guard<std::mutex> lock(_task_mutex);

    // Find existing task or add new one
    for (auto& t : _known_tasks) {
      if (t.id == msg->task_id) {
        t.assigned_drone = msg->assigned_drone_id;
        t.status = msg->status;
        return;
      }
    }

    // New task
    SwarmTaskData task;
    task.id = msg->task_id;
    task.location = Eigen::Vector3d(msg->location.x, msg->location.y, msg->location.z);
    task.duration = msg->duration;
    task.priority = msg->priority;
    task.assigned_drone = msg->assigned_drone_id;
    task.status = msg->status;
    task.spawn_time = msg->spawn_time;
    _known_tasks.push_back(task);

    RCLCPP_INFO(this->get_logger(), "[TASK] Received task %u at (%.1f,%.1f) dur=%.0fs pri=%d status=%d",
                task.id, task.location.x(), task.location.y(), task.duration, task.priority, task.status);

    // If unassigned and we're idle (swarming), try to bid
    if (task.status == 0 && _task_state == 0) {
      should_auction = true;
    }
  }
  // Run auction OUTSIDE the lock to avoid deadlock
  // (run_task_auction() acquires _task_mutex internally)
  if (should_auction) {
    run_task_auction();
  }
}

void PlannerNode::spawn_tasks() {
  if (!_enable_task_allocation || _drone_id != 1) return;

  // Count active tasks (not completed)
  int active_count = 0;
  {
    const std::lock_guard<std::mutex> lock(_task_mutex);
    for (const auto& t : _known_tasks) {
      if (t.status < 3) active_count++;
    }
  }
  int max_tasks = static_cast<int>(0.75 * _num_drones);
  if (active_count >= max_tasks) return;

  // Probabilistic spawning
  std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
  if (prob_dist(_task_rng) > _task_spawn_probability) return;

  // Random location within grid bounds
  std::uniform_real_distribution<double> x_dist(_grid_origin_x + _r_separation,
                                                 _grid_origin_x + _grid_width - _r_separation);
  std::uniform_real_distribution<double> y_dist(_grid_origin_y + _r_separation,
                                                 _grid_origin_y + _grid_height - _r_separation);
  std::uniform_real_distribution<double> dur_dist(10.0, 60.0);
  std::uniform_int_distribution<int> pri_dist(1, 5);

  auto msg = ground_system_msgs::msg::SwarmTask();
  msg.header.stamp = this->now();
  msg.task_id = _next_task_id++;
  msg.location.x = x_dist(_task_rng);
  msg.location.y = y_dist(_task_rng);
  msg.location.z = _swarm_altitude;
  msg.duration = dur_dist(_task_rng);
  msg.priority = pri_dist(_task_rng);
  msg.assigned_drone_id = 0;
  msg.status = 0;  // UNASSIGNED
  msg.spawn_time = this->now().seconds();

  swarm_task_pub->publish(msg);

  RCLCPP_INFO(this->get_logger(), "[TASK] Spawned task %u at (%.1f,%.1f) dur=%.0fs pri=%d",
              msg.task_id, msg.location.x, msg.location.y, msg.duration, msg.priority);
}

void PlannerNode::run_task_auction() {
  if (!_enable_task_allocation || _task_state != 0) return;

  const std::lock_guard<std::mutex> lock(_task_mutex);
  Eigen::Vector3d my_pos;
  {
    const std::lock_guard<std::mutex> slock(state_mutex_);
    my_pos = Eigen::Vector3d(_state.pose.position.x, _state.pose.position.y, _state.pose.position.z);
  }

  // Find best unassigned task for this drone
  // Bid = priority / (1 + distance) — works across all map scales
  double best_bid = -std::numeric_limits<double>::max();
  uint32_t best_task_id = 0;

  for (const auto& t : _known_tasks) {
    if (t.status != 0) continue;  // Only bid on unassigned tasks

    double travel_dist = (t.location - my_pos).norm();
    double bid = static_cast<double>(t.priority) / (1.0 + travel_dist);

    if (bid > best_bid) {
      best_bid = bid;
      best_task_id = t.id;
    }
  }

  if (best_task_id == 0) return;

  // Winner determination: each drone bids independently.
  // Assign if no closer neighbor within R_comm exists (tie-break by lower drone_id).
  if (best_bid > 0) {
    // Check if another drone within R_comm is closer (paper: auction bid propagation)
    bool another_closer = false;
    for (const auto& t : _known_tasks) {
      if (t.id != best_task_id) continue;
      const std::lock_guard<std::mutex> nlock(_neighbor_mutex);
      for (auto& [nid, ns] : _neighbor_states) {
        if (!ns.valid) continue;
        double age = (this->now() - ns.last_update).seconds();
        if (age > 2.0) continue;
        // Only consider neighbors within communication range R_comm
        double neighbor_dist_to_me = (ns.position - my_pos).norm();
        if (neighbor_dist_to_me > _r_comm) continue;
        double n_dist = (t.location - ns.position).norm();
        double my_dist = (t.location - my_pos).norm();
        if (n_dist < my_dist || (std::abs(n_dist - my_dist) < 0.1 && nid < _drone_id)) {
          another_closer = true;
          break;
        }
      }
      break;
    }

    if (!another_closer) {
      // Win the auction — assign to self
      for (auto& t : _known_tasks) {
        if (t.id == best_task_id) {
          t.assigned_drone = _drone_id;
          t.status = 1;  // ASSIGNED
          _task_state = 1;  // NAVIGATING_TO_TASK
          _current_task_id = best_task_id;

          // Publish assignment
          auto msg = ground_system_msgs::msg::SwarmTask();
          msg.header.stamp = this->now();
          msg.task_id = t.id;
          msg.location.x = t.location.x();
          msg.location.y = t.location.y();
          msg.location.z = t.location.z();
          msg.duration = t.duration;
          msg.priority = t.priority;
          msg.assigned_drone_id = _drone_id;
          msg.status = 1;
          msg.spawn_time = t.spawn_time;
          swarm_task_pub->publish(msg);

          RCLCPP_INFO(this->get_logger(), "[TASK] Won auction for task %u, navigating to (%.1f,%.1f)",
                      t.id, t.location.x(), t.location.y());
          break;
        }
      }
    }
  }
}

void PlannerNode::execute_task_state() {
  if (!_enable_task_allocation || _current_task_id == 0) return;

  Eigen::Vector3d my_pos;
  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    my_pos = Eigen::Vector3d(_state.pose.position.x, _state.pose.position.y, _state.pose.position.z);
  }

  const std::lock_guard<std::mutex> lock(_task_mutex);
  SwarmTaskData* current_task = nullptr;
  for (auto& t : _known_tasks) {
    if (t.id == _current_task_id) { current_task = &t; break; }
  }
  if (!current_task) {
    _task_state = 0;
    _current_task_id = 0;
    return;
  }

  if (_task_state == 1) {  // NAVIGATING_TO_TASK
    double dist = (current_task->location - my_pos).norm();
    if (dist < _task_proximity_threshold) {
      // Arrived at task location — start execution
      _task_state = 2;  // EXECUTING_TASK
      _task_exec_start_time = this->now().seconds();
      current_task->status = 2;  // IN_PROGRESS

      auto msg = ground_system_msgs::msg::SwarmTask();
      msg.header.stamp = this->now();
      msg.task_id = current_task->id;
      msg.location.x = current_task->location.x();
      msg.location.y = current_task->location.y();
      msg.location.z = current_task->location.z();
      msg.duration = current_task->duration;
      msg.priority = current_task->priority;
      msg.assigned_drone_id = _drone_id;
      msg.status = 2;
      msg.spawn_time = current_task->spawn_time;
      swarm_task_pub->publish(msg);

      RCLCPP_INFO(this->get_logger(), "[TASK] Arrived at task %u, executing for %.0fs",
                  current_task->id, current_task->duration);
    }
  } else if (_task_state == 2) {  // EXECUTING_TASK
    double elapsed = this->now().seconds() - _task_exec_start_time;
    if (elapsed >= current_task->duration) {
      // Task complete
      current_task->status = 3;  // COMPLETED

      auto msg = ground_system_msgs::msg::SwarmTask();
      msg.header.stamp = this->now();
      msg.task_id = current_task->id;
      msg.location.x = current_task->location.x();
      msg.location.y = current_task->location.y();
      msg.location.z = current_task->location.z();
      msg.duration = current_task->duration;
      msg.priority = current_task->priority;
      msg.assigned_drone_id = _drone_id;
      msg.status = 3;
      msg.spawn_time = current_task->spawn_time;
      swarm_task_pub->publish(msg);

      RCLCPP_INFO(this->get_logger(), "[TASK] Completed task %u, returning to swarming",
                  current_task->id);
      _task_state = 0;  // SWARMING
      _current_task_id = 0;
    }
  }
}

// =============================================================================
// Main Swarm Exploration Loop
// =============================================================================

void PlannerNode::swarm_exploration_loop() {
  if (!_swarm_mode) return;
  if (_planner_state == PlanningStates::OFF || _planner_state == PlanningStates::TAKING_OFF) return;

  // Increment metrics counter
  _metrics_total_timesteps++;

  // 1. Update occupancy grid from depth observations
  update_occupancy_grid();

  // 2. Task spawning (drone 1 only)
  spawn_tasks();

  // 3. Task state machine
  execute_task_state();

  // 4. Detect and group frontier regions
  auto regions = group_frontier_regions();
  _frontiers_remaining = 0;
  for (const auto& r : regions) _frontiers_remaining += r.size;

  // 5. Select best frontier region and target its centroid (paper Eq. 26,
  //    MATLAB calculateFrontierVelocity.m: v_frontier = centroid - robot_pos)
  if (!regions.empty()) {
    auto best_region = select_frontier_region(regions);
    _assigned_frontier = best_region.centroid;
    _has_frontier = true;
  } else {
    _has_frontier = false;
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
      "[SWARM] No frontiers remaining — exploration complete!");
  }

  // 6. Compute forces based on paper's decision logic (main.m lines 430-475)
  //
  // Paper algorithm:
  //   if is_critical (any neighbor < R_critical):
  //     v_flock = W_s*vs + W_w*vw;  v_frontier = [0,0,0]
  //   else if swarming:
  //     v_flock = W_c*vc + W_a*va + W_s*vs + W_w*vw
  //     v_frontier = calculateFrontierVelocity(...)    (already scaled by w_f)
  //   else if on_task:
  //     v_flock = W_s*vs + W_w*vw;  v_frontier = [0,0,0]
  //     (updateRobotState handles task navigation separately)
  //
  // Then: v_fused = w_f * v_frontier + (1 - w_f) * v_flock
  //       (where w_f ∈ [0,1] is the frontier blend weight)

  // Check R_critical emergency condition
  bool is_critical = false;
  {
    const std::lock_guard<std::mutex> lock(_neighbor_mutex);
    Eigen::Vector3d my_pos_cr;
    {
      const std::lock_guard<std::mutex> slock(state_mutex_);
      my_pos_cr = Eigen::Vector3d(_state.pose.position.x, _state.pose.position.y, _state.pose.position.z);
    }
    for (auto& [id, ns] : _neighbor_states) {
      if (!ns.valid) continue;
      double age = (this->now() - ns.last_update).seconds();
      if (age > 2.0) continue;
      double dist = (ns.position - my_pos_cr).norm();
      if (dist < _r_critical) {
        is_critical = true;
        break;
      }
    }
  }

  Eigen::Vector3d v_flock(0, 0, 0);
  Eigen::Vector3d v_frontier_vec(0, 0, 0);

  if (is_critical) {
    // EMERGENCY: separation + wall avoidance only, no frontier
    Eigen::Vector3d separation_force = compute_separation();
    Eigen::Vector3d boundary_force = compute_boundary_repulsion();
    v_flock = _w_separation * separation_force + _w_obstacle * boundary_force;
    v_frontier_vec = Eigen::Vector3d(0, 0, 0);
  } else if (_task_state == 1 && _current_task_id > 0) {
    // ON_TASK / NAVIGATING: v_flock = W_s*vs + W_w*vw, navigate to task
    Eigen::Vector3d separation_force = compute_separation();
    Eigen::Vector3d boundary_force = compute_boundary_repulsion();
    v_flock = _w_separation * separation_force + _w_obstacle * boundary_force;
    // Task navigation handled by updateRobotState-like logic: steer toward task
    Eigen::Vector3d task_loc(0, 0, 0);
    {
      const std::lock_guard<std::mutex> lock(_task_mutex);
      for (const auto& t : _known_tasks) {
        if (t.id == _current_task_id) { task_loc = t.location; break; }
      }
    }
    Eigen::Vector2d task_2d(task_loc.x(), task_loc.y());
    v_frontier_vec = compute_frontier_attraction(task_2d);
    // For on_task, w_f blending still applies with task as "frontier target"
  } else if (_task_state == 2) {
    // EXECUTING_TASK: hover at current position
    Eigen::Vector3d separation_force = compute_separation();
    Eigen::Vector3d boundary_force = compute_boundary_repulsion();
    v_flock = _w_separation * separation_force + _w_obstacle * boundary_force;
    v_frontier_vec = Eigen::Vector3d(0, 0, 0);
  } else {
    // SWARMING: full Reynolds flocking + frontier
    Eigen::Vector3d cohesion_force = compute_cohesion();
    Eigen::Vector3d separation_force = compute_separation();
    Eigen::Vector3d alignment_force = compute_alignment();
    Eigen::Vector3d boundary_force = compute_boundary_repulsion();
    v_flock = _w_cohesion * cohesion_force +
              _w_alignment * alignment_force +
              _w_separation * separation_force +
              _w_obstacle * boundary_force;

    if (_has_frontier) {
      v_frontier_vec = compute_frontier_attraction(_assigned_frontier);
      // Paper: v_frontier already scaled by w_f inside calculateFrontierVelocity
      // We keep it as unit vector here and apply w_f in the fusion below
    }
  }

  // 7. Velocity fusion (paper: updateRobotState.m)
  //    v_fused = w_f * v_frontier + (1 - w_f) * v_flock
  //    where w_f ∈ [0,1] controls frontier vs flocking priority
  double wf = std::clamp(_w_frontier, 0.0, 1.0);
  Eigen::Vector3d combined_velocity;
  if (is_critical || _task_state == 2) {
    // Emergency or executing task: no blending, just v_flock
    combined_velocity = v_flock;
  } else {
    // Normalize v_flock and v_frontier before blending (paper normalizes each)
    Eigen::Vector3d v_flock_norm = v_flock;
    double flock_mag = v_flock_norm.norm();
    if (flock_mag > 0.01) v_flock_norm = v_flock_norm / flock_mag;

    Eigen::Vector3d v_frontier_norm = v_frontier_vec;
    double frontier_mag = v_frontier_norm.norm();
    if (frontier_mag > 0.01) v_frontier_norm = v_frontier_norm / frontier_mag;

    combined_velocity = wf * v_frontier_norm + (1.0 - wf) * v_flock_norm;
  }

  // Normalize v_fused to unit direction (paper: v_fused / ||v_fused|| * linear_vel, then
  // only the direction is used by the heading controller)
  double cv_mag = combined_velocity.norm();
  if (cv_mag > 0.01) {
    combined_velocity = combined_velocity / cv_mag;  // unit direction for heading target
  }

  // If executing task and arrived, stop: zero velocity, skip heading update
  bool hovering = (_task_state == 2);

  // 8. Heading-rate-limited motion (paper Section 3.1.4, Eq. 28-31)
  //    MATLAB updateRobotState.m: drone always moves forward in heading direction
  //    at max_speed. Heading steers toward v_fused via proportional controller.
  //    This is a unicycle kinematic model, NOT point-mass Boid theory.
  double linear_vel = hovering ? 0.0 : _max_swarm_speed;

  if (cv_mag > 0.01 && !hovering) {
    // Compute desired heading from v_fused direction
    // MATLAB: [desired_azimuth, desired_elevation] = cart2sph(vx, vy, vz)
    double desired_azimuth = atan2(combined_velocity.y(), combined_velocity.x());
    double desired_elevation = atan2(combined_velocity.z(),
        sqrt(combined_velocity.x() * combined_velocity.x() +
             combined_velocity.y() * combined_velocity.y()));

    if (_motion_mode == 0) {
      desired_elevation = 0.0;  // 2D: no pitch
    }

    // Wrapped angular error (MATLAB: atan2(sin(desired-current), cos(desired-current)))
    double az_error = atan2(sin(desired_azimuth - _current_azimuth),
                            cos(desired_azimuth - _current_azimuth));
    double el_error = atan2(sin(desired_elevation - _current_elevation),
                            cos(desired_elevation - _current_elevation));

    // Proportional heading controller (MATLAB: angular_vel = angular_gain_k * error)
    double angular_vel_az = _heading_gain * az_error;
    double angular_vel_el = (_motion_mode == 0) ? 0.0 : (_heading_gain * el_error);

    // Integrate heading (MATLAB: updated = current + angular_vel * dt)
    _current_azimuth = fmod(_current_azimuth + angular_vel_az * _swarm_dt, 2.0 * M_PI);
    if (_current_azimuth < 0) _current_azimuth += 2.0 * M_PI;
    _current_elevation += angular_vel_el * _swarm_dt;
  }

  if (_motion_mode == 0) {
    _current_elevation = 0.0;
  }

  // Reconstruct velocity from heading (MATLAB: [dx,dy,dz] = sph2cart(az, el, 1))
  double dx = cos(_current_azimuth) * cos(_current_elevation);
  double dy = sin(_current_azimuth) * cos(_current_elevation);
  double dz = sin(_current_elevation);
  Eigen::Vector3d heading_velocity(dx * linear_vel, dy * linear_vel, dz * linear_vel);

  // Altitude control overlays
  if (_motion_mode == 0) {
    // 2D MODE: altitude hold via proportional controller
    double cur_z;
    {
      const std::lock_guard<std::mutex> lock(state_mutex_);
      cur_z = _state.pose.position.z;
    }
    heading_velocity.z() = (_swarm_altitude - cur_z) * 2.0;
  } else {
    // 3D MODE: altitude floor recovery
    double cur_z;
    {
      const std::lock_guard<std::mutex> lock(state_mutex_);
      cur_z = _state.pose.position.z;
    }
    if (cur_z < _fence_min_z + _wall_buffer) {
      double recovery_vz = (_fence_min_z + _wall_buffer - cur_z) * 2.0;
      heading_velocity.z() = std::max(heading_velocity.z(), recovery_vz);
    }
  }

  // Set goal position = current + velocity * lookahead
  double lookahead = 2.0;
  Eigen::Vector3d my_pos;
  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    my_pos = Eigen::Vector3d(_state.pose.position.x, _state.pose.position.y, _state.pose.position.z);
  }
  Eigen::Vector3d target_pos = my_pos + heading_velocity * lookahead;
  target_pos.x() = std::clamp(target_pos.x(), _fence_min_x, _fence_max_x);
  target_pos.y() = std::clamp(target_pos.y(), _fence_min_y, _fence_max_y);
  target_pos.z() = std::clamp(target_pos.z(), _fence_min_z, _fence_max_z);

  // Heading from current azimuth (heading-rate model: heading IS the state)
  _goal_heading = _current_azimuth;

  // Replace combined_velocity with heading_velocity for publishing and logging
  combined_velocity = heading_velocity;

  // 9. Publish velocity setpoint (paper's algorithm outputs velocity directly)
  _setpoint_count++;
  TrajectoryPoint swarm_ref;
  swarm_ref.position = target_pos;
  swarm_ref.velocity = combined_velocity;
  swarm_ref.acceleration = Eigen::Vector3d(0.0, 0.0, 0.0);
  swarm_ref.heading = _goal_heading;
  if (_runtime_mode == RuntimeModes::OMNIDRONES) {
    publish_velocity_command(swarm_ref);
  } else {
    public_ref_pos(swarm_ref);
  }

  RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
    "[HEADING] az=%.1f deg el=%.1f deg speed=%.1f",
    _current_azimuth * 180.0 / M_PI, _current_elevation * 180.0 / M_PI, linear_vel);

  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
    "[HEADING-RATE] drone=%d az=%.1f° vel=(%.2f,%.2f,%.2f) cv_dir=(%.2f,%.2f) hovering=%d",
    _drone_id, _current_azimuth * 180.0 / M_PI,
    heading_velocity.x(), heading_velocity.y(), heading_velocity.z(),
    combined_velocity.x(), combined_velocity.y(), hovering ? 1 : 0);

  // 10. Collision detection for metrics (edge-detect: count transitions into zone)
  {
    const std::lock_guard<std::mutex> lock(_neighbor_mutex);
    bool agent_collision = false;
    for (auto& [id, ns] : _neighbor_states) {
      if (!ns.valid) continue;
      double age = (this->now() - ns.last_update).seconds();
      if (age > 2.0) continue;
      double dist = (ns.position - my_pos).norm();
      if (dist < _collision_radius) {
        agent_collision = true;
        break;
      }
    }
    if (agent_collision && !_was_in_agent_collision) _metrics_agent_collision_count++;
    _was_in_agent_collision = agent_collision;
  }
  {
    double min_x = _grid_origin_x;
    double max_x = _grid_origin_x + _grid_width;
    double min_y = _grid_origin_y;
    double max_y = _grid_origin_y + _grid_height;
    bool wall_collision =
      (my_pos.x() - min_x < _wall_collision_distance) ||
      (max_x - my_pos.x() < _wall_collision_distance) ||
      (my_pos.y() - min_y < _wall_collision_distance) ||
      (max_y - my_pos.y() < _wall_collision_distance);
    if (wall_collision && !_was_in_wall_collision) _metrics_wall_collision_count++;
    _was_in_wall_collision = wall_collision;
  }

  // Compute setpoint frequency every 2 seconds (persist across calls for display)
  double now_sec = this->now().seconds();
  double dt_freq = now_sec - _last_freq_time;
  if (dt_freq >= 2.0) {
    _last_freq_hz = static_cast<double>(_setpoint_count - _last_freq_count) / dt_freq;
    _last_freq_count = _setpoint_count;
    _last_freq_time = now_sec;
  }

  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
    "[SWARM] drone=%d regions=%zu frontiers=%u task_state=%d pos=(%.1f,%.1f) vel=(%.2f,%.2f) setpoint_hz=%.1f",
    _drone_id, regions.size(), _frontiers_remaining, _task_state,
    my_pos.x(), my_pos.y(),
    combined_velocity.x(), combined_velocity.y(), _last_freq_hz);
}

// =============================================================================
// Status & Grid Publishing
// =============================================================================

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

  // Task state
  msg.task_state = _task_state;
  msg.current_task_id = _current_task_id;
  if (_current_task_id > 0) {
    const std::lock_guard<std::mutex> lock(_task_mutex);
    for (const auto& t : _known_tasks) {
      if (t.id == _current_task_id) {
        msg.task_location.x = t.location.x();
        msg.task_location.y = t.location.y();
        msg.task_location.z = t.location.z();
        break;
      }
    }
  }

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
      if (dist <= _r_comm) {
        num_neighbors++;
        nearest_dist = std::min(nearest_dist, dist);
      }
    }
  }
  msg.num_neighbors = num_neighbors;
  msg.nearest_neighbor_dist = (nearest_dist < 1e6) ? nearest_dist : 0.0;

  swarm_status_pub->publish(msg);
}

// =============================================================================
// Map Merging (MATLAB: mergeMaps.m)
// Merges neighbor occupancy grids received via /occupancy_grid topic.
// For each neighbor within communication_range: merged = max(local, neighbor)
// =============================================================================
void PlannerNode::neighbor_grid_callback(
    const ground_system_msgs::msg::OccupancyGrid2D::SharedPtr msg) {
  if (!_swarm_mode) return;
  if (msg->drone_id == _drone_id) return;  // Skip own messages

  // Check if neighbor is within communication range (MATLAB: comm_range_sq check)
  Eigen::Vector3d my_pos;
  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    my_pos = Eigen::Vector3d(_state.pose.position.x, _state.pose.position.y, _state.pose.position.z);
  }

  Eigen::Vector3d neighbor_pos;
  {
    const std::lock_guard<std::mutex> lock(_neighbor_mutex);
    auto it = _neighbor_states.find(static_cast<int>(msg->drone_id));
    if (it == _neighbor_states.end() || !it->second.valid) return;
    double age = (this->now() - it->second.last_update).seconds();
    if (age > 2.0) return;
    neighbor_pos = it->second.position;
  }

  double dist = (neighbor_pos - my_pos).norm();
  if (dist > _r_comm) return;  // Not within communication range

  // Grid compatibility check
  if (msg->width != static_cast<uint32_t>(_grid_cols) ||
      msg->height != static_cast<uint32_t>(_grid_rows)) return;

  // Merge: element-wise max (MATLAB: max(coverage_i, coverage_k))
  double merge_time = this->now().seconds();
  {
    const std::lock_guard<std::mutex> lock(_grid_mutex);
    uint32_t new_cells = 0;
    size_t grid_size = _occupancy_grid.size();
    for (size_t i = 0; i < grid_size && i < msg->data.size(); ++i) {
      if (msg->data[i] > _occupancy_grid[i]) {
        if (_occupancy_grid[i] == 0 && msg->data[i] == 1) new_cells++;
        _occupancy_grid[i] = msg->data[i];
        // Approximate last_visit_time for merged cells (MATLAB uses max of timestamps;
        // we don't have the neighbor's timestamps, so use current time as proxy)
        _last_visit_time[i] = merge_time;
      }
    }
    _cells_explored += new_cells;

    // Cache this neighbor's full grid for global-map frontier computation
    // (MATLAB: global_map.coverage = max across all robot_local_maps each timestep)
    _neighbor_grids[static_cast<int>(msg->drone_id)].assign(msg->data.begin(), msg->data.end());

    if (new_cells > 0) {
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
        "[MAP-MERGE] drone=%d merged %u cells from drone %u (total=%d/%d)",
        _drone_id, new_cells, msg->drone_id, _cells_explored, _grid_cols * _grid_rows);
    }
  }
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

void PlannerNode::publish_swarm_metrics() {
  if (!_swarm_mode) return;

  auto msg = ground_system_msgs::msg::SwarmMetrics();
  msg.header.stamp = this->now();
  msg.drone_id = _drone_id;

  uint32_t total = _grid_cols * _grid_rows;
  msg.coverage_percent = (total > 0) ? (100.0f * _cells_explored / total) : 0.0f;

  // Revisit heatmap
  {
    const std::lock_guard<std::mutex> lock(_grid_mutex);
    msg.visit_counts = _visit_counts;
  }

  // Collision metrics
  msg.total_timesteps = _metrics_total_timesteps;
  msg.agent_collision_count = _metrics_agent_collision_count;
  msg.wall_collision_count = _metrics_wall_collision_count;
  msg.agent_collision_rate = (_metrics_total_timesteps > 0) ?
    static_cast<float>(_metrics_agent_collision_count) / _metrics_total_timesteps : 0.0f;
  msg.wall_collision_rate = (_metrics_total_timesteps > 0) ?
    static_cast<float>(_metrics_wall_collision_count) / _metrics_total_timesteps : 0.0f;

  // Grid metadata
  msg.cell_size = _grid_cell_size;
  msg.origin_x = _grid_origin_x;
  msg.origin_y = _grid_origin_y;
  msg.width = _grid_cols;
  msg.height = _grid_rows;

  swarm_metrics_pub->publish(msg);
}
