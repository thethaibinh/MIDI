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

  // Dedicated callback groups so heavy planning in img_callback (default MX
  // group) cannot starve state updates or the control timer.
  state_callback_group_ = this->create_callback_group(
    rclcpp::CallbackGroupType::MutuallyExclusive);
  control_callback_group_ = this->create_callback_group(
    rclcpp::CallbackGroupType::MutuallyExclusive);
  rclcpp::SubscriptionOptions state_sub_opts;
  state_sub_opts.callback_group = state_callback_group_;
  // Mission / reset / takeoff events mutate the same flight state
  // as control_loop (trajectory_queue_, reference_trajectory_, _goal_set,
  // planner state). Running them on control_callback_group_ serialises them
  // with the control timer so no extra locks are needed.
  rclcpp::SubscriptionOptions control_sub_opts;
  control_sub_opts.callback_group = control_callback_group_;

  // Publishers
  point_cloud_pub = this->create_publisher<sm::PointCloud2>("/cloud_out", 10);
  visual_pub = this->create_publisher<visualization_msgs::msg::Marker>("/visualization", 10);

  // Publishers based on runtime mode
  if (_runtime_mode == RuntimeModes::MAVROS) {
    // MAVROS runs at root namespace — use absolute topic path
    raw_ref_pos_pub = this->create_publisher<mavros_msgs::msg::PositionTarget>("/mavros/setpoint_raw/local", 10);
  } else if (_runtime_mode == RuntimeModes::OMNIDRONES) {
    // OmniDrones subscribes to /Drone{i}/mavros/... — use relative topic so node namespace applies
    raw_ref_pos_pub = this->create_publisher<mavros_msgs::msg::PositionTarget>("mavros/setpoint_raw/local", 10);
  }

  // Subscribers
  image_sub = this->create_subscription<sm::Image>(
    _depth_topic, rclcpp::SensorDataQoS().keep_last(1),
    std::bind(&PlannerNode::img_callback, this, std::placeholders::_1));

  if (_visualise) {
    visual_sub = this->create_subscription<sm::Image>(
      _depth_topic, rclcpp::SensorDataQoS().keep_last(1),
      std::bind(&PlannerNode::visualise, this, std::placeholders::_1));
  }

  // Mission command subscriber (start trigger — after upload + ACK)
  mission_sub = this->create_subscription<ground_system_msgs::msg::StartSwarmMission>(
    "/start_swarm_mission", 10,
    std::bind(&PlannerNode::mission_callback, this, std::placeholders::_1),
    control_sub_opts);

  // Mission upload subscriber (waypoints — Phase 1, before start)
  mission_upload_sub = this->create_subscription<ground_system_msgs::msg::SwarmMissionUpload>(
    "/swarm_mission_upload", 10,
    std::bind(&PlannerNode::mission_upload_callback, this, std::placeholders::_1),
    control_sub_opts);

  // Mission ACK publisher
  mission_ack_pub_ = this->create_publisher<ground_system_msgs::msg::SwarmMissionAck>(
    "/swarm_mission_ack", 10);

  // Takeoff command subscriber - triggers takeoff only (no goal)
  takeoff_sub = this->create_subscription<ground_system_msgs::msg::Takeoff>(
    "/takeoff", 10,
    std::bind(&PlannerNode::takeoff_callback, this, std::placeholders::_1),
    control_sub_opts);

  reset_sub = this->create_subscription<std_msgs::msg::Empty>(
    "/reset_planner", 10,
    std::bind(&PlannerNode::reset_callback, this, std::placeholders::_1),
    control_sub_opts);

  // Subscribe to odometry - use relative topic so namespace remapping works
  // When running in /Drone1 namespace, this becomes /Drone1/odometry
  odom_sub = this->create_subscription<nav_msgs::msg::Odometry>(
    "odometry", 5,
    std::bind(&PlannerNode::odometry_callback, this, std::placeholders::_1),
    state_sub_opts);

  // For MAVROS mode: subscribe to pose/twist/state from MAVROS
  if (_runtime_mode == RuntimeModes::MAVROS) {
    // MAVROS publishes with BEST_EFFORT QoS - must match for subscription to work
    // Use absolute paths since MAVROS is at root namespace, not Drone1 namespace
    rclcpp::QoS mavros_qos(5);
    mavros_qos.best_effort();

    // Pose/twist also use BEST_EFFORT QoS from MAVROS
  mav_pose_sub = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/mavros/local_position/pose", mavros_qos,
    std::bind(&PlannerNode::mav_pose_callback, this, std::placeholders::_1),
    state_sub_opts);

  mav_twist_sub = this->create_subscription<geometry_msgs::msg::TwistStamped>(
      "/mavros/local_position/velocity_body", mavros_qos,
    std::bind(&PlannerNode::mav_twist_callback, this, std::placeholders::_1),
    state_sub_opts);

    // mav_accel_sub = this->create_subscription<sensor_msgs::msg::Imu>(
    //   "/mavros/imu/data_raw", mavros_qos,
    //   std::bind(&PlannerNode::mav_accel_callback, this, std::placeholders::_1));

    mav_state_sub = this->create_subscription<mavros_msgs::msg::State>(
      "/mavros/state", 10,
      std::bind(&PlannerNode::ardupilot_status_callback, this, std::placeholders::_1),
      state_sub_opts);

    // MAVROS service clients (for real FC) - use absolute paths since services are at root namespace
    arming_srv = this->create_client<mavros_msgs::srv::CommandBool>("/mavros/cmd/arming");
    takeoff_srv = this->create_client<mavros_msgs::srv::CommandTOL>("/mavros/cmd/takeoff");
    land_srv = this->create_client<mavros_msgs::srv::CommandTOL>("/mavros/cmd/land");
    mode_srv = this->create_client<mavros_msgs::srv::SetMode>("/mavros/set_mode");
  }
  // Timer — runs on its own MX group so planning (default group) cannot
  // delay setpoint publication. It may run concurrently with img_callback;
  // trajectory_mutex_ serialises access to trajectory_queue_ / _steered.
  control_loop_timer_ = this->create_wall_timer(
    std::chrono::duration<double>(_trajectory_discretisation_cycle),
    std::bind(&PlannerNode::control_loop, this),
    control_callback_group_);

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
    // Reentrant callback group so OPUS service responses can fire
    // concurrently with the high-frequency depth image callback.
    opus_callback_group_ = this->create_callback_group(
      rclcpp::CallbackGroupType::Reentrant);

    rclcpp::QoS opus_qos(10);
    opus_qos.reliable();

    // Topic publisher: lock request/cancel (no reply — coordinator responds via /opus/status)
    opus_lock_req_pub_ = this->create_publisher<OpusPlanLockReqMsg>(
      "/opus/plan_lock_request", opus_qos);

    // Topic publisher for abort
    opus_plan_abort_pub_ = this->create_publisher<ground_system_msgs::msg::OpusPlanAbort>(
      "/opus/plan_abort", opus_qos);

    // Topic publisher: trajectory submission (coordinator replies on /opus/trajectory_ack)
    opus_traj_submit_pub_ = this->create_publisher<ground_system_msgs::msg::OpusTrajectorySubmit>(
      "/opus/trajectory_submit", opus_qos);

    // Subscription: coordinator ack of submitted trajectory
    rclcpp::SubscriptionOptions opus_cb_opts;
    opus_cb_opts.callback_group = opus_callback_group_;
    opus_traj_ack_sub_ = this->create_subscription<ground_system_msgs::msg::OpusTrajectoryAck>(
      "/opus/trajectory_ack", opus_qos,
      std::bind(&PlannerNode::opus_trajectory_ack_callback, this, std::placeholders::_1),
      opus_cb_opts);

    // Status subscription: detects "lock granted to me"
    opus_status_sub_ = this->create_subscription<ground_system_msgs::msg::OpusStatus>(
      "/opus/status", opus_qos,
      std::bind(&PlannerNode::opus_status_callback, this, std::placeholders::_1),
      opus_cb_opts);

    RCLCPP_INFO(this->get_logger(), "OPUS coordination enabled (drone_id=%d)", opus_drone_id_);
  } else {
    RCLCPP_INFO(this->get_logger(), "OPUS coordination disabled (solo mode)");
  }

  // Initial position publisher for GCS (TRANSIENT_LOCAL so late-joining GUI receives it)
  rclcpp::QoS latched_qos(1);
  latched_qos.reliable().transient_local();
  initial_position_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>(
    "initial_position", latched_qos);

  RCLCPP_INFO(this->get_logger(), "MIDI Planner initialized");
}

// All other PlannerNode method implementations are in:
//   planner_node_state_callbacks.cpp  — odometry, pose, twist, accel, FC status
//   planner_node_state_machine.cpp   — update_planner_state, set_auto_pilot_state_forced
//   planner_node_tracker.cpp         — control_loop, trajectory tracking, setpoint publishing
//   planner_node_planner.cpp         — img_callback, depth preprocessing, point cloud
//   planner_node_opus.cpp            — OPUS coordination (lock, submit, ack, abort)
//   planner_node_mission.cpp         — mission upload/start, takeoff, reset, waypoints
//   planner_node_visualiser.cpp      — visualise (point cloud, trajectory, goal markers)
//   planner_node_parameters.cpp      — loadParameters
