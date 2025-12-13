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
  // Initialize TF2 buffers and listeners
  to_world_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  to_vehicle_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  to_world_tf2_ = std::make_shared<tf2_ros::TransformListener>(*to_world_buffer_);
  to_vehicle_tf2_ = std::make_shared<tf2_ros::TransformListener>(*to_vehicle_buffer_);

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
  control_command_pub_ = this->create_publisher<dodgeros_msgs::msg::Command>(
    "/kingfisher/dodgeros_pilot/feedthrough_command", 1);
  point_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_out", 1);
  visual_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/visualization", 1);
  raw_ref_pos_pub_ = this->create_publisher<mavros_msgs::msg::PositionTarget>(
    "mavros/setpoint_raw/local", 1);
  att_ctrl_pub_ = this->create_publisher<mavros_msgs::msg::AttitudeTarget>(
    "/mavros/setpoint_raw/attitude", 1);

  // Subscribers
  image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    _depth_topic, sensor_qos,
    std::bind(&PlannerNode::img_callback, this, std::placeholders::_1));

  if (_visualise) {
    visual_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      _depth_topic, sensor_qos,
      std::bind(&PlannerNode::visualise, this, std::placeholders::_1));
  }

  start_sub_ = this->create_subscription<std_msgs::msg::Empty>(
    "/kingfisher/start_navigation", 1,
    std::bind(&PlannerNode::start_callback, this, std::placeholders::_1));

  reset_sub_ = this->create_subscription<std_msgs::msg::Empty>(
    "/kingfisher/dodgeros_pilot/reset_sim", 1,
    std::bind(&PlannerNode::reset_callback, this, std::placeholders::_1));

  state_sub_ = this->create_subscription<dodgeros_msgs::msg::QuadState>(
    "/kingfisher/dodgeros_pilot/state", rclcpp::QoS(1).best_effort(),
    std::bind(&PlannerNode::state_callback, this, std::placeholders::_1));

  odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
    "/delta/odometry_sensor1/odometry", rclcpp::QoS(1).best_effort(),
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

  // Service clients
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
  } else if (_runtime_mode == RuntimeModes::MAVROS) {
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

void PlannerNode::start_callback(const std_msgs::msg::Empty::SharedPtr msg) {
  (void)msg;
  RCLCPP_WARN(this->get_logger(), "[%s] Planner: Start publishing commands!", this->get_name());
  set_auto_pilot_state_forced(PlanningStates::START);
  // Clear global reference trajectory
  steering_value = 0.0f;
  _steered = false;
  trajectory_queue_.clear();
  reference_trajectory_ = ruckig::Trajectory<3>();
  had_reference_trajectory = false;
}

void PlannerNode::reset_callback(const std_msgs::msg::Empty::SharedPtr msg) {
  (void)msg;
  RCLCPP_WARN(this->get_logger(), "[%s] Planner: Reset quadrotor simulator!", this->get_name());
  set_auto_pilot_state_forced(PlanningStates::OFF);
  steering_value = 0.0f;
  _steered = false;
  if (_runtime_mode == RuntimeModes::MAVROS) {
    _goal_set = false;
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
  // Store latest pose timestamp
  _latest_pose_stamp = msg->header.stamp;

  // pose in MAVROS "map" ENU
  _state.pose = msg->pose;
  _agi_state.p = agi::fromRosVec3(msg->pose.position);
  _agi_state.q(agi::Quaternion(msg->pose.orientation.w, msg->pose.orientation.x,
                               msg->pose.orientation.y,
                               msg->pose.orientation.z));
  // Update state timestamp with oldest of the latest messages
  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    auto min_stamp = std::min({_latest_pose_stamp, _latest_twist_stamp, _latest_accel_stamp},
      [](const auto& a, const auto& b) {
        return rclcpp::Time(a) < rclcpp::Time(b);
      });
    _agi_state.t = rclcpp::Time(min_stamp).seconds();
  }
  dodgeros_msgs::msg::QuadState ros_state = toRosQuadState(_agi_state);
  state_callback(std::make_shared<dodgeros_msgs::msg::QuadState>(ros_state));
}

// Body: "base_link" - FLU
void PlannerNode::mav_twist_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg) {
  // Store latest twist timestamp
  _latest_twist_stamp = msg->header.stamp;

  // linear velocity
  _agi_state.v = agi::fromRosVec3(msg->twist.linear);
  // angular velocity
  _agi_state.w = agi::fromRosVec3(msg->twist.angular);

  const std::lock_guard<std::mutex> lock(state_mutex_);
  // Update state timestamp with oldest of the latest messages
  if (_collision_checking_method == CollisionCheckingMethod::PYRAMID) {
    auto min_stamp = std::min({_latest_pose_stamp, _latest_twist_stamp, _latest_accel_stamp},
      [](const auto& a, const auto& b) {
        return rclcpp::Time(a) < rclcpp::Time(b);
      });
    _agi_state.t = rclcpp::Time(min_stamp).seconds();
  } else {
    auto min_stamp = std::min(_latest_pose_stamp, _latest_twist_stamp,
      [](const auto& a, const auto& b) {
        return rclcpp::Time(a) < rclcpp::Time(b);
      });
    _agi_state.t = rclcpp::Time(min_stamp).seconds();
  }
}

// Body: "base_link" - FLU
void PlannerNode::mav_accel_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
  if (_collision_checking_method != CollisionCheckingMethod::PYRAMID) {
    return;
  }
  // Store latest accel timestamp
  _latest_accel_stamp = msg->header.stamp;

  // imu accel is a sum of translational and anti-gravity in FLU (positive
  // gravity is measured in the Downward direction)
  geometry_msgs::msg::Vector3 acceleration_body_frame = msg->linear_acceleration;
  geometry_msgs::msg::Vector3 anti_gravity_world_frame, anti_gravity_body_frame;
  anti_gravity_world_frame.z = ANTI_G_ENU;

  // transform ENU ("map") to FLU ("base_link")
  geometry_msgs::msg::TransformStamped w2b_transform;
  try {
    w2b_transform = to_vehicle_buffer_->lookupTransform(_vehicle_frame, _world_frame, tf2::TimePointZero);
  } catch (tf2::TransformException& ex) {
    RCLCPP_WARN(this->get_logger(), "%s", ex.what());
    return;
  }
  tf2::doTransform(anti_gravity_world_frame, anti_gravity_body_frame, w2b_transform);
  _agi_state.a = agi::fromRosVec3(eigenToGeometry(toEigen(acceleration_body_frame) - toEigen(anti_gravity_body_frame)));
  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    // Update state timestamp with oldest of the latest messages
    auto min_stamp = std::min({_latest_pose_stamp, _latest_twist_stamp, _latest_accel_stamp},
      [](const auto& a, const auto& b) {
        return rclcpp::Time(a) < rclcpp::Time(b);
      });
    _agi_state.t = rclcpp::Time(min_stamp).seconds();
  }
}

void PlannerNode::odometry_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
  agi::QuadState state;

  state.setZero();
  state.t = rclcpp::Time(msg->header.stamp).seconds();
  state.p = agi::fromRosVec3(msg->pose.pose.position);
  state.q(agi::Quaternion(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                     msg->pose.pose.orientation.y,
                     msg->pose.pose.orientation.z));
  state.v = agi::fromRosVec3(msg->twist.twist.linear);
  state.w = agi::fromRosVec3(msg->twist.twist.angular);

  dodgeros_msgs::msg::QuadState ros_state = toRosQuadState(state);
  state_callback(std::make_shared<dodgeros_msgs::msg::QuadState>(ros_state));
}

void PlannerNode::state_callback(const dodgeros_msgs::msg::QuadState::SharedPtr state) {
  const std::lock_guard<std::mutex> lock(state_mutex_);
  // assign new state
  _state = *state;
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
  if (trajectory_queue_.size()) {
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
  // Update autopilot state
  // Start/Takeoff switching (turn on)
  if (_runtime_mode == RuntimeModes::MAVROS && _planner_state == PlanningStates::OFF &&
      flight_controller_status.mode == "GUIDED" && flight_controller_status.armed) {
    auto request = std::make_shared<mavros_msgs::srv::CommandTOL::Request>();
    request->altitude = _goal_in_world_frame.z;
    auto result = takeoff_srv_->async_send_request(request);
    if (result.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready) {
      auto response = result.get();
      if (response->success) {
      set_auto_pilot_state_forced(PlanningStates::START);
        RCLCPP_WARN(this->get_logger(), "[%s] Taken off!", this->get_name());
        usleep(100000);  // 0.1 second
      }
    }
    // Recording instant goal for an Ardupilot flight trial
    if (!_goal_set) {
      // East
      _goal_in_world_frame.x = _state.pose.position.x - _goal_west_coordinate;
      // North
      _goal_in_world_frame.y = _state.pose.position.y + _goal_north_coordinate;
      // Up
      _goal_in_world_frame.z = _goal_up_coordinate;
      RCLCPP_WARN(this->get_logger(), "[%s] Setting goal to (%.2f, %.2f, %.2f)",
                  this->get_name(), _goal_in_world_frame.x,
               _goal_in_world_frame.y, _goal_in_world_frame.z);
      _goal_set = true;
    }
  }

  // Reset the planner when the quadrotor is disarmed
  if (_runtime_mode == RuntimeModes::MAVROS && _planner_state != PlanningStates::OFF &&
      !flight_controller_status.armed)
    reset_callback(nullptr);

  // Skip updating planner state if goal is not set
  if (!_goal_set) return;

  // Trajectory control switching
  geometry_msgs::msg::Point goal_in_world_frame = _goal_in_world_frame;
  // only check distance to goal horizontally for switching control
  goal_in_world_frame.z = _state.pose.position.z;
  double distance_to_goal =
    (geometryToEigen(_state.pose.position) - geometryToEigen(goal_in_world_frame)).norm();
  if (_state.pose.position.z >= (_goal_in_world_frame.z - 0.1) &&
      _planner_state == PlanningStates::START) {
    set_auto_pilot_state_forced(PlanningStates::TRAJECTORY_CONTROL);
  }

  // Stop planning when the goal is less than 1.5m close.
  else if (_planner_state == PlanningStates::TRAJECTORY_CONTROL &&
           (distance_to_goal < _go_to_goal_threshold ||
            // NWU
            ((_state.pose.position.x + _go_to_goal_threshold / 3) > _goal_in_world_frame.x &&
             _runtime_mode == RuntimeModes::FLIGHTMARE) ||
            // ENU
            ((_state.pose.position.y + _go_to_goal_threshold / 10) > _goal_in_world_frame.y &&
             _runtime_mode == RuntimeModes::MAVROS))) {
    set_auto_pilot_state_forced(PlanningStates::GO_TO_GOAL);
    _stop_planning_point_in_world_frame = _state.pose.position;
  } else if (_runtime_mode == RuntimeModes::MAVROS &&
             _planner_state == PlanningStates::GO_TO_GOAL &&
             distance_to_goal < _go_to_goal_threshold * 0.2) {
    auto request = std::make_shared<mavros_msgs::srv::CommandTOL::Request>();
    auto result = land_srv_->async_send_request(request);
    if (result.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready) {
      auto response = result.get();
      if (response->success) {
      set_auto_pilot_state_forced(PlanningStates::LAND);
        RCLCPP_WARN(this->get_logger(), "[%s] Landing!", this->get_name());
      }
    }
  // In Flightmare, the evaluation script will restart the node,
  // therefore we don't need to reset the planner state
  } else if (_runtime_mode == RuntimeModes::MAVROS &&
             _planner_state == PlanningStates::LAND && !flight_controller_status.armed) {
    reset_callback(nullptr);
  }
}

void PlannerNode::track_trajectory() {
  if (_planner_state == PlanningStates::LAND ||
      _planner_state == PlanningStates::OFF || !_goal_set)
    return;
  if (!had_reference_trajectory && _runtime_mode == RuntimeModes::MAVROS)
    return;

  double control_command_delay = 0.0;
  rclcpp::Time wall_time_now = this->now();
  rclcpp::Time command_execution_time = wall_time_now + rclcpp::Duration::from_seconds(control_command_delay);

  // Trajectory discretization
  TrajectoryPoint reference_point;
  if (_planner_state == PlanningStates::START) {
    _reference_trajectory_start_time = command_execution_time;
    steering_value = 0.0f;
    if (_runtime_mode == RuntimeModes::MAVROS)
      return;
    if (_runtime_mode == RuntimeModes::FLIGHTMARE) {
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
    Eigen::Vector3d current_euler_angles =
      quaternionToEulerAnglesZYX(geometryToEigen(_state.pose.orientation));
    reference_point.heading = current_euler_angles(2);
  }

  // Run position controller to track trajectory sample point
  if (_state.t - this->now().seconds() > 0.2) {
    RCLCPP_WARN(this->get_logger(), "[%s] State is too old, skipping control command", this->get_name());
    return;
  }
  if (_runtime_mode == RuntimeModes::MAVROS &&
      _mavros_control_mode == MavrosControlModes::KINEMATIC) {
    public_ref_pos(reference_point);
    return;
  }
  QuadStateEstimate quad_state_ = quad_common_state_from_dodgedrone_state(_state);
  ControlCommand command = base_controller_.run(quad_state_, reference_point, base_controller_params_);
  if (_runtime_mode == RuntimeModes::FLIGHTMARE) {
    command.timestamp = wall_time_now;
    command.expected_execution_time = command_execution_time;
    command.control_mode = ControlMode::BODY_RATES;
    publish_control_command(command);
  } else if (_runtime_mode == RuntimeModes::MAVROS &&
             _mavros_control_mode == MavrosControlModes::ATTITUDE) {
    public_ref_att(command);
  }
}

void PlannerNode::public_ref_pos(const quadrotor_common::TrajectoryPoint& reference_point) {
  mavros_msgs::msg::PositionTarget msg;
  msg.header.stamp = this->now();
  // FRAME_LOCAL_NED
  msg.coordinate_frame = 1;
  msg.type_mask = 0;
  // reference_point in ENU
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

void PlannerNode::public_ref_att(const ControlCommand& control_cmd) {
  mavros_msgs::msg::AttitudeTarget att_setpoint;
  att_setpoint.type_mask = 0;
  att_setpoint.header.stamp = this->now();
  att_setpoint.orientation.w = control_cmd.orientation.w();
  att_setpoint.orientation.x = control_cmd.orientation.x();
  att_setpoint.orientation.y = control_cmd.orientation.y();
  att_setpoint.orientation.z = control_cmd.orientation.z();
  att_setpoint.thrust = control_cmd.collective_thrust * 0.05;
  att_setpoint.body_rate.x = control_cmd.bodyrates(0);
  att_setpoint.body_rate.y = control_cmd.bodyrates(1);
  att_setpoint.body_rate.z = control_cmd.bodyrates(2);
  att_ctrl_pub_->publish(att_setpoint);
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

void PlannerNode::publish_control_command(const ControlCommand& control_cmd) {
  if (control_cmd.control_mode == ControlMode::NONE) {
    RCLCPP_ERROR(this->get_logger(), "[%s] Control mode is NONE, will not publish ControlCommand",
                 this->get_name());
  } else {
    dodgeros_msgs::msg::Command ros_command;
    ros_command.header.stamp = control_cmd.timestamp;
    ros_command.t = _state.t;
    ros_command.is_single_rotor_thrust = false;
    ros_command.collective_thrust = control_cmd.collective_thrust;
    ros_command.bodyrates.x = control_cmd.bodyrates.x();
    ros_command.bodyrates.y = control_cmd.bodyrates.y();
    ros_command.bodyrates.z = control_cmd.bodyrates.z();

    control_command_pub_->publish(ros_command);
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
  switch (_planner_state) {
    case PlanningStates::OFF:
      state_name = "OFF";
      break;
    case PlanningStates::START:
      state_name = "START";
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
    if (_runtime_mode == RuntimeModes::FLIGHTMARE) {
      // in Flightmare, raw velocity and acceleration are in NWU (world frame)
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
  } else if (_runtime_mode == RuntimeModes::MAVROS) {
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

  // Only consider a valid trajectory
  if (!check_valid_trajectory(position_world_frame, opt_traj)) return;

  // New traj generated
  {
    const std::lock_guard<std::mutex> lock(trajectory_mutex_);
    steering_value = 0.0f;
    _steered = false;
    opt_traj.assign_body_to_world_transform(body_to_world);
    opt_traj.assign_world_to_body_transform(world_to_body);
    trajectory_queue_.push_back(opt_traj);
  }

  // Visualisation
  // parameters for visualization
  visualization_msgs::msg::Marker pyramids_bases, pyramids_edges, polynomial_trajectory, goal_marker;
  pyramids_bases.header.frame_id = pyramids_edges.header.frame_id = _vehicle_frame;
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
  pyramids_bases.scale.x = pyramids_edges.scale.x =
    polynomial_trajectory.scale.x = 0.02;
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

  goal_marker.pose.position.x = _goal_in_world_frame.x;
  goal_marker.pose.position.y = _goal_in_world_frame.y;
  goal_marker.pose.position.z = _goal_in_world_frame.z;
  goal_marker.pose.orientation.w = 1.0;
  // Publish goal marker
  visual_pub_->publish(goal_marker);

  // Publish polynomial trajectory
  double trajectory_duration = reference_trajectory_.get_duration();
  if (trajectory_duration < 0.01) {
    polynomial_trajectory.points.clear();
    visual_pub_->publish(polynomial_trajectory);
    return;
  }
  for (int i = 0; i <= 100; i++) {
    geometry_msgs::msg::Point position = reference_trajectory_.get_position_in_world_frame(trajectory_duration * i / 100);
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
  if (_planner_state == PlanningStates::START)
    polynomial_trajectory.points.clear();
  visual_pub_->publish(pyramids_bases);
  visual_pub_->publish(polynomial_trajectory);

  // Publish pyramids
  double trajectory_depth = 4;
  std::vector<Pyramid> pyramids;
  Eigen::Vector3d corners[4];

  if (_collision_checking_method == CollisionCheckingMethod::PYRAMID) {
    // pyramids = planner.get_pyramids();
  } else {
    // FOV bounding box
    corners[0] = camera.deproject_pixel_to_point(0, 0, trajectory_depth);
    corners[1] = camera.deproject_pixel_to_point(camera.get_width(), 0, trajectory_depth);
    corners[2] = camera.deproject_pixel_to_point(camera.get_width(), camera.get_height(), trajectory_depth);
    corners[3] = camera.deproject_pixel_to_point(0, camera.get_height(), trajectory_depth);
    Pyramid fov_pyramid(trajectory_depth, corners);
    pyramids.push_back(fov_pyramid);

    // Sampling bounding box
    std::vector<int> sampling_range = trajectory_sampler.get_sampling_range();
    corners[0] = camera.deproject_pixel_to_point(sampling_range[0], sampling_range[2], trajectory_depth);
    corners[1] = camera.deproject_pixel_to_point(sampling_range[1], sampling_range[2], trajectory_depth);
    corners[2] = camera.deproject_pixel_to_point(sampling_range[1], sampling_range[3], trajectory_depth);
    corners[3] = camera.deproject_pixel_to_point(sampling_range[0], sampling_range[3], trajectory_depth);
    Pyramid sampling_pyramid(trajectory_depth, corners);
    pyramids.push_back(sampling_pyramid);
  }
  if (pyramids.empty()) {
    return;
  }
  for (std::vector<Pyramid>::iterator it = pyramids.begin();
       it != pyramids.end(); ++it) {
    for (uint8_t i = 0; i < 4; i++) {
      p.x = p.y = p.z = 0;
      pyramids_edges.points.push_back(p);
      p.x = (*it)._corners[i].z();
      p.y = -(*it)._corners[i].x();
      p.z = -(*it)._corners[i].y();
      pyramids_bases.points.push_back(p);
      pyramids_edges.points.push_back(p);
      // close the rectangle base
      if (i == 3) {
        p.x = (*it)._corners[0].z();
        p.y = -(*it)._corners[0].x();
        p.z = -(*it)._corners[0].y();
      } else {
        p.x = (*it)._corners[i+1].z();
        p.y = -(*it)._corners[i+1].x();
        p.z = -(*it)._corners[i+1].y();
      }
      pyramids_bases.points.push_back(p);
    }
  }

  // Goal vector
  p.x = p.y = p.z = 0;
  pyramids_edges.points.push_back(p);
  p.x = goal_in_camera_frame.point.x;
  p.y = goal_in_camera_frame.point.y;
  p.z = goal_in_camera_frame.point.z;
  pyramids_edges.points.push_back(p);

  visual_pub_->publish(pyramids_bases);
  visual_pub_->publish(pyramids_edges);
}

QuadStateEstimate PlannerNode::quad_common_state_from_dodgedrone_state(
  const dodgeros_msgs::msg::QuadState& _state) {
  QuadStateEstimate quad_state_;

  // frame ID
  quad_state_.coordinate_frame = QuadStateEstimate::CoordinateFrame::WORLD;

  // velocity
  quad_state_.velocity =
    Eigen::Vector3d(_state.velocity.linear.x, _state.velocity.linear.y,
                    _state.velocity.linear.z);

  // position
  quad_state_.position = Eigen::Vector3d(
    _state.pose.position.x, _state.pose.position.y, _state.pose.position.z);

  // attitude
  quad_state_.orientation =
    Eigen::Quaterniond(_state.pose.orientation.w, _state.pose.orientation.x,
                       _state.pose.orientation.y, _state.pose.orientation.z);

  // angular velocity
  quad_state_.bodyrates =
    Eigen::Vector3d(_state.velocity.angular.x, _state.velocity.angular.y,
                    _state.velocity.angular.z);

  return quad_state_;
}

void PlannerNode::visualise(const sensor_msgs::msg::Image::SharedPtr depth_msg) {
  if (!rclcpp::ok()) {
    return;
  }
  // convert depth image to point cloud
  pointcloud_type* cloud = create_point_cloud(depth_msg);
  sensor_msgs::msg::PointCloud2 cloudMessage;
  pcl::toROSMsg(*cloud, cloudMessage);
  point_cloud_pub_->publish(cloudMessage);

  // Visualisation
  // parameters for visualization
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
  if (_planner_state == PlanningStates::START)
    polynomial_trajectory.points.clear();
  visual_pub_->publish(pyramids_bases);
  visual_pub_->publish(polynomial_trajectory);
}
