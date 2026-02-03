#include "planner_node.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>

bool PlannerNode::loadParameters() {

  // Declare and get parameters
  this->declare_parameter<std::string>("scenario", "sitl");
  this->declare_parameter<std::string>("planner_config_path", "");
  
  std::string scenario_str;
  if (!this->get_parameter("scenario", scenario_str)) {
    RCLCPP_ERROR(this->get_logger(), "Failed to get scenario parameter");
    return false;
  }

  // Get planner config path from parameter or environment
  std::string planner_config_path;
  if (this->get_parameter("planner_config_path", planner_config_path) && !planner_config_path.empty()) {
    // Use provided path
  } else {
    // Try to construct from environment variable
    const char* planner_path_env = std::getenv("PLANNER_PATH");
    if (planner_path_env) {
      planner_config_path = std::string(planner_path_env) + "/configs/" + scenario_str + ".yaml";
    } else {
      // Use relative path in the install directory
      planner_config_path = ament_index_cpp::get_package_share_directory("midi") + "/configs/" + scenario_str + ".yaml";
    }
  }

  RCLCPP_INFO(this->get_logger(), "Loading planner config from: %s", planner_config_path.c_str());

  // Load planner parameters
  YAML::Node planner_config;
  try {
    planner_config = YAML::LoadFile(planner_config_path);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to load planning config file: %s", e.what());
    return false;
  }

  if (!planner_config) {
    RCLCPP_WARN(this->get_logger(), "Planning config file not found");
    return false;
  }

  // Runtime mode
  std::string runtime_mode_str = planner_config["runtime_mode"].as<std::string>();
  if (runtime_mode_str == "omnidrones") {
    _runtime_mode = RuntimeModes::OMNIDRONES;
  } else if (runtime_mode_str == "mavros") {
    _runtime_mode = RuntimeModes::MAVROS;
  } else if (runtime_mode_str == "omnidrones") {
    _runtime_mode = RuntimeModes::OMNIDRONES;
  }

  // Collision checking method
  std::string collision_checking_method_str = planner_config["collision_checking_method"].as<std::string>();
  if (collision_checking_method_str == "midi") {
    _collision_checking_method = CollisionCheckingMethod::MIDI;
  } else if (collision_checking_method_str == "pyramid") {
    _collision_checking_method = CollisionCheckingMethod::PYRAMID;
  }

  // Visualisation
  _visualise = planner_config["visualise"].as<bool>();

  // Frame names
  _world_frame = planner_config["world_frame_name"].as<std::string>();
  _vehicle_frame = planner_config["vehicle_frame_name"].as<std::string>();

  // Topics
  _depth_topic = planner_config["topics"]["depth"].as<std::string>();

  // Goal coordinates
  _goal_north_coordinate = planner_config["goal_coordinate"]["north"].as<double>();
  _goal_west_coordinate = planner_config["goal_coordinate"]["west"].as<double>();
  _goal_up_coordinate = planner_config["goal_coordinate"]["up"].as<double>();

  if (_runtime_mode == RuntimeModes::OMNIDRONES) {
    // OmniDrones uses NWU world frame (env rotated so obstacles face drone)
    // NWU: X=North(forward), Y=West(left), Z=Up
    // Drone body is FLU: X=Forward, Y=Left, Z=Up
    // At yaw=0, body FLU aligns with world NWU
    _goal_in_world_frame.x = _goal_north_coordinate;  // X = North = forward
    _goal_in_world_frame.y = _goal_west_coordinate;   // Y = West = left
    _goal_in_world_frame.z = _goal_up_coordinate;     // Z = Up
    _goal_set = true;
    RCLCPP_INFO(this->get_logger(), "OmniDrones goal: north=%.1f, west=%.1f, up=%.1f -> NWU (%.1f, %.1f, %.1f)",
                _goal_north_coordinate, _goal_west_coordinate, _goal_up_coordinate,
                _goal_in_world_frame.x, _goal_in_world_frame.y, _goal_in_world_frame.z);
  }

  // Depth camera parameters
  _depth_scale = planner_config["depth_camera"]["depth_scale"].as<double>();
  
  // Load camera intrinsics - used for both OmniDrones and MAVROS modes
  double config_focal_length = planner_config["depth_camera"]["focal_length"].as<double>();
  double config_cx = planner_config["depth_camera"]["cx"].as<double>();
  double config_cy = planner_config["depth_camera"]["cy"].as<double>();
  
  if (_runtime_mode == RuntimeModes::MAVROS) {
    _decimation_factor = planner_config["depth_camera"]["decimation_factor"].as<int>();
    _real_focal_length = config_focal_length / _decimation_factor;
    _real_cx = config_cx / _decimation_factor;
    _real_cy = config_cy / _decimation_factor;
  } else if (_runtime_mode == RuntimeModes::OMNIDRONES) {
    // For OmniDrones, use focal_length directly (already at simulated resolution)
    _decimation_factor = 1;
    _real_focal_length = config_focal_length;
    _real_cx = config_cx;
    _real_cy = config_cy;
  }

  std::vector<double> temp;
  temp.push_back(planner_config["depth_camera"]["depth_uncertainty"]["ca0"].as<double>());
  temp.push_back(planner_config["depth_camera"]["depth_uncertainty"]["ca1"].as<double>());
  temp.push_back(planner_config["depth_camera"]["depth_uncertainty"]["ca2"].as<double>());
  temp.push_back(planner_config["depth_camera"]["depth_uncertainty"]["cl0"].as<double>());
  temp.push_back(planner_config["depth_camera"]["depth_uncertainty"]["cl1"].as<double>());
  temp.push_back(planner_config["depth_camera"]["depth_uncertainty"]["cl2"].as<double>());
  _depth_uncertainty_coeffs = temp;

  // Planning parameters
  _planning_cycle_time = planner_config["planning_cycle_time"].as<double>();
  _checking_time_ratio = planner_config["checking_time_ratio"].as<double>();
  _acc_planning_threshold = planner_config["acc_planning_threshold"].as<double>();
  _vel_planning_threshold = planner_config["vel_planning_threshold"].as<double>();
  _checked_trajectories_threshold = planner_config["checked_trajectories_threshold"].as<int>();
  _3d_planning = planner_config["3d_planning"].as<bool>();
  _2d_z_margin = planner_config["2d_z_margin"].as<double>();
  
  std::string traveling_cost_str = planner_config["traveling_cost"].as<std::string>();
  if (traveling_cost_str == "direction") {
    _traveling_cost = TravelingCost::DIRECTION;
  } else if (traveling_cost_str == "distance") {
    _traveling_cost = TravelingCost::DISTANCE;
  }
  
  _replan_factor = planner_config["replan_factor"].as<double>();
  _debug_num_trajectories = planner_config["debug_num_trajectories"].as<bool>();
  _collision_probability_threshold = planner_config["collision_probability_threshold"].as<double>();
  _sampled_trajectories_threshold = planner_config["sampled_trajectories_threshold"].as<uint32_t>();
  _openmp_chunk_size = planner_config["openmp_chunk_size"].as<uint32_t>();

  // Collision checking parameters
  _true_vehicle_radius = planner_config["true_vehicle_radius"].as<double>();
  _planning_vehicle_radius = planner_config["planning_vehicle_radius"].as<double>();
  _minimum_clear_distance = planner_config["minimum_clear_distance"].as<double>();

  // Fence/world limits (keep drone within safe bounds)
  _fence_min_x = planner_config["fence_limits"]["min_x"].as<double>();
  _fence_max_x = planner_config["fence_limits"]["max_x"].as<double>();
  _fence_min_y = planner_config["fence_limits"]["min_y"].as<double>();
  _fence_max_y = planner_config["fence_limits"]["max_y"].as<double>();
  _fence_min_z = planner_config["fence_limits"]["min_z"].as<double>();
  _fence_max_z = planner_config["fence_limits"]["max_z"].as<double>();
  RCLCPP_INFO(this->get_logger(), "Fence limits: X[%.1f, %.1f], Y[%.1f, %.1f], Z[%.1f, %.1f]",
              _fence_min_x, _fence_max_x, _fence_min_y, _fence_max_y, _fence_min_z, _fence_max_z);

  // SSTO trajectory generation parameters
  _depth_upper_bound = planner_config["depth_upper_bound"].as<double>();
  _depth_lower_bound = planner_config["depth_lower_bound"].as<double>();
  _depth_sampling_margin = planner_config["depth_sampling_margin"].as<double>();
  _is_spiral_sampling = planner_config["is_spiral_sampling"].as<bool>();
  _spiral_sampling_step = planner_config["spiral_sampling_step"].as<uint8_t>();

  // Kinematic constraints
  _max_velocity_x = planner_config["max_velocity_x"].as<double>();
  _max_velocity_y = planner_config["max_velocity_y"].as<double>();
  _max_velocity_z = planner_config["max_velocity_z"].as<double>();
  _max_acceleration_x = planner_config["max_acceleration_x"].as<double>();
  _max_acceleration_y = planner_config["max_acceleration_y"].as<double>();
  _max_acceleration_z = planner_config["max_acceleration_z"].as<double>();

  // Control parameters
  _trajectory_discretisation_cycle = planner_config["trajectory_discretisation_cycle"].as<double>();
  _go_to_goal_threshold = planner_config["go_to_goal_threshold"].as<double>();
  
  std::string mavros_control_mode_str = planner_config["mavros_control_mode"].as<std::string>();
  if (mavros_control_mode_str == "kinematic")
    _mavros_control_mode = MavrosControlModes::KINEMATIC;
  else if (mavros_control_mode_str == "attitude")
    _mavros_control_mode = MavrosControlModes::ATTITUDE;
  std::string setpoint_type_str = planner_config["setpoint_type"].as<std::string>();
  if (setpoint_type_str == "position")
    _setpoint_type = SetpointTypes::POSITION_ONLY;
  else if (setpoint_type_str == "full")
    _setpoint_type = SetpointTypes::FULL_STATE;

  RCLCPP_INFO(this->get_logger(), "Parameters loaded successfully");
  return true;
}
