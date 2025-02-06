#include "planner_node.hpp"

bool PlannerNode::loadParameters() {
  // Load trajectory tracking controller parameters
  if (!base_controller_params_.loadParameters(pnh_)) return false;

  // Load sim parameters
  std::string sim_config_path = std::string(getenv("FLIGHTMARE_PATH")) +
                                "/flightpy/configs/vision/config.yaml";
  YAML::Node sim_config = YAML::LoadFile(sim_config_path);
  // Load fov from Flightmare sim config
  if (!sim_config["rgb_camera"]) {
    ROS_WARN("RGB camera not found in sim config file");
    return false;
  }
  _flightmare_fov = sim_config["rgb_camera"]["fov"].as<double>();

  // Load scenario parameters
  std::string scenario_str;
  if (!quadrotor_common::getParam("scenario", scenario_str, pnh_))
    return false;
  const std::string planner_config_path = std::string(getenv("PLANNER_PATH")) + "/configs/" + scenario_str + ".yaml";

  // Load planner parameters
  YAML::Node planner_config = YAML::LoadFile(planner_config_path);
  if (!planner_config) {
    ROS_WARN("Planning config file not found");
    return false;
  }
  ROS_WARN("Planner config file: %s", planner_config_path.c_str());
  // Scenario parameters
  // Runtime mode
  std::string runtime_mode_str = planner_config["runtime_mode"].as<std::string>();
  if (runtime_mode_str == "flightmare") {
    _runtime_mode = RuntimeModes::FLIGHTMARE;
  } else if (runtime_mode_str == "mavros") {
    _runtime_mode = RuntimeModes::MAVROS;
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
  // topics
  _depth_topic = planner_config["topics"]["depth"].as<std::string>();
  _scene_flow_topic = planner_config["topics"]["scene_flow"].as<std::string>();
  // Goal coordinates
  // For MAVROS ENU, record goal coordinates once taken off due to the takeoff
  // place is not the origin (0,0,0)
  _goal_north_coordinate = planner_config["goal_coordinate"]["north"].as<double>();
  _goal_west_coordinate = planner_config["goal_coordinate"]["west"].as<double>();
  _goal_up_coordinate = planner_config["goal_coordinate"]["up"].as<double>();
  // For Flightmare NWU, take goal coordinates from config file
  if (_runtime_mode == RuntimeModes::FLIGHTMARE) {
    _goal_in_world_frame.x = _goal_north_coordinate;
    _goal_in_world_frame.y = _goal_west_coordinate;
    _goal_in_world_frame.z = _goal_up_coordinate;
    _goal_set = true;
  }
  // Depth camera parameters
  _depth_scale = planner_config["depth_camera"]["depth_scale"].as<double>();
  if (_runtime_mode == RuntimeModes::MAVROS) {
    _decimation_factor = planner_config["depth_camera"]["decimation_factor"].as<int>();
    _real_focal_length = planner_config["depth_camera"]["focal_length"].as<double>() / _decimation_factor;
    _real_cx = planner_config["depth_camera"]["cx"].as<double>() / _decimation_factor;
    _real_cy = planner_config["depth_camera"]["cy"].as<double>() / _decimation_factor;
  }
  std::vector<double> temp;
  temp.push_back(planner_config["depth_camera"]["depth_uncertainty"]["ca0"].as<double>());
  temp.push_back(planner_config["depth_camera"]["depth_uncertainty"]["ca1"].as<double>());
  temp.push_back(planner_config["depth_camera"]["depth_uncertainty"]["ca2"].as<double>());
  temp.push_back(planner_config["depth_camera"]["depth_uncertainty"]["cl0"].as<double>());
  temp.push_back(planner_config["depth_camera"]["depth_uncertainty"]["cl1"].as<double>());
  temp.push_back(planner_config["depth_camera"]["depth_uncertainty"]["cl2"].as<double>());
  _depth_uncertainty_coeffs = temp;


  // Dynamic position uncertainty coefficients
  temp.clear();
  temp.push_back(planner_config["dynamic_pos_uncertainty"]["x"].as<double>());
  temp.push_back(planner_config["dynamic_pos_uncertainty"]["y"].as<double>());
  temp.push_back(planner_config["dynamic_pos_uncertainty"]["z"].as<double>());
  _dynamic_pos_cov_coeffs = temp;
  // Planning parameters
  _planning_cycle_time = planner_config["planning_cycle_time"].as<double>();
  _checking_time_ratio = planner_config["checking_time_ratio"].as<double>();
  _acc_planning_threshold = planner_config["acc_planning_threshold"].as<double>();
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
  _is_periodic_replanning = planner_config["is_periodic_replanning"].as<bool>();
  _replanning_interval = planner_config["replanning_interval"].as<double>();
  _debug_num_trajectories = planner_config["debug_num_trajectories"].as<bool>();
  _collision_probability_threshold = planner_config["collision_probability_threshold"].as<double>();
  _sampled_trajectories_threshold = planner_config["sampled_trajectories_threshold"].as<uint32_t>();
  _openmp_chunk_size = planner_config["openmp_chunk_size"].as<uint32_t>();

  // Collision checking parameters
  _true_vehicle_radius = planner_config["true_vehicle_radius"].as<double>();
  _planning_vehicle_radius = planner_config["planning_vehicle_radius"].as<double>();
  _minimum_clear_distance = planner_config["minimum_clear_distance"].as<double>();

  // SSTO trajectory generation parameters
  // Sampling parameters
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

  return true;
}
