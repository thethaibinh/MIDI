#ifndef PLANNER_NODE_HPP
#define PLANNER_NODE_HPP

#pragma once
#include <unistd.h>

#include <boost/program_options.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <yaml-cpp/yaml.h>

#include "depth_uncertainty_planner/base_planner.hpp"
#include "depth_uncertainty_planner/sampling.hpp"
#include <common_math/frame_transforms.hpp>

// ROS2 base
#include <rclcpp/rclcpp.hpp>

// autopilot
#include "autopilot_states.h"

// math
// #include "common.hpp"

// ROS2 messages
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/int8.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/empty.hpp>

// mavros messages (for Ardupilot mode)
#include <mavros_msgs/msg/position_target.hpp>
#include <mavros_msgs/msg/state.hpp>
#include <mavros_msgs/srv/command_bool.hpp>
#include <mavros_msgs/srv/command_tol.hpp>
#include <mavros_msgs/srv/set_mode.hpp>

// Ground system messages
#include <ground_system_msgs/msg/start_swarm_mission.hpp>
#include <ground_system_msgs/msg/fbv_goal.hpp>
#include <ground_system_msgs/msg/benchmark_status.hpp>
#include <ground_system_msgs/msg/takeoff.hpp>
#include <ground_system_msgs/msg/fly_to.hpp>
#include <ground_system_msgs/msg/swarm_params.hpp>
#include <ground_system_msgs/msg/swarm_exploration_status.hpp>
#include <ground_system_msgs/msg/occupancy_grid2_d.hpp>

// CV
#include <cv_bridge/cv_bridge.h>

#include <sstream>

// Replacement headers for quadrotor_common dependencies
#include "common_math/ros_eigen.hpp"

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

// Eigen
#include <Eigen/Dense>

// Replacement headers for quadrotor dependencies
#include <common_math/trajectory_point.h>
#include <common_math/geometry_eigen_conversions.h>

// Ruckig
#include <ruckig/ruckig.hpp>

// ROS2 TF2
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// PCL
// PCL for point cloud visualization
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace sm = sensor_msgs::msg;
typedef pcl::PointXYZ point_type;
typedef pcl::PointCloud<point_type> pointcloud_type;

using namespace common_math;
using namespace depth_uncertainty_planner;
using namespace autopilot;
using namespace quadrotor_common;

class PlannerNode : public rclcpp::Node {
 public:
  PlannerNode();  // Constructor

 private:
  // Member variables and private functions
  std::string _vehicle_frame, _world_frame, _depth_topic;
  std::shared_ptr<tf2_ros::Buffer> to_world_buffer;
  std::shared_ptr<tf2_ros::Buffer> to_vehicle_buffer;
  std::shared_ptr<tf2_ros::TransformListener> to_world_tf2;
  std::shared_ptr<tf2_ros::TransformListener> to_vehicle_tf2;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr visual_pub;
  rclcpp::Publisher<sm::PointCloud2>::SharedPtr point_cloud_pub;
  rclcpp::Publisher<mavros_msgs::msg::PositionTarget>::SharedPtr raw_ref_pos_pub;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr vel_cmd_pub;
  rclcpp::Publisher<ground_system_msgs::msg::BenchmarkStatus>::SharedPtr benchmark_status_pub;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_throttled_pub_;  // Throttled odom for zenoh
  
  rclcpp::Subscription<sm::Image>::SharedPtr image_sub;
  rclcpp::Subscription<sm::Image>::SharedPtr visual_sub;
  rclcpp::Subscription<ground_system_msgs::msg::StartSwarmMission>::SharedPtr mission_sub;
  rclcpp::Subscription<ground_system_msgs::msg::FBVGoal>::SharedPtr fbv_goal_sub;
  rclcpp::Subscription<ground_system_msgs::msg::Takeoff>::SharedPtr takeoff_sub;
  rclcpp::Subscription<ground_system_msgs::msg::FlyTo>::SharedPtr fly_to_sub;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr reset_sub;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr mavros_odom_sub_;  // For throttling
  rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr mav_state_sub;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr mav_pose_sub;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr mav_twist_sub;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr mav_accel_sub;
  
  // MAVROS service clients (non-blocking)
  rclcpp::Client<mavros_msgs::srv::CommandBool>::SharedPtr arming_srv;
  rclcpp::Client<mavros_msgs::srv::CommandTOL>::SharedPtr takeoff_srv;
  rclcpp::Client<mavros_msgs::srv::CommandTOL>::SharedPtr land_srv;
  rclcpp::Client<mavros_msgs::srv::SetMode>::SharedPtr mode_srv;
  
  // Service call state tracking (to avoid duplicate calls and blocking)
  std::atomic<bool> mission_received_{false};
  std::atomic<bool> mode_switch_pending_{false};
  std::atomic<bool> arming_pending_{false};
  std::atomic<bool> takeoff_pending_{false};
  std::atomic<bool> land_pending_{false};
  std::atomic<bool> takeoff_requested_{false};  // Triggers GUIDED->ARM->TAKEOFF without goal
  
  rclcpp::TimerBase::SharedPtr control_loop_timer_;
  
  QuadState _state;
  double steering_value;
  bool _steered;
  std::mutex state_mutex_, trajectory_mutex_;

  // Autopilot
  ruckig::Trajectory<3> reference_trajectory_;
  bool had_reference_trajectory, _goal_set;
  PlanningStates _planner_state;
  Eigen::Vector3d targetPos_, targetVel_, targetAcc_, targetJerk_, targetSnap_, targetPos_prev_, targetVel_prev_;
  Eigen::Vector3d mavPos_, mavVel_, mavRate_;

  // State switching variables
  bool state_estimate_available_;
  rclcpp::Time time_of_switch_to_current_state_{0, 0, RCL_ROS_TIME};
  rclcpp::Time _latest_pose_stamp{0, 0, RCL_ROS_TIME};
  rclcpp::Time _latest_twist_stamp{0, 0, RCL_ROS_TIME};
  mavros_msgs::msg::State flight_controller_status;
  Eigen::Vector3d initial_start_position_;
  Eigen::Vector3d initial_land_position_;

  // Trajectory execution variables
  std::list<ruckig::Trajectory<3>> trajectory_queue_;
  rclcpp::Time _reference_trajectory_start_time{0, 0, RCL_ROS_TIME};

  // Callback functions
  void sampling_mode_callback(const std_msgs::msg::Int8::SharedPtr msg);
  void mission_callback(const ground_system_msgs::msg::StartSwarmMission::SharedPtr msg);
  void fbv_goal_callback(const ground_system_msgs::msg::FBVGoal::SharedPtr msg);
  void takeoff_callback(const ground_system_msgs::msg::Takeoff::SharedPtr msg);
  void fly_to_callback(const ground_system_msgs::msg::FlyTo::SharedPtr msg);
  void reset_callback(const std_msgs::msg::Empty::SharedPtr msg);
  void img_callback(const sm::Image::SharedPtr depth_msg);
  void visualise(const sm::Image::SharedPtr depth_msg);
  void odometry_callback(const nav_msgs::msg::Odometry::SharedPtr odom_msg);
  // Ardupilot state callbacks
  void ardupilot_status_callback(const mavros_msgs::msg::State::SharedPtr msg);
  void mav_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void mav_twist_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg);
  void mav_accel_callback(const sensor_msgs::msg::Imu::SharedPtr msg);

  // Helper functions
  cv::Mat preprocess_depth_image(const sm::Image::SharedPtr depth_msg);
  void control_loop();
  void update_reference_trajectory();
  void track_trajectory();
  void update_planner_state();
  void publish_velocity_command(const TrajectoryPoint& reference_point);
  void public_ref_pos(const TrajectoryPoint& reference_point);
  void asign_reference_trajectory(rclcpp::Time wall_time_now);
  void get_reference_point_at_time(
    const ruckig::Trajectory<3>& reference_trajectory, const double& point_time,
    TrajectoryPoint& reference_point);
  bool loadParameters();
  void set_auto_pilot_state_forced(const PlanningStates& new_state);
  pointcloud_type* create_point_cloud (const sm::Image::SharedPtr depth_msg);
  
  // Benchmark helpers
  void publish_benchmark_status(uint8_t status);

  // ===== Frontier-Led Swarming =====
  void swarm_params_callback(const ground_system_msgs::msg::SwarmParams::SharedPtr msg);
  void neighbor_odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg, int neighbor_id);
  void swarm_exploration_loop();
  void update_occupancy_grid();
  std::vector<Eigen::Vector2d> detect_frontiers();
  Eigen::Vector2d select_frontier(const std::vector<Eigen::Vector2d>& frontiers);
  Eigen::Vector3d compute_cohesion();
  Eigen::Vector3d compute_separation();
  Eigen::Vector3d compute_alignment();
  Eigen::Vector3d compute_frontier_attraction(const Eigen::Vector2d& target_frontier);
  void publish_swarm_status();
  void publish_occupancy_grid();
  Eigen::Vector2d world_to_grid(double wx, double wy) const;
  Eigen::Vector2d grid_to_world(int gx, int gy) const;
  bool is_in_grid(int gx, int gy) const;

  // Constants
  static constexpr double kPositionJumpTolerance_ = 0.5;
  RuntimeModes _runtime_mode;
  MavrosControlModes _mavros_control_mode;
  SetpointTypes _setpoint_type;
  double _trajectory_discretisation_cycle, _planning_cycle_time, _2d_z_margin, _replan_factor;
  uint8_t _spiral_sampling_step;
  bool _visualise, _3d_planning, _debug_num_trajectories, _is_spiral_sampling;
  std::vector<double> _depth_uncertainty_coeffs;
  double _depth_upper_bound, _depth_lower_bound, _checking_time_ratio, _depth_sampling_margin;
  double _go_to_goal_threshold, _goal_north_coordinate, _goal_west_coordinate, _goal_up_coordinate;
  double _flightmare_fov, _depth_scale, _real_focal_length, _real_cx, _real_cy, _decimation_factor;
  geometry_msgs::msg::Point _goal_in_world_frame, _home_in_world_frame;
  double _goal_heading;  // Heading to goal (computed once when goal is set)
  double _max_velocity_x, _max_velocity_y, _max_velocity_z;
  double _max_acceleration_x, _max_acceleration_y, _max_acceleration_z;
  double _acc_planning_threshold, _vel_planning_threshold;
  double _depth_age_threshold, _state_age_threshold, _transform_age_threshold;
  uint32_t _checked_trajectories_threshold, _sampled_trajectories_threshold;
  double _collision_probability_threshold;
  uint32_t _openmp_chunk_size;
  CollisionCheckingMethod _collision_checking_method;
  TravelingCost _traveling_cost;
  // New member variables for vehicle parameters
  double _true_vehicle_radius;
  double _planning_vehicle_radius;
  double _minimum_clear_distance;
  
  // Fence/world limits (to keep drone within safe bounds)
  double _fence_min_x, _fence_max_x;
  double _fence_min_y, _fence_max_y;
  double _fence_min_z, _fence_max_z;
  
  // Last valid setpoint (for fence breach recovery - stop at last valid position)
  Eigen::Vector3d _last_valid_position{0.0, 0.0, 0.0};
  double _last_valid_heading{0.0};
  bool _has_valid_setpoint{false};
  
  // Benchmark tracking
  int32_t _current_trial_id = 0;
  rclcpp::Time _trial_start_time{0, 0, RCL_ROS_TIME};
  bool _trial_started = false;
  
  // Odom throttle for zenoh (100Hz -> 10Hz)
  rclcpp::Time last_odom_throttle_time_{0, 0, RCL_ROS_TIME};
  static constexpr double kOdomThrottleInterval_ = 0.1;  // 10 Hz

  // ===== Frontier-Led Swarming Members =====
  bool _swarm_mode{false};
  int _drone_id{1};
  int _num_drones{3};

  // Swarm behavior weights (runtime-tunable via /swarm_params)
  double _w_cohesion{0.8};
  double _w_separation{1.5};
  double _w_alignment{0.5};
  double _w_frontier{1.2};
  double _w_obstacle{2.0};
  double _separation_radius{3.0};
  double _neighbor_radius{10.0};
  double _max_swarm_speed{1.5};
  double _swarm_altitude{1.5};

  // Occupancy grid
  double _grid_cell_size{0.5};
  double _grid_width{60.0};
  double _grid_height{60.0};
  int _grid_cols{0};
  int _grid_rows{0};
  double _grid_origin_x{0.0};
  double _grid_origin_y{0.0};
  std::vector<uint8_t> _occupancy_grid;  // 0=unknown, 1=free, 2=occupied
  std::mutex _grid_mutex;
  uint32_t _cells_explored{0};

  // Neighbor state tracking
  struct NeighborState {
    Eigen::Vector3d position{0, 0, 0};
    Eigen::Vector3d velocity{0, 0, 0};
    double yaw{0.0};
    rclcpp::Time last_update{0, 0, RCL_ROS_TIME};
    bool valid{false};
  };
  std::map<int, NeighborState> _neighbor_states;
  std::mutex _neighbor_mutex;

  // Frontier tracking
  Eigen::Vector2d _assigned_frontier{0, 0};
  bool _has_frontier{false};
  uint32_t _frontiers_remaining{0};

  // Swarm pub/sub
  rclcpp::Subscription<ground_system_msgs::msg::SwarmParams>::SharedPtr swarm_params_sub;
  std::vector<rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr> neighbor_odom_subs;
  rclcpp::Publisher<ground_system_msgs::msg::SwarmExplorationStatus>::SharedPtr swarm_status_pub;
  rclcpp::Publisher<ground_system_msgs::msg::OccupancyGrid2D>::SharedPtr occupancy_grid_pub;
  rclcpp::TimerBase::SharedPtr swarm_exploration_timer_;
  rclcpp::TimerBase::SharedPtr swarm_status_timer_;
  rclcpp::TimerBase::SharedPtr occupancy_pub_timer_;
};

#endif  // PLANNER_NODE_HPP
