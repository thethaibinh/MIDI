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
#include "common.hpp"

// ROS2 msg
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/int8.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/empty.hpp>
#include <sensor_msgs/msg/imu.hpp>

// mavros msg (ROS2)
#include <mavros_msgs/msg/attitude_target.hpp>
#include <mavros_msgs/msg/position_target.hpp>
#include <mavros_msgs/msg/state.hpp>
#include <mavros_msgs/srv/command_tol.hpp>
#include <mavros_msgs/srv/set_mode.hpp>

// CV
#include <cv_bridge/cv_bridge.h>

#include <sstream>

// dodgelib
#include "dodgelib/math/types.hpp"
#include "dodgeros_msgs/msg/quad_state.hpp"
#include "dodgeros/ros_eigen.hpp"

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

// Eigen
#include <Eigen/Dense>

// quadrotor message (ROS2)
#include <quadrotor_msgs/msg/control_command.hpp>
#include <dodgeros_msgs/msg/command.hpp>

// RPG quad common and control
#include <position_controller/position_controller.h>
#include <position_controller/position_controller_params.h>
#include <quadrotor_common/control_command.h>
#include <quadrotor_common/quad_state_estimate.h>
#include <quadrotor_common/trajectory.h>
#include <quadrotor_common/trajectory_point.h>
#include <quadrotor_common/geometry_eigen_conversions.h>
#include <quadrotor_common/math_common.h>

// Ruckig
#include <ruckig/ruckig.hpp>

// ROS TF2
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

// PCL
#include <pcl/common/io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/msg/marker.hpp>

#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
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
  
  /// @brief Initialize the node - must be called after construction
  /// This is required because some initialization requires shared_from_this()
  /// which cannot be called in the constructor
  bool init();

 private:
  // Member variables and private functions
  std::string _vehicle_frame, _world_frame, _depth_topic;
  std::shared_ptr<tf2_ros::Buffer> to_world_buffer_, to_vehicle_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> to_world_tf2_, to_vehicle_tf2_;
  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr trajectoty_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr visual_pub_;
  rclcpp::Publisher<dodgeros_msgs::msg::Command>::SharedPtr control_command_pub_;
  rclcpp::Publisher<mavros_msgs::msg::PositionTarget>::SharedPtr raw_ref_pos_pub_;
  rclcpp::Publisher<mavros_msgs::msg::AttitudeTarget>::SharedPtr att_ctrl_pub_;
  
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr reset_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr start_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<dodgeros_msgs::msg::QuadState>::SharedPtr state_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr visual_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr mav_state_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr mav_pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr mav_twist_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr mav_accel_sub_;
  
  rclcpp::Client<mavros_msgs::srv::CommandTOL>::SharedPtr takeoff_srv_;
  rclcpp::Client<mavros_msgs::srv::CommandTOL>::SharedPtr land_srv_;
  rclcpp::Client<mavros_msgs::srv::SetMode>::SharedPtr mode_srv_;
  
  rclcpp::TimerBase::SharedPtr statusloop_timer_;
  rclcpp::TimerBase::SharedPtr control_loop_timer_;
  
  dodgeros_msgs::msg::QuadState _state;
  agi::QuadState _agi_state;
  double steering_value;
  bool _steered;
  std::mutex state_mutex_, trajectory_mutex_;

  // Autopilot
  ruckig::Trajectory<3> reference_trajectory_;
  position_controller::PositionController base_controller_;
  position_controller::PositionControllerParams base_controller_params_;
  bool had_reference_trajectory, _goal_set;
  PlanningStates _planner_state;
  Eigen::Vector3d targetPos_, targetVel_, targetAcc_, targetJerk_, targetSnap_, targetPos_prev_, targetVel_prev_;
  Eigen::Vector3d mavPos_, mavVel_, mavRate_;

  // State switching variables
  bool state_estimate_available_;
  rclcpp::Time time_of_switch_to_current_state_, _latest_pose_stamp,
    _latest_twist_stamp, _latest_accel_stamp;
  mavros_msgs::msg::State flight_controller_status;
  Eigen::Vector3d initial_start_position_;
  Eigen::Vector3d initial_land_position_;

  // Trajectory execution variables
  std::list<ruckig::Trajectory<3>> trajectory_queue_;
  rclcpp::Time _reference_trajectory_start_time;

  // Callback functions
  void sampling_mode_callback(const std_msgs::msg::Int8::SharedPtr msg);
  void start_callback(const std_msgs::msg::Empty::SharedPtr msg);
  void reset_callback(const std_msgs::msg::Empty::SharedPtr msg);
  void state_callback(const dodgeros_msgs::msg::QuadState::SharedPtr state);
  void img_callback(const sensor_msgs::msg::Image::SharedPtr depth_msg);
  void visualise(const sensor_msgs::msg::Image::SharedPtr depth_msg);
  void odometry_callback(const nav_msgs::msg::Odometry::SharedPtr odom_msg);
  // Ardupilot state callbacks
  void ardupilot_status_callback(const mavros_msgs::msg::State::SharedPtr msg);
  void mav_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void mav_twist_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg);
  void mav_accel_callback(const sensor_msgs::msg::Imu::SharedPtr msg);

  // position controller functions
  QuadStateEstimate quad_common_state_from_dodgedrone_state(
    const dodgeros_msgs::msg::QuadState& _state);
  Eigen::Vector3d array3d_to_eigen3d(const std::array<double, 3>& arr);
  cv::Mat preprocess_depth_image(const sensor_msgs::msg::Image::SharedPtr depth_msg);
  void control_loop();
  void update_reference_trajectory();
  void track_trajectory();
  void update_planner_state();
  void publish_control_command(const ControlCommand& control_cmd);
  void public_ref_att(const ControlCommand& control_cmd);
  void public_ref_pos(const TrajectoryPoint& reference_point);
  void asign_reference_trajectory(rclcpp::Time wall_time_now);
  bool check_valid_trajectory(const geometry_msgs::msg::Point& current_position, const ruckig::Trajectory<3>& trajectory);
  void get_reference_point_at_time(
    const ruckig::Trajectory<3>& reference_trajectory, const double& point_time,
    TrajectoryPoint& reference_point);
  bool loadParameters();
  void set_auto_pilot_state_forced(const PlanningStates& new_state);
  pointcloud_type* create_point_cloud(const sensor_msgs::msg::Image::SharedPtr depth_msg);

  // Constants
  static constexpr double kPositionJumpTolerance_ = 0.5;
  RuntimeModes _runtime_mode;
  MavrosControlModes _mavros_control_mode;
  double _trajectory_discretisation_cycle, _planning_cycle_time, _2d_z_margin, _replan_factor;
  uint8_t _spiral_sampling_step;
  bool _visualise, _3d_planning, _debug_num_trajectories, _is_spiral_sampling;
  std::vector<double> _depth_uncertainty_coeffs;
  double _depth_upper_bound, _depth_lower_bound, _checking_time_ratio, _depth_sampling_margin;
  double _go_to_goal_threshold, _goal_north_coordinate, _goal_west_coordinate, _goal_up_coordinate;
  double _flightmare_fov, _depth_scale, _real_focal_length, _real_cx, _real_cy, _decimation_factor;
  geometry_msgs::msg::Point _goal_in_world_frame, _stop_planning_point_in_world_frame;
  double _max_velocity_x, _max_velocity_y, _max_velocity_z;
  double _max_acceleration_x, _max_acceleration_y, _max_acceleration_z;
  double _acc_planning_threshold;
  uint32_t _checked_trajectories_threshold, _sampled_trajectories_threshold;
  double _collision_probability_threshold;
  uint32_t _openmp_chunk_size;
  CollisionCheckingMethod _collision_checking_method;
  TravelingCost _traveling_cost;
  // New member variables for vehicle parameters
  double _true_vehicle_radius;
  double _planning_vehicle_radius;
  double _minimum_clear_distance;
};

#endif  // PLANNER_NODE_HPP
