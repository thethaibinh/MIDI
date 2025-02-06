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

// ROS base
#include <ros/console.h>
#include "ros/ros.h"

// autopilot
#include "autopilot_states.h"

// math
#include "common.hpp"

// msg
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/AccelWithCovarianceStamped.h>
#include "geometry_msgs/Vector3.h"
#include "nav_msgs/Odometry.h"
#include "sensor_msgs/Image.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Int8.h"
#include "std_msgs/String.h"
#include "sensor_msgs/Imu.h"

// mavros msg
#include <mavros_msgs/AttitudeTarget.h>
#include <mavros_msgs/PositionTarget.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/CommandTOL.h>
#include <mavros_msgs/SetMode.h>

// CV
#include <cv_bridge/cv_bridge.h>

#include <sstream>

// dodgelib
#include "dodgelib/math/types.hpp"
#include "dodgeros_msgs/QuadState.h"
#include "dodgeros/ros_eigen.hpp"

#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>

// Eigen catkin
#include <Eigen/Dense>

// quadrotor message
#include <quadrotor_msgs/ControlCommand.h>
#include <dodgeros_msgs/Command.h>

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
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

// this is a global definition of the points to be used
// changes to omit color would need adaptations in
// the visualization too
#include <pcl/io/io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/Marker.h>

#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
namespace sm = sensor_msgs;
typedef pcl::PointXYZ point_type;
typedef pcl::PointCloud<point_type> pointcloud_type;

using namespace common_math;
using namespace depth_uncertainty_planner;
using namespace autopilot;
using namespace quadrotor_common;

class PlannerNode {
 public:
  PlannerNode();  // Constructor

 private:
  // Member variables and private functions
  std::string _vehicle_frame, _world_frame, _depth_topic, _scene_flow_topic;
  tf2_ros::Buffer to_world_buffer, to_vehicle_buffer;
  tf2_ros::TransformListener to_world_tf2, to_vehicle_tf2;
  ros::NodeHandle n_;
  ros::NodeHandle pnh_;
  ros::Publisher trajectoty_pub, point_cloud_pub, visual_pub,
    control_command_pub_, raw_ref_pos_pub, att_ctrl_pub;
  ros::Subscriber reset_sub, start_sub, image_sub, scene_flow_sub, state_sub, visual_sub,
    odom_sub, mav_state_sub, mav_pose_sub, mav_twist_sub, mav_accel_sub;
  ros::ServiceClient takeoff_srv, land_srv, mode_srv;
  ros::Timer statusloop_timer_, control_loop_timer_;
  dodgeros_msgs::QuadState _state;
  agi::QuadState _agi_state;
  double steering_value;
  bool _steered;
  // int8_t sampling_mode;
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
  ros::Time time_of_switch_to_current_state_, _latest_pose_stamp,
    _latest_twist_stamp, _latest_accel_stamp;
  mavros_msgs::State flight_controller_status;
  Eigen::Vector3d initial_start_position_;
  Eigen::Vector3d initial_land_position_;

  // Trajectory execution variables
  std::list<ruckig::Trajectory<3>> trajectory_queue_;
  ros::Time _reference_trajectory_start_time;

  // Callback functions
  void sampling_mode_callback(const std_msgs::Int8::ConstPtr& msg);
  void start_callback(const std_msgs::Empty::ConstPtr& msg);
  void reset_callback(const std_msgs::Empty::ConstPtr& msg);
  void state_callback(const dodgeros_msgs::QuadState& state);
  void plan(const sensor_msgs::ImageConstPtr& depth_msg);
  void visualise(const sensor_msgs::ImageConstPtr& depth_msg);
  void odometry_callback(const nav_msgs::OdometryConstPtr& odom_msg);
  // Ardupilot state callbacks
  void ardupilot_status_callback(const mavros_msgs::State::ConstPtr& msg);
  void mav_pose_callback(const geometry_msgs::PoseStamped &msg);
  void mav_twist_callback(const geometry_msgs::TwistStamped &msg);
  void mav_accel_callback(const sensor_msgs::Imu &msg);

  // position controller functions
  QuadStateEstimate quad_common_state_from_dodgedrone_state(
    const dodgeros_msgs::QuadState& _state);
  Eigen::Vector3d array3d_to_eigen3d(const std::array<double, 3>& arr);
  cv::Mat preprocess_depth_image(const sm::ImageConstPtr& depth_msg);
  cv::Mat preprocess_scene_flow_image(const sm::ImageConstPtr& scene_flow_msg);
  void control_loop(const ros::TimerEvent &event);
  void update_reference_trajectory();
  void track_trajectory();
  void update_planner_state();
  void publish_control_command(const ControlCommand& control_cmd);
  void public_ref_att(const ControlCommand& control_cmd);
  void public_ref_pos(const TrajectoryPoint& reference_point);
  void asign_reference_trajectory(ros::Time wall_time_now);
  bool check_valid_trajectory(const geometry_msgs::Point& current_position, const ruckig::Trajectory<3>& trajectory);
  void get_reference_point_at_time(
    const ruckig::Trajectory<3>& reference_trajectory, const double& point_time,
    TrajectoryPoint& reference_point);
  bool loadParameters();
  void set_auto_pilot_state_forced(const PlanningStates& new_state);
  pointcloud_type* create_point_cloud (const sm::ImageConstPtr& depth_msg);

  // Constants
  static constexpr double kPositionJumpTolerance_ = 0.5;
  RuntimeModes _runtime_mode;
  MavrosControlModes _mavros_control_mode;
  double _trajectory_discretisation_cycle, _planning_cycle_time, _2d_z_margin, _replan_factor, _replanning_interval;
  bool _is_periodic_replanning;
  uint8_t _spiral_sampling_step;
  bool _visualise, _3d_planning, _debug_num_trajectories, _is_spiral_sampling, _is_dynamic_planning;
  std::vector<double> _depth_uncertainty_coeffs, _dynamic_pos_cov_coeffs;
  double _depth_upper_bound, _depth_lower_bound, _checking_time_ratio, _depth_sampling_margin;
  double _go_to_goal_threshold, _goal_north_coordinate, _goal_west_coordinate, _goal_up_coordinate;
  double _flightmare_fov, _depth_scale, _real_focal_length, _real_cx, _real_cy, _decimation_factor;
  geometry_msgs::Point _goal_in_world_frame, _stop_planning_point_in_world_frame;
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
