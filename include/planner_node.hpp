#ifndef PLANNER_NODE_HPP
#define PLANNER_NODE_HPP

#pragma once
#include <unistd.h>
#include <cstdlib>

#include <boost/program_options.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <atomic>
#include <optional>
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
#include <ground_system_msgs/msg/swarm_mission_upload.hpp>
#include <ground_system_msgs/msg/swarm_mission_ack.hpp>
#include <ground_system_msgs/msg/takeoff.hpp>

// OPUS coordination (services + status topic)
#include <ground_system_msgs/msg/opus_plan_lock_request.hpp>
#include <ground_system_msgs/msg/opus_trajectory_submit.hpp>
#include <ground_system_msgs/msg/opus_trajectory_ack.hpp>
#include <ground_system_msgs/msg/opus_plan_abort.hpp>
#include <ground_system_msgs/msg/opus_status.hpp>
#include <ground_system_msgs/msg/opus_trajectory.hpp>
#include <ground_system_msgs/msg/opus_phase_segment.hpp>
#include <ground_system_msgs/msg/waypoint.hpp>

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
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr initial_position_pub_;
  bool initial_position_published_ = false;
  
  rclcpp::Subscription<sm::Image>::SharedPtr image_sub;
  rclcpp::Subscription<sm::Image>::SharedPtr visual_sub;
  rclcpp::Subscription<ground_system_msgs::msg::StartSwarmMission>::SharedPtr mission_sub;
  rclcpp::Subscription<ground_system_msgs::msg::SwarmMissionUpload>::SharedPtr mission_upload_sub;
  rclcpp::Publisher<ground_system_msgs::msg::SwarmMissionAck>::SharedPtr mission_ack_pub_;
  rclcpp::Subscription<ground_system_msgs::msg::Takeoff>::SharedPtr takeoff_sub;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr reset_sub;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr reinitialise_sub;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr brake_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr land_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
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
  std::atomic<bool> mission_uploaded_{false};  // true after waypoints uploaded, before start
  std::atomic<bool> mode_switch_pending_{false};
  std::atomic<bool> arming_pending_{false};
  std::atomic<bool> takeoff_pending_{false};
  std::atomic<bool> land_pending_{false};
  // Timestamps at which the corresponding request was dispatched. Used to
  // implement a retry cooldown so we don't spam the FC while /mavros/state
  // catches up. Pending flag is cleared by observed-state transition or by
  // timeout (whichever comes first). Protected by service-response thread
  // writes + update_planner_state reads; single-writer per flag, so atomic
  // time_point is unnecessary — the pending atomic bool gates access.
  std::chrono::steady_clock::time_point mode_switch_sent_time_{};
  std::chrono::steady_clock::time_point arming_sent_time_{};
  std::chrono::steady_clock::time_point takeoff_sent_time_{};
  std::chrono::steady_clock::time_point land_sent_time_{};
  std::atomic<bool> takeoff_requested_{false};  // Triggers GUIDED->ARM->TAKEOFF without goal
  bool _reinitialise_requested{false};  // True after reinitialise_callback, cleared on auto-reset
  std::atomic<bool> brake_mode_switch_sent_{false};  // MAVROS: BRAKE mode switch sent once
  
  rclcpp::TimerBase::SharedPtr control_loop_timer_;
  
  QuadState _state;
  double steering_value;
  bool _steered;
  std::mutex state_mutex_, trajectory_mutex_;
  std::mutex fc_status_mutex_;  // Protects flight_controller_status (written on state_callback_group_, read on control_callback_group_)

  // Autopilot
  ruckig::Trajectory<3> reference_trajectory_;
  bool had_reference_trajectory, _goal_set;
  std::atomic<PlanningStates> _planner_state;
  Eigen::Vector3d targetPos_, targetVel_, targetAcc_, targetJerk_, targetSnap_, targetPos_prev_, targetVel_prev_;
  Eigen::Vector3d mavPos_, mavVel_, mavRate_;

  // State switching variables
  bool state_estimate_available_;
  rclcpp::Time time_of_switch_to_current_state_{0, 0, RCL_ROS_TIME};
  rclcpp::Time _latest_pose_stamp{0, 0, RCL_ROS_TIME};
  rclcpp::Time _latest_twist_stamp{0, 0, RCL_ROS_TIME};
  mavros_msgs::msg::State flight_controller_status;  // Protected by fc_status_mutex_
  Eigen::Vector3d initial_start_position_;
  Eigen::Vector3d initial_land_position_;

  // Trajectory execution variables
  std::list<ruckig::Trajectory<3>> trajectory_queue_;
  rclcpp::Time _reference_trajectory_start_time{0, 0, RCL_ROS_TIME};
  // When set, the next trajectory installed by update_reference_trajectory()
  // uses this wall-clock time as its t=0 anchor instead of "now". Used to
  // compensate for the OPUS grant/ack round-trip so the first reference
  // sampled from a just-accepted trajectory skips the latency gap rather
  // than replaying a stale t=0. Protected by trajectory_mutex_.
  std::optional<rclcpp::Time> pending_trajectory_start_time_override_;

  // Callback functions
  void sampling_mode_callback(const std_msgs::msg::Int8::SharedPtr msg);
  void mission_callback(const ground_system_msgs::msg::StartSwarmMission::SharedPtr msg);
  void mission_upload_callback(const ground_system_msgs::msg::SwarmMissionUpload::SharedPtr msg);
  void takeoff_callback(const ground_system_msgs::msg::Takeoff::SharedPtr msg);
  void reset_planner();
  void brake_callback(const std_msgs::msg::Empty::SharedPtr msg);
  void land_swarm_callback(const std_msgs::msg::Empty::SharedPtr msg);
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
  void public_ref_pos(const TrajectoryPoint& reference_point);
  void get_reference_point_at_time(
    const ruckig::Trajectory<3>& reference_trajectory, const double& point_time,
    TrajectoryPoint& reference_point);
  bool loadParameters();
  void set_auto_pilot_state_forced(const PlanningStates& new_state);
  pointcloud_type* create_point_cloud(const sm::Image::SharedPtr depth_msg);

  // OPUS coordination helpers
  using OpusPlanLockReqMsg = ground_system_msgs::msg::OpusPlanLockRequest;
  void opus_send_lock_request();
  void opus_status_callback(const ground_system_msgs::msg::OpusStatus::SharedPtr msg);
  void opus_abort_planning(const std::string& reason);
  bool opus_should_abort_replanning(double* elapsed_sec = nullptr);
  void opus_submit_trajectory(const ruckig::Trajectory<3>& traj,
                              const geometry_msgs::msg::TransformStamped& body_to_world,
                              const geometry_msgs::msg::Point& world_position);
  void opus_trajectory_ack_callback(
      const ground_system_msgs::msg::OpusTrajectoryAck::SharedPtr msg);
  // Transform vector from camera frame (RDF) to world frame (ENU)
  // For positions: applies rotation + translation (current world position)
  // For velocity/accel: applies rotation only
  Eigen::Vector3d transform_camera_to_world(
      const Eigen::Vector3d& camera_vec,
      const geometry_msgs::msg::TransformStamped& body_to_world,
      bool is_position,
      const geometry_msgs::msg::Point& world_position);

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
  double _go_to_goal_threshold, _goal_up_coordinate;
  double _flightmare_fov, _depth_scale, _real_focal_length, _real_cx, _real_cy, _decimation_factor;
  geometry_msgs::msg::Point _goal_in_world_frame, _home_in_world_frame;
  double _goal_heading;  // Heading to goal (computed once when goal is set)
  double _initial_heading = 0.0;  // Heading at mission upload (for reinitialise)
  double _trajectory_heading = 0.0;  // Heading toward trajectory terminal (updated per trajectory)

  // Waypoint mission tracking
  std::vector<ground_system_msgs::msg::Waypoint> _waypoint_list;  // Original FLU waypoints (for logging)
  std::vector<Eigen::Vector3d> _world_waypoints;  // Pre-computed world-frame positions (reused across loops)
  std::vector<double> _waypoint_headings;   // Pre-computed world headings per waypoint
  std::vector<double> _waypoint_hold_times; // Hold time per waypoint
  size_t _current_waypoint_index = 0;
  uint32_t _remaining_loops = 0;
  bool _waypoint_mission_active = false;
  std::string _mission_name;
  // Advance to next waypoint; returns false when mission complete
  bool advance_waypoint();
  // Set _goal_in_world_frame from pre-computed _world_waypoints at _current_waypoint_index
  void set_goal_from_waypoint();
  // Convert FLU waypoints to world frame using agent's initial yaw, store in _world_waypoints
  void convert_waypoints_to_world(double initial_yaw);
  // Compute heading from current position toward _goal_in_world_frame
  double compute_heading_to_goal() const;
  // Re-initialise callback: return to initial position (only from FINISHED state)
  void reinitialise_callback(const std_msgs::msg::Empty::SharedPtr msg);
  // Log received mission to YAML file for history/replay
  void log_mission_to_yaml(const ground_system_msgs::msg::SwarmMissionUpload::SharedPtr& msg);
  // Heading alignment threshold (radians, ~10 degrees)
  static constexpr double kHeadingAlignThreshold_ = 0.07;
  // Maximum yaw rate for heading alignment (rad/s)
  static constexpr double kHeadingAlignYawRate_ = 0.5;

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

  // Dedicated callback groups so the heavy img_callback (default group) cannot
  // block odometry/pose/twist updates or the control-loop timer. Without this
  // split, long planning iterations freeze _state.t and the TF broadcast,
  // causing state_age / transform_age checks in img_callback to trip.
  rclcpp::CallbackGroup::SharedPtr state_callback_group_;    // MX: odom/pose/twist/mav_state
  rclcpp::CallbackGroup::SharedPtr control_callback_group_;  // MX: control_loop_timer_

  // OPUS coordination (lock-request service + trajectory-check service
  //                    + status subscription + abort topic)
  rclcpp::CallbackGroup::SharedPtr opus_callback_group_;  // Reentrant group for OPUS clients
  rclcpp::Publisher<OpusPlanLockReqMsg>::SharedPtr opus_lock_req_pub_;
  rclcpp::Publisher<ground_system_msgs::msg::OpusTrajectorySubmit>::SharedPtr opus_traj_submit_pub_;
  rclcpp::Subscription<ground_system_msgs::msg::OpusTrajectoryAck>::SharedPtr opus_traj_ack_sub_;
  rclcpp::Publisher<ground_system_msgs::msg::OpusPlanAbort>::SharedPtr opus_plan_abort_pub_;
  rclcpp::Subscription<ground_system_msgs::msg::OpusStatus>::SharedPtr opus_status_sub_;
  bool opus_enabled_ = false;
  bool opus_granted_ = false;          // Lock granted by coordinator (from /opus/status)
  bool opus_check_pending_ = false;    // Waiting for TrajectoryCheck service response
  bool opus_lock_pending_ = false;     // Lock request sent, waiting for grant via status
  uint8_t opus_drone_id_ = 0;
  uint32_t opus_plan_sequence_ = 0;    // Monotonic counter for correlating lock/check/abort
  ruckig::Trajectory<3> opus_pending_trajectory_;  // Trajectory awaiting check response
  // Wall-clock time of the most recent /opus/trajectory_submit publish. On
  // ack, this is forwarded to pending_trajectory_start_time_override_ so the
  // accepted trajectory's t=0 aligns with the moment of submission (before
  // the grant/check round-trip), not the moment of ack reception.
  rclcpp::Time opus_submission_time_{0, 0, RCL_ROS_TIME};
  std::mutex opus_mutex_;
  double opus_local_replan_timeout_ = kOpusLocalReplanTimeout_;

  // OPUS pre-queue: the latest locally planned trajectory waiting for OPUS submission.
  // img_callback always plans locally and overwrites this. At replan time,
  // update_reference_trajectory sets opus_submission_needed_ and the next
  // img_callback picks this entry and submits to the GCS (collision check is
  // done by the coordinator).
  struct OpusPreQueueEntry {
    ruckig::Trajectory<3> trajectory;
    geometry_msgs::msg::TransformStamped body_to_world;
    geometry_msgs::msg::Point world_position;
  };
  std::optional<OpusPreQueueEntry> opus_pre_queue_;
  std::atomic<bool> opus_submission_needed_{false};  // Set at replan trigger, cleared on check response. Atomic: read in img_callback without opus_mutex_.

  // OPUS timeout tracking (monotonic clock)
  std::chrono::steady_clock::time_point opus_grant_time_{};
  static constexpr double kOpusLocalReplanTimeout_ = 30.0;  // seconds before aborting local replanning
  static constexpr double kOpusAckTimeout_ = 0.02;  // seconds before treating a missing ack as lost
  double opus_ack_timeout_ = kOpusAckTimeout_;
};

#endif  // PLANNER_NODE_HPP
