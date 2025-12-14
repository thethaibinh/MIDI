// quad_state.hpp
/*!
 * Copyright 2024 by Binh Nguyen <thethaibinh@gmail.com>
 *
 * This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
 * For commercial use, please contact the author for licensing terms.
 */
#ifndef QUAD_STATE_HPP
#define QUAD_STATE_HPP

#include <Eigen/Dense>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/quaternion.hpp>

namespace quad_state {

// Quaternion type alias for convenience
using Quaternion = Eigen::Quaterniond;

/**
 * @brief QuadState represents the full state of a quadrotor
 * 
 * This replaces the agi::QuadState from dodgelib and dodgeros_msgs::msg::QuadState.
 * Position is in world frame, velocity can be in world or body frame depending on context.
 */
struct QuadState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double t{0.0};           // Timestamp in seconds
  Eigen::Vector3d p{Eigen::Vector3d::Zero()};  // Position in world frame (NWU or ENU)
  Quaternion q{Quaternion::Identity()};         // Orientation (world to body)
  Eigen::Vector3d v{Eigen::Vector3d::Zero()};  // Linear velocity
  Eigen::Vector3d w{Eigen::Vector3d::Zero()};  // Angular velocity (body frame)
  Eigen::Vector3d a{Eigen::Vector3d::Zero()};  // Linear acceleration (body frame)

  QuadState() = default;

  void setZero() {
    t = 0.0;
    p.setZero();
    q.setIdentity();
    v.setZero();
    w.setZero();
    a.setZero();
  }

  // Get position components
  double x() const { return p.x(); }
  double y() const { return p.y(); }
  double z() const { return p.z(); }

  // Get velocity magnitude
  double speed() const { return v.norm(); }

  // Get body-frame forward direction in world frame
  Eigen::Vector3d forward() const {
    return q * Eigen::Vector3d::UnitX();
  }
};

/**
 * @brief ROS-compatible state for callbacks and internal storage
 * This replaces dodgeros_msgs::msg::QuadState for internal use
 */
struct RosQuadState {
  double t{0.0};
  geometry_msgs::msg::Pose pose;
  geometry_msgs::msg::Twist velocity;
  geometry_msgs::msg::Twist acceleration;

  RosQuadState() {
    pose.orientation.w = 1.0;  // Identity quaternion
  }
};

/**
 * @brief State estimate used by controllers
 * Replaces quadrotor_common::QuadStateEstimate
 */
struct QuadStateEstimate {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double timestamp{0.0};
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
  Quaternion orientation{Quaternion::Identity()};
  Eigen::Vector3d bodyrates{Eigen::Vector3d::Zero()};

  QuadStateEstimate() = default;
};

// ============================================================================
// Conversion functions (replace agi:: conversions)
// ============================================================================

/**
 * @brief Convert geometry_msgs Point/Vector3 to Eigen::Vector3d
 */
inline Eigen::Vector3d fromRosVec3(const geometry_msgs::msg::Point& p) {
  return Eigen::Vector3d(p.x, p.y, p.z);
}

inline Eigen::Vector3d fromRosVec3(const geometry_msgs::msg::Vector3& v) {
  return Eigen::Vector3d(v.x, v.y, v.z);
}

/**
 * @brief Convert Eigen::Vector3d to geometry_msgs Point
 */
inline geometry_msgs::msg::Point toRosPoint(const Eigen::Vector3d& v) {
  geometry_msgs::msg::Point p;
  p.x = v.x();
  p.y = v.y();
  p.z = v.z();
  return p;
}

/**
 * @brief Convert Eigen::Vector3d to geometry_msgs Vector3
 */
inline geometry_msgs::msg::Vector3 toRosVec3(const Eigen::Vector3d& v) {
  geometry_msgs::msg::Vector3 vec;
  vec.x = v.x();
  vec.y = v.y();
  vec.z = v.z();
  return vec;
}

/**
 * @brief Convert QuadState to RosQuadState
 */
inline RosQuadState toRosQuadState(const QuadState& state) {
  RosQuadState ros_state;
  ros_state.t = state.t;
  ros_state.pose.position = toRosPoint(state.p);
  ros_state.pose.orientation.w = state.q.w();
  ros_state.pose.orientation.x = state.q.x();
  ros_state.pose.orientation.y = state.q.y();
  ros_state.pose.orientation.z = state.q.z();
  ros_state.velocity.linear = toRosVec3(state.v);
  ros_state.velocity.angular = toRosVec3(state.w);
  ros_state.acceleration.linear = toRosVec3(state.a);
  return ros_state;
}

/**
 * @brief Convert RosQuadState to QuadState
 */
inline QuadState fromRosQuadState(const RosQuadState& ros_state) {
  QuadState state;
  state.t = ros_state.t;
  state.p = fromRosVec3(ros_state.pose.position);
  state.q = Quaternion(ros_state.pose.orientation.w,
                       ros_state.pose.orientation.x,
                       ros_state.pose.orientation.y,
                       ros_state.pose.orientation.z);
  state.v = fromRosVec3(ros_state.velocity.linear);
  state.w = fromRosVec3(ros_state.velocity.angular);
  state.a = fromRosVec3(ros_state.acceleration.linear);
  return state;
}

/**
 * @brief Convert RosQuadState to QuadStateEstimate (for controllers)
 */
inline QuadStateEstimate toQuadStateEstimate(const RosQuadState& ros_state) {
  QuadStateEstimate est;
  est.timestamp = ros_state.t;
  est.position = fromRosVec3(ros_state.pose.position);
  est.velocity = fromRosVec3(ros_state.velocity.linear);
  est.orientation = Quaternion(ros_state.pose.orientation.w,
                                ros_state.pose.orientation.x,
                                ros_state.pose.orientation.y,
                                ros_state.pose.orientation.z);
  est.bodyrates = fromRosVec3(ros_state.velocity.angular);
  return est;
}

} // namespace quad_state

#endif // QUAD_STATE_HPP
