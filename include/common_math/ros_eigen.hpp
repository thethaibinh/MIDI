#ifndef DODGEROS_ROS_EIGEN_HPP
#define DODGEROS_ROS_EIGEN_HPP

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/accel.hpp>
#include <Eigen/Dense>
#include <std_msgs/msg/header.hpp>

namespace quadrotor_common {

// Simple state structure to replace dodgeros_msgs::QuadState
struct QuadState {
  double t;
  geometry_msgs::msg::Pose pose;
  geometry_msgs::msg::Twist velocity;
  geometry_msgs::msg::Accel acceleration;
  
  QuadState() : t(0.0) {}
};

inline geometry_msgs::msg::Point toRosPoint(const Eigen::Vector3d& v) {
  geometry_msgs::msg::Point p;
  p.x = v.x();
  p.y = v.y();
  p.z = v.z();
  return p;
}

inline geometry_msgs::msg::Vector3 toRosVec3(const Eigen::Vector3d& v) {
  geometry_msgs::msg::Vector3 vec;
  vec.x = v.x();
  vec.y = v.y();
  vec.z = v.z();
  return vec;
}

inline geometry_msgs::msg::Quaternion toRosQuaternion(const Eigen::Quaterniond& q) {
  geometry_msgs::msg::Quaternion quat;
  quat.w = q.w();
  quat.x = q.x();
  quat.y = q.y();
  quat.z = q.z();
  return quat;
}

inline Eigen::Vector3d toEigen(const geometry_msgs::msg::Point& p) {
  return Eigen::Vector3d(p.x, p.y, p.z);
}

inline Eigen::Vector3d toEigen(const geometry_msgs::msg::Vector3& v) {
  return Eigen::Vector3d(v.x, v.y, v.z);
}

} // namespace quadrotor_common

#endif // DODGEROS_ROS_EIGEN_HPP

