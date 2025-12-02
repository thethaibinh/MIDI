#ifndef QUADROTOR_COMMON_GEOMETRY_EIGEN_CONVERSIONS_H
#define QUADROTOR_COMMON_GEOMETRY_EIGEN_CONVERSIONS_H

#include <Eigen/Dense>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/quaternion.hpp>

namespace quadrotor_common {

inline Eigen::Vector3d geometryToEigen(const geometry_msgs::msg::Point& p) {
  return Eigen::Vector3d(p.x, p.y, p.z);
}

inline Eigen::Vector3d geometryToEigen(const geometry_msgs::msg::Vector3& v) {
  return Eigen::Vector3d(v.x, v.y, v.z);
}

inline Eigen::Quaterniond geometryToEigen(const geometry_msgs::msg::Quaternion& q) {
  return Eigen::Quaterniond(q.w, q.x, q.y, q.z);
}

inline geometry_msgs::msg::Point eigenToGeometryPoint(const Eigen::Vector3d& v) {
  geometry_msgs::msg::Point p;
  p.x = v.x();
  p.y = v.y();
  p.z = v.z();
  return p;
}

inline geometry_msgs::msg::Vector3 eigenToGeometryVec3(const Eigen::Vector3d& v) {
  geometry_msgs::msg::Vector3 vec;
  vec.x = v.x();
  vec.y = v.y();
  vec.z = v.z();
  return vec;
}

// Keep one generic version for backward compatibility (returns Point)
inline geometry_msgs::msg::Point eigenToGeometry(const Eigen::Vector3d& v) {
  return eigenToGeometryPoint(v);
}

inline geometry_msgs::msg::Quaternion eigenToGeometry(const Eigen::Quaterniond& q) {
  geometry_msgs::msg::Quaternion quat;
  quat.w = q.w();
  quat.x = q.x();
  quat.y = q.y();
  quat.z = q.z();
  return quat;
}

} // namespace quadrotor_common

#endif // QUADROTOR_COMMON_GEOMETRY_EIGEN_CONVERSIONS_H

