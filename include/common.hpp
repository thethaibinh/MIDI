#ifndef COMMON_H
#define COMMON_H

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Vector3.h>
#include <Eigen/Dense>

#define ANTI_G_ENU 9.80665


static Eigen::Vector3d body_rdf_from_flu_eigen(const Eigen::Vector3d &m) {
  Eigen::Vector3d v;
  v << -m(1), -m(2), m(0);
  return v;
}

static Eigen::Vector3d body_rdf_from_flu_eigen(const geometry_msgs::Vector3 &m) {
  Eigen::Vector3d v;
  v << -m.y, -m.z, m.x;
  return v;
}

static geometry_msgs::Vector3 body_rdf_from_flu_geometry_vector3(const geometry_msgs::Vector3 &m) {
  geometry_msgs::Vector3 v;
  v.x = -m.y;
  v.y = -m.z;
  v.z =  m.x;
  return v;
}

static geometry_msgs::Vector3 map_enu_from_nwu_double(const double &x, const double &y, const double &z) {
  geometry_msgs::Vector3 v;
  v.x = -y;
  v.y = x;
  v.z = z;
  return v;
}

static geometry_msgs::Vector3 body_rdf_from_flu_geometry_vector3(const Eigen::Vector3d &m) {
  geometry_msgs::Vector3 v;
  v.x = -m(1);
  v.y = -m(2);
  v.z =  m(0);
  return v;
}

static geometry_msgs::Point body_rdf_from_flu_geometry_point(const Eigen::Vector3d &m) {
  geometry_msgs::Point v;
  v.x = -m(1);
  v.y = -m(2);
  v.z =  m(0);
  return v;
}

static geometry_msgs::Point body_rdf_from_flu_geometry_point(const geometry_msgs::Point &m) {
  geometry_msgs::Point v;
  v.x = -m.y;
  v.y = -m.z;
  v.z =  m.x;
  return v;
}

static geometry_msgs::Vector3 body_flu_from_rdf_geometry_vector3(const Eigen::Vector3d &m) {
  geometry_msgs::Vector3 v;
  v.x = m(2);
  v.y = -m(0);
  v.z = -m(1);
  return v;
}

static geometry_msgs::Vector3 body_flu_from_rdf_geometry_vector3(const geometry_msgs::Vector3 &m) {
  geometry_msgs::Vector3 v;
  v.x = m.z;
  v.y = -m.x;
  v.z = -m.y;
  return v;
}

static geometry_msgs::Point body_flu_from_rdf_geometry_point(const Eigen::Vector3d &m) {
  geometry_msgs::Point v;
  v.x = m(2);
  v.y = -m(0);
  v.z = -m(1);
  return v;
}

static geometry_msgs::Point body_flu_from_rdf_geometry_point(const geometry_msgs::Point &m) {
  geometry_msgs::Point v;
  v.x = m.z;
  v.y = -m.x;
  v.z = -m.y;
  return v;
}

static geometry_msgs::Point neu_from_enu_geometry_point(const geometry_msgs::Point &m) {
  geometry_msgs::Point v;
  v.x = m.y;
  v.y = m.x;
  v.z = m.z;
  return v;
}

static geometry_msgs::Vector3 neu_from_enu_geometry_vector3(const geometry_msgs::Vector3 &m) {
  geometry_msgs::Vector3 v;
  v.x = m.y;
  v.y = m.x;
  v.z = m.z;
  return v;
}

// static geometry_msgs::Vector3 body_flu_from_brd_geometry(const geometry_msgs::Vector3 &m) {
//   geometry_msgs::Vector3 v;
//   v.x = -m.x;
//   v.y = -m.y;
//   v.z = -m.z;
//   return v;
// }

// inline Eigen::Vector3d toEigen(const Vector<double> &p) {
//   Eigen::Vector3d ev3(p[0], p[1], p[2]);
//   return ev3;
// }

inline Eigen::Vector3d toEigen(const geometry_msgs::Point &p) {
  Eigen::Vector3d ev3(p.x, p.y, p.z);
  return ev3;
}

inline Eigen::Vector3d toEigen(const geometry_msgs::Vector3 &v3) {
  Eigen::Vector3d ev3(v3.x, v3.y, v3.z);
  return ev3;
}

inline geometry_msgs::Vector3 toGeometry(const Eigen::Vector3d &v3) {
  geometry_msgs::Vector3 gev3;
  gev3.x = v3(0),
  gev3.y = v3(1),
  gev3.z = v3(2);
  return gev3;
}

inline Eigen::Vector4d quatMultiplication(const Eigen::Vector4d &q, const Eigen::Vector4d &p) {
  Eigen::Vector4d quat;
  quat << p(0) * q(0) - p(1) * q(1) - p(2) * q(2) - p(3) * q(3), p(0) * q(1) + p(1) * q(0) - p(2) * q(3) + p(3) * q(2),
      p(0) * q(2) + p(1) * q(3) + p(2) * q(0) - p(3) * q(1), p(0) * q(3) - p(1) * q(2) + p(2) * q(1) + p(3) * q(0);
  return quat;
}

inline Eigen::Matrix3d quat2RotMatrix(const Eigen::Vector4d &q) {
  Eigen::Matrix3d rotmat;
  rotmat << q(0) * q(0) + q(1) * q(1) - q(2) * q(2) - q(3) * q(3), 2 * q(1) * q(2) - 2 * q(0) * q(3),
      2 * q(0) * q(2) + 2 * q(1) * q(3),

      2 * q(0) * q(3) + 2 * q(1) * q(2), q(0) * q(0) - q(1) * q(1) + q(2) * q(2) - q(3) * q(3),
      2 * q(2) * q(3) - 2 * q(0) * q(1),

      2 * q(1) * q(3) - 2 * q(0) * q(2), 2 * q(0) * q(1) + 2 * q(2) * q(3),
      q(0) * q(0) - q(1) * q(1) - q(2) * q(2) + q(3) * q(3);
  return rotmat;
}

inline Eigen::Vector4d rot2Quaternion(const Eigen::Matrix3d &R) {
  Eigen::Vector4d quat;
  double tr = R.trace();
  if (tr > 0.0) {
    double S = sqrt(tr + 1.0) * 2.0;  // S=4*qw
    quat(0) = 0.25 * S;
    quat(1) = (R(2, 1) - R(1, 2)) / S;
    quat(2) = (R(0, 2) - R(2, 0)) / S;
    quat(3) = (R(1, 0) - R(0, 1)) / S;
  } else if ((R(0, 0) > R(1, 1)) & (R(0, 0) > R(2, 2))) {
    double S = sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2)) * 2.0;  // S=4*qx
    quat(0) = (R(2, 1) - R(1, 2)) / S;
    quat(1) = 0.25 * S;
    quat(2) = (R(0, 1) + R(1, 0)) / S;
    quat(3) = (R(0, 2) + R(2, 0)) / S;
  } else if (R(1, 1) > R(2, 2)) {
    double S = sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2)) * 2.0;  // S=4*qy
    quat(0) = (R(0, 2) - R(2, 0)) / S;
    quat(1) = (R(0, 1) + R(1, 0)) / S;
    quat(2) = 0.25 * S;
    quat(3) = (R(1, 2) + R(2, 1)) / S;
  } else {
    double S = sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1)) * 2.0;  // S=4*qz
    quat(0) = (R(1, 0) - R(0, 1)) / S;
    quat(1) = (R(0, 2) + R(2, 0)) / S;
    quat(2) = (R(1, 2) + R(2, 1)) / S;
    quat(3) = 0.25 * S;
  }
  return quat;
}

#endif
