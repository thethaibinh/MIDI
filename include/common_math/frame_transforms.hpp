// frame_transform.h
/*!
 * Copyright 2024 by Binh Nguyen <thethaibinh@gmail.com>
 *
 * This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
 * You may use, redistribute, and modify this code for non-commercial purposes only, provided that:
 * 1. You give appropriate credit to the original author
 * 2. You indicate if changes were made
 *
 * This code is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * For commercial use, please contact the author for licensing terms.
 * Full license text: https://creativecommons.org/licenses/by-nc/4.0/
 */
#ifndef FRAME_TRANSFORM_H
#define FRAME_TRANSFORM_H

#include <geometry_msgs/Point.h>
#include <geometry_msgs/Vector3.h>
#include <Eigen/Dense>
#include <array>

namespace frame_transform {

// Overloaded function to transform a point from body frame (FLU) to camera frame (RDF)
// std::array<double, 3> --> geometry_msgs::Point
inline void transform_body_to_camera(const std::array<double, 3>& body_point,
                              geometry_msgs::Point& camera_point) {
  camera_point.x = -body_point[1];  // Left to Right
  camera_point.y = -body_point[2];  // Up to Down
  camera_point.z = body_point[0];   // Forward remains Forward
}
// std::array<double, 3> --> geometry_msgs::Vector3
inline void transform_body_to_camera(const std::array<double, 3>& body_point,
                              geometry_msgs::Vector3& camera_vector) {
  camera_vector.x = -body_point[1];  // Left to Right
  camera_vector.y = -body_point[2];  // Up to Down
  camera_vector.z = body_point[0];   // Forward remains Forward
}
// geometry_msgs::Point --> geometry_msgs::Point
inline void transform_body_to_camera(
  const geometry_msgs::Point& body_point, geometry_msgs::Point& camera_point) {
  camera_point.x = -body_point.y;  // Left to Right
  camera_point.y = -body_point.z;  // Up to Down
  camera_point.z = body_point.x;   // Forward remains Forward
}
// geometry_msgs::Vector3 --> geometry_msgs::Vector3
inline void transform_body_to_camera(const geometry_msgs::Vector3& body_vector,
                              geometry_msgs::Vector3& camera_vector) {
  camera_vector.x = -body_vector.y;  // Left to Right
  camera_vector.y = -body_vector.z;  // Up to Down
  camera_vector.z = body_vector.x;   // Forward remains Forward
}


// Overloaded function to transform a point/vector from camera frame (RDF) to
// body frame (FLU) std::array<double, 3> --> geometry_msgs::Point
inline void transform_camera_to_body(const std::array<double, 3>& camera_point,
                              geometry_msgs::Point& body_point) {
  body_point.x = camera_point[2];   // Forward remains Forward
  body_point.y = -camera_point[0];  // Right to Left
  body_point.z = -camera_point[1];  // Down to Up
}
// std::array<double, 3> --> geometry_msgs::Vector3
inline void transform_camera_to_body(const std::array<double, 3>& camera_vector,
                              geometry_msgs::Vector3& body_vector) {
  body_vector.x = camera_vector[2];   // Forward remains Forward
  body_vector.y = -camera_vector[0];  // Right to Left
  body_vector.z = -camera_vector[1];  // Down to Up
}
// geometry_msgs::Point --> geometry_msgs::Point
inline void transform_camera_to_body(const geometry_msgs::Point& camera_point,
                              geometry_msgs::Point& body_point) {
  body_point.x = camera_point.z;   // Forward remains Forward
  body_point.y = -camera_point.x;  // Right to Left
  body_point.z = -camera_point.y;  // Down to Up
}
// Eigen::Vector3d --> geometry_msgs::Point
inline void transform_camera_to_body(const Eigen::Vector3d& camera_point,
                              geometry_msgs::Point& body_point) {
  body_point.x = camera_point.z();   // Forward remains Forward
  body_point.y = -camera_point.x();  // Right to Left
  body_point.z = -camera_point.y();  // Down to Up
}
// geometry_msgs::Vector3 --> geometry_msgs::Vector3
inline void transform_camera_to_body(const geometry_msgs::Vector3& camera_vector,
                              geometry_msgs::Vector3& body_vector) {
  body_vector.x = camera_vector.z;   // Forward remains Forward
  body_vector.y = -camera_vector.x;  // Right to Left
  body_vector.z = -camera_vector.y;  // Down to Up
}
// geometry_msgs::Vector3 --> geometry_msgs::Point
inline void transform_camera_to_body(const geometry_msgs::Vector3& camera_point,
                              geometry_msgs::Point& body_point) {
  body_point.x = camera_point.z;   // Forward remains Forward
  body_point.y = -camera_point.x;  // Right to Left
  body_point.z = -camera_point.y;  // Down to Up
}
// geometry_msgs::Point --> geometry_msgs::Vector3
inline void transform_camera_to_body(const geometry_msgs::Point& camera_vector,
                              geometry_msgs::Vector3& body_vector) {
  body_vector.x = camera_vector.z;   // Forward remains Forward
  body_vector.y = -camera_vector.x;  // Right to Left
  body_vector.z = -camera_vector.y;  // Down to Up
}

}  // namespace frame_transform

#endif  // FRAME_TRANSFORM_H
