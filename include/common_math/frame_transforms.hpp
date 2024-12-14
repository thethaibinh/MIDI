// frame_transform.h
/*!
 * Copyright 2024 by Binh Nguyen <thethaibinh@gmail.com>
 *
 * This code is free software: you can redistribute
 * it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * This code is distributed in the hope that it will
 * be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with the code.  If not, see <http://www.gnu.org/licenses/>.
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
