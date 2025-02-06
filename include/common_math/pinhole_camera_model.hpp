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

#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Dense>

namespace common_math {

class PinholeCamera {
 public:
  //! Constructor
  /*!
   * @param minimum_clear_distance The minimum distance for collision checking
   */
  PinholeCamera(const double focal_length, const double cx, const double cy,
                const uint16_t width = 320, const uint16_t height = 240,
                const std::vector<double>& cov_coeffs = {},
                const std::vector<double>& dynamic_pos_cov_coeffs = {},
                const double true_radius = 0.26f,
                const double planning_radius = 0.55f,
                const double minimum_clear_distance = 0.5)
    : _focal_length(focal_length),
      _cx(cx),
      _cy(cy),
      _width(width),
      _height(height),
      _cov_coeffs(cov_coeffs),
      _dynamic_pos_cov_coeffs(dynamic_pos_cov_coeffs),
      _true_vehicle_radius(true_radius),
      _planning_vehicle_radius(planning_radius),
      _minimum_clear_distance(minimum_clear_distance) {}

  Eigen::Vector3d get_depth_noise_covariance_matrix(const Eigen::Vector3d& depth_point) const {
    double ca0 = _cov_coeffs[0];
    double ca1 = _cov_coeffs[1];
    double ca2 = _cov_coeffs[2];
    double cl0 = _cov_coeffs[3];
    double cl1 = _cov_coeffs[4];
    double cl2 = _cov_coeffs[5];

    double sigma_a = ca0 + ca1 * depth_point.z() + ca2 * depth_point.z() * depth_point.z();
    double sigma_lx = cl0 + cl1 * depth_point.z() + cl2 * std::abs(depth_point.x());
    double sigma_ly = cl0 + cl1 * depth_point.z() + cl2 * std::abs(depth_point.y());
    return Eigen::Vector3d(sigma_lx, sigma_ly, sigma_a);
  }

  Eigen::Vector3d get_dynamic_pos_covariance_matrix() const
  {
      // These are dynamic position uncertainty due to uncertainty in velocity
      double sigma_x = _dynamic_pos_cov_coeffs[0];
      double sigma_y = _dynamic_pos_cov_coeffs[1];
      double sigma_z = _dynamic_pos_cov_coeffs[2];
      return Eigen::Vector3d(sigma_x, sigma_y, sigma_z);
  }

  //! Computes the 3D position of a point given a position in pixel coordinates
  //! and a depth
  /*!
   * @param x Input: The horizontal position of the pixel (should be between
   * zero and the image width)
   * @param y Input: The vertical position of the pixel (should be between zero
   * and the image height)
   * @param depth Input: The Z depth of the point [meters]
   * @param out_point Output: The 3D position of the point (X points towards
   * right edge of image, Y towards bottom edge of image, and Z into the image)
   */
  Eigen::Vector3d deproject_pixel_to_point(const int& x, const int& y,
                                           const double& depth) const {
    return depth * Eigen::Vector3d((x - _cx) / _focal_length,
                                   (y - _cy) / _focal_length, 1);
  }

  //! Projects 3D point into the image, returning the pixel coordinates
  /*!
   * @param point Input: The position of the point in 3D space
   * @param out_x Output: The horizontal position of the projected point in pixel
   * coordinates
   * @param out_y Output: The vertical position of the projected point in pixel
   * coordinates
   */
  Eigen::Vector2i project_point_to_pixel(const Eigen::Vector3d& point) const {
    double x = point.x() * _focal_length / std::abs(point.z()) + _cx;
    double y = point.y() * _focal_length / std::abs(point.z()) + _cy;

    // Round to nearest integer without clamping
    return Eigen::Vector2i(static_cast<int16_t>(std::round(x)), static_cast<int16_t>(std::round(y)));
  }

  void project_point_to_pixel_adding_margin(const Eigen::Vector3d& point,
                                            std::vector<double>& out_x,
                                            std::vector<double>& out_y) const {

    double safety_margin = _true_vehicle_radius * _focal_length / _minimum_clear_distance;

    // left
    double left = point.x() * _focal_length / std::abs(point.z()) + _cx - safety_margin;
    out_x.push_back(left);

    // right
    double right = point.x() * _focal_length / std::abs(point.z()) + _cx + safety_margin;
    out_x.push_back(right);

    // top
    double top = point.y() * _focal_length / std::abs(point.z()) + _cy - safety_margin;
    out_y.push_back(top);

    // bottom
    double bottom = point.y() * _focal_length / std::abs(point.z()) + _cy + safety_margin;
    out_y.push_back(bottom);
  }

  /*!
   * @return The frame dimensions minus a margin based on true/planning vehicle radius
   * left, right, top, bottom
   */
  std::vector<uint16_t> get_frame_dimensions_with_true_radius_margin() const {
    uint16_t margin = static_cast<uint16_t>(_true_vehicle_radius * _focal_length / _minimum_clear_distance);
    return {margin, static_cast<uint16_t>(_width - margin), margin, static_cast<uint16_t>(_height - margin)};
  }

  // std::vector<uint16_t> get_frame_dimensions_with_planning_radius_margin() const {
  //   uint16_t margin = static_cast<uint16_t>(_planning_vehicle_radius * _focal_length / _minimum_clear_distance);
  //   return {margin, static_cast<uint16_t>(_width - margin), margin, static_cast<uint16_t>(_height - margin)};
  // }

  // Getters for the attributes
  double get_focal_length() const { return _focal_length; }
  double get_cx() const { return _cx; }
  double get_cy() const { return _cy; }
  double get_fx() const { return _focal_length; }
  double get_fy() const { return _focal_length; }
  double get_true_vehicle_radius() const { return _true_vehicle_radius; }
  double get_planning_vehicle_radius() const { return _planning_vehicle_radius; }
  uint16_t get_width() const { return _width; }
  uint16_t get_height() const { return _height; }
  double get_minimum_clear_distance() const { return _minimum_clear_distance; }

  Eigen::Vector2i clamp_to_frame_with_margin(const int& x, const int& y) const {
    std::vector<uint16_t> frame_dims = get_frame_dimensions_with_true_radius_margin();
    uint16_t left = frame_dims[0];
    uint16_t right = frame_dims[1];
    uint16_t top = frame_dims[2];
    uint16_t bottom = frame_dims[3];

    // Find the closest point on the boundary
    if (y < top) {
        if (x < left) {
            return Eigen::Vector2i(left, top);
        } else if (x > right) {
            return Eigen::Vector2i(right, top);
        } else {
            return Eigen::Vector2i(x, top);
        }
    } else if (y > bottom) {
        if (x < left) {
            return Eigen::Vector2i(left, bottom);
        } else if (x > right) {
            return Eigen::Vector2i(right, bottom);
        } else {
            return Eigen::Vector2i(x, bottom);
        }
    } else { // top <= y <= bottom
        if (x < left) {
            return Eigen::Vector2i(left, y);
        } else if (x > right) {
            return Eigen::Vector2i(right, y);
        } else {
            return Eigen::Vector2i(x, y);
        }
    }
  }

 private:
  double _focal_length;  // Focal length of the camera
  double _cx;            // Principal point x-coordinate
  double _cy;            // Principal point y-coordinate
  uint16_t _width;            // Image width in pixels
  uint16_t _height;           // Image height in pixels
  // The true radius of the vehicle. Any depth values closer than
  // this distance to the camera will be ignored. [meters]
  double _true_vehicle_radius;  // vehicle radius for planning
  // We plan as if the vehicle has this radius. This value should be slightly
  // larger than _trueVehicleRadius to account for pose estimation errors and
  // trajectory tracking errors. [meters]
  double _planning_vehicle_radius;  // vehicle radius for planning
  double _minimum_clear_distance;  // Minimum distance for collision checking
  std::vector<double> _cov_coeffs;
  std::vector<double> _dynamic_pos_cov_coeffs;
};

} // namespace common_math
