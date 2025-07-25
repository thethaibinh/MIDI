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

#pragma once
#include <vector>
// Eigen catkin
#include <Eigen/Dense>
// ROS base
#include <ros/console.h>
#include "ros/ros.h"

namespace common_math {

//! Represents a triple integrator segment in 3D with constant jerk as input.
/*!
 *  The polynomial is a function of time and is of the form
 *  p = p0 + t * (v0 + t * (a0 / 2 + t * j / 6))
 *  v = v0 + t * (a0 + t * j / 2)
 *  a = a0 + t * j
 *  where each coefficient (p0, v0, a0, j) is a vector with three dimensions.
 *  The polynomial is only defined between given start and end times.
 */
class Segment {
 public:
  // Constructor to initialize coefficients
  Segment(const std::vector<Eigen::Vector3d>& coefficients,
          const double& start_time, const double& end_time)
    : _coeffs(coefficients), _start_time(start_time), _end_time(end_time) {}
  virtual ~Segment() {}

  //! Returns the position of the polynomial along the given axis at a given
  //! time.
  /*!
   * @param i Axis of the trajectory to evaluate
   * @param t Time at which to evaluate the polynomial (must be between
   * startTime and endTime)
   * @return Position of trajectory along axis i at time t
   */
  virtual double get_axis_value(const int& i, const double& t) const = 0;

  virtual double get_axis_start_value(const int& i) const = 0;

  virtual double get_axis_end_value(const int& i) const = 0;

  bool is_monotonically_increasing_depth() const {
    std::vector<double> roots;
    get_depth_switching_points(roots);
    if (roots.size() >= 1)
      return false;
    else
      return true;
  }

  std::vector<double> get_depth_switching_points_and_terminals() const {
    std::vector<double> roots;
    roots.push_back(get_start_time());
    get_depth_switching_points(roots);
    roots.push_back(get_end_time());
    return roots;
  }

  virtual void get_depth_switching_points(std::vector<double>& roots) const = 0;

  //! Returns the 3D position of the polynomial at a given time.
  /*!
   * @param t Time at which to evaluate the polynomial (must be between
   * startTime and endTime)
   * @return Position of trajectory at time t
   */
  virtual Eigen::Vector3d get_point(const double& t) const = 0;
  virtual Eigen::Vector3d get_start_point() const = 0;
  virtual Eigen::Vector3d get_end_point() const = 0;

  //! Return the 3D vector of coefficients
  std::vector<Eigen::Vector3d> get_coeffs() const { return _coeffs; }

  //! Return the end time of the trajectory
  double get_start_time() const { return _start_time; }

  //! Return the end time of the trajectory
  double get_end_time() const { return _end_time; }

  //! Return the trajectory duration
  double get_duration() const { return _end_time - _start_time; }

  //! Returns the coefficient of the time derivative of the trajectory
  /*!
   * @return A 5D vector of 3D coefficients representing the time derivative
   * of the trajectory, where derivCoeffs[0] is a 3D vector of coefficients
   * corresponding to t^4, derivCoeffs[1] corresponds to t^3, etc.
   */
  virtual std::vector<Eigen::Vector3d> get_derivative_coeffs() const = 0;

  //! Returns the i-th 3D vector of coefficients.
  virtual Eigen::Vector3d operator[](const int& i) const {
    switch (i) {
      case 0:
        return _coeffs[0];
      case 1:
        return _coeffs[1];
      case 2:
        return _coeffs[2];
      default:
        // We should never get here (index out of bounds)
        assert(false);
        double nan_value = std::numeric_limits<double>::quiet_NaN();
        return Eigen::Vector3d(nan_value, nan_value,
                               nan_value);  // Returns vector of NaNs
    }
  }

 private:
  std::vector<Eigen::Vector3d> _coeffs;
  double _start_time, _end_time;
};

}  // namespace common_math
