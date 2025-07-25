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
#include "common_math/segment.hpp"
#include "common_math/polynomial2.hpp"

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
class ThirdOrderSegment : public Segment {
 public:
  //! Constructor.
  /*!
   * @param coeffs The six 3D coefficients defining the polynomial.
   * The coefficients are ordered such that coeffs[0] corresponds to t^5,
   * coeffs[1] to t^4 and so forth.
   * @param end_time The latest time at which the polynomial is defined
   */
  ThirdOrderSegment(std::vector<Eigen::Vector3d> coeffs, double start_time,
                    double end_time)
    : Segment(coeffs, start_time, end_time) {
    if (get_start_time() > get_end_time()) {
      throw std::invalid_argument(
        "The end time must be greater than the start time.");
    }
    std::vector<Eigen::Vector3d> _coeffs = get_coeffs();
    if (_coeffs.size() != 4) {
      throw std::invalid_argument(
        "Third-order segment must have exactly 4 coefficients.");
    }
  }

  //! Constructor.
  /*!
   * @param j/6   Coefficients for t^3
   * @param a0/2  Coefficients for t^2
   * @param v0    Coefficients for t
   * @param p0    Constant terms
   * @param end_time The latest time at which the polynomial is defined
   */
  // ThirdOrderSegment(Eigen::Vector3d j, Eigen::Vector3d a0, Eigen::Vector3d v0, Eigen::Vector3d p0, double start_time, double end_time) {
  //   assert(start_time <= end_time);
  //   _coeffs = {j / 6, a0 / 2, v0, p0};
  //   get_start_time() = start_time;
  //   get_end_time() = end_time;
  // }

  //! Returns the 3D position of the polynomial at a given time.
  /*!
   * @param t Time at which to evaluate the polynomial (must be between
   * get_start_time() and get_end_time())
   * @return Position of trajectory at time t
   */
  Eigen::Vector3d get_point(const double& t) const override {
    if ((t - get_start_time()) * (t - get_end_time()) > 0) {
      throw std::invalid_argument("Reference time is not in range.");
    }
    std::vector<Eigen::Vector3d> _coeffs = get_coeffs();
    if (_coeffs.size() != 4) {
      throw std::invalid_argument(
        "Third-order segment must have exactly 4 coefficients.");
    }
    return _coeffs[0] * t * t * t + _coeffs[1] * t * t + _coeffs[2] * t + _coeffs[3];
  }
  Eigen::Vector3d get_start_point() const override {
    std::vector<Eigen::Vector3d> _coeffs = get_coeffs();
    if (_coeffs.size() != 4) {
      throw std::invalid_argument(
        "Third-order segment must have exactly 4 coefficients.");
    }
    return get_point(get_start_time());
  }
  Eigen::Vector3d get_end_point() const override {
    std::vector<Eigen::Vector3d> _coeffs = get_coeffs();
    if (_coeffs.size() != 4) {
      throw std::invalid_argument(
        "Third-order segment must have exactly 4 coefficients.");
    }
    return get_point(get_end_time());
  }


  //! Returns the position of the polynomial along the given axis at a given
  //! time.
  /*!
   * @param i Axis of the trajectory to evaluate
   * @param t Time at which to evaluate the polynomial (must be between
   * get_start_time() and get_end_time())
   * @return Position of trajectory along axis i at time t
   */
  double get_axis_value(const int& i, const double& t) const override {
    if ((t - get_start_time()) * (t - get_end_time()) > 0) {
      throw std::invalid_argument("Reference time is not in range.");
    }
    std::vector<Eigen::Vector3d> _coeffs = get_coeffs();
    if (_coeffs.size() != 4) {
      throw std::invalid_argument(
        "Third-order segment must have exactly 4 coefficients.");
    }
    return _coeffs[0][i] * t * t * t + _coeffs[1][i] * t * t + _coeffs[2][i] * t + _coeffs[3][i];
  }

  double get_axis_start_value(const int& i) const override {
    std::vector<Eigen::Vector3d> _coeffs = get_coeffs();
    if (_coeffs.size() != 4) {
      throw std::invalid_argument(
        "Third-order segment must have exactly 4 coefficients.");
    }
    return get_axis_value(i, get_start_time());
  }

  double get_axis_end_value(const int& i) const override {
    std::vector<Eigen::Vector3d> _coeffs = get_coeffs();
    if (_coeffs.size() != 4) {
      throw std::invalid_argument(
        "Third-order segment must have exactly 4 coefficients.");
    }
    return get_axis_value(i, get_end_time());
  }

  void get_depth_switching_points(std::vector<double>& roots) const override {
    // Compute the coefficients of \dot{d}_z(t)
    std::vector<Eigen::Vector3d> derivative_coeffs = get_derivative_coeffs();
    std::vector<double> c = {derivative_coeffs[0].z(), derivative_coeffs[1].z(), derivative_coeffs[2].z()};

    // Compute the times at which the segment changes direction along the z-axis
    SecondOrderPolynomial pol2(c, get_start_time(), get_end_time());
    pol2.solve_roots(roots);
    std::sort(roots.begin(), roots.end());
  }

  //! Returns the coefficient of the time derivative of the trajectory
  /*!
   * @return A 5D vector of 3D coefficients representing the time derivative
   * of the trajectory, where deriv_coeffs[0] is a 3D vector of coefficients
   * corresponding to t^4, deriv_coeffs[1] corresponds to t^3, etc.
   */
  std::vector<Eigen::Vector3d> get_derivative_coeffs() const override {
    std::vector<Eigen::Vector3d> _coeffs = get_coeffs();
    if (_coeffs.size() != 4) {
      throw std::invalid_argument(
        "Third-order segment must have exactly 4 coefficients.");
    }
    std::vector<Eigen::Vector3d> deriv_coeffs;
    deriv_coeffs.reserve(3);
    for (int i = 0; i < 3; i++) {
      deriv_coeffs.push_back((3 - i) * _coeffs[i]);
    }
    return deriv_coeffs;
  }

  //! Returns the i-th 3D vector of coefficients.
  Eigen::Vector3d operator[](const int& i) const override {
    std::vector<Eigen::Vector3d> _coeffs = get_coeffs();
    if (_coeffs.size() != 4) {
      throw std::invalid_argument(
        "Third-order segment must have exactly 4 coefficients.");
    }
    switch (i) {
      case 0:
        return _coeffs[0];
      case 1:
        return _coeffs[1];
      case 2:
        return _coeffs[2];
      case 3:
        return _coeffs[3];
      default:
        // We should never get here (index out of bounds)
        assert(false);
        double nan_value = std::numeric_limits<double>::quiet_NaN();
        return Eigen::Vector3d(nan_value, nan_value,
                               nan_value);  // Returns vector of NaNs
    }
  }
};

}  // namespace common_math
