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
#include "common_math/segment.hpp"
#include "common_math/pinhole_camera_model.hpp"
#include "common_math/polynomial4.hpp"

namespace common_math {

//! Represents a double integrator segment in 3D with constant acceleration a as
//! input.
/*!
 *  The polynomial is a function of time and is of the form
 *  p = p0 + t * (v0 + t * (a / 2))
 *  v = v0 + t * a
 *  where each coefficient (p0, v0, a) is a vector with three dimensions.
 *  The polynomial is only defined between given start and end times.
 */
class SecondOrderSegment : public Segment {
 public:
  //! Constructor.
  /*!
   * @param coeffs The six 3D coefficients defining the polynomial.
   * The coefficients are ordered such that coeffs[0] corresponds to t^2,
   * coeffs[1] to t and so forth.
   * @param startTime The earliest time at which the polynomial is defined
   * @param endTime The latest time at which the polynomial is defined
   */
  SecondOrderSegment(const std::vector<Eigen::Vector3d>& coeffs,
                     const double& start_time, const double& end_time)
    : Segment(coeffs, start_time, end_time) {
    if (get_start_time() > get_end_time()) {
      throw std::invalid_argument(
        "The end time must be greater than the start time.");
    }
    if (get_coeffs().size() != 3) {
      throw std::invalid_argument(
        "Second-order segment must have exactly 3 coefficients.");
    }
  }

  //! Returns the 3D position of the polynomial at a given time.
  /*!
   * @param t Time at which to evaluate the polynomial (must be between
   * startTime and endTime)
   * @return Position of trajectory at time t
   */
  Eigen::Vector3d get_point(const double& t) const override {
    if ((t - get_start_time()) * (t - get_end_time()) > 0) {
      throw std::invalid_argument("Reference time is not in range.");
    }
    std::vector<Eigen::Vector3d> _coeffs = get_coeffs();
    if (_coeffs.size() != 3) {
      throw std::invalid_argument(
        "Second-order segment must have exactly 3 coefficients.");
    }
    return _coeffs[0] * t * t + _coeffs[1] * t + _coeffs[2];
  }

  Eigen::Vector3d get_start_point() const override {
    std::vector<Eigen::Vector3d> _coeffs = get_coeffs();
    if (_coeffs.size() != 3) {
      throw std::invalid_argument(
        "Second-order segment must have exactly 3 coefficients.");
    }
    return get_point(get_start_time());
  }

  Eigen::Vector3d get_end_point() const override {
    std::vector<Eigen::Vector3d> _coeffs = get_coeffs();
    if (_coeffs.size() != 3) {
      throw std::invalid_argument(
        "Second-order segment must have exactly 3 coefficients.");
    }
    return get_point(get_end_time());
  }

  uint8_t solve_first_time_at_depth(const double depth, double& time) const {
    std::vector<Eigen::Vector3d> coeffs = get_coeffs();
    std::vector<double> z_coeffs = {coeffs[0].z(), coeffs[1].z(), coeffs[2].z() - depth};

    const SecondOrderPolynomial pol2(z_coeffs, get_start_time(), get_end_time());
    std::vector<double> times;
    const uint8_t num_roots = pol2.solve_roots(times);
    time = times[0];
    return num_roots;
  }

  //! Returns the position of the polynomial along the given axis at a given
  //! time.
  /*!
   * @param i Axis of the trajectory to evaluate
   * @param t Time at which to evaluate the polynomial (must be between
   * startTime and endTime)
   * @return Position of trajectory along axis i at time t
   */
  double get_axis_value(const int& i, const double& t) const override {
    if ((t - get_start_time()) * (t - get_end_time()) > 0) {
      throw std::invalid_argument("Reference time is not in range.");
    }
    std::vector<Eigen::Vector3d> _coeffs = get_coeffs();
    if (_coeffs.size() != 3) {
      throw std::invalid_argument(
        "Second-order segment must have exactly 3 coefficients.");
    }
    return _coeffs[0][i] * t * t + _coeffs[1][i] * t + _coeffs[2][i];
  }

  double get_axis_start_value(const int& i) const override {
    std::vector<Eigen::Vector3d> _coeffs = get_coeffs();
    if (_coeffs.size() != 3) {
      throw std::invalid_argument(
        "Second-order segment must have exactly 3 coefficients.");
    }
    return get_axis_value(i, get_start_time());
  }

  double get_axis_end_value(const int& i) const override {
    std::vector<Eigen::Vector3d> _coeffs = get_coeffs();
    if (_coeffs.size() != 3) {
      throw std::invalid_argument(
        "Second-order segment must have exactly 3 coefficients.");
    }
    return get_axis_value(i, get_end_time());
  }

  double get_euclidean_distance(const Eigen::Vector3d& pixel) const {
    return sqrt(get_euclidean_distance_square(pixel));
  }
  // Function to calculate Euclidean distance to a depth pixel
  double get_euclidean_distance_square(const Eigen::Vector3d& pixel) const {
    // The closest point in the trajectory to pixels with depth greater
    // than endpoint_depth is the endpoint.
    double endpoint_depth = get_axis_end_value(2);
    double start_point_depth = get_axis_start_value(2);

    if (pixel.z() > endpoint_depth) return (pixel - get_end_point()).squaredNorm();
    if (pixel.z() < start_point_depth) return (pixel - get_start_point()).squaredNorm();

    std::vector<Eigen::Vector3d> _coeffs = get_coeffs();
    double a1 = _coeffs[0][0], b1 = _coeffs[1][0], c1 = _coeffs[2][0];
    double a2 = _coeffs[0][1], b2 = _coeffs[1][1], c2 = _coeffs[2][1];
    double a3 = _coeffs[0][2], b3 = _coeffs[1][2], c3 = _coeffs[2][2];

    // Time-parameterized polynomial of Euclidean distance from a point to the trajectory
    std::vector<double> eu_dist_coeffs = {
      a1 * a1 + a2 * a2 + a3 * a3,
      a1 * b1 * 2.0 + a2 * b2 * 2.0 + a3 * b3 * 2.0,
      a1 * (c1 - pixel.x()) * 2.0 + a2 * (c2 - pixel.y()) * 2.0 + a3 * (c3 - pixel.z()) * 2.0 + b1 * b1 + b2 * b2 + b3 * b3,
      b1 * (c1 - pixel.x()) * 2.0 + b2 * (c2 - pixel.y()) * 2.0 + b3 * (c3 - pixel.z()) * 2.0,
      (c1 - pixel.x()) * (c1 - pixel.x()) + (c2 - pixel.y()) * (c2 - pixel.y()) + (c3 - pixel.z()) * (c3 - pixel.z())};

    FourthOrderPolynomial pol4(eu_dist_coeffs, get_start_time(), get_end_time());
    return pol4.get_min();
    // if (eu_dist_coeffs[0] != 0) {
    //   FourthOrderPolynomial pol4(eu_dist_coeffs, get_start_time(), get_end_time());
    //   return pol4.get_min();
    // } else if (eu_dist_coeffs[1] != 0) {
    //   eu_dist_coeffs.erase(eu_dist_coeffs.begin());
    //   ThirdOrderPolynomial pol3(eu_dist_coeffs, get_start_time(), get_end_time());
    //   return pol3.get_min();
    // } else {
    //   eu_dist_coeffs.erase(eu_dist_coeffs.begin());
    //   eu_dist_coeffs.erase(eu_dist_coeffs.begin());
    //   SecondOrderPolynomial pol2(eu_dist_coeffs, get_start_time(), get_end_time());
    //   return pol2.get_min();
    // }
  }

  double get_mahalanobis_distance(const Eigen::Vector3d& mean,
                                  const Eigen::Vector3d& cov_diag) const {
    const double mds = get_du_mds(mean, cov_diag);
    return sqrt(mds);
  }

  // Function to calculate a half of the Mahalanobis distance square for dynamic uncertainty
  double get_du_mds(const Eigen::Vector3d& mean,
                                  const Eigen::Vector3d& cov_diag) const {
    std::vector<Eigen::Vector3d> _coeffs = get_coeffs();
    double a1 = _coeffs[0][0], b1 = _coeffs[1][0], c1 = _coeffs[2][0];
    double a2 = _coeffs[0][1], b2 = _coeffs[1][1], c2 = _coeffs[2][1];
    double a3 = _coeffs[0][2], b3 = _coeffs[1][2], c3 = _coeffs[2][2];

    std::vector<double> ma_dist_coeffs = {
      // A fourth orders polynomial from PDF expanded equation in the paper
        a1 * a1 / cov_diag.x() +
        a2 * a2 / cov_diag.y() +
        a3 * a3 / cov_diag.z(),
        a1 * b1 / cov_diag.x() * 2.0 +
        a2 * b2 / cov_diag.y() * 2.0 +
        a3 * b3 / cov_diag.z() * 2.0,
        (b1 * b1 + 2.0 * a1 * (c1 - mean.x())) / cov_diag.x() +
        (b2 * b2 + 2.0 * a2 * (c2 - mean.y())) / cov_diag.y() +
        (b3 * b3 + 2.0 * a3 * (c3 - mean.z())) / cov_diag.z(),
        b1 * (c1 - mean.x()) / cov_diag.x() * 2.0 +
        b2 * (c2 - mean.y()) / cov_diag.y() * 2.0 +
        b3 * (c3 - mean.z()) / cov_diag.z() * 2.0,
        (c1 - mean.x()) * (c1 - mean.x()) / cov_diag.x() +
        (c2 - mean.y()) * (c2 - mean.y()) / cov_diag.y() +
        (c3 - mean.z()) * (c3 - mean.z()) / cov_diag.z()};

    FourthOrderPolynomial pol4(ma_dist_coeffs, get_start_time(), get_end_time());
    return pol4.get_min();
    // if (ma_dist_coeffs[0] != 0) {
    //   FourthOrderPolynomial pol4(ma_dist_coeffs, get_start_time(), get_end_time());
    //   return pol4.get_min();
    // } else if (ma_dist_coeffs[1] != 0) {
    //   ma_dist_coeffs.erase(ma_dist_coeffs.begin());
    //   ThirdOrderPolynomial pol3(ma_dist_coeffs, get_start_time(), get_end_time());
    //   return pol3.get_min();
    // } else {
    //   ma_dist_coeffs.erase(ma_dist_coeffs.begin());
    //   ma_dist_coeffs.erase(ma_dist_coeffs.begin());
    //   SecondOrderPolynomial pol2(ma_dist_coeffs, get_start_time(), get_end_time());
    //   return pol2.get_min();
    // }
  }

  // Function to calculate a half of the Mahalanobis distance square for dynamic uncertainty
  double get_dy_mds(const Eigen::Vector3d& velocity,
                                  const Eigen::Vector3d& cov_diag) const {
    std::vector<Eigen::Vector3d> _coeffs = get_coeffs();
    double a1 = _coeffs[0][0], b1 = _coeffs[1][0], c1 = _coeffs[2][0];
    double a2 = _coeffs[0][1], b2 = _coeffs[1][1], c2 = _coeffs[2][1];
    double a3 = _coeffs[0][2], b3 = _coeffs[1][2], c3 = _coeffs[2][2];
    double vx = velocity.x(), vy = velocity.y(), vz = velocity.z();
    std::vector<double> ma_dist_coeffs = {
        // A fourth orders polynomial from PDF expanded equation in the paper
        (a1 * a1) / cov_diag.x() +
        (a2 * a2) / cov_diag.y() +
        (a3 * a3) / cov_diag.z(),
        a1 / cov_diag.x() * (b1 - vx) * 2.0 +
        a2 / cov_diag.y() * (b2 - vy) * 2.0 +
        a3 / cov_diag.z() * (b3 - vz) * 2.0,
        (a1 * c1 * 2.0 + (b1 - vx) * (b1 - vx)) / cov_diag.x() +
        (a2 * c2 * 2.0 + (b2 - vy) * (b2 - vy)) / cov_diag.y() +
        (a3 * c3 * 2.0 + (b3 - vz) * (b3 - vz)) / cov_diag.z(),
        c1 / cov_diag.x() * (b1 - vx) * 2.0 +
        c2 / cov_diag.y() * (b2 - vy) * 2.0 +
        c3 / cov_diag.z() * (b3 - vz) * 2.0,
        (c1 * c1) / cov_diag.x() +
        (c2 * c2) / cov_diag.y() +
        (c3 * c3) / cov_diag.z()};

    FourthOrderPolynomial pol4(ma_dist_coeffs, get_start_time(), get_end_time());
    return pol4.get_min();
    // if (ma_dist_coeffs[0] != 0) {
    //   FourthOrderPolynomial pol4(ma_dist_coeffs, get_start_time(), get_end_time());
    //   return pol4.get_min();
    // } else if (ma_dist_coeffs[1] != 0) {
    //   ma_dist_coeffs.erase(ma_dist_coeffs.begin());
    //   ThirdOrderPolynomial pol3(ma_dist_coeffs, get_start_time(), get_end_time());
    //   return pol3.get_min();
    // } else {
    //   ma_dist_coeffs.erase(ma_dist_coeffs.begin());
    //   ma_dist_coeffs.erase(ma_dist_coeffs.begin());
    //   SecondOrderPolynomial pol2(ma_dist_coeffs, get_start_time(), get_end_time());
    //   return pol2.get_min();
    // }
  }

  // Function to calculate joint PDF for three independent normal variables
  double get_du_collision_probability(const Eigen::Vector3d& mean,
                                   const PinholeCamera& camera,
                                   double& mahalanobis_distance) const {
    // Calculate the diagonal of the covariance matrix
    const Eigen::Vector3d cov_diag = camera.get_depth_noise_covariance_matrix(mean);

    // Calculate the Mahalanobis distance
    const double mds = get_du_mds(mean, cov_diag);
    mahalanobis_distance = std::min(mahalanobis_distance, sqrt(mds));


    // Calculate the denominator constant (2π)^(3/2) * sigma_x * sigma_y * sigma_z
    double denominator = pow(2 * M_PI, 1.5) * sqrt(cov_diag.x() * cov_diag.y() * cov_diag.z());
    return exp(-mds / 2) / denominator;
  }

  // Function to calculate joint PDF for three independent normal variables
  double get_dy_collision_probability(const Eigen::Vector3d& velocity,
                                   const PinholeCamera& camera,
                                   double& mahalanobis_distance) const {
    // Calculate the diagonal of the covariance matrix for dynamic position uncertainty
    const Eigen::Vector3d cov_diag = camera.get_dynamic_pos_covariance_matrix();

    // Calculate the Mahalanobis distance
    const double mds = get_dy_mds(velocity, cov_diag);
    mahalanobis_distance = std::min(mahalanobis_distance, sqrt(mds));


    // Calculate the denominator constant (2π)^(3/2) * sigma_x * sigma_y * sigma_z
    double denominator = pow(2 * M_PI, 1.5) * sqrt(cov_diag.x() * cov_diag.y() * cov_diag.z());
    return exp(-mds / 2) / denominator;
  }

  void get_depth_switching_points(std::vector<double>& roots) const override {
    std::vector<Eigen::Vector3d> derivative_coeffs = get_derivative_coeffs();
    std::vector<double> c = {derivative_coeffs[0].z(), derivative_coeffs[1].z()};

    // Compute the times at which the segment changes direction along the z-axis
    if (c[0] == 0) return;
    double root_t = -c[1] / c[0];
    if ((root_t > get_start_time()) && (root_t < get_end_time()))
      roots.push_back(root_t);
  }

  //! Returns the coefficient of the time derivative of the trajectory
  /*!
   * @return A 3D vector of 3D coefficients representing the time derivative
   * of the trajectory, where deriv_coeffs[0] is a 3D vector of coefficients
   * corresponding to t, deriv_coeffs[1] corresponds to the constant.
   */
  std::vector<Eigen::Vector3d> get_derivative_coeffs() const override {
    std::vector<Eigen::Vector3d> _coeffs = get_coeffs();
    std::vector<Eigen::Vector3d> deriv_coeffs;
    deriv_coeffs.reserve(2);
    for (int i = 0; i < 2; i++) {
      deriv_coeffs.push_back((2 - i) * _coeffs[i]);
    }
    return deriv_coeffs;
  }

  std::vector<int16_t> get_projection_boundary(const PinholeCamera& camera) const {
    // Finding time points at all terminals and extremes of the trajectory
    std::vector<double> extreme_and_terminal_times;
    extreme_and_terminal_times.push_back(get_start_time());
    extreme_and_terminal_times.push_back(get_end_time());

    // Now projecting the trajectory onto the focal plane and find the maximum
    // coordinates of the projection.

    // The trajectory's spatial coordinates equation in time
    // x = a1*t.^2 + b1*t + c1;
    // y = a2*t.^2 + b2*t + c2;
    // z = a3*t.^2 + b3*t + c3;

    // Calculate the projected coordinates using the pinhole camera model with f, cx, cy are given.
    // projected_x = x * f / z + cx;
    // projected_y = y * f / z + cy;

    // Calculate the first derivatives with respect to t
    // deriv_proj_x = diff(projected_x, t);
    // deriv_proj_y = diff(projected_y, t);

    std::vector<Eigen::Vector3d> _coeffs = get_coeffs();
    double a1 = _coeffs[0][0], b1 = _coeffs[1][0], c1 = _coeffs[2][0];
    double a2 = _coeffs[0][1], b2 = _coeffs[1][1], c2 = _coeffs[2][1];
    double a3 = _coeffs[0][2], b3 = _coeffs[1][2], c3 = _coeffs[2][2];

    // Symbolically computed by Matlab
    SecondOrderPolynomial deriv_proj_x(
      {a1 * b3 - a3 * b1, (a1 * c3 - a3 * c1) * 2.0, b1 * c3 - b3 * c1},
      get_start_time(), get_end_time());
    deriv_proj_x.solve_roots(extreme_and_terminal_times);
    SecondOrderPolynomial deriv_proj_y(
      {a2 * b3 - a3 * b2, (a2 * c3 - a3 * c2) * 2.0, b2 * c3 - b3 * c2},
      get_start_time(), get_end_time());
    deriv_proj_y.solve_roots(extreme_and_terminal_times);

    // Sorting time points in ascending order
    std::sort(extreme_and_terminal_times.begin(),
              extreme_and_terminal_times.end());

    // Computing coordinates for every extremes and terminals
    // Projecting them onto the focal plane and
    // adding safety margin to the projection.
    std::vector<double> projected_adding_margin_x_coords,
                        projected_adding_margin_y_coords;
    for (uint8_t i = 0; i < extreme_and_terminal_times.size(); i++) {
      // Projecting valid points onto the focal plane and adding safety margin
      double time_point = extreme_and_terminal_times[i];
      if ((time_point - get_start_time()) * (time_point - get_end_time()) <= 0) {
        camera.project_point_to_pixel_adding_margin(
          get_point(time_point), projected_adding_margin_x_coords,
          projected_adding_margin_y_coords);
      }
    }

    // Find the maximum and minimum elements
    int16_t min_projected_x = static_cast<int16_t>(*std::min_element(projected_adding_margin_x_coords.begin(), projected_adding_margin_x_coords.end()));
    int16_t max_projected_x = static_cast<int16_t>(*std::max_element(projected_adding_margin_x_coords.begin(), projected_adding_margin_x_coords.end()));
    int16_t min_projected_y = static_cast<int16_t>(*std::min_element(projected_adding_margin_y_coords.begin(), projected_adding_margin_y_coords.end()));
    int16_t max_projected_y = static_cast<int16_t>(*std::max_element(projected_adding_margin_y_coords.begin(), projected_adding_margin_y_coords.end()));
    min_projected_x = std::clamp(min_projected_x, static_cast<int16_t>(0), static_cast<int16_t>(camera.get_width()));
    min_projected_y = std::clamp(min_projected_y, static_cast<int16_t>(0), static_cast<int16_t>(camera.get_height()));
    max_projected_x = std::clamp(max_projected_x, static_cast<int16_t>(0), static_cast<int16_t>(camera.get_width()));
    max_projected_y = std::clamp(max_projected_y, static_cast<int16_t>(0), static_cast<int16_t>(camera.get_height()));
    return {min_projected_x, min_projected_y, max_projected_x, max_projected_y};
  }

  //! Returns the i-th 3D vector of coefficients.
  Eigen::Vector3d operator[](const int& i) const override { return Segment::operator[](i); }
};

}  // namespace common_math
