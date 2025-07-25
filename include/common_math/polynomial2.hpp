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
#include "common_math/polynomial.hpp"

class SecondOrderPolynomial : public Polynomial {
 public:
  // Constructor initializes the coefficients for a fourth-order polynomial
  SecondOrderPolynomial(const std::vector<double>& coefficients,
                        const double& start_time, const double& end_time)
    : Polynomial(coefficients, start_time, end_time) {
    if (get_coeffs().size() != 3) {
      throw std::invalid_argument(
        "Quadratic polynomials must have exactly 3 coefficients.");
    }
    if (get_start_time() > get_end_time()) {
      throw std::invalid_argument(
        "The end time must be greater than the start time.");
    }
  }

  // Evaluate the polynomial at a given point x
  double get_value(const double& x) const override {
    std::vector<double> _coeffs = get_coeffs();
    if (_coeffs.size() != 3) {
      throw std::invalid_argument(
        "Quadratic polynomials must have exactly 3 coefficients.");
    }
    return _coeffs[0] * x * x + _coeffs[1] * x + _coeffs[2];
  }

  void solve_derivative_roots(std::vector<double>& roots) const override {
    std::vector<double> diff_coeffs = get_derivative_coeffs();
    if (diff_coeffs[0] == 0 || diff_coeffs.size() != 2) {
      throw std::invalid_argument(
        "The derivative of this quadratic polynomial must be a linear "
        "polynomial.");
    } else {
      roots.push_back(-diff_coeffs[1] / diff_coeffs[0]);
      if ((roots.back() < get_start_time()) || (roots.back() > get_end_time())) {
        roots.pop_back();
      }
    }
  }

  uint8_t solve_roots(std::vector<double>& roots) const override {
    std::vector<double> coeffs = get_coeffs();
    if (coeffs.size() != 3) {
      throw std::invalid_argument(
        "Quadratic polynomials must have exactly 3 coefficients.");
    }
    double A = coeffs[0];
    double B = coeffs[1];
    double C = coeffs[2];
    // Contingency: if A = 0, not a quadratic = linear
    // initialise counters for real and imaginary roots
    if (A == 0) {
      // If B is zero then we have a NaN
      if (B == 0) return 0;
      roots.push_back(-1.0 * C / B);
      uint8_t num_roots = 1;
      if ((roots.back() < get_start_time()) || (roots.back() > get_end_time())) {
        roots.pop_back();
        num_roots = 0;
      }
      return num_roots;
    }

    double discriminant = (B * B) - (4 * A * C);

    // Cannot do imaginary numbers, yet
    if (discriminant < 0) return 0;

    // This avoids a cancellation of errors. See
    // http://en.wikipedia.org/wiki/Quadratic_equation#Floating_point_implementation
    double t = -0.5 * (B + ((B < 0) ? -1 : 1) * std::sqrt(discriminant));

    roots.push_back(t / A);
    if ((roots.back() < get_start_time()) || (roots.back() > get_end_time())) {
      roots.pop_back();
    }
    roots.push_back(C / t);
    if ((roots.back() < get_start_time()) || (roots.back() > get_end_time())) {
      roots.pop_back();
    }
    return 2;
  }
};
