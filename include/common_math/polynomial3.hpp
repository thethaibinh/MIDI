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
#include "common_math/polynomial2.hpp"
// ROS base
#include <ros/console.h>
#include "ros/ros.h"

class ThirdOrderPolynomial : public Polynomial {
 public:
  // Constructor initializes the coefficients for a fourth-order polynomial
  ThirdOrderPolynomial(const std::vector<double>& coefficients,
                        const double& start_time, const double& end_time)
    : Polynomial(coefficients, start_time, end_time) {
    if (get_coeffs().size() != 4) {
      throw std::invalid_argument(
        "Cubic polynomials must have exactly 4 coefficients.");
    }
    if (get_start_time() > get_end_time()) {
      throw std::invalid_argument(
        "The end time must be greater than the start time.");
    }
  }

  // Evaluate the polynomial at a given point x
  double get_value(const double& x) const override {
    std::vector<double> _coeffs = get_coeffs();
    if (_coeffs.size() != 4) {
      throw std::invalid_argument(
        "Cubic polynomial must have exactly 4 coefficients.");
    }
    return _coeffs[0] * x * x * x + _coeffs[1] * x * x + _coeffs[2] * x +
           _coeffs[3];
  }

  void solve_derivative_roots(std::vector<double>& roots) const override {
    std::vector<double> diff_coeffs = get_derivative_coeffs();
    if (diff_coeffs.size() != 3) {
      throw std::invalid_argument(
        "The derivative of a cubic polynomial must have exactly 3 "
        "coefficients.");
    }
    SecondOrderPolynomial pol2(diff_coeffs, get_start_time(), get_end_time());
    pol2.solve_roots(roots);
  }

  //---------------------------------------------------------------------------
  // solve cubic equation x^3 + a*x^2 + b*x + c
  // x - array of size 3
  // In case 3 real roots: => x[0], x[1], x[2], return 3
  //         2 real roots: x[0], x[1],          return 2
  //         1 real root : x[0], x[1] ± i*x[2], return 1
  uint8_t solve_roots(std::vector<double>& roots) const override {
    std::vector<double> _coeffs = get_coeffs();
    if (_coeffs[0] == 0 || _coeffs.size() != 4) {
      throw std::invalid_argument(
        "solve_roots: Cubic polynomial must have exactly 4 coefficients.");
    }
    double a = _coeffs[1] / _coeffs[0];
    double b = _coeffs[2] / _coeffs[0];
    double c = _coeffs[3] / _coeffs[0];
    double a2 = a * a;
    double q = (a2 - 3 * b) / 9;
    double r = (a * (2 * a2 - 9 * b) + 27 * c) / 54;
    double r2 = r * r;
    double q3 = q * q * q;
    double A, B;
    if (r2 < q3) {
      double t = r / sqrt(q3);
      if (t < -1) t = -1;
      if (t > 1) t = 1;
      t = acos(t);
      a /= 3;
      q = -2 * sqrt(q);

      uint8_t num_roots = 0;
      roots.push_back(q * cos(t / 3) - a);
      num_roots++;
      if ((roots.back() < get_start_time()) || (roots.back() > get_end_time())) {
        roots.pop_back();
        num_roots--;
      }
      roots.push_back(q * cos((t + M_2PI) / 3) - a);
      num_roots++;
      if ((roots.back() < get_start_time()) || (roots.back() > get_end_time())) {
        roots.pop_back();
        num_roots--;
      }
      roots.push_back(q * cos((t - M_2PI) / 3) - a);
      num_roots++;
      if ((roots.back() < get_start_time()) || (roots.back() > get_end_time())) {
        roots.pop_back();
        num_roots--;
      }
      return num_roots;
    } else {
      A = -pow(fabs(r) + sqrt(r2 - q3), 1. / 3);
      if (r < 0) A = -A;
      B = (0 == A ? 0 : q / A);

      a /= 3;
      roots.push_back((A + B) - a);
      roots.push_back(-0.5 * (A + B) - a);
      roots.push_back(0.5 * sqrt(3.) * (A - B));
      uint8_t num_roots = 3;
      if (fabs(roots.back()) < eps) {
        roots.pop_back();
        num_roots--;
        if ((roots.back() < get_start_time()) || (roots.back() > get_end_time())) {
          roots.pop_back();
          num_roots--;
        }
        if ((roots.back() < get_start_time()) || (roots.back() > get_end_time())) {
          roots.pop_back();
          num_roots--;
        }
        return num_roots;
      }
      roots.pop_back();
      num_roots--;
      roots.pop_back();
      num_roots--;
      if ((roots.back() < get_start_time()) || (roots.back() > get_end_time())) {
        roots.pop_back();
        num_roots--;
      }
      return num_roots;
    }
  }
};