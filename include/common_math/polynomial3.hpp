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
    if (_coeffs.size() != 4) {
      throw std::invalid_argument(
        "solve_roots: Cubic polynomial must have exactly 4 coefficients.");
    }
    // Handle degenerate cases first
    if(fabs(_coeffs[0]) < EPS) {
      // Reduce to cubic equation if leading coefficient is nearly zero
      const std::vector<double> cubic_coeffs = {_coeffs[1], _coeffs[2], _coeffs[3]};
      SecondOrderPolynomial cubic(cubic_coeffs, get_start_time(), get_end_time());
      return cubic.solve_roots(roots);
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
      if (fabs(roots.back()) < EPS) {
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