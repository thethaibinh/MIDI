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
#include "common_math/polynomial3.hpp"

class FourthOrderPolynomial : public Polynomial {
 public:
  // Constructor initializes the coefficients for a fourth-order polynomial
  FourthOrderPolynomial(const std::vector<double>& coefficients,
                        const double& start_time, const double& end_time)
    : Polynomial(coefficients, start_time, end_time) {
    if (get_coeffs().size() != 5) {
      throw std::invalid_argument(
        "Fourth-order polynomial must have exactly 5 coefficients.");
    }
    if (get_start_time() > get_end_time()) {
      throw std::invalid_argument(
        "The end time must be greater than the start time.");
    }
  }

  // Evaluate the polynomial at a given point x
  double get_value(const double& x) const override {
    std::vector<double> _coeffs = get_coeffs();
    if (_coeffs.size() != 5) {
      throw std::invalid_argument(
        "Quartic polynomials must have exactly 5 coefficients.");
    }
    return _coeffs[0] * x * x * x * x + _coeffs[1] * x * x * x +
           _coeffs[2] * x * x + _coeffs[3] * x + _coeffs[4];
  }

  void solve_derivative_roots(std::vector<double>& roots) const override {
    std::vector<double> diff_coeffs = get_derivative_coeffs();
    if (diff_coeffs.size() != 4) {
      throw std::invalid_argument(
        "The derivative of a quartic polynomial must have exactly 4 "
        "coefficients.");
    }
    ThirdOrderPolynomial pol3(diff_coeffs, get_start_time(), get_end_time());
    pol3.solve_roots(roots);
  }

  //---------------------------------------------------------------------------
  // solve quartic equation x^4 + a*x^3 + b*x^2 + c*x + d
  // Attention - this function returns dynamically allocated array. It has to be
  // released afterwards.
  uint8_t solve_roots(std::vector<double>& roots) const override {
    std::vector<double> _coeffs = get_coeffs();
    if (_coeffs.size() != 5) {
      throw std::invalid_argument(
        "Quartic polynomials must have exactly 5 coefficients.");
    }
    double a = _coeffs[1] / _coeffs[0];
    double b = _coeffs[2] / _coeffs[0];
    double c = _coeffs[3] / _coeffs[0];
    double d = _coeffs[4] / _coeffs[0];
    double a3 = -b;
    double b3 = a * c - 4. * d;
    double c3 = -a * a * d - c * c + 4. * b * d;

    // initialise counters for real and imaginary roots
    int rCnt = 0;
    // cubic resolvent
    // y^3 − b*y^2 + (ac−4d)*y − a^2*d−c^2+4*b*d = 0

    std::vector<double> coeffs = {1, a3, b3, c3};
    ThirdOrderPolynomial pol3(coeffs, get_start_time(), get_end_time());
    std::vector<double> x3;
    unsigned int iZeroes = pol3.solve_roots(x3);

    double q1, q2, p1, p2, D, sqD, y;

    y = x3[0];
    // The essence - choosing Y with maximal absolute value.
    if (iZeroes != 1) {
      if (fabs(x3[1]) > fabs(y)) y = x3[1];
      if (fabs(x3[2]) > fabs(y)) y = x3[2];
    }

    // h1+h2 = y && h1*h2 = d  <=>  h^2 -y*h + d = 0    (h === q)

    D = y * y - 4 * d;
    if (fabs(D) < eps)  // in other words - D==0
    {
      q1 = q2 = y * 0.5;
      // g1+g2 = a && g1+g2 = b-y   <=>   g^2 - a*g + b-y = 0    (p === g)
      D = a * a - 4 * (b - y);
      if (fabs(D) < eps)  // in other words - D==0
        p1 = p2 = a * 0.5;

      else {
        sqD = sqrt(D);
        p1 = (a + sqD) * 0.5;
        p2 = (a - sqD) * 0.5;
      }
    } else {
      sqD = sqrt(D);
      q1 = (y + sqD) * 0.5;
      q2 = (y - sqD) * 0.5;
      // g1+g2 = a && g1*h2 + g2*h1 = c       ( && g === p )  Krammer
      p1 = (a * q1 - c) / (q1 - q2);
      p2 = (c - a * q2) / (q1 - q2);
    }


    // solving quadratic eq. - x^2 + p1*x + q1 = 0
    D = p1 * p1 - 4 * q1;
    if (!(D < 0.0)) {
      // real roots filled from left
      sqD = sqrt(D);
      roots.push_back((-p1 + sqD) * 0.5);
      ++rCnt;
      roots.push_back((-p1 - sqD) * 0.5);
      ++rCnt;
    }

    // solving quadratic eq. - x^2 + p2*x + q2 = 0
    D = p2 * p2 - 4 * q2;
    if (!(D < 0.0)) {
      sqD = sqrt(D);
      roots.push_back((-p2 + sqD) * 0.5);
      ++rCnt;
      roots.push_back((-p2 - sqD) * 0.5);
      ++rCnt;
    }

    return rCnt;
  }
};
