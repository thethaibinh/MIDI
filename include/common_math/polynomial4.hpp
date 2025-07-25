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
  uint8_t solve_roots(std::vector<double> &roots) const override
  {
    std::vector<double> _coeffs = get_coeffs();
    if (_coeffs.size() != 5)
    {
      throw std::invalid_argument(
          "Quartic polynomials must have exactly 5 coefficients.");
    }

    // Handle degenerate cases first
    if (fabs(_coeffs[0]) < EPS)
    {
      // Reduce to cubic equation if leading coefficient is nearly zero
      const std::vector<double> cubic_coeffs = {_coeffs[1], _coeffs[2], _coeffs[3], _coeffs[4]};
      ThirdOrderPolynomial cubic(cubic_coeffs, get_start_time(), get_end_time());
      return cubic.solve_roots(roots);
    }

    // Normalize coefficients
    double a = _coeffs[1] / _coeffs[0];
    double b = _coeffs[2] / _coeffs[0];
    double c = _coeffs[3] / _coeffs[0];
    double d = _coeffs[4] / _coeffs[0];

    // Handle special case: biquadratic equation (x⁴ + px² + q = 0)
    if (fabs(a) < EPS && fabs(c) < EPS)
    {
      return solve_biquadratic(b, d, roots);
    }

    // Cubic resolvent coefficients
    double a2 = a * a;
    double a3 = -b;
    double b3 = a * c - 4.0 * d;
    double c3 = -a2 * d - c * c + 4.0 * b * d;

    // Solve cubic resolvent
    std::vector<double> resolvent_coeffs = {1.0, a3, b3, c3};
    ThirdOrderPolynomial resolvent(resolvent_coeffs, -INFINITY, INFINITY);
    std::vector<double> x3;
    uint8_t iZeroes = resolvent.solve_roots(x3);

    // Find y with maximum absolute value for better numerical stability
    double y = x3[0];
    if (iZeroes != 1)
    {
      if (fabs(x3[1]) > fabs(y))
        y = x3[1];
      if (fabs(x3[2]) > fabs(y))
        y = x3[2];
    }

    // Handle nearly equal roots case
    double D = y * y - 4.0 * d;
    double q1, q2, p1, p2;

    if (fabs(D) < EPS)
    {
      // Case of repeated roots
      q1 = q2 = y * 0.5;
      D = a2 - 4.0 * (b - y);

      if (fabs(D) < EPS)
      {
        // Four equal roots case
        p1 = p2 = a * 0.5;
      }
      else
      {
        double sqD = sqrt(D);
        p1 = (a + sqD) * 0.5;
        p2 = (a - sqD) * 0.5;
      }
    }
    else
    {
      // Regular case
      double sqD = sqrt(D);
      q1 = (y + sqD) * 0.5;
      q2 = (y - sqD) * 0.5;

      // Handle division by small numbers
      double denom = q1 - q2;
      if (fabs(denom) < EPS)
      {
        // Alternative computation for nearly equal roots
        p1 = p2 = a * 0.5;
      }
      else
      {
        p1 = (a * q1 - c) / denom;
        p2 = (c - a * q2) / denom;
      }
    }

    return solve_quadratic_pair(p1, q1, p2, q2, roots);
  }

private:
  // Helper method for biquadratic case
  uint8_t solve_biquadratic(const double &p, const double &q, std::vector<double> &roots) const
  {
    std::vector<double> quad_coeffs = {1.0, p, q};
    SecondOrderPolynomial quad(quad_coeffs, -INFINITY, INFINITY);
    std::vector<double> quad_roots;
    uint8_t n = quad.solve_roots(quad_roots);

    uint8_t num_roots = 0;
    for (uint8_t i = 0; i < n; i++)
    {
      if (quad_roots[i] >= 0)
      {
        double sqrt_root = sqrt(quad_roots[i]);
        if (sqrt_root >= get_start_time() && sqrt_root <= get_end_time())
        {
          roots.push_back(sqrt_root);
          num_roots++;
        }
        if (-sqrt_root >= get_start_time() && -sqrt_root <= get_end_time())
        {
          roots.push_back(-sqrt_root);
          num_roots++;
        }
      }
    }
    return num_roots;
  }

  // Helper method for solving the two quadratic equations
  uint8_t solve_quadratic_pair(const double &p1, const double &q1,
                               const double &p2, const double &q2,
                               std::vector<double> &roots) const
  {
    std::vector<double> temp_roots;

    // First quadratic: x² + p1x + q1 = 0
    double D1 = p1 * p1 - 4.0 * q1;
    if (D1 >= -EPS)
    { // Allow slightly negative discriminant due to numerical errors
      if (D1 < 0)
        D1 = 0;
      double sqD = sqrt(D1);
      temp_roots.push_back((-p1 + sqD) * 0.5);
      temp_roots.push_back((-p1 - sqD) * 0.5);
    }

    // Second quadratic: x² + p2x + q2 = 0
    double D2 = p2 * p2 - 4.0 * q2;
    if (D2 >= -EPS)
    {
      if (D2 < 0)
        D2 = 0;
      double sqD = sqrt(D2);
      temp_roots.push_back((-p2 + sqD) * 0.5);
      temp_roots.push_back((-p2 - sqD) * 0.5);
    }

    // Filter roots within bounds and handle duplicates
    for (const double &root : temp_roots)
    {
      if (root >= get_start_time() && root <= get_end_time())
      {
        // Check if this root is already in the results (within epsilon)
        bool is_duplicate = false;
        for (const double &existing_root : roots)
        {
          if (fabs(root - existing_root) < EPS)
          {
            is_duplicate = true;
            break;
          }
        }
        if (!is_duplicate)
        {
          roots.push_back(root);
        }
      }
    }
    return roots.size();
  }
};
