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
#include <cmath>
#include <iostream>
#include <vector>
// Eigen catkin
#include <Eigen/Dense>

const double PI = 3.141592653589793238463L;
const double M_2PI = 2 * PI;
const double EPS = 1e-12;

class Polynomial {
 private:
  std::vector<double> _coeffs;  // Coefficients of the polynomial
  double _start_time, _end_time;

 public:
  // Virtual destructor
  virtual ~Polynomial() = default;
  // Constructor to initialize coefficients
  Polynomial(const std::vector<double>& coefficients, const double& start_time,
             const double& end_time)
    : _coeffs(coefficients), _start_time(start_time), _end_time(end_time) {
    if (_start_time > _end_time) {
      throw std::invalid_argument(
        "The end time must be greater than the start time.");
    }
  }

  virtual uint8_t solve_roots(std::vector<double>& roots) const = 0;

  virtual void solve_derivative_roots(std::vector<double>& roots) const = 0;

  std::vector<double> get_extremes_and_terminals() const {
    // Determining polynomial terminals and extremes
    std::vector<double> deriv_roots_and_terminals;
    deriv_roots_and_terminals.push_back(get_start_time());
    deriv_roots_and_terminals.push_back(get_end_time());
    solve_derivative_roots(deriv_roots_and_terminals);
    std::sort(deriv_roots_and_terminals.begin(),
              deriv_roots_and_terminals.end());
    // Computing extremes and terminals
    std::vector<double> extreme_and_terminal_vals;
    for (unsigned i = 0; i < deriv_roots_and_terminals.size(); i++) {
      if (deriv_roots_and_terminals[i] < get_start_time()) {
        // Skip root if it's before start time
        continue;
      } else if (deriv_roots_and_terminals[i] <= get_end_time()) {
        extreme_and_terminal_vals.push_back(get_value(deriv_roots_and_terminals[i]));
      }
    }
    return extreme_and_terminal_vals;
  }

  double get_max() const {
    std::vector<double> vals = get_extremes_and_terminals();
    return *std::max_element(vals.begin(), vals.end());
  }
  double get_min() const {
    std::vector<double> vals = get_extremes_and_terminals();
    return *std::min_element(vals.begin(), vals.end());
  }

  virtual double get_value(const double& t) const = 0;

  double get_start_value() const { return get_value(get_start_time()); }
  double get_end_value() const { return get_value(get_end_time()); }

  //! Return the 1D vector of coefficients
  std::vector<double> get_coeffs() const { return _coeffs; }

  //! Return the end time of the trajectory
  double get_start_time() const { return _start_time; }

  //! Return the end time of the trajectory
  double get_end_time() const { return _end_time; }

  //! Return the trajectory duration
  double get_duration() const { return _end_time - _start_time; }

  std::vector<double> get_derivative_coeffs() const {
    // Polynomial coefficient vector
    std::vector<double> coeffs = get_coeffs();
    // Derivative coefficient vector
    std::vector<double> diff_coeffs;
    uint8_t diff_coeffs_size = coeffs.size() - 1;
    diff_coeffs.reserve(diff_coeffs_size);
    for (int i = 0; i < diff_coeffs_size; i++) {
      diff_coeffs.push_back((diff_coeffs_size - i) * coeffs[i]);
    }
    return diff_coeffs;
  }
};
