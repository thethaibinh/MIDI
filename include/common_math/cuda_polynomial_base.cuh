#pragma once
#include "cuda_math.cuh"

namespace common_math {

const double CUDA_PI = 3.141592653589793238463L;
const double CUDA_2PI = 2 * CUDA_PI;
const double CUDA_EPS = 1e-12;

class CudaPolynomialBase {
public:
    __device__ __host__ CudaPolynomialBase(const double* coeffs, uint8_t size, double t0, double tf)
        : _size(size), _t0(t0), _tf(tf) {
        for(uint8_t i = 0; i < size; i++) {
            _coeffs[i] = coeffs[i];
        }
    }

    __device__ __host__ virtual ~CudaPolynomialBase() {}

    __device__ __host__ virtual double get_value(double t) const = 0;
    __device__ __host__ virtual void get_derivative_coeffs(double* diff_coeffs) const = 0;
    __device__ __host__ virtual uint8_t solve_derivative_roots(double* roots) const = 0;
    __device__ __host__ virtual uint8_t solve_roots(double* roots) const = 0;

    __device__ __host__ double get_start_time() const { return _t0; }
    __device__ __host__ double get_end_time() const { return _tf; }
    __device__ __host__ double get_duration() const { return _tf - _t0; }

    __device__ __host__ double get_min() const {
      double roots[5];  // max extrema and terminal time
      uint8_t num_extrema_and_terminals = 0;

      // Add start time
      roots[num_extrema_and_terminals++] = _t0;

      // Get derivative roots
      uint8_t num_extrema = solve_derivative_roots(roots + 1);

      num_extrema_and_terminals += num_extrema;

      // Add end time
      roots[num_extrema_and_terminals++] = _tf;

      // Find minimum value
      double min_val = INFINITY;
      for (uint8_t i = 0; i < num_extrema_and_terminals; i++) {
        if (roots[i] >= _t0 && roots[i] <= _tf) {
          min_val = cuda_min(min_val, get_value(roots[i]));
        }
      }
      return min_val;
    }

protected:
    double _coeffs[5];  // Max size for 4th order polynomial
    int _size;
    double _t0, _tf;
};

} // namespace common_math