#pragma once
#include "cuda_polynomial_base.cuh"
#include "cuda_third_order_polynomial.cuh"

namespace common_math {

class CudaFourthOrderPolynomial : public CudaPolynomialBase {
public:
    __device__ __host__ CudaFourthOrderPolynomial(const double* coeffs, double t0, double tf)
        : CudaPolynomialBase(coeffs, 5, t0, tf) {}

    __device__ __host__ double get_value(double x) const override {
        return _coeffs[0] * x * x * x * x + _coeffs[1] * x * x * x +
               _coeffs[2] * x * x + _coeffs[3] * x + _coeffs[4];
    }

    __device__ __host__ void get_derivative_coeffs(double* diff_coeffs) const override {
        diff_coeffs[0] = 4.0 * _coeffs[0];
        diff_coeffs[1] = 3.0 * _coeffs[1];
        diff_coeffs[2] = 2.0 * _coeffs[2];
        diff_coeffs[3] = _coeffs[3];
    }

    __device__ __host__ uint8_t solve_derivative_roots(double* roots) const override {
        double diff_coeffs[4];
        get_derivative_coeffs(diff_coeffs);
        CudaThirdOrderPolynomial pol3(diff_coeffs, _t0, _tf);
        return pol3.solve_roots(roots);
    }

    __device__ __host__ uint8_t solve_roots(double* roots) const override {
        // Handle degenerate cases first
        if(fabs(_coeffs[0]) < CUDA_EPS) {
            // Reduce to cubic equation if leading coefficient is nearly zero
            double cubic_coeffs[4] = {_coeffs[1], _coeffs[2], _coeffs[3], _coeffs[4]};
            CudaThirdOrderPolynomial cubic(cubic_coeffs, _t0, _tf);
            return cubic.solve_roots(roots);
        }

        // Normalize coefficients
        double a = _coeffs[1] / _coeffs[0];
        double b = _coeffs[2] / _coeffs[0];
        double c = _coeffs[3] / _coeffs[0];
        double d = _coeffs[4] / _coeffs[0];

        // Handle special case: biquadratic equation (x⁴ + px² + q = 0)
        if(fabs(a) < CUDA_EPS && fabs(c) < CUDA_EPS) {
            return solve_biquadratic(b, d, roots);
        }

        // Cubic resolvent with improved numerical stability
        double a2 = a * a;
        double a3 = -b;
        double b3 = a * c - 4.0 * d;
        double c3 = -a2 * d - c * c + 4.0 * b * d;

        // Solve cubic resolvent with full range
        double resolvent_coeffs[4] = {1.0, a3, b3, c3};
        double x3[3];
        CudaThirdOrderPolynomial resolvent(resolvent_coeffs, -INFINITY, INFINITY);  // Use full range for resolvent
        uint8_t iZeroes = resolvent.solve_roots(x3);

        // Find y with maximum absolute value for better numerical stability
        double y = x3[0];
        if(iZeroes != 1) {
            if(fabs(x3[1]) > fabs(y)) y = x3[1];
            if(fabs(x3[2]) > fabs(y)) y = x3[2];
        }

        // Handle nearly equal roots case
        double D = y * y - 4.0 * d;
        double q1, q2, p1, p2;

        if(fabs(D) < CUDA_EPS) {
            // Case of repeated roots
            q1 = q2 = y * 0.5;
            D = a2 - 4.0 * (b - y);

            if(fabs(D) < CUDA_EPS) {
                // Four equal roots case
                p1 = p2 = a * 0.5;
            } else {
                double sqD = sqrt(D);
                p1 = (a + sqD) * 0.5;
                p2 = (a - sqD) * 0.5;
            }
        } else {
            // Regular case
            double sqD = sqrt(D);
            q1 = (y + sqD) * 0.5;
            q2 = (y - sqD) * 0.5;

            // Handle division by small numbers
            double denom = q1 - q2;
            if(fabs(denom) < CUDA_EPS) {
                // Alternative computation for nearly equal roots
                p1 = p2 = a * 0.5;
            } else {
                p1 = (a * q1 - c) / denom;
                p2 = (c - a * q2) / denom;
            }
        }

        return solve_quadratic_pair(p1, q1, p2, q2, roots);
    }

    // Helper method for biquadratic case
    __device__ __host__ uint8_t solve_biquadratic(double p, double q, double* roots) const {
        double quad_roots[2];
        double quad_coeffs[3] = {1.0, p, q};
        CudaSecondOrderPolynomial quad(quad_coeffs, -INFINITY, INFINITY);
        uint8_t n = quad.solve_roots(quad_roots);

        uint8_t num_roots = 0;
        for(uint8_t i = 0; i < n; i++) {
            if(quad_roots[i] >= 0) {
                double sqrt_root = sqrt(quad_roots[i]);
                if(sqrt_root >= _t0 && sqrt_root <= _tf) {
                    roots[num_roots++] = sqrt_root;
                }
                if(-sqrt_root >= _t0 && -sqrt_root <= _tf) {
                    roots[num_roots++] = -sqrt_root;
                }
            }
        }
        return num_roots;
    }

    // Helper method for solving the two quadratic equations
    __device__ __host__ uint8_t solve_quadratic_pair(
        double p1, double q1, double p2, double q2, double* roots) const {

        uint8_t num_roots = 0;
        double temp_roots[4];

        // First quadratic: x² + p1x + q1 = 0
        double D1 = p1 * p1 - 4.0 * q1;
        if(D1 >= -CUDA_EPS) {  // Allow slightly negative discriminant due to numerical errors
            if(D1 < 0) D1 = 0;
            double sqD = sqrt(D1);
            temp_roots[num_roots++] = (-p1 + sqD) * 0.5;
            temp_roots[num_roots++] = (-p1 - sqD) * 0.5;
        }

        // Second quadratic: x² + p2x + q2 = 0
        double D2 = p2 * p2 - 4.0 * q2;
        if(D2 >= -CUDA_EPS) {
            if(D2 < 0) D2 = 0;
            double sqD = sqrt(D2);
            temp_roots[num_roots++] = (-p2 + sqD) * 0.5;
            temp_roots[num_roots++] = (-p2 - sqD) * 0.5;
        }

        // Filter roots within bounds and handle duplicates
        uint8_t valid_roots = 0;
        for(uint8_t i = 0; i < num_roots; i++) {
            if(temp_roots[i] >= _t0 && temp_roots[i] <= _tf) {
                roots[valid_roots++] = temp_roots[i];
            }
        }
        // Handle duplicates
        for(uint8_t i = 0; i < valid_roots; i++) {
            for(uint8_t j = i + 1; j < valid_roots; j++) {
                if(fabs(roots[i] - roots[j]) < CUDA_EPS) {
                    valid_roots--;
                    for(uint8_t k = j; k < valid_roots; k++) {
                        roots[k] = roots[k + 1];
                    }
                }
            }
        }
        return valid_roots;
    }
};

} // namespace common_math