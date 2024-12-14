#pragma once
#include "cuda_polynomial_base.cuh"

namespace common_math {

class CudaSecondOrderPolynomial : public CudaPolynomialBase {
public:
    __device__ __host__ CudaSecondOrderPolynomial(const double* coeffs, double t0, double tf)
        : CudaPolynomialBase(coeffs, 3, t0, tf) {}

    __device__ __host__ double get_value(double x) const override {
        return _coeffs[0] * x * x + _coeffs[1] * x + _coeffs[2];
    }

    __device__ __host__ void get_derivative_coeffs(double* diff_coeffs) const override {
        diff_coeffs[0] = 2.0 * _coeffs[0];
        diff_coeffs[1] = _coeffs[1];
    }

    __device__ __host__ uint8_t solve_derivative_roots(double* roots) const override {
        double diff_coeffs[2];
        get_derivative_coeffs(diff_coeffs);

        if(diff_coeffs[0] == 0) {
            return 0;
        }

        double root = -diff_coeffs[1] / diff_coeffs[0];
        if(root >= _t0 && root <= _tf) {
            roots[0] = root;
            return 1;
        } else {
            return 0;
        }
    }

    __device__ __host__ uint8_t solve_roots(double* roots) const override {
        double A = _coeffs[0];
        double B = _coeffs[1];
        double C = _coeffs[2];

        if(A == 0) {
            if(B == 0) return 0;
            double root = -C / B;
            if(root >= _t0 && root <= _tf) {
                roots[0] = root;
                return 1;
            }
            return 0;
        }

        double discriminant = B * B - 4 * A * C;
        if(discriminant < 0) return 0;

        double t = -0.5 * (B + ((B < 0) ? -1 : 1) * sqrt(discriminant));
        uint8_t num_roots = 0;

        double root1 = t / A;
        if(root1 >= _t0 && root1 <= _tf) {
            roots[num_roots++] = root1;
        }

        double root2 = C / t;
        if(root2 >= _t0 && root2 <= _tf) {
            roots[num_roots++] = root2;
        }

        return num_roots;
    }
};

} // namespace common_math