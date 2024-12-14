#pragma once
#include "cuda_polynomial_base.cuh"
#include "cuda_second_order_polynomial.cuh"

namespace common_math {

class CudaThirdOrderPolynomial : public CudaPolynomialBase {
public:
    __device__ __host__ CudaThirdOrderPolynomial(const double* coeffs, double t0, double tf)
        : CudaPolynomialBase(coeffs, 4, t0, tf) {}

    __device__ __host__ double get_value(double x) const override {
        return _coeffs[0] * x * x * x + _coeffs[1] * x * x + _coeffs[2] * x + _coeffs[3];
    }

    __device__ __host__ void get_derivative_coeffs(double* diff_coeffs) const override {
        diff_coeffs[0] = 3.0 * _coeffs[0];
        diff_coeffs[1] = 2.0 * _coeffs[1];
        diff_coeffs[2] = _coeffs[2];
    }

    __device__ __host__ uint8_t solve_derivative_roots(double* roots) const override {
        double diff_coeffs[3];
        get_derivative_coeffs(diff_coeffs);
        CudaSecondOrderPolynomial pol2(diff_coeffs, _t0, _tf);
        return pol2.solve_roots(roots);
    }

    __device__ __host__ uint8_t solve_roots(double* roots) const override {
        if(_coeffs[0] == 0) {
            // Handle degenerate case - fall back to quadratic
            double quad_coeffs[3] = {_coeffs[1], _coeffs[2], _coeffs[3]};
            CudaSecondOrderPolynomial quad(quad_coeffs, _t0, _tf);
            return quad.solve_roots(roots);
        }

        double a = _coeffs[1] / _coeffs[0];
        double b = _coeffs[2] / _coeffs[0];
        double c = _coeffs[3] / _coeffs[0];

        double a2 = a * a;
        double q = (a2 - 3 * b) / 9;
        double r = (a * (2 * a2 - 9 * b) + 27 * c) / 54;
        double r2 = r * r;
        double q3 = q * q * q;

        if(r2 < q3) {
            // Three real roots case
            double t = r / sqrt(q3);
            t = cuda_clamp(t, -1.0, 1.0);  // Use clamp for numerical stability
            t = acos(t);
            a /= 3;
            q = -2 * sqrt(q);

            uint8_t num_roots = 0;
            double roots_temp[3] = {
                q * cos(t / 3) - a,
                q * cos((t + CUDA_2PI) / 3) - a,
                q * cos((t - CUDA_2PI) / 3) - a
            };

            // Filter roots within bounds
            for(int i = 0; i < 3; i++) {
                if(roots_temp[i] >= _t0 && roots_temp[i] <= _tf) {
                    roots[num_roots++] = roots_temp[i];
                }
            }
            return num_roots;
        } else {
            // One real root case
            double A = -cuda_sign(r) * pow(fabs(r) + sqrt(r2 - q3), 1.0 / 3.0);
            double B = (A == 0 ? 0 : q / A);

            double root = (A + B) - a/3;
            if(root >= _t0 && root <= _tf) {
                roots[0] = root;
                return 1;
            }
            return 0;
        }
    }
};

} // namespace common_math