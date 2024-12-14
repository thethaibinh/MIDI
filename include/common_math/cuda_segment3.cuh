#pragma once
#include "cuda_segment.cuh"
#include "cuda_second_order_polynomial.cuh"

namespace common_math {

class CudaThirdOrderSegment : public CudaSegment {
public:
    __device__ __host__ CudaThirdOrderSegment(const CudaVector3d* coeffs, double start_time, double end_time)
        : CudaSegment(coeffs, 4, start_time, end_time) {}

    __device__ __host__ CudaVector3d get_point(double t) const override {
        t = cuda_clamp(t, _start_time, _end_time);
        return _coeffs[0] * (t * t * t) +
               _coeffs[1] * (t * t) +
               _coeffs[2] * t +
               _coeffs[3];
    }

    __device__ __host__ CudaVector3d get_start_point() const override {
        return get_point(_start_time);
    }

    __device__ __host__ CudaVector3d get_end_point() const override {
        return get_point(_end_time);
    }

    __device__ __host__ double get_axis_value(int i, double t) const {
        t = cuda_clamp(t, _start_time, _end_time);
        return _coeffs[0][i] * (t * t * t) +
               _coeffs[1][i] * (t * t) +
               _coeffs[2][i] * t +
               _coeffs[3][i];
    }

    __device__ __host__ double get_axis_start_value(int i) const {
        return get_axis_value(i, _start_time);
    }

    __device__ __host__ double get_axis_end_value(int i) const {
        return get_axis_value(i, _end_time);
    }

    __device__ __host__ uint8_t get_depth_switching_points(double* switching_points) const override {
        CudaVector3d deriv_coeffs[3];
        get_derivative_coeffs(deriv_coeffs);

        // Create second order polynomial from z-component of derivative
        double z_coeffs[3] = {
            deriv_coeffs[0].z,
            deriv_coeffs[1].z,
            deriv_coeffs[2].z
        };

        CudaSecondOrderPolynomial pol2(z_coeffs, _start_time, _end_time);
        uint8_t num_points = pol2.solve_roots(switching_points);
        return num_points;
    }

    __device__ __host__ void get_derivative_coeffs(CudaVector3d* deriv_coeffs) const override {
        deriv_coeffs[0] = _coeffs[0] * 3.0;  // t^2 term
        deriv_coeffs[1] = _coeffs[1] * 2.0;  // t term
        deriv_coeffs[2] = _coeffs[2];        // constant term
    }
};

} // namespace common_math