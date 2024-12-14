#pragma once
#include "cuda_math.cuh"

namespace common_math {

class CudaSegment {
public:
    __device__ __host__ CudaSegment(const CudaVector3d* coeffs, int num_coeffs, double start_time, double end_time)
        : _start_time(start_time), _end_time(end_time), _num_coeffs(num_coeffs) {
        for(int i = 0; i < num_coeffs; i++) {
            _coeffs[i] = coeffs[i];
        }
    }

    __device__ __host__ virtual ~CudaSegment() {}

    __device__ __host__ virtual CudaVector3d get_point(double t) const = 0;
    __device__ __host__ virtual CudaVector3d get_start_point() const = 0;
    __device__ __host__ virtual CudaVector3d get_end_point() const = 0;
    __device__ __host__ virtual void get_derivative_coeffs(CudaVector3d* deriv_coeffs) const = 0;

    __device__ __host__ double get_start_time() const { return _start_time; }
    __device__ __host__ double get_end_time() const { return _end_time; }
    __device__ __host__ double get_duration() const { return _end_time - _start_time; }

    __device__ __host__ CudaVector3d get_coeff(int i) const {
        if(i >= 0 && i < _num_coeffs) {
            return _coeffs[i];
        }
        return CudaVector3d();
    }

    __device__ __host__ uint8_t get_coeffs(CudaVector3d* coeffs) const {
        for(int i = 0; i < _num_coeffs; i++) {
            coeffs[i] = _coeffs[i];
        }
        return _num_coeffs;
    }

    __device__ __host__ bool is_monotonically_increasing_depth() const {
        double switching_points[6];
        uint8_t num_points = get_depth_switching_points(switching_points);
        if (num_points >= 1)
            return false;
        else
            return true;
    }

    __device__ __host__ virtual uint8_t get_depth_switching_points(double* switching_points) const = 0;

    __device__ __host__ uint8_t get_depth_switching_points_and_terminals(double* switching_points_and_terminals) const {
        switching_points_and_terminals[0] = get_start_time();
        uint8_t num_points = get_depth_switching_points(switching_points_and_terminals + 1);
        switching_points_and_terminals[num_points + 1] = get_end_time();
        return num_points + 2;
    }
protected:
    CudaVector3d _coeffs[4];  // Max 4 coefficients for cubic
    int _num_coeffs;
    double _start_time, _end_time;
};

} // namespace common_math