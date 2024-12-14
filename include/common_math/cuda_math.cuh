#pragma once
#include <cuda_runtime.h>
#include <algorithm>
// #include <Eigen/Dense>

namespace common_math {

struct CudaVector3d {
    double x, y, z;

    __device__ __host__ CudaVector3d() : x(0), y(0), z(0) {}
    __device__ __host__ CudaVector3d(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    // __device__ __host__ CudaVector3d(const Eigen::Vector3d& v) : x(v.x()), y(v.y()), z(v.z()) {}

    __device__ __host__ double norm() const {
        return sqrt(square_norm());
    }

    __device__ __host__ double square_norm() const {
        return x*x + y*y + z*z;
    }

    __device__ __host__ double dot(const CudaVector3d& other) const {
        return x*other.x + y*other.y + z*other.z;
    }

    __device__ __host__ double operator[](const int& i) const {
        switch(i) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            default: return 0;
        }
    }
};

// Core CUDA math utilities
__device__ __host__ inline double cuda_min(double a, double b) {
    #ifdef __CUDA_ARCH__
    return fmin(a, b);
    #else
    return std::min(a, b);
    #endif
}

__device__ __host__ inline double cuda_max(double a, double b) {
    #ifdef __CUDA_ARCH__
    return fmax(a, b);
    #else
    return std::max(a, b);
    #endif
}

__device__ __host__ inline double cuda_clamp(double val, double min, double max) {
    return cuda_min(cuda_max(val, min), max);
}

__device__ __host__ inline double cuda_sign(double val) {
    return val > 0 ? 1 : val < 0 ? -1 : 0;
}

__device__ __host__ inline double cuda_abs(double val) {
    return val > 0 ? val : -val;
}

// __device__ __host__ inline void cuda_sort(double* arr, int n) {
//     std::sort(arr, arr + n);
// }

// __device__ __host__ inline double cuda_min_array(double* arr, int n) {
//     return *std::min_element(arr, arr + n);
// }

// __device__ __host__ inline double cuda_max_array(double* arr, int n) {
//     return *std::max_element(arr, arr + n);
// }

// Atomic operations
__device__ __host__ inline double atomicMin(double* address, double val) {
    double old = *address;
    *address = cuda_min(old, val);
    return old;
}

__device__ __host__ inline double atomicMax(double* address, double val) {
    double old = *address;
    *address = cuda_max(old, val);
    return old;
}

// Vector operations
__device__ __host__ inline CudaVector3d operator+(const CudaVector3d& a, const CudaVector3d& b) {
    return CudaVector3d(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __host__ inline CudaVector3d operator-(const CudaVector3d& a, const CudaVector3d& b) {
    return CudaVector3d(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __host__ inline CudaVector3d operator*(const CudaVector3d& v, double s) {
    return CudaVector3d(v.x * s, v.y * s, v.z * s);
}

__device__ __host__ inline CudaVector3d operator*(double s, const CudaVector3d& v) {
    return v * s;
}

__device__ __host__ inline CudaVector3d operator/(const CudaVector3d& v, double s) {
    return CudaVector3d(v.x / s, v.y / s, v.z / s);
}

} // namespace common_math
