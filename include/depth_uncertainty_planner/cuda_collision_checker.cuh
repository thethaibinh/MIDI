#pragma once

#include "common_math/segment2.hpp"
#include "common_math/pinhole_camera_model.hpp"
#include <cuda_runtime.h>
#include <iostream>

namespace depth_uncertainty_planner {

class CudaCollisionChecker {
public:
    static bool check_for_collision(
        const common_math::SecondOrderSegment* segment,
        double& trajectory_collision_probability,
        double& mahalanobis_distance,
        const float* depth_data,
        const common_math::PinholeCamera& camera);

private:
    static cudaError_t handleCudaError(cudaError_t err, const char* file, int line);
};

#define CUDA_CHECK(err) do { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::cerr << "[ERROR] CUDA Error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err_) << std::endl; \
        return false; \
    } \
} while(0)

} // namespace depth_uncertainty_planner