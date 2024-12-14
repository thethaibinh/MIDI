#include "depth_uncertainty_planner/cuda_collision_checker.cuh"
#include "common_math/cuda_conversions.cuh"
#include <vector>
#include <ros/ros.h>

namespace depth_uncertainty_planner {

using common_math::CudaVector3d;
using common_math::atomicMin;
using common_math::atomicMax;

// CUDA kernel
__global__ void collision_checking_kernel(
    const float* depth_data,
    const common_math::CudaPinholeCamera cuda_camera,
    const common_math::CudaSecondOrderSegment cuda_segment,
    bool* collision_free,
    double* max_collision_probability,
    double* min_mahalanobis_distance,
    int16_t left, int16_t top, int16_t right, int16_t bottom,
    uint32_t img_width) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < left || x > right || y < top || y > bottom) return;

    const double spatial_z = depth_data[y * img_width + x];

    if (spatial_z < cuda_camera.get_true_vehicle_radius()) return;

    const CudaVector3d depth_point = cuda_camera.deproject_pixel_to_point(x, y, spatial_z);

    if (spatial_z < cuda_camera.get_minimum_clear_distance()) {
        if (cuda_segment.get_euclidean_distance(depth_point) < cuda_camera.get_planning_vehicle_radius()) {
            *collision_free = false;
        }
        return;
    }

    if (spatial_z > cuda_camera.get_planning_vehicle_radius()) {
        double prob = cuda_segment.get_collision_probability(depth_point, cuda_camera, *min_mahalanobis_distance);
        atomicMax(max_collision_probability, prob);
        if (prob > 5) {
            *collision_free = false;
        }
        return;
    }

    if (cuda_segment.get_euclidean_distance(depth_point) < cuda_camera.get_planning_vehicle_radius()) {
        *collision_free = false;
        return;
    }

    double prob = cuda_segment.get_collision_probability(depth_point, cuda_camera, *min_mahalanobis_distance);
    atomicMax(max_collision_probability, prob);
    if (prob > 5) {
        *collision_free = false;
    }
}

// Error handling
cudaError_t CudaCollisionChecker::handleCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        ROS_ERROR("CUDA Error at %s:%d: %s", file, line, cudaGetErrorString(err));
    }
    return err;
}

// Make sure static matches the declaration
bool CudaCollisionChecker::check_for_collision(
    const common_math::SecondOrderSegment* segment,
    double& trajectory_collision_probability,
    double& mahalanobis_distance,
    const float* depth_data,
    const common_math::PinholeCamera& camera) {

    // Check CUDA device
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        ROS_ERROR("No CUDA-capable device found: %s", cudaGetErrorString(err));
        return false;
    }

    // Print device info
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    ROS_INFO("Using CUDA device: %s", deviceProp.name);

    // Calculate depth data size
    size_t depth_size = camera.get_width() * camera.get_height() * sizeof(float);

    // Debug print
    // ROS_WARN("Allocating depth data: width=%u, height=%u, size=%lu bytes",
    //          camera.get_width(), camera.get_height(), depth_size);

    // Device pointers
    float* d_depth_data = nullptr;
    bool* d_collision_free = nullptr;
    double* d_max_collision_probability = nullptr;
    double* d_min_mahalanobis_distance = nullptr;

    try {
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_depth_data, depth_size));
        CUDA_CHECK(cudaMalloc(&d_collision_free, sizeof(bool)));
        CUDA_CHECK(cudaMalloc(&d_max_collision_probability, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_min_mahalanobis_distance, sizeof(double)));

        // Copy depth data to device
        CUDA_CHECK(cudaMemcpy(d_depth_data, depth_data, depth_size, cudaMemcpyHostToDevice));

        // Initialize other device memory
        bool init_collision_free = true;
        double init_max_prob = 0.0;
        double init_min_dist = INFINITY;

        CUDA_CHECK(cudaMemcpy(d_collision_free, &init_collision_free, sizeof(bool), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_max_collision_probability, &init_max_prob, sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_min_mahalanobis_distance, &init_min_dist, sizeof(double), cudaMemcpyHostToDevice));

        // Convert to CUDA types
        common_math::CudaPinholeCamera cuda_camera = common_math::CudaConverter::toCuda(camera);
        common_math::CudaSecondOrderSegment cuda_segment = common_math::CudaConverter::toCuda(*segment);

        // Get projection boundary
        int16_t boundary[4];  // [min_x, min_y, max_x, max_y]
        cuda_segment.get_projection_boundary(cuda_camera, boundary);

        // Debug print boundaries
        // ROS_WARN("Projection boundaries: x[%d, %d], y[%d, %d]",
        //          boundary[0], boundary[2], boundary[1], boundary[3]);

        const int16_t left = boundary[0];
        const int16_t top = boundary[1];
        const int16_t right = boundary[2];
        const int16_t bottom = boundary[3];

        if (left < 0 || right > cuda_camera.get_width() || top < 0 ||
            bottom > cuda_camera.get_height())
            return false;

        // Configure block size
        dim3 block(16, 16);

        // Calculate grid size
        // int dx = right - left + 1;  // add 1 because boundaries are inclusive
        // int dy = bottom - top + 1;

        // ROS_WARN("Region dimensions: dx=%d, dy=%d", dx, dy);

        // dim3 grid(
        //     (dx + block.x - 1) / block.x,
        //     (dy + block.y - 1) / block.y
        // );
        dim3 grid(
            (cuda_camera.get_width() + block.x - 1) / block.x,
            (cuda_camera.get_height() + block.y - 1) / block.y
        );

        ROS_INFO("Grid dimensions: (%d, %d)", grid.x, grid.y);

        // Ensure grid dimensions are valid
        if (grid.x == 0 || grid.y == 0) {
            ROS_ERROR("Invalid grid dimensions. Check if right >= left and bottom >= top");
            ROS_ERROR("Boundaries: x[%d, %d], y[%d, %d]", left, right, top, bottom);
            return false;
        }

        // Launch kernel
        collision_checking_kernel<<<grid, block>>>(
            d_depth_data,
            cuda_camera,
            cuda_segment,
            d_collision_free,
            d_max_collision_probability,
            d_min_mahalanobis_distance,
            left, top, right, bottom,
            cuda_camera.get_width()
        );

        // Check for kernel errors
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy results back
        bool collision_free;
        CUDA_CHECK(cudaMemcpy(&collision_free, d_collision_free, sizeof(bool), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&trajectory_collision_probability, d_max_collision_probability, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&mahalanobis_distance, d_min_mahalanobis_distance, sizeof(double), cudaMemcpyDeviceToHost));

        // Cleanup
        cudaFree(d_depth_data);
        cudaFree(d_collision_free);
        cudaFree(d_max_collision_probability);
        cudaFree(d_min_mahalanobis_distance);

        return collision_free;

    } catch (const std::exception& e) {
        ROS_ERROR("CUDA error: %s", e.what());
        ROS_WARN("CUDA error: %s", e.what());

        // Cleanup in case of error
        if (d_depth_data) cudaFree(d_depth_data);
        if (d_collision_free) cudaFree(d_collision_free);
        if (d_max_collision_probability) cudaFree(d_max_collision_probability);
        if (d_min_mahalanobis_distance) cudaFree(d_min_mahalanobis_distance);

        throw;
    }
    return true;
}

} // namespace depth_uncertainty_planner