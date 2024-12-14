#pragma once
#include "pinhole_camera_model.hpp"
#include "cuda_pinhole_camera_model.cuh"
#include "segment2.hpp"
#include "segment3.hpp"
#include "cuda_segment2.cuh"
#include "cuda_segment3.cuh"
#include <cuda_runtime.h>

namespace common_math {

class CudaConverter {
public:
    static __host__ CudaPinholeCamera toCuda(const PinholeCamera& camera) {
        return CudaPinholeCamera(
            camera.get_focal_length(),
            camera.get_cx(),
            camera.get_cy(),
            camera.get_width(),
            camera.get_height(),
            nullptr,
            camera.get_true_vehicle_radius(),
            camera.get_planning_vehicle_radius(),
            camera.get_minimum_clear_distance()
        );
    }

    static __host__ CudaSecondOrderSegment toCuda(const SecondOrderSegment& segment) {
        // Get coefficients from CPU segment
        std::vector<Eigen::Vector3d> coeffs = segment.get_coeffs();


        CudaVector3d cuda_coeffs[3];
        for(int i = 0; i < 3; i++) {
            cuda_coeffs[i] = CudaVector3d(
                coeffs[i].x(),
                coeffs[i].y(),
                coeffs[i].z()
            );
        }
        return CudaSecondOrderSegment(
            cuda_coeffs,
            segment.get_start_time(),
            segment.get_end_time()
        );
    }

    static __host__ CudaThirdOrderSegment toCudaThirdOrder(const ThirdOrderSegment& segment) {
        std::vector<Eigen::Vector3d> coeffs = segment.get_coeffs();
        CudaVector3d cuda_coeffs[4];

        for(int i = 0; i < 4; i++) {
            cuda_coeffs[i] = CudaVector3d(
                coeffs[i].x(),
                coeffs[i].y(),
                coeffs[i].z()
            );
        }

        return CudaThirdOrderSegment(
            cuda_coeffs,
            segment.get_start_time(),
            segment.get_end_time()
        );
    }
};

} // namespace common_math