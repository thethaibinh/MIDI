#pragma once
#include "cuda_math.cuh"

namespace common_math {

class CudaPinholeCamera {
public:
    __device__ __host__ CudaPinholeCamera(
        double focal_length,
        double cx,
        double cy,
        uint16_t width,
        uint16_t height,
        const double* cov_coeffs,
        double true_radius,
        double planning_radius,
        double minimum_clear_distance)
        : _focal_length(focal_length),
          _cx(cx),
          _cy(cy),
          _width(width),
          _height(height),
          _true_vehicle_radius(true_radius),
          _planning_vehicle_radius(planning_radius),
          _minimum_clear_distance(minimum_clear_distance) {
        // Copy covariance coefficients if provided
        if(cov_coeffs != nullptr) {
            for(int i = 0; i < 6; i++) {
                _cov_coeffs[i] = cov_coeffs[i];
            }
        }
    }

    __device__ __host__ CudaVector3d get_covariance_matrix(const CudaVector3d& point) const {
        double ca0 = _cov_coeffs[0];
        double ca1 = _cov_coeffs[1];
        double ca2 = _cov_coeffs[2];
        double cl0 = _cov_coeffs[3];
        double cl1 = _cov_coeffs[4];
        double cl2 = _cov_coeffs[5];

        const double depth = point[2];
        const double depth_sq = depth * depth;
        const double sigma_a = ca0 + ca1 * depth + ca2 * depth_sq;
        const double sigma_lx = cl0 + cl1 * depth + cl2 * fabs(point[0]);
        const double sigma_ly = cl0 + cl1 * depth + cl2 * fabs(point[1]);

        return CudaVector3d(sigma_lx, sigma_ly, sigma_a);
    }

    __device__ __host__ CudaVector3d deproject_pixel_to_point(
        int x, int y, double depth) const {
        CudaVector3d point;
        point.x = depth * (x - _cx) / _focal_length;
        point.y = depth * (y - _cy) / _focal_length;
        point.z = depth;
        return point;
    }

    __device__ __host__ void project_point_to_pixel(
        const CudaVector3d& point, int16_t* pixel) const {
        double x = point[0] * _focal_length / cuda_abs(point[2]) + _cx;
        double y = point[1] * _focal_length / cuda_abs(point[2]) + _cy;

        pixel[0] = static_cast<int16_t>(round(x));
        pixel[1] = static_cast<int16_t>(round(y));
    }

    __device__ __host__ void project_point_to_pixel_adding_margin(
        const double* point,
        double* out_x,
        double* out_y) const {
        double margin = _true_vehicle_radius * _focal_length / _minimum_clear_distance;

        // left
        out_x[0] = point[0] * _focal_length / cuda_abs(point[2]) + _cx - margin;

        // top
        out_y[0] = point[1] * _focal_length / cuda_abs(point[2]) + _cy - margin;

        // right
        out_x[1] = point[0] * _focal_length / cuda_abs(point[2]) + _cx + margin;

        // bottom
        out_y[1] = point[1] * _focal_length / cuda_abs(point[2]) + _cy + margin;
    }

    // __device__ __host__ void get_frame_dimensions_with_planning_radius_margin(uint16_t* dims) const {
    //     uint16_t margin = static_cast<uint16_t>(_planning_vehicle_radius * _focal_length / _minimum_clear_distance);
    //     dims[0] = margin;                    // left
    //     dims[1] = _width - margin;           // right
    //     dims[2] = margin;                    // top
    //     dims[3] = _height - margin;          // bottom
    // }

    __device__ __host__ void get_frame_dimensions_with_true_radius_margin(uint16_t* dims) const {
        uint16_t margin = static_cast<uint16_t>(_true_vehicle_radius * _focal_length / _minimum_clear_distance);
        dims[0] = margin;                    // left
        dims[1] = _width - margin;           // right
        dims[2] = margin;                    // top
        dims[3] = _height - margin;          // bottom
    }

    // Getters
    __device__ __host__ double get_focal_length() const { return _focal_length; }
    __device__ __host__ double get_cx() const { return _cx; }
    __device__ __host__ double get_cy() const { return _cy; }
    __device__ __host__ double get_true_vehicle_radius() const { return _true_vehicle_radius; }
    __device__ __host__ double get_planning_vehicle_radius() const { return _planning_vehicle_radius; }
    __device__ __host__ uint16_t get_width() const { return _width; }
    __device__ __host__ uint16_t get_height() const { return _height; }
    __device__ __host__ double get_minimum_clear_distance() const { return _minimum_clear_distance; }

private:
    double _focal_length;
    double _cx;
    double _cy;
    uint16_t _width;
    uint16_t _height;
    double _true_vehicle_radius;
    double _planning_vehicle_radius;
    double _minimum_clear_distance;
    double _cov_coeffs[6];
};

} // namespace common_math