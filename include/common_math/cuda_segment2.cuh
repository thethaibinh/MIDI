#pragma once
#include "cuda_segment.cuh"
#include "cuda_second_order_polynomial.cuh"
#include "cuda_fourth_order_polynomial.cuh"
#include "cuda_third_order_polynomial.cuh"
#include "cuda_pinhole_camera_model.cuh"

namespace common_math {

class CudaSecondOrderSegment : public CudaSegment {
public:
    __device__ __host__ CudaSecondOrderSegment(const CudaVector3d* coeffs, double start_time, double end_time)
        : CudaSegment(coeffs, 3, start_time, end_time) {}

    __device__ __host__ CudaVector3d get_point(double t) const override {
        // if ((t - _start_time) * (t - _end_time) > 0) return CudaVector3d();
        t = cuda_clamp(t, _start_time, _end_time);

        return _coeffs[0] * (t * t) +
               _coeffs[1] * t +
               _coeffs[2];
    }

    __device__ __host__ uint8_t solve_first_time_at_depth(const double depth, double& time) const {
        const double a = _coeffs[0].z;
        const double b = _coeffs[1].z;
        const double c = _coeffs[2].z - depth;
        const double coeffs[3] = {a, b, c};
        const CudaSecondOrderPolynomial pol2(coeffs, _start_time, _end_time);
        double times[2];
        const uint8_t num_roots = pol2.solve_roots(times);
        time = times[0];
        return num_roots;
    }

    __device__ __host__ CudaVector3d get_start_point() const override {
        return get_point(_start_time);
    }

    __device__ __host__ CudaVector3d get_end_point() const override {
        return get_point(_end_time);
    }

    __device__ __host__ CudaVector3d get_acceleration() const { return _coeffs[0] * 2.0; }
    __device__ __host__ CudaVector3d get_initial_velocity() const { return _coeffs[1]; }
    __device__ __host__ CudaVector3d get_initial_position() const { return _coeffs[2]; }

    __device__ __host__ double get_acceleration_x() const { return _coeffs[0].x * 2.0; }
    __device__ __host__ double get_acceleration_y() const { return _coeffs[0].y * 2.0; }
    __device__ __host__ double get_acceleration_z() const { return _coeffs[0].z * 2.0; }

    __device__ __host__ double get_initial_velocity_x() const { return _coeffs[1].x; }
    __device__ __host__ double get_initial_velocity_y() const { return _coeffs[1].y; }
    __device__ __host__ double get_initial_velocity_z() const { return _coeffs[1].z; }

    __device__ __host__ double get_initial_position_x() const { return _coeffs[2].x; }
    __device__ __host__ double get_initial_position_y() const { return _coeffs[2].y; }
    __device__ __host__ double get_initial_position_z() const { return _coeffs[2].z; }

    __device__ __host__ double get_axis_value(int i, double t) const {
        // if ((t - _start_time) * (t - _end_time) > 0) return 0.0;
        t = cuda_clamp(t, _start_time, _end_time);
        return _coeffs[0][i] * t * t +
               _coeffs[1][i] * t +
               _coeffs[2][i];
    }

    __device__ __host__ double get_axis_start_value(int i) const {
        return get_axis_value(i, _start_time);
    }

    __device__ __host__ double get_axis_end_value(int i) const {
        return get_axis_value(i, _end_time);
    }

    __device__ __host__ double get_euclidean_distance(const CudaVector3d& pixel) const {
        return sqrt(get_euclidean_distance_square(pixel));
    }

    __device__ __host__ double get_euclidean_distance_square(const CudaVector3d& pixel) const {
        double endpoint_depth = get_axis_end_value(2);
        double start_point_depth = get_axis_start_value(2);

        if (pixel.z > endpoint_depth) {
            CudaVector3d diff = pixel - get_end_point();
            return diff.square_norm();
        }
        if (pixel.z < start_point_depth) {
            CudaVector3d diff = pixel - get_start_point();
            return diff.square_norm();
        }

        double a1 = _coeffs[0].x, b1 = _coeffs[1].x, c1 = _coeffs[2].x;
        double a2 = _coeffs[0].y, b2 = _coeffs[1].y, c2 = _coeffs[2].y;
        double a3 = _coeffs[0].z, b3 = _coeffs[1].z, c3 = _coeffs[2].z;

        double eu_dist_coeffs[5] = {
            a1 * a1 + a2 * a2 + a3 * a3,
            a1 * b1 * 2.0 + a2 * b2 * 2.0 + a3 * b3 * 2.0,
            a1 * (c1 - pixel.x) * 2.0 + a2 * (c2 - pixel.y) * 2.0 + a3 * (c3 - pixel.z) * 2.0 + b1 * b1 + b2 * b2 + b3 * b3,
            b1 * (c1 - pixel.x) * 2.0 + b2 * (c2 - pixel.y) * 2.0 + b3 * (c3 - pixel.z) * 2.0,
            (c1 - pixel.x) * (c1 - pixel.x) + (c2 - pixel.y) * (c2 - pixel.y) + (c3 - pixel.z) * (c3 - pixel.z)
        };

        CudaFourthOrderPolynomial pol4(eu_dist_coeffs, _start_time, _end_time);
        return pol4.get_min();

        // if (eu_dist_coeffs[0] != 0) {
        //     CudaFourthOrderPolynomial pol4(eu_dist_coeffs, _start_time, _end_time);
        //     return pol4.get_min();
        // } else if (eu_dist_coeffs[1] != 0) {
        //     CudaThirdOrderPolynomial pol3(&eu_dist_coeffs[1], _start_time, _end_time);
        //     return pol3.get_min();
        // } else {
        //     CudaSecondOrderPolynomial pol2(&eu_dist_coeffs[2], _start_time, _end_time);
        //     return pol2.get_min();
        // }
    }

    __device__ __host__ double get_half_mahalanobis_distance_square(
        const CudaVector3d& mean, const CudaVector3d& cov_diag) const {

        double a1 = _coeffs[0].x, b1 = _coeffs[1].x, c1 = _coeffs[2].x;
        double a2 = _coeffs[0].y, b2 = _coeffs[1].y, c2 = _coeffs[2].y;
        double a3 = _coeffs[0].z, b3 = _coeffs[1].z, c3 = _coeffs[2].z;

        double ma_dist_coeffs[5] = {
            a1 * a1 / cov_diag.x / 2.0 + a2 * a2 / cov_diag.y / 2.0 + a3 * a3 / cov_diag.z / 2.0,
            a1 * b1 / cov_diag.x + a2 * b2 / cov_diag.y + a3 * b3 / cov_diag.z,
            (b1 * b1 + 2.0 * a1 * (c1 - mean.x)) / cov_diag.x / 2.0 +
            (b2 * b2 + 2.0 * a2 * (c2 - mean.y)) / cov_diag.y / 2.0 +
            (b3 * b3 + 2.0 * a3 * (c3 - mean.z)) / cov_diag.z / 2.0,
            b1 * (c1 - mean.x) / cov_diag.x + b2 * (c2 - mean.y) / cov_diag.y + b3 * (c3 - mean.z) / cov_diag.z,
            (c1 - mean.x) * (c1 - mean.x) / cov_diag.x / 2.0 +
            (c2 - mean.y) * (c2 - mean.y) / cov_diag.y / 2.0 +
            (c3 - mean.z) * (c3 - mean.z) / cov_diag.z / 2.0
        };

        CudaFourthOrderPolynomial pol4(ma_dist_coeffs, _start_time, _end_time);
        return pol4.get_min();
        // if (ma_dist_coeffs[0] != 0) {
        //     CudaFourthOrderPolynomial pol4(ma_dist_coeffs, _start_time, _end_time);
        //     return pol4.get_min();
        // } else if (ma_dist_coeffs[1] != 0) {
        //     CudaThirdOrderPolynomial pol3(&ma_dist_coeffs[1], _start_time, _end_time);
        //     return pol3.get_min();
        // } else {
        //     CudaSecondOrderPolynomial pol2(&ma_dist_coeffs[2], _start_time, _end_time);
        //     return pol2.get_min();
        // }
    }

    __device__ __host__ double get_collision_probability(
        const CudaVector3d& mean,
        const CudaPinholeCamera& camera,
        double& mahalanobis_distance) const {

        const CudaVector3d cov_diag = camera.get_covariance_matrix(mean);
        const double hmds = get_half_mahalanobis_distance_square(mean, cov_diag);
        atomicMin(&mahalanobis_distance, sqrt(hmds * 2.0));

        double denominator = pow(2 * CUDA_PI, 1.5) * sqrt(cov_diag.x * cov_diag.y * cov_diag.z);
        return exp(-hmds) / denominator;
    }

    __device__ __host__ uint8_t get_depth_switching_points(double* switching_points) const override {
        CudaVector3d deriv_coeffs[2];
        get_derivative_coeffs(deriv_coeffs);
        uint8_t num_points = 0;

        // No switching points if the z-component of the derivative is zero
        if (deriv_coeffs[0].z == 0) {
            num_points = 0;
            return num_points;
        }

        double root_t = -deriv_coeffs[1].z / deriv_coeffs[0].z;
        if ((root_t > _start_time) && (root_t < _end_time)) {
            switching_points[0] = root_t;
            num_points = 1;
        } else {
            num_points = 0;
        }
        return num_points;
    }

    __device__ __host__ void get_derivative_coeffs(CudaVector3d* deriv_coeffs) const override {
        deriv_coeffs[0] = _coeffs[0] * 2.0;
        deriv_coeffs[1] = _coeffs[1];
    }

    __device__ __host__ void get_projection_boundary(
        const CudaPinholeCamera& camera,
        int16_t* boundary) const {

        // Initialize boundaries: left, top, right, bottom
        boundary[0] = INT16_MAX;  // min_x
        boundary[1] = INT16_MAX;  // min_y
        boundary[2] = INT16_MIN;  // max_x
        boundary[3] = INT16_MIN;  // max_y

        // Get derivative coefficients for projection
        const double a1 = _coeffs[0].x, b1 = _coeffs[1].x, c1 = _coeffs[2].x;
        const double a2 = _coeffs[0].y, b2 = _coeffs[1].y, c2 = _coeffs[2].y;
        const double a3 = _coeffs[0].z, b3 = _coeffs[1].z, c3 = _coeffs[2].z;

        // Create array for extreme and terminal times
        double times[6];
        int num_times = 2;  // Start with start and end times
        times[0] = _start_time;
        times[1] = _end_time;

        // Add roots from x projection derivative
        const double x_coeffs[3] = {a1 * b3 - a3 * b1, (a1 * c3 - a3 * c1) * 2.0, b1 * c3 - b3 * c1};
        CudaSecondOrderPolynomial deriv_proj_x(x_coeffs, _start_time, _end_time);
        deriv_proj_x.solve_roots(times + num_times);

        // Add roots from y projection derivative
        const double y_coeffs[3] = {a2 * b3 - a3 * b2, (a2 * c3 - a3 * c2) * 2.0, b2 * c3 - b3 * c2};
        CudaSecondOrderPolynomial deriv_proj_y(y_coeffs, _start_time, _end_time);
        deriv_proj_y.solve_roots(times + num_times + 2);

        // Process each time point
        for (int i = 0; i < 6; i++) {
            double t = times[i];
            if (t >= _start_time && t <= _end_time) {
                CudaVector3d point = get_point(t);
                double proj_x[2], proj_y[2];

                camera.project_point_to_pixel_adding_margin(
                    reinterpret_cast<const double*>(&point),
                    proj_x, proj_y);

                // Update boundaries: left, top, right, bottom
                boundary[0] = cuda_min(cuda_min(boundary[0], static_cast<int16_t>(proj_x[0])), static_cast<int16_t>(proj_x[1]));
                boundary[1] = cuda_min(cuda_min(boundary[1], static_cast<int16_t>(proj_y[0])), static_cast<int16_t>(proj_y[1]));
                boundary[2] = cuda_max(cuda_max(boundary[2], static_cast<int16_t>(proj_x[0])), static_cast<int16_t>(proj_x[1]));
                boundary[3] = cuda_max(cuda_max(boundary[3], static_cast<int16_t>(proj_y[0])), static_cast<int16_t>(proj_y[1]));
            }
        }
        boundary[0] = cuda_clamp(boundary[0], 0, camera.get_width());
        boundary[1] = cuda_clamp(boundary[1], 0, camera.get_height());
        boundary[2] = cuda_clamp(boundary[2], 0, camera.get_width());
        boundary[3] = cuda_clamp(boundary[3], 0, camera.get_height());
    }
};

} // namespace common_math