/*!
 * Copyright 2024 by Binh Nguyen <thethaibinh@gmail.com>
 *
 * Stub types for non-CUDA builds. These provide type definitions
 * so the code compiles without CUDA, but collision checking will 
 * always return true (no collision) - trajectories are not validated.
 */

#ifndef CUDA_STUBS_HPP
#define CUDA_STUBS_HPP

#ifndef CUDA_AVAILABLE

#include <vector>
#include <cmath>
#include <cstdint>

namespace common_math {

// Stub vector type
struct CudaVector3d {
  double x, y, z;
  CudaVector3d() : x(0), y(0), z(0) {}
  CudaVector3d(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
};

// Forward declaration
class CudaSecondOrderSegment;
class CudaThirdOrderSegment;

// Stub camera type
class CudaPinholeCamera {
public:
  CudaPinholeCamera() = default;
  double get_true_vehicle_radius() const { return 0.5; }
  double get_planning_vehicle_radius() const { return 0.6; }
  double get_minimum_clear_distance() const { return 1.0; }
  double get_min_collision_distance() const { return 1.0; }
  uint16_t get_width() const { return 640; }
  uint16_t get_height() const { return 480; }
  CudaVector3d deproject_pixel_to_point(uint16_t, uint16_t, double z) const { 
    return CudaVector3d(0, 0, z); 
  }
};

// Converter stub
class CudaConverter {
public:
  template<typename T>
  static CudaPinholeCamera toCuda(const T&) { return CudaPinholeCamera(); }
};

// Base segment class stub
class CudaSegment {
public:
  virtual ~CudaSegment() = default;
  virtual double get_duration() const { return 0.0; }
};

// Second order segment stub
class CudaSecondOrderSegment : public CudaSegment {
  double start_time_, end_time_;
  CudaVector3d coeffs_[3];
public:
  CudaSecondOrderSegment() : start_time_(0), end_time_(0) {}
  CudaSecondOrderSegment(const CudaVector3d coeffs[3], double duration) 
    : start_time_(0), end_time_(duration) {
    for (int i = 0; i < 3; i++) coeffs_[i] = coeffs[i];
  }
  CudaSecondOrderSegment(const CudaVector3d coeffs[3], double start, double end) 
    : start_time_(start), end_time_(end) {
    for (int i = 0; i < 3; i++) coeffs_[i] = coeffs[i];
  }
  bool is_monotonically_increasing_depth() const { return true; }
  double get_duration() const override { return end_time_ - start_time_; }
  double get_start_time() const { return start_time_; }
  double get_end_time() const { return end_time_; }
  CudaVector3d get_start_point() const { return coeffs_[2]; }
  CudaVector3d get_end_point() const { return coeffs_[2]; }
  void get_coeffs(CudaVector3d out[3]) const { for (int i = 0; i < 3; i++) out[i] = coeffs_[i]; }
  uint8_t solve_first_time_at_depth(double, double&) const { return 0; }
  void get_projection_boundary(const CudaPinholeCamera&, int16_t out[4]) const {
    out[0] = 0; out[1] = 0; out[2] = 640; out[3] = 480;
  }
  double get_euclidean_distance(const CudaVector3d&) const { return 100.0; }
  double get_collision_probability(const CudaVector3d&, const CudaPinholeCamera&, double&) const { return 0.0; }
};

// Third order segment stub
class CudaThirdOrderSegment : public CudaSegment {
  double start_time_, end_time_;
  CudaVector3d coeffs_[4];
public:
  CudaThirdOrderSegment() : start_time_(0), end_time_(0) {}
  CudaThirdOrderSegment(const CudaVector3d coeffs[4], double duration) 
    : start_time_(0), end_time_(duration) {
    for (int i = 0; i < 4; i++) coeffs_[i] = coeffs[i];
  }
  CudaThirdOrderSegment(const CudaVector3d coeffs[4], double start, double end) 
    : start_time_(start), end_time_(end) {
    for (int i = 0; i < 4; i++) coeffs_[i] = coeffs[i];
  }
  double get_duration() const override { return end_time_ - start_time_; }
  double get_start_time() const { return start_time_; }
  double get_end_time() const { return end_time_; }
  void get_coeffs(CudaVector3d out[4]) const { for (int i = 0; i < 4; i++) out[i] = coeffs_[i]; }
  void get_derivative_coeffs(CudaVector3d out[3]) const { for (int i = 0; i < 3; i++) out[i] = CudaVector3d(); }
  uint8_t get_depth_switching_points_and_terminals(double out[6]) const { 
    out[0] = start_time_; out[1] = end_time_; return 2; 
  }
};

// Monotonic segment stub
class CudaMonotonicSegment3 {
  CudaVector3d coeffs_[4];
  double start_time_, end_time_;
public:
  CudaMonotonicSegment3() : start_time_(0), end_time_(0) {}
  CudaMonotonicSegment3(const CudaVector3d coeffs[4], double start, double end)
    : start_time_(start), end_time_(end) {
    for (int i = 0; i < 4; i++) coeffs_[i] = coeffs[i];
  }
  double get_start_time() const { return start_time_; }
  double get_end_time() const { return end_time_; }
  double get_duration() const { return end_time_ - start_time_; }
  CudaVector3d get_start_point() const { return coeffs_[3]; }
  CudaVector3d get_end_point() const { return coeffs_[3]; }
  void get_coeffs(CudaVector3d out[4]) const { for (int i = 0; i < 4; i++) out[i] = coeffs_[i]; }
  bool is_increasing_depth() const { return true; }
  void get_projection_boundary(const CudaPinholeCamera&, int16_t out[4]) const {
    out[0] = 0; out[1] = 0; out[2] = 640; out[3] = 480;
  }
  double get_euclidean_distance(const CudaVector3d&) const { return 100.0; }
  bool operator<(const CudaMonotonicSegment3& other) const { return start_time_ < other.start_time_; }
};

} // namespace common_math

#endif // !CUDA_AVAILABLE

#endif // CUDA_STUBS_HPP
