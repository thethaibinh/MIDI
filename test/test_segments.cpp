/**
 * Unit tests for common_math segment classes:
 *   - SecondOrderSegment (segment2.hpp) — distance functions
 *   - ThirdOrderSegment  (segment3.hpp) — depth switching
 *
 * Copyright 2024 by Binh Nguyen <thethaibinh@gmail.com>
 * Licensed under CC BY-NC 4.0.
 */
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>

#include <common_math/segment2.hpp>
#include <common_math/segment3.hpp>

using namespace common_math;

// ============================================================================
// SecondOrderSegment — construction and evaluation
// ============================================================================

TEST(SecondOrderSegment, PointEvaluation) {
  // p(t) = [0.5, 0, 0]*t^2 + [1, 0, 0]*t + [0, 0, 0]
  // At t=2: p = [0.5*4 + 2, 0, 0] = [4, 0, 0]
  std::vector<Eigen::Vector3d> coeffs = {
    Eigen::Vector3d(0.5, 0, 0),
    Eigen::Vector3d(1, 0, 0),
    Eigen::Vector3d(0, 0, 0)};
  SecondOrderSegment seg(coeffs, 0.0, 3.0);

  Eigen::Vector3d pt = seg.get_point(2.0);
  EXPECT_NEAR(pt.x(), 4.0, 1e-10);
  EXPECT_NEAR(pt.y(), 0.0, 1e-10);
  EXPECT_NEAR(pt.z(), 0.0, 1e-10);
}

TEST(SecondOrderSegment, StartAndEndPoints) {
  std::vector<Eigen::Vector3d> coeffs = {
    Eigen::Vector3d(1, 0, 0),
    Eigen::Vector3d(0, 1, 0),
    Eigen::Vector3d(0, 0, 1)};
  SecondOrderSegment seg(coeffs, 0.0, 2.0);

  Eigen::Vector3d start = seg.get_start_point();
  EXPECT_NEAR(start.x(), 0.0, 1e-10);
  EXPECT_NEAR(start.y(), 0.0, 1e-10);
  EXPECT_NEAR(start.z(), 1.0, 1e-10);

  // At t=2: [4, 2, 1]
  Eigen::Vector3d end = seg.get_end_point();
  EXPECT_NEAR(end.x(), 4.0, 1e-10);
  EXPECT_NEAR(end.y(), 2.0, 1e-10);
  EXPECT_NEAR(end.z(), 1.0, 1e-10);
}

// ============================================================================
// SecondOrderSegment — segment-to-segment distance
// ============================================================================

TEST(SecondOrderSegment, OffsetSegmentsKnownDistance) {
  // Segment A: p(t) = (t^2, 0, 0) for t in [0, 1]  — accelerating in X
  // Segment B: p(t) = (0, 3, 0) for t in [0, 1]    — stationary at Y=3
  // d(t) = (t^2, -3, 0), ||d||^2 = t^4 + 9, min at t=0 → 9
  SecondOrderSegment seg_a(
    {Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0)},
    0.0, 1.0);
  SecondOrderSegment seg_b(
    {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 3, 0)},
    0.0, 1.0);

  double min_dist_sq = seg_a.get_min_distance_square_to_segment(seg_b);
  EXPECT_NEAR(min_dist_sq, 9.0, 1e-6);
}

TEST(SecondOrderSegment, CrossingSegmentsZeroDistance) {
  // Segment A: p(t) = (t, 0, 0) for t in [-1, 1]
  // Segment B: p(t) = (0, t, 0) for t in [-1, 1]
  // Cross at origin at t=0
  SecondOrderSegment seg_a(
    {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 0, 0)},
    -1.0, 1.0);
  SecondOrderSegment seg_b(
    {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 1, 0), Eigen::Vector3d(0, 0, 0)},
    -1.0, 1.0);

  double min_dist_sq = seg_a.get_min_distance_square_to_segment(seg_b);
  EXPECT_NEAR(min_dist_sq, 0.0, 1e-6);
}

TEST(SecondOrderSegment, NonOverlappingTimeInfiniteDistance) {
  SecondOrderSegment seg_a(
    {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 0, 0)},
    0.0, 1.0);
  SecondOrderSegment seg_b(
    {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 3, 0)},
    2.0, 3.0);

  double min_dist_sq = seg_a.get_min_distance_square_to_segment(seg_b);
  EXPECT_TRUE(std::isinf(min_dist_sq));
}

TEST(SecondOrderSegment, AcceleratingSegmentVsStationary) {
  // Segment A: accelerating in X: p(t) = (0.5*t^2, 0, 0)
  // Segment B: stationary at (0, 2, 0)
  // At t=0: dist = 2.0, dist^2 = 4.0
  SecondOrderSegment seg_a(
    {Eigen::Vector3d(0.5, 0, 0), Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0)},
    0.0, 2.0);
  SecondOrderSegment seg_b(
    {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 2, 0)},
    0.0, 2.0);

  double min_dist_sq = seg_a.get_min_distance_square_to_segment(seg_b);
  EXPECT_NEAR(min_dist_sq, 4.0, 1e-4);
}

TEST(SecondOrderSegment, DistanceVsBruteForce) {
  // Arbitrary accelerating segments
  SecondOrderSegment seg_a(
    {Eigen::Vector3d(0.1, -0.2, 0.05),
     Eigen::Vector3d(1.0, 0.5, -0.3),
     Eigen::Vector3d(0.0, 0.0, 1.0)},
    0.0, 3.0);
  SecondOrderSegment seg_b(
    {Eigen::Vector3d(-0.15, 0.1, 0.0),
     Eigen::Vector3d(0.5, -0.8, 0.2),
     Eigen::Vector3d(2.0, 1.0, 1.0)},
    0.5, 2.5);

  double analytical = seg_a.get_min_distance_square_to_segment(seg_b);

  // Brute force over overlapping interval [0.5, 2.5]
  double sampled_min = 1e18;
  for (int i = 0; i <= 50000; ++i) {
    double t = 0.5 + 2.0 * i / 50000.0;
    Eigen::Vector3d pa = seg_a.get_point(t);
    Eigen::Vector3d pb = seg_b.get_point(t);
    double d = (pa - pb).squaredNorm();
    sampled_min = std::min(sampled_min, d);
  }

  EXPECT_NEAR(analytical, sampled_min, 1e-2);
  // Closed-form finds true minimum, should be <= brute force
  EXPECT_LE(analytical, sampled_min + 1e-4);
}

TEST(SecondOrderSegment, MinDistanceEuclidean) {
  // Verify get_min_distance_to_segment = sqrt(get_min_distance_square_to_segment)
  // seg_a accelerates in X, seg_b stationary at Y=3 → min dist = 3 at t=0
  SecondOrderSegment seg_a(
    {Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0)},
    0.0, 1.0);
  SecondOrderSegment seg_b(
    {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 3, 0)},
    0.0, 1.0);

  double dist = seg_a.get_min_distance_to_segment(seg_b);
  double dist_sq = seg_a.get_min_distance_square_to_segment(seg_b);
  EXPECT_NEAR(dist, std::sqrt(dist_sq), 1e-10);
  EXPECT_NEAR(dist, 3.0, 1e-6);
}

// ============================================================================
// SecondOrderSegment — point distance
// ============================================================================

TEST(SecondOrderSegment, DistanceToPoint) {
  // Straight line along X: p(t) = (t, 0, 1) for t in [0, 5]
  // Point at (2.5, 3, 1) — closest on segment at t=2.5, dist = 3
  SecondOrderSegment seg(
    {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 0, 1)},
    0.0, 5.0);

  Eigen::Vector3d pt(2.5, 3.0, 1.0);
  double dist_sq = seg.get_euclidean_distance_square(pt);
  EXPECT_NEAR(dist_sq, 9.0, 1e-4);

  double dist = seg.get_euclidean_distance(pt);
  EXPECT_NEAR(dist, 3.0, 1e-4);
}

// ============================================================================
// SecondOrderSegment — depth switching
// ============================================================================

TEST(SecondOrderSegment, DepthSwitching) {
  // p(t) = [0]*t^2 + [0, 0, 1]*t + [0, 0, 0]
  // z(t) = t — monotonically increasing, no switching
  SecondOrderSegment seg(
    {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(0, 0, 0)},
    0.0, 5.0);

  std::vector<double> switches;
  seg.get_depth_switching_points(switches);
  EXPECT_EQ(switches.size(), 0u);
  EXPECT_TRUE(seg.is_monotonically_increasing_depth());
}

TEST(SecondOrderSegment, DepthSwitchingWithExtremum) {
  // z(t) = -t^2 + 4t on [0, 5]
  // z'(t) = -2t + 4 = 0 at t=2 (switches from increasing to decreasing)
  SecondOrderSegment seg(
    {Eigen::Vector3d(0, 0, -1), Eigen::Vector3d(0, 0, 4), Eigen::Vector3d(0, 0, 0)},
    0.0, 5.0);

  std::vector<double> switches;
  seg.get_depth_switching_points(switches);
  ASSERT_EQ(switches.size(), 1u);
  EXPECT_NEAR(switches[0], 2.0, 1e-10);
  EXPECT_FALSE(seg.is_monotonically_increasing_depth());
}

// ============================================================================
// ThirdOrderSegment — construction and evaluation
// ============================================================================

TEST(ThirdOrderSegment, PointEvaluation) {
  // p(t) = [1,0,0]*t^3 + [0,0,0]*t^2 + [0,0,0]*t + [0,0,0]
  // At t=2: [8, 0, 0]
  std::vector<Eigen::Vector3d> coeffs = {
    Eigen::Vector3d(1, 0, 0),
    Eigen::Vector3d(0, 0, 0),
    Eigen::Vector3d(0, 0, 0),
    Eigen::Vector3d(0, 0, 0)};
  ThirdOrderSegment seg(coeffs, 0.0, 3.0);

  Eigen::Vector3d pt = seg.get_point(2.0);
  EXPECT_NEAR(pt.x(), 8.0, 1e-10);
  EXPECT_NEAR(pt.y(), 0.0, 1e-10);
  EXPECT_NEAR(pt.z(), 0.0, 1e-10);
}

TEST(ThirdOrderSegment, DepthSwitching) {
  // z(t) = t^3 - 3t on [0, 3]
  // z'(t) = 3t^2 - 3 = 0 at t=1 (in range)
  std::vector<Eigen::Vector3d> coeffs = {
    Eigen::Vector3d(0, 0, 1),
    Eigen::Vector3d(0, 0, 0),
    Eigen::Vector3d(0, 0, -3),
    Eigen::Vector3d(0, 0, 0)};
  ThirdOrderSegment seg(coeffs, 0.0, 3.0);

  std::vector<double> switches;
  seg.get_depth_switching_points(switches);
  ASSERT_EQ(switches.size(), 1u);
  EXPECT_NEAR(switches[0], 1.0, 1e-6);
}

TEST(ThirdOrderSegment, StartEndPoints) {
  std::vector<Eigen::Vector3d> coeffs = {
    Eigen::Vector3d(1, 0, 0),
    Eigen::Vector3d(0, 1, 0),
    Eigen::Vector3d(0, 0, 1),
    Eigen::Vector3d(1, 2, 3)};
  ThirdOrderSegment seg(coeffs, 0.0, 1.0);

  Eigen::Vector3d start = seg.get_start_point();
  EXPECT_NEAR(start.x(), 1.0, 1e-10);
  EXPECT_NEAR(start.y(), 2.0, 1e-10);
  EXPECT_NEAR(start.z(), 3.0, 1e-10);

  // At t=1: [1+0+0+1, 0+1+0+2, 0+0+1+3] = [2, 3, 4]
  Eigen::Vector3d end = seg.get_end_point();
  EXPECT_NEAR(end.x(), 2.0, 1e-10);
  EXPECT_NEAR(end.y(), 3.0, 1e-10);
  EXPECT_NEAR(end.z(), 4.0, 1e-10);
}

// ============================================================================
// SecondOrderSegment — derivative coeffs
// ============================================================================

TEST(SecondOrderSegment, DerivativeCoeffs) {
  // p(t) = [a]*t^2 + [b]*t + [c], velocity = [2a]*t + [b]
  Eigen::Vector3d a(1, 2, 3), b(4, 5, 6), c(7, 8, 9);
  SecondOrderSegment seg({a, b, c}, 0.0, 1.0);

  auto deriv = seg.get_derivative_coeffs();
  ASSERT_EQ(deriv.size(), 2u);
  EXPECT_TRUE(deriv[0].isApprox(2.0 * a, 1e-10));
  EXPECT_TRUE(deriv[1].isApprox(b, 1e-10));
}

// ============================================================================
// Segment — invalid construction
// ============================================================================

TEST(SecondOrderSegment, InvalidTimeRange) {
  EXPECT_THROW(
    SecondOrderSegment(
      {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()},
      5.0, 1.0),
    std::invalid_argument);
}

TEST(SecondOrderSegment, WrongCoefficientCount) {
  // 2 coefficients instead of 3
  EXPECT_THROW(
    SecondOrderSegment(
      {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()},
      0.0, 1.0),
    std::invalid_argument);
}

// ============================================================================
// SecondOrderSegment — degenerate / edge case tests
// ============================================================================

TEST(SecondOrderSegment, StationaryDronesKnownDistance) {
  // Two stationary drones at (0,0,0) and (0,2,0)
  // All-zero velocity & acceleration: should return constant distance = 4
  SecondOrderSegment seg_a(
    {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d(0, 0, 0)},
    0.0, 5.0);
  SecondOrderSegment seg_b(
    {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d(0, 2, 0)},
    0.0, 5.0);

  double min_dist_sq = seg_a.get_min_distance_square_to_segment(seg_b);
  EXPECT_NEAR(min_dist_sq, 4.0, 1e-10);
}

TEST(SecondOrderSegment, PartialTimeOverlap) {
  // Seg A: [0, 3], Seg B: [2, 5] — overlap [2, 3]
  // A: p(t) = (t, 0, 0), B: p(t) = (0, t, 0)
  // d(t) = (t, -t, 0), ||d||^2 = 2t^2, min at t=2 → 8
  SecondOrderSegment seg_a(
    {Eigen::Vector3d::Zero(), Eigen::Vector3d(1, 0, 0), Eigen::Vector3d::Zero()},
    0.0, 3.0);
  SecondOrderSegment seg_b(
    {Eigen::Vector3d::Zero(), Eigen::Vector3d(0, 1, 0), Eigen::Vector3d::Zero()},
    2.0, 5.0);

  double min_dist_sq = seg_a.get_min_distance_square_to_segment(seg_b);
  EXPECT_NEAR(min_dist_sq, 8.0, 1e-6);
}

TEST(SecondOrderSegment, IdenticalSegmentsZeroDistance) {
  // Same trajectory → distance should be 0
  SecondOrderSegment seg(
    {Eigen::Vector3d(0.5, 0, 0), Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 0, 0)},
    0.0, 2.0);

  double min_dist_sq = seg.get_min_distance_square_to_segment(seg);
  EXPECT_NEAR(min_dist_sq, 0.0, 1e-10);
}

TEST(SecondOrderSegment, FuzzMinDistVsBruteForce) {
  srand(42);
  for (int trial = 0; trial < 50; ++trial) {
    auto rv = [](double scale) {
      return Eigen::Vector3d(
        scale * (rand() / (double)RAND_MAX - 0.5),
        scale * (rand() / (double)RAND_MAX - 0.5),
        scale * (rand() / (double)RAND_MAX - 0.5));
    };
    double t0 = (rand() / (double)RAND_MAX) * 2.0;
    double t1 = t0 + 0.5 + (rand() / (double)RAND_MAX) * 3.0;

    SecondOrderSegment seg_a({rv(2.0), rv(3.0), rv(5.0)}, t0, t1);
    SecondOrderSegment seg_b({rv(2.0), rv(3.0), rv(5.0)}, t0, t1);

    double analytical = seg_a.get_min_distance_square_to_segment(seg_b);

    double sampled = 1e18;
    for (int i = 0; i <= 10000; ++i) {
      double t = t0 + (t1 - t0) * i / 10000.0;
      double d = (seg_a.get_point(t) - seg_b.get_point(t)).squaredNorm();
      sampled = std::min(sampled, d);
    }

    EXPECT_LE(analytical, sampled + 1e-3)
      << "Fuzz trial " << trial << " on [" << t0 << "," << t1 << "]";
    EXPECT_NEAR(analytical, sampled, 0.1)
      << "Fuzz trial " << trial << " diverged too far from brute force";
  }
}
