/**
 * Unit tests for common_math polynomial root-finding:
 *   - SecondOrderPolynomial (polynomial2.hpp)
 *   - ThirdOrderPolynomial  (polynomial3.hpp)
 *   - FourthOrderPolynomial (polynomial4.hpp)
 *
 * Copyright 2024 by Binh Nguyen <thethaibinh@gmail.com>
 * Licensed under CC BY-NC 4.0.
 */
#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>
#include <vector>

#include <common_math/polynomial2.hpp>
#include <common_math/polynomial3.hpp>
#include <common_math/polynomial4.hpp>

// ============================================================================
// SecondOrderPolynomial
// ============================================================================

TEST(SecondOrderPolynomial, KnownRoots) {
  // (x-1)(x-3) = x^2 - 4x + 3
  SecondOrderPolynomial p({1.0, -4.0, 3.0}, 0.0, 5.0);

  std::vector<double> roots;
  p.solve_roots(roots);
  std::sort(roots.begin(), roots.end());

  ASSERT_EQ(roots.size(), 2u);
  EXPECT_NEAR(roots[0], 1.0, 1e-10);
  EXPECT_NEAR(roots[1], 3.0, 1e-10);
}

TEST(SecondOrderPolynomial, RootsOutsideRange) {
  // (x-1)(x-3) on [1.5, 2.5] — roots excluded
  SecondOrderPolynomial p({1.0, -4.0, 3.0}, 1.5, 2.5);

  std::vector<double> roots;
  p.solve_roots(roots);
  EXPECT_EQ(roots.size(), 0u);
}

TEST(SecondOrderPolynomial, NoRealRoots) {
  // x^2 + 1 has no real roots
  SecondOrderPolynomial p({1.0, 0.0, 1.0}, -10.0, 10.0);

  std::vector<double> roots;
  p.solve_roots(roots);
  EXPECT_EQ(roots.size(), 0u);
}

TEST(SecondOrderPolynomial, GetMinAtVertex) {
  // x^2 - 4x + 5 has minimum at x=2 with value 1
  SecondOrderPolynomial p({1.0, -4.0, 5.0}, 0.0, 5.0);
  EXPECT_NEAR(p.get_min(), 1.0, 1e-10);
}

TEST(SecondOrderPolynomial, GetMinVsSampling) {
  // 2x^2 - 3x + 7 on [-1, 4]
  SecondOrderPolynomial p({2.0, -3.0, 7.0}, -1.0, 4.0);

  double analytical = p.get_min();

  // Brute force sampling
  double sampled_min = 1e18;
  for (int i = 0; i <= 10000; ++i) {
    double t = -1.0 + 5.0 * i / 10000.0;
    double val = p.get_value(t);
    sampled_min = std::min(sampled_min, val);
  }

  EXPECT_NEAR(analytical, sampled_min, 1e-3);
}

TEST(SecondOrderPolynomial, LinearDegenerate) {
  // 0*x^2 + 2x - 6 = 0 => x = 3
  SecondOrderPolynomial p({0.0, 2.0, -6.0}, 0.0, 10.0);

  std::vector<double> roots;
  p.solve_roots(roots);
  ASSERT_EQ(roots.size(), 1u);
  EXPECT_NEAR(roots[0], 3.0, 1e-10);
}

TEST(SecondOrderPolynomial, DerivativeRoots) {
  // 3x^2 - 12x + 5, derivative = 6x - 12 = 0 => x = 2
  SecondOrderPolynomial p({3.0, -12.0, 5.0}, 0.0, 5.0);

  std::vector<double> droots;
  p.solve_derivative_roots(droots);
  ASSERT_EQ(droots.size(), 1u);
  EXPECT_NEAR(droots[0], 2.0, 1e-10);
}

// ============================================================================
// ThirdOrderPolynomial
// ============================================================================

TEST(ThirdOrderPolynomial, ThreeRealRoots) {
  // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
  ThirdOrderPolynomial p({1.0, -6.0, 11.0, -6.0}, 0.0, 5.0);

  std::vector<double> roots;
  p.solve_roots(roots);
  std::sort(roots.begin(), roots.end());

  ASSERT_EQ(roots.size(), 3u);
  EXPECT_NEAR(roots[0], 1.0, 1e-6);
  EXPECT_NEAR(roots[1], 2.0, 1e-6);
  EXPECT_NEAR(roots[2], 3.0, 1e-6);
}

TEST(ThirdOrderPolynomial, OneRealRoot) {
  // x^3 + x + 2 = 0, one real root at x = -cbrt(2) ~ -1.2599
  ThirdOrderPolynomial p({1.0, 0.0, 0.0, 2.0}, -5.0, 5.0);

  std::vector<double> roots;
  p.solve_roots(roots);
  ASSERT_EQ(roots.size(), 1u);
  EXPECT_NEAR(roots[0], -std::cbrt(2.0), 1e-6);
}

TEST(ThirdOrderPolynomial, GetMinVsSampling) {
  // x^3 - 3x^2 + 4 on [-2, 3]
  ThirdOrderPolynomial p({1.0, -3.0, 0.0, 4.0}, -2.0, 3.0);

  double analytical = p.get_min();

  double sampled_min = 1e18;
  for (int i = 0; i <= 10000; ++i) {
    double t = -2.0 + 5.0 * i / 10000.0;
    sampled_min = std::min(sampled_min, p.get_value(t));
  }

  EXPECT_NEAR(analytical, sampled_min, 1e-2);
}

TEST(ThirdOrderPolynomial, DerivativeRoots) {
  // x^3 - 6x^2 + 11x - 6, derivative = 3x^2 - 12x + 11
  // Roots: (12 ± sqrt(144-132)) / 6 = (12 ± sqrt(12)) / 6
  ThirdOrderPolynomial p({1.0, -6.0, 11.0, -6.0}, 0.0, 5.0);

  std::vector<double> droots;
  p.solve_derivative_roots(droots);
  std::sort(droots.begin(), droots.end());
  ASSERT_EQ(droots.size(), 2u);
  EXPECT_NEAR(droots[0], (12.0 - std::sqrt(12.0)) / 6.0, 1e-6);
  EXPECT_NEAR(droots[1], (12.0 + std::sqrt(12.0)) / 6.0, 1e-6);
}

// ============================================================================
// FourthOrderPolynomial
// ============================================================================

TEST(FourthOrderPolynomial, FourDistinctRoots) {
  // (x-1)(x-2)(x-3)(x-4) = x^4 - 10x^3 + 35x^2 - 50x + 24
  FourthOrderPolynomial p({1.0, -10.0, 35.0, -50.0, 24.0}, 0.0, 5.0);

  // get_min() should find the minimum value over [0, 5]
  double min_val = p.get_min();
  // Minimum is between roots, brute-force check:
  double sampled_min = 1e18;
  for (int i = 0; i <= 10000; ++i) {
    double t = 5.0 * i / 10000.0;
    sampled_min = std::min(sampled_min, p.get_value(t));
  }
  EXPECT_NEAR(min_val, sampled_min, 1e-2);
}

TEST(FourthOrderPolynomial, PerfectSquare) {
  // (x^2 - 1)^2 = x^4 - 2x^2 + 1, min = 0 at x = ±1
  FourthOrderPolynomial p({1.0, 0.0, -2.0, 0.0, 1.0}, -2.0, 2.0);
  EXPECT_NEAR(p.get_min(), 0.0, 1e-6);
}

TEST(FourthOrderPolynomial, GetMinVsSampling) {
  // Random quartic on [-1, 3]
  FourthOrderPolynomial p({0.5, -2.0, 1.0, 3.0, -1.0}, -1.0, 3.0);

  double analytical = p.get_min();
  double sampled_min = 1e18;
  for (int i = 0; i <= 50000; ++i) {
    double t = -1.0 + 4.0 * i / 50000.0;
    sampled_min = std::min(sampled_min, p.get_value(t));
  }

  EXPECT_NEAR(analytical, sampled_min, 1e-2);
  EXPECT_LE(analytical, sampled_min + 1e-4);
}

TEST(FourthOrderPolynomial, GetMax) {
  // (x-1.5)^4 on [0, 3], max at endpoints
  // At x=0: 1.5^4 = 5.0625, at x=3: 1.5^4 = 5.0625
  FourthOrderPolynomial p({1.0, -6.0, 13.5, -13.5, 5.0625}, 0.0, 3.0);

  EXPECT_NEAR(p.get_max(), 5.0625, 1e-3);
  EXPECT_NEAR(p.get_min(), 0.0, 1e-3);
}

// ============================================================================
// Polynomial base class: get_extremes_and_terminals, get_derivative_coeffs
// ============================================================================

TEST(PolynomialBase, ExtremesAndTerminals) {
  // x^2 - 4x + 5 on [0, 5]: values at 0, 2 (vertex), 5
  SecondOrderPolynomial p({1.0, -4.0, 5.0}, 0.0, 5.0);

  auto vals = p.get_extremes_and_terminals();
  // Should contain: p(0)=5, p(2)=1, p(5)=10
  ASSERT_GE(vals.size(), 3u);

  double min_val = *std::min_element(vals.begin(), vals.end());
  double max_val = *std::max_element(vals.begin(), vals.end());
  EXPECT_NEAR(min_val, 1.0, 1e-10);
  EXPECT_NEAR(max_val, 10.0, 1e-10);
}

TEST(PolynomialBase, InvalidTimeRange) {
  EXPECT_THROW(SecondOrderPolynomial({1.0, 0.0, 0.0}, 5.0, 1.0), std::invalid_argument);
}
