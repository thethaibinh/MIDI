/**
 * Unit tests for ExplorationCost (trajectory_cost.hpp):
 *   - Direction cost: negative dot product of normalized vectors
 *   - Distance cost: Euclidean distance to goal
 *   - Cost wrapper dispatch
 *
 * Copyright 2024 by Binh Nguyen <thethaibinh@gmail.com>
 * Licensed under CC BY-NC 4.0.
 */
#include <gtest/gtest.h>
#include <cmath>
#include <Eigen/Dense>

#include <depth_uncertainty_planner/trajectory_cost.hpp>

using namespace autopilot;

// ============================================================================
// Direction cost
// ============================================================================

TEST(ExplorationCost, DirectionCost_SameDirection) {
  // Endpoint in same direction as exploration → cost = -1 (best)
  ExplorationCost cost(Eigen::Vector3d(0, 0, 1), TravelingCost::DIRECTION);
  Eigen::Vector3d endpoint(0, 0, 5);
  EXPECT_NEAR(cost.get_direction_cost(endpoint), -1.0, 1e-10);
}

TEST(ExplorationCost, DirectionCost_OppositeDirection) {
  // Endpoint in opposite direction → cost = +1 (worst)
  ExplorationCost cost(Eigen::Vector3d(0, 0, 1), TravelingCost::DIRECTION);
  Eigen::Vector3d endpoint(0, 0, -3);
  EXPECT_NEAR(cost.get_direction_cost(endpoint), 1.0, 1e-10);
}

TEST(ExplorationCost, DirectionCost_Perpendicular) {
  // Endpoint perpendicular → cost = 0
  ExplorationCost cost(Eigen::Vector3d(1, 0, 0), TravelingCost::DIRECTION);
  Eigen::Vector3d endpoint(0, 1, 0);
  EXPECT_NEAR(cost.get_direction_cost(endpoint), 0.0, 1e-10);
}

TEST(ExplorationCost, DirectionCost_45Degrees) {
  // 45° angle → cost = -cos(45°) = -sqrt(2)/2
  ExplorationCost cost(Eigen::Vector3d(1, 0, 0), TravelingCost::DIRECTION);
  Eigen::Vector3d endpoint(1, 1, 0);
  EXPECT_NEAR(cost.get_direction_cost(endpoint), -std::sqrt(2.0) / 2.0, 1e-6);
}

TEST(ExplorationCost, DirectionCost_MagnitudeInvariant) {
  // Cost depends only on direction, not magnitude
  ExplorationCost cost(Eigen::Vector3d(0, 0, 1), TravelingCost::DIRECTION);
  Eigen::Vector3d ep1(0, 0, 1);
  Eigen::Vector3d ep2(0, 0, 100);
  EXPECT_NEAR(cost.get_direction_cost(ep1), cost.get_direction_cost(ep2), 1e-10);
}

// ============================================================================
// Distance cost
// ============================================================================

TEST(ExplorationCost, DistanceCost_AtGoal) {
  ExplorationCost cost(Eigen::Vector3d(5, 0, 0), TravelingCost::DISTANCE);
  Eigen::Vector3d endpoint(5, 0, 0);
  EXPECT_NEAR(cost.get_distance_to_goal_cost(endpoint), 0.0, 1e-10);
}

TEST(ExplorationCost, DistanceCost_KnownDistance) {
  ExplorationCost cost(Eigen::Vector3d(3, 4, 0), TravelingCost::DISTANCE);
  Eigen::Vector3d endpoint(0, 0, 0);
  EXPECT_NEAR(cost.get_distance_to_goal_cost(endpoint), 5.0, 1e-10);
}

TEST(ExplorationCost, DistanceCost_3D) {
  ExplorationCost cost(Eigen::Vector3d(1, 2, 3), TravelingCost::DISTANCE);
  Eigen::Vector3d endpoint(4, 6, 3);
  // sqrt((4-1)^2 + (6-2)^2 + 0^2) = sqrt(9+16) = 5
  EXPECT_NEAR(cost.get_distance_to_goal_cost(endpoint), 5.0, 1e-10);
}

// ============================================================================
// Cost wrapper dispatch
// ============================================================================

TEST(ExplorationCost, WrapperDispatchesDirection) {
  ExplorationCost cost(Eigen::Vector3d(0, 0, 1), TravelingCost::DIRECTION);
  Eigen::Vector3d endpoint(0, 0, 5);
  double wrapper_val = ExplorationCost::get_cost_wrapper(&cost, endpoint);
  double direct_val = cost.get_direction_cost(endpoint);
  EXPECT_NEAR(wrapper_val, direct_val, 1e-10);
}

TEST(ExplorationCost, WrapperDispatchesDistance) {
  ExplorationCost cost(Eigen::Vector3d(3, 4, 0), TravelingCost::DISTANCE);
  Eigen::Vector3d endpoint(0, 0, 0);
  double wrapper_val = ExplorationCost::get_cost_wrapper(&cost, endpoint);
  double direct_val = cost.get_distance_to_goal_cost(endpoint);
  EXPECT_NEAR(wrapper_val, direct_val, 1e-10);
}

TEST(ExplorationCost, GetTravelingCost) {
  ExplorationCost dir_cost(Eigen::Vector3d::UnitX(), TravelingCost::DIRECTION);
  ExplorationCost dist_cost(Eigen::Vector3d::UnitX(), TravelingCost::DISTANCE);
  EXPECT_EQ(dir_cost.get_traveling_cost(), TravelingCost::DIRECTION);
  EXPECT_EQ(dist_cost.get_traveling_cost(), TravelingCost::DISTANCE);
}
