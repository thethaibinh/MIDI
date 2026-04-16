/**
 * @file test_planner_state_machine.cpp
 * @brief Unit tests for PlanningStates transitions that are NOT OPUS-specific.
 *
 * Companion to test_opus_state_machine.cpp — covers the broader state machine:
 *   1. TAKING_OFF → ALIGNING_HEADING (altitude check)
 *   2. HOLDING_WAYPOINT → ALIGNING_HEADING (hold_time expiry, waypoint advance)
 *   3. HOLDING_WAYPOINT → GO_TO_GOAL (mission complete after last waypoint)
 *   4. GO_TO_GOAL → FINISHED (trajectory duration elapsed)
 *   5. FINISHED + reinitialise → OFF (position + heading convergence)
 *   6. BRAKE dead-end (no transitions out)
 *   7. LAND → OFF (OmniDrones altitude convergence)
 *   8. reset_planner() completeness
 *   9. brake_callback / land_swarm_callback guard conditions
 *  10. advance_waypoint loop logic
 *
 * No ROS2 / Ruckig / sensor dependencies — pure state + flag logic.
 */

#include <gtest/gtest.h>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <vector>

#include <Eigen/Core>

#include "autopilot_states.h"

using autopilot::PlanningStates;
using autopilot::RuntimeModes;

// ============================================================================
// Lightweight model of planner state + the transition logic under test.
// Extracted from planner_node_state_machine.cpp / planner_node_mission.cpp.
// ============================================================================

struct PlannerStateMachine {
  // --- Core state ---
  std::atomic<PlanningStates> planner_state{PlanningStates::OFF};
  RuntimeModes runtime_mode{RuntimeModes::OMNIDRONES};

  // --- Position / orientation (simulated) ---
  Eigen::Vector3d position{0.0, 0.0, 0.0};
  double current_yaw{0.0};  // radians

  // --- Goal ---
  Eigen::Vector3d goal{0.0, 0.0, 0.0};
  double goal_heading{0.0};
  double goal_up_coordinate{0.0};
  bool goal_set{false};

  // --- Home / reinitialise ---
  Eigen::Vector3d home{0.0, 0.0, 0.0};
  double initial_heading{0.0};
  bool reinitialise_requested{false};

  // --- Mission flags ---
  bool mission_received{false};
  bool mission_uploaded{false};
  bool takeoff_requested{false};

  // --- Waypoint mission ---
  bool waypoint_mission_active{false};
  size_t current_waypoint_index{0};
  uint32_t remaining_loops{0};
  std::vector<Eigen::Vector3d> world_waypoints;
  std::vector<double> waypoint_headings;
  std::vector<double> waypoint_hold_times;

  // --- Trajectory tracking ---
  bool had_reference_trajectory{false};
  double trajectory_duration{0.0};   // seconds
  double trajectory_elapsed{0.0};    // seconds since trajectory start

  // --- OPUS (minimal, for reset test) ---
  bool opus_enabled{false};
  std::mutex opus_mutex;
  bool opus_granted{false};
  bool opus_check_pending{false};
  bool opus_lock_pending{false};
  bool opus_submission_needed{false};
  uint32_t opus_plan_sequence{0};

  // --- Misc flags ---
  bool mode_switch_pending{false};
  bool arming_pending{false};
  bool takeoff_pending{false};
  bool land_pending{false};
  bool brake_mode_switch_sent{false};
  float steering_value{0.0f};
  bool steered{false};
  bool has_valid_setpoint{false};

  // --- Timing (simulated) ---
  double time_in_current_state{0.0};  // seconds since last state switch

  // --- Thresholds (match planner defaults) ---
  static constexpr double kHeadingAlignThreshold = 0.17;  // ~10 deg
  static constexpr double kGoToGoalThreshold = 0.5;
  static constexpr double kLandAltitudeThreshold = 0.1;
  static constexpr double kReinitPositionThreshold = 0.1;

  // ==========================================================================
  // State transitions (extracted from update_planner_state)
  // ==========================================================================

  void set_state(PlanningStates s) {
    planner_state.store(s, std::memory_order_release);
    time_in_current_state = 0.0;
  }

  /// TAKING_OFF → ALIGNING_HEADING when altitude reached
  void check_takeoff_complete() {
    if (planner_state != PlanningStates::TAKING_OFF) return;
    double takeoff_complete_altitude = goal_up_coordinate - 0.1;
    if (position.z() >= takeoff_complete_altitude) {
      if (waypoint_mission_active && current_waypoint_index < waypoint_headings.size()) {
        goal_heading = waypoint_headings[current_waypoint_index];
      } else {
        // compute_heading_to_goal equivalent
        double dx = goal.x() - position.x();
        double dy = goal.y() - position.y();
        goal_heading = std::atan2(dy, dx);
      }
      set_state(PlanningStates::ALIGNING_HEADING);
    }
  }

  /// ALIGNING_HEADING → WAITING_FOR_OPUS or TRAJECTORY_CONTROL
  void check_heading_aligned() {
    if (planner_state != PlanningStates::ALIGNING_HEADING) return;
    double yaw_error = goal_heading - current_yaw;
    while (yaw_error > M_PI) yaw_error -= 2.0 * M_PI;
    while (yaw_error < -M_PI) yaw_error += 2.0 * M_PI;

    if (std::abs(yaw_error) < kHeadingAlignThreshold) {
      had_reference_trajectory = false;
      if (opus_enabled) {
        const std::lock_guard<std::mutex> lock(opus_mutex);
        opus_submission_needed = true;
        set_state(PlanningStates::WAITING_FOR_OPUS);
      } else {
        set_state(PlanningStates::TRAJECTORY_CONTROL);
      }
    }
  }

  /// TRAJECTORY_CONTROL → HOLDING_WAYPOINT / GO_TO_GOAL on near goal OR passing half-space
  void check_goal_reached() {
    if (planner_state != PlanningStates::TRAJECTORY_CONTROL) return;

    // Half-space check: (drone_xy - goal_xy) · (goal_xy - prev_xy) > 0
    Eigen::Vector2d segment_start_xy;
    if (waypoint_mission_active && current_waypoint_index > 0) {
      segment_start_xy << world_waypoints[current_waypoint_index - 1].x(),
                          world_waypoints[current_waypoint_index - 1].y();
    } else {
      segment_start_xy << home.x(), home.y();
    }
    Eigen::Vector2d goal_xy(goal.x(), goal.y());
    Eigen::Vector2d drone_xy(position.x(), position.y());
    Eigen::Vector2d seg_dir = goal_xy - segment_start_xy;
    double seg_len_sq = seg_dir.squaredNorm();

    double dx = position.x() - goal.x();
    double dy = position.y() - goal.y();
    double distance = std::sqrt(dx * dx + dy * dy);

    bool passed = (seg_len_sq > 1e-6)
        ? (drone_xy - goal_xy).dot(seg_dir) > 0.0
        : false;

    if (distance < kGoToGoalThreshold || passed) {
      if (waypoint_mission_active) {
        set_state(PlanningStates::HOLDING_WAYPOINT);
      } else {
        set_state(PlanningStates::GO_TO_GOAL);
      }
    }
  }

  /// HOLDING_WAYPOINT: after hold_time, advance or finish
  /// Returns true if state changed.
  bool check_holding_waypoint() {
    if (planner_state != PlanningStates::HOLDING_WAYPOINT) return false;

    double hold_time = 1.0;
    if (current_waypoint_index < waypoint_hold_times.size()) {
      hold_time = waypoint_hold_times[current_waypoint_index];
    }

    if (time_in_current_state >= hold_time) {
      if (advance_waypoint()) {
        set_state(PlanningStates::ALIGNING_HEADING);
      } else {
        set_state(PlanningStates::GO_TO_GOAL);
      }
      return true;
    }
    return false;
  }

  /// GO_TO_GOAL → FINISHED
  void check_trajectory_complete() {
    if (planner_state != PlanningStates::GO_TO_GOAL) return;
    if (!had_reference_trajectory) return;
    if (trajectory_elapsed > trajectory_duration) {
      set_state(PlanningStates::FINISHED);
    }
  }

  /// FINISHED + reinitialise → OFF
  void check_reinitialise_complete() {
    if (planner_state != PlanningStates::FINISHED) return;
    if (!reinitialise_requested) return;

    double dist_to_home = (position - home).norm();
    double yaw_error = initial_heading - current_yaw;
    while (yaw_error > M_PI) yaw_error -= 2.0 * M_PI;
    while (yaw_error < -M_PI) yaw_error += 2.0 * M_PI;

    if (dist_to_home < kReinitPositionThreshold &&
        std::abs(yaw_error) < kHeadingAlignThreshold) {
      reset_planner();
    }
  }

  /// LAND → OFF (OmniDrones)
  void check_land_complete() {
    if (planner_state != PlanningStates::LAND) return;
    if (runtime_mode != RuntimeModes::OMNIDRONES) return;
    double altitude_error = std::abs(position.z() - home.z());
    if (altitude_error < kLandAltitudeThreshold) {
      reset_planner();
    }
  }

  // ==========================================================================
  // Callbacks (extracted from planner_node_mission.cpp)
  // ==========================================================================

  /// brake_callback guard and transition
  bool try_brake() {
    auto s = planner_state.load();
    if (s == PlanningStates::LAND || s == PlanningStates::FINISHED ||
        s == PlanningStates::OFF || s == PlanningStates::BRAKE) {
      return false;  // rejected
    }
    had_reference_trajectory = false;
    set_state(PlanningStates::BRAKE);
    return true;
  }

  /// land_swarm_callback guard and transition (OmniDrones path)
  bool try_land() {
    auto s = planner_state.load();
    if (s != PlanningStates::FINISHED && s != PlanningStates::BRAKE) {
      return false;  // rejected
    }
    had_reference_trajectory = false;
    set_state(PlanningStates::LAND);
    return true;
  }

  /// reinitialise_callback guard
  bool try_reinitialise() {
    if (planner_state != PlanningStates::FINISHED) return false;
    goal = home;
    goal_heading = initial_heading;
    reinitialise_requested = true;
    return true;
  }

  // ==========================================================================
  // Waypoint helpers
  // ==========================================================================

  bool advance_waypoint() {
    current_waypoint_index++;
    if (current_waypoint_index >= world_waypoints.size()) {
      if (remaining_loops > 0) {
        remaining_loops--;
        current_waypoint_index = 0;
      } else {
        waypoint_mission_active = false;
        return false;
      }
    }
    // set_goal_from_waypoint equivalent
    const auto& wp = world_waypoints[current_waypoint_index];
    goal = wp;
    goal_heading = waypoint_headings[current_waypoint_index];
    return true;
  }

  // ==========================================================================
  // reset_planner (extracted from planner_node_mission.cpp)
  // ==========================================================================

  void reset_planner() {
    set_state(PlanningStates::OFF);
    steering_value = 0.0f;
    steered = false;
    goal_set = false;
    mission_received = false;
    mission_uploaded = false;
    mode_switch_pending = false;
    arming_pending = false;
    takeoff_pending = false;
    land_pending = false;
    takeoff_requested = false;
    reinitialise_requested = false;
    brake_mode_switch_sent = false;
    had_reference_trajectory = false;
    has_valid_setpoint = false;
    goal_up_coordinate = 0.0;
    // Waypoint state
    world_waypoints.clear();
    waypoint_headings.clear();
    waypoint_hold_times.clear();
    current_waypoint_index = 0;
    remaining_loops = 0;
    waypoint_mission_active = false;
    // OPUS
    if (opus_enabled) {
      const std::lock_guard<std::mutex> lock(opus_mutex);
      opus_granted = false;
      opus_check_pending = false;
      opus_lock_pending = false;
      opus_submission_needed = false;
      ++opus_plan_sequence;
    }
  }

  // ==========================================================================
  // Test helpers
  // ==========================================================================

  /// Set up a simple 3-waypoint mission
  void setup_waypoint_mission(uint32_t loops = 0) {
    world_waypoints = {
      Eigen::Vector3d(1.0, 0.0, 1.5),
      Eigen::Vector3d(2.0, 1.0, 1.5),
      Eigen::Vector3d(3.0, 0.0, 1.5),
    };
    waypoint_headings = {0.0, M_PI / 4.0, -M_PI / 4.0};
    waypoint_hold_times = {1.0, 2.0, 1.5};
    current_waypoint_index = 0;
    remaining_loops = loops;
    waypoint_mission_active = true;
    goal = world_waypoints[0];
    goal_heading = waypoint_headings[0];
    goal_set = true;
    goal_up_coordinate = 1.5;
    home = Eigen::Vector3d(0.0, 0.0, 0.0);
  }
};


// ============================================================================
// TAKING_OFF → ALIGNING_HEADING
// ============================================================================

TEST(PlannerStateMachine, TakeoffToAligningWhenAltitudeReached) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::TAKING_OFF);
  sm.goal_up_coordinate = 1.5;
  sm.goal = Eigen::Vector3d(5.0, 0.0, 1.5);
  sm.position = Eigen::Vector3d(0.0, 0.0, 0.5);  // below threshold

  sm.check_takeoff_complete();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::TAKING_OFF);

  // Reach threshold (goal_up - 0.1 = 1.4)
  sm.position.z() = 1.4;
  sm.check_takeoff_complete();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::ALIGNING_HEADING);
}

TEST(PlannerStateMachine, TakeoffToAligningAboveThreshold) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::TAKING_OFF);
  sm.goal_up_coordinate = 1.5;
  sm.goal = Eigen::Vector3d(5.0, 0.0, 1.5);
  sm.position = Eigen::Vector3d(0.0, 0.0, 2.0);  // overshoot

  sm.check_takeoff_complete();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::ALIGNING_HEADING);
}

TEST(PlannerStateMachine, TakeoffUsesWaypointHeadingWhenActive) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::TAKING_OFF);
  sm.setup_waypoint_mission();
  sm.position = Eigen::Vector3d(0.0, 0.0, 1.5);

  sm.check_takeoff_complete();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::ALIGNING_HEADING);
  EXPECT_DOUBLE_EQ(sm.goal_heading, sm.waypoint_headings[0]);
}

TEST(PlannerStateMachine, TakeoffComputesHeadingWhenNoWaypointMission) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::TAKING_OFF);
  sm.goal_up_coordinate = 1.5;
  sm.goal = Eigen::Vector3d(0.0, 5.0, 1.5);  // due north
  sm.position = Eigen::Vector3d(0.0, 0.0, 1.5);

  sm.check_takeoff_complete();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::ALIGNING_HEADING);
  EXPECT_NEAR(sm.goal_heading, M_PI / 2.0, 0.01);  // atan2(5,0) = π/2
}

TEST(PlannerStateMachine, TakeoffIgnoredFromWrongState) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::TRAJECTORY_CONTROL);
  sm.goal_up_coordinate = 1.5;
  sm.position = Eigen::Vector3d(0.0, 0.0, 2.0);

  sm.check_takeoff_complete();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::TRAJECTORY_CONTROL);
}


// ============================================================================
// ALIGNING_HEADING → TRAJECTORY_CONTROL (no OPUS)
// ============================================================================

TEST(PlannerStateMachine, AligningToTrajectoryControlWhenHeadingMatched) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::ALIGNING_HEADING);
  sm.goal_heading = 1.0;
  sm.current_yaw = 1.05;  // within 0.17 rad threshold
  sm.opus_enabled = false;

  sm.check_heading_aligned();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::TRAJECTORY_CONTROL);
  EXPECT_FALSE(sm.had_reference_trajectory);
}

TEST(PlannerStateMachine, AligningStaysWhenHeadingNotAligned) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::ALIGNING_HEADING);
  sm.goal_heading = 1.0;
  sm.current_yaw = 0.5;  // error = 0.5, above threshold
  sm.opus_enabled = false;

  sm.check_heading_aligned();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::ALIGNING_HEADING);
}

TEST(PlannerStateMachine, AligningToWaitingForOpusWhenEnabled) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::ALIGNING_HEADING);
  sm.goal_heading = 0.0;
  sm.current_yaw = 0.05;
  sm.opus_enabled = true;

  sm.check_heading_aligned();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::WAITING_FOR_OPUS);
  EXPECT_TRUE(sm.opus_submission_needed);
}

TEST(PlannerStateMachine, HeadingWrapAround) {
  // Goal near +π, current near -π  →  small error after wrapping
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::ALIGNING_HEADING);
  sm.goal_heading = M_PI - 0.05;
  sm.current_yaw = -M_PI + 0.05;  // actual diff = 0.1 (< 0.17)
  sm.opus_enabled = false;

  sm.check_heading_aligned();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::TRAJECTORY_CONTROL);
}


// ============================================================================
// TRAJECTORY_CONTROL → HOLDING_WAYPOINT / GO_TO_GOAL
// ============================================================================

TEST(PlannerStateMachine, GoalReachedWaypointMission) {
  // Drone has crossed the perpendicular line at the waypoint (past goal along segment)
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::TRAJECTORY_CONTROL);
  sm.waypoint_mission_active = true;
  sm.home = Eigen::Vector3d(0.0, 0.0, 1.5);      // segment start
  sm.goal = Eigen::Vector3d(1.0, 0.0, 1.5);       // segment end
  sm.position = Eigen::Vector3d(1.1, 0.1, 1.5);   // past the goal along +X

  sm.check_goal_reached();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::HOLDING_WAYPOINT);
}

TEST(PlannerStateMachine, GoalReachedFinalGoal) {
  // Drone has crossed the perpendicular line at the final goal
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::TRAJECTORY_CONTROL);
  sm.waypoint_mission_active = false;
  sm.home = Eigen::Vector3d(0.0, 0.0, 1.5);      // segment start
  sm.goal = Eigen::Vector3d(1.0, 0.0, 1.5);       // segment end
  sm.position = Eigen::Vector3d(1.01, 0.0, 1.5);  // just past the goal

  sm.check_goal_reached();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::GO_TO_GOAL);
}

TEST(PlannerStateMachine, GoalNotReachedStaysInTrajectoryControl) {
  // Drone is still on the start side of the perpendicular line at the goal
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::TRAJECTORY_CONTROL);
  sm.home = Eigen::Vector3d(0.0, 0.0, 1.5);       // segment start
  sm.goal = Eigen::Vector3d(10.0, 0.0, 1.5);       // segment end
  sm.position = Eigen::Vector3d(5.0, 0.0, 1.5);    // halfway — not past goal

  sm.check_goal_reached();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::TRAJECTORY_CONTROL);
}

TEST(PlannerStateMachine, GoalReachedViaHalfSpaceNotDistance) {
  // Drone crosses the perpendicular line but is far from the goal in absolute
  // distance — half-space check triggers anyway (correct behaviour: the drone
  // has overshot sideways but is past the goal along the segment direction).
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::TRAJECTORY_CONTROL);
  sm.waypoint_mission_active = false;
  sm.home = Eigen::Vector3d(0.0, 0.0, 1.5);
  sm.goal = Eigen::Vector3d(10.0, 0.0, 1.5);
  sm.position = Eigen::Vector3d(10.1, 5.0, 1.5);  // past goal, but 5m off-axis

  sm.check_goal_reached();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::GO_TO_GOAL);
}

TEST(PlannerStateMachine, GoalNotReachedFarAndBeforeHalfSpace) {
  // Drone is before the perpendicular line AND farther than the distance
  // threshold — neither condition fires, stays in TRAJECTORY_CONTROL.
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::TRAJECTORY_CONTROL);
  sm.waypoint_mission_active = false;
  sm.home = Eigen::Vector3d(0.0, 0.0, 1.5);
  sm.goal = Eigen::Vector3d(10.0, 0.0, 1.5);
  sm.position = Eigen::Vector3d(8.0, 0.0, 1.5);   // dist = 2.0, before line

  sm.check_goal_reached();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::TRAJECTORY_CONTROL);
}

TEST(PlannerStateMachine, GoalReachedCloseButBeforeHalfSpace) {
  // Drone is within distance threshold but hasn't crossed the half-space —
  // should still transition because distance alone is sufficient.
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::TRAJECTORY_CONTROL);
  sm.waypoint_mission_active = false;
  sm.home = Eigen::Vector3d(0.0, 0.0, 1.5);
  sm.goal = Eigen::Vector3d(10.0, 0.0, 1.5);
  sm.position = Eigen::Vector3d(9.8, 0.1, 1.5);   // dist ~0.22 < 0.5, before line

  sm.check_goal_reached();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::GO_TO_GOAL);
}

TEST(PlannerStateMachine, GoalReachedSecondWaypointUsesFirstAsSegmentStart) {
  // Second waypoint: segment direction is from WP0 to WP1, not from home
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::TRAJECTORY_CONTROL);
  sm.waypoint_mission_active = true;
  sm.home = Eigen::Vector3d(0.0, 0.0, 1.5);
  sm.world_waypoints = {
    Eigen::Vector3d(5.0, 0.0, 1.5),   // WP0
    Eigen::Vector3d(5.0, 5.0, 1.5),   // WP1 — segment goes in +Y from WP0
  };
  sm.current_waypoint_index = 1;
  sm.goal = sm.world_waypoints[1];
  sm.position = Eigen::Vector3d(5.0, 5.1, 1.5);   // past WP1 along +Y

  sm.check_goal_reached();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::HOLDING_WAYPOINT);
}


// ============================================================================
// HOLDING_WAYPOINT → advance / mission complete
// ============================================================================

TEST(PlannerStateMachine, HoldWaypointAdvancesAfterHoldTime) {
  PlannerStateMachine sm;
  sm.setup_waypoint_mission();
  sm.set_state(PlanningStates::HOLDING_WAYPOINT);
  sm.current_waypoint_index = 0;  // hold_time = 1.0

  sm.time_in_current_state = 0.5;
  EXPECT_FALSE(sm.check_holding_waypoint());
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::HOLDING_WAYPOINT);

  sm.time_in_current_state = 1.0;
  EXPECT_TRUE(sm.check_holding_waypoint());
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::ALIGNING_HEADING);
  EXPECT_EQ(sm.current_waypoint_index, 1u);
}

TEST(PlannerStateMachine, HoldWaypointUsesPerWaypointHoldTime) {
  PlannerStateMachine sm;
  sm.setup_waypoint_mission();
  sm.set_state(PlanningStates::HOLDING_WAYPOINT);
  sm.current_waypoint_index = 1;  // hold_time = 2.0

  sm.time_in_current_state = 1.5;
  EXPECT_FALSE(sm.check_holding_waypoint());

  sm.time_in_current_state = 2.0;
  EXPECT_TRUE(sm.check_holding_waypoint());
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::ALIGNING_HEADING);
}

TEST(PlannerStateMachine, HoldWaypointMissionCompleteGoesToGoal) {
  PlannerStateMachine sm;
  sm.setup_waypoint_mission();  // 3 waypoints, 0 loops
  sm.set_state(PlanningStates::HOLDING_WAYPOINT);
  sm.current_waypoint_index = 2;  // last waypoint

  sm.time_in_current_state = 2.0;
  EXPECT_TRUE(sm.check_holding_waypoint());
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::GO_TO_GOAL);
  EXPECT_FALSE(sm.waypoint_mission_active);
}

TEST(PlannerStateMachine, HoldWaypointDefaultHoldTimeWhenMissing) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::HOLDING_WAYPOINT);
  sm.waypoint_mission_active = true;
  // No waypoints set up → index out of bounds → default hold_time = 1.0
  sm.world_waypoints = {};
  sm.waypoint_hold_times = {};

  sm.time_in_current_state = 0.9;
  EXPECT_FALSE(sm.check_holding_waypoint());

  sm.time_in_current_state = 1.0;
  // advance_waypoint will return false (empty waypoints)
  EXPECT_TRUE(sm.check_holding_waypoint());
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::GO_TO_GOAL);
}


// ============================================================================
// GO_TO_GOAL → FINISHED
// ============================================================================

TEST(PlannerStateMachine, GoToGoalFinishesWhenTrajectoryComplete) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::GO_TO_GOAL);
  sm.had_reference_trajectory = true;
  sm.trajectory_duration = 5.0;

  sm.trajectory_elapsed = 3.0;
  sm.check_trajectory_complete();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::GO_TO_GOAL);

  sm.trajectory_elapsed = 5.1;
  sm.check_trajectory_complete();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::FINISHED);
}

TEST(PlannerStateMachine, GoToGoalWaitsForReferenceTrajectory) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::GO_TO_GOAL);
  sm.had_reference_trajectory = false;
  sm.trajectory_elapsed = 100.0;
  sm.trajectory_duration = 1.0;

  sm.check_trajectory_complete();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::GO_TO_GOAL);
}


// ============================================================================
// FINISHED + reinitialise → OFF
// ============================================================================

TEST(PlannerStateMachine, ReinitialiseCompletesWhenAtHome) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::FINISHED);
  sm.home = Eigen::Vector3d(0.0, 0.0, 1.0);
  sm.initial_heading = 0.5;
  sm.reinitialise_requested = true;

  // Far from home
  sm.position = Eigen::Vector3d(5.0, 3.0, 1.0);
  sm.current_yaw = 0.5;
  sm.check_reinitialise_complete();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::FINISHED);

  // Close to home position, yaw still off
  sm.position = Eigen::Vector3d(0.01, 0.01, 1.01);
  sm.current_yaw = 2.0;
  sm.check_reinitialise_complete();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::FINISHED);

  // At home, heading aligned → reset
  sm.current_yaw = 0.5;
  sm.check_reinitialise_complete();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::OFF);
  EXPECT_FALSE(sm.reinitialise_requested);
}

TEST(PlannerStateMachine, ReinitialiseIgnoredWhenNotRequested) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::FINISHED);
  sm.home = Eigen::Vector3d(0.0, 0.0, 1.0);
  sm.position = sm.home;
  sm.initial_heading = 0.0;
  sm.current_yaw = 0.0;
  sm.reinitialise_requested = false;

  sm.check_reinitialise_complete();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::FINISHED);
}

TEST(PlannerStateMachine, ReinitialiseHeadingWrapAround) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::FINISHED);
  sm.home = Eigen::Vector3d(0.0, 0.0, 0.0);
  sm.position = sm.home;
  sm.initial_heading = M_PI - 0.05;
  sm.current_yaw = -M_PI + 0.05;
  sm.reinitialise_requested = true;

  sm.check_reinitialise_complete();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::OFF);
}


// ============================================================================
// BRAKE dead-end
// ============================================================================

TEST(PlannerStateMachine, BrakeIsDeadEnd) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::BRAKE);
  sm.goal_set = true;
  sm.goal_up_coordinate = 1.5;
  sm.position = Eigen::Vector3d(0.0, 0.0, 1.5);
  sm.goal = sm.position;
  sm.had_reference_trajectory = true;
  sm.trajectory_elapsed = 100.0;
  sm.trajectory_duration = 1.0;

  // None of the normal transitions should fire from BRAKE
  sm.check_takeoff_complete();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::BRAKE);

  sm.check_heading_aligned();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::BRAKE);

  sm.check_goal_reached();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::BRAKE);

  sm.check_trajectory_complete();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::BRAKE);

  sm.check_reinitialise_complete();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::BRAKE);

  sm.check_land_complete();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::BRAKE);
}

TEST(PlannerStateMachine, BrakeAcceptedFromTrajectoryControl) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::TRAJECTORY_CONTROL);
  EXPECT_TRUE(sm.try_brake());
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::BRAKE);
  EXPECT_FALSE(sm.had_reference_trajectory);
}

TEST(PlannerStateMachine, BrakeAcceptedFromTakingOff) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::TAKING_OFF);
  EXPECT_TRUE(sm.try_brake());
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::BRAKE);
}

TEST(PlannerStateMachine, BrakeRejectedFromOff) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::OFF);
  EXPECT_FALSE(sm.try_brake());
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::OFF);
}

TEST(PlannerStateMachine, BrakeRejectedFromFinished) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::FINISHED);
  EXPECT_FALSE(sm.try_brake());
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::FINISHED);
}

TEST(PlannerStateMachine, BrakeRejectedFromLand) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::LAND);
  EXPECT_FALSE(sm.try_brake());
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::LAND);
}

TEST(PlannerStateMachine, BrakeRejectedFromBrake) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::BRAKE);
  EXPECT_FALSE(sm.try_brake());
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::BRAKE);
}


// ============================================================================
// LAND → OFF (OmniDrones altitude convergence)
// ============================================================================

TEST(PlannerStateMachine, LandCompletesWhenAltitudeReachedOmniDrones) {
  PlannerStateMachine sm;
  sm.runtime_mode = RuntimeModes::OMNIDRONES;
  sm.set_state(PlanningStates::LAND);
  sm.home = Eigen::Vector3d(0.0, 0.0, 0.2);

  sm.position = Eigen::Vector3d(0.0, 0.0, 1.0);  // still high
  sm.check_land_complete();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::LAND);

  sm.position = Eigen::Vector3d(0.0, 0.0, 0.25);  // within 0.1 of 0.2
  sm.check_land_complete();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::OFF);
}

TEST(PlannerStateMachine, LandDoesNotResetInMavrosMode) {
  PlannerStateMachine sm;
  sm.runtime_mode = RuntimeModes::MAVROS;
  sm.set_state(PlanningStates::LAND);
  sm.home = Eigen::Vector3d(0.0, 0.0, 0.0);
  sm.position = sm.home;  // at home altitude

  sm.check_land_complete();
  // MAVROS resets via disarm detection, not altitude check
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::LAND);
}


// ============================================================================
// land_swarm_callback guard conditions
// ============================================================================

TEST(PlannerStateMachine, LandAcceptedFromFinished) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::FINISHED);
  EXPECT_TRUE(sm.try_land());
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::LAND);
}

TEST(PlannerStateMachine, LandAcceptedFromBrake) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::BRAKE);
  EXPECT_TRUE(sm.try_land());
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::LAND);
}

TEST(PlannerStateMachine, LandRejectedFromTrajectoryControl) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::TRAJECTORY_CONTROL);
  EXPECT_FALSE(sm.try_land());
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::TRAJECTORY_CONTROL);
}

TEST(PlannerStateMachine, LandRejectedFromOff) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::OFF);
  EXPECT_FALSE(sm.try_land());
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::OFF);
}

TEST(PlannerStateMachine, LandRejectedFromTakingOff) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::TAKING_OFF);
  EXPECT_FALSE(sm.try_land());
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::TAKING_OFF);
}


// ============================================================================
// reinitialise_callback guard
// ============================================================================

TEST(PlannerStateMachine, ReinitialiseRejectedFromTrajectoryControl) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::TRAJECTORY_CONTROL);
  EXPECT_FALSE(sm.try_reinitialise());
  EXPECT_FALSE(sm.reinitialise_requested);
}

TEST(PlannerStateMachine, ReinitialiseAcceptedFromFinished) {
  PlannerStateMachine sm;
  sm.set_state(PlanningStates::FINISHED);
  sm.home = Eigen::Vector3d(1.0, 2.0, 3.0);
  sm.initial_heading = 0.5;

  EXPECT_TRUE(sm.try_reinitialise());
  EXPECT_TRUE(sm.reinitialise_requested);
  EXPECT_DOUBLE_EQ(sm.goal.x(), 1.0);
  EXPECT_DOUBLE_EQ(sm.goal.y(), 2.0);
  EXPECT_DOUBLE_EQ(sm.goal.z(), 3.0);
  EXPECT_DOUBLE_EQ(sm.goal_heading, 0.5);
}


// ============================================================================
// advance_waypoint loop logic
// ============================================================================

TEST(PlannerStateMachine, AdvanceWaypointNormal) {
  PlannerStateMachine sm;
  sm.setup_waypoint_mission();  // 3 WPs, 0 loops

  EXPECT_EQ(sm.current_waypoint_index, 0u);
  EXPECT_TRUE(sm.advance_waypoint());
  EXPECT_EQ(sm.current_waypoint_index, 1u);
  EXPECT_TRUE(sm.advance_waypoint());
  EXPECT_EQ(sm.current_waypoint_index, 2u);
  // Past last waypoint → done
  EXPECT_FALSE(sm.advance_waypoint());
  EXPECT_FALSE(sm.waypoint_mission_active);
}

TEST(PlannerStateMachine, AdvanceWaypointWithLoops) {
  PlannerStateMachine sm;
  sm.setup_waypoint_mission(1);  // 3 WPs, remaining_loops=1 → 2 total passes

  // First pass (starts at WP0, advance through WP1, WP2)
  EXPECT_TRUE(sm.advance_waypoint());   // → WP1
  EXPECT_TRUE(sm.advance_waypoint());   // → WP2
  EXPECT_TRUE(sm.advance_waypoint());   // wraps → WP0, remaining=0
  EXPECT_EQ(sm.current_waypoint_index, 0u);
  EXPECT_EQ(sm.remaining_loops, 0u);

  // Second (final) pass
  EXPECT_TRUE(sm.advance_waypoint());   // → WP1
  EXPECT_TRUE(sm.advance_waypoint());   // → WP2
  EXPECT_FALSE(sm.advance_waypoint());  // done
  EXPECT_FALSE(sm.waypoint_mission_active);
}

TEST(PlannerStateMachine, AdvanceWaypointUpdatesGoal) {
  PlannerStateMachine sm;
  sm.setup_waypoint_mission();

  sm.advance_waypoint();
  EXPECT_DOUBLE_EQ(sm.goal.x(), sm.world_waypoints[1].x());
  EXPECT_DOUBLE_EQ(sm.goal.y(), sm.world_waypoints[1].y());
  EXPECT_DOUBLE_EQ(sm.goal_heading, sm.waypoint_headings[1]);
}


// ============================================================================
// reset_planner completeness
// ============================================================================

TEST(PlannerStateMachine, ResetClearsAllFlags) {
  PlannerStateMachine sm;
  // Set everything to non-default values
  sm.set_state(PlanningStates::TRAJECTORY_CONTROL);
  sm.steering_value = 1.0f;
  sm.steered = true;
  sm.goal_set = true;
  sm.mission_received = true;
  sm.mission_uploaded = true;
  sm.mode_switch_pending = true;
  sm.arming_pending = true;
  sm.takeoff_pending = true;
  sm.land_pending = true;
  sm.takeoff_requested = true;
  sm.reinitialise_requested = true;
  sm.brake_mode_switch_sent = true;
  sm.had_reference_trajectory = true;
  sm.has_valid_setpoint = true;
  sm.goal_up_coordinate = 5.0;
  sm.setup_waypoint_mission(3);

  sm.reset_planner();

  EXPECT_EQ(sm.planner_state.load(), PlanningStates::OFF);
  EXPECT_FLOAT_EQ(sm.steering_value, 0.0f);
  EXPECT_FALSE(sm.steered);
  EXPECT_FALSE(sm.goal_set);
  EXPECT_FALSE(sm.mission_received);
  EXPECT_FALSE(sm.mission_uploaded);
  EXPECT_FALSE(sm.mode_switch_pending);
  EXPECT_FALSE(sm.arming_pending);
  EXPECT_FALSE(sm.takeoff_pending);
  EXPECT_FALSE(sm.land_pending);
  EXPECT_FALSE(sm.takeoff_requested);
  EXPECT_FALSE(sm.reinitialise_requested);
  EXPECT_FALSE(sm.brake_mode_switch_sent);
  EXPECT_FALSE(sm.had_reference_trajectory);
  EXPECT_FALSE(sm.has_valid_setpoint);
  EXPECT_DOUBLE_EQ(sm.goal_up_coordinate, 0.0);
  EXPECT_TRUE(sm.world_waypoints.empty());
  EXPECT_TRUE(sm.waypoint_headings.empty());
  EXPECT_TRUE(sm.waypoint_hold_times.empty());
  EXPECT_EQ(sm.current_waypoint_index, 0u);
  EXPECT_EQ(sm.remaining_loops, 0u);
  EXPECT_FALSE(sm.waypoint_mission_active);
}

TEST(PlannerStateMachine, ResetClearsOpusState) {
  PlannerStateMachine sm;
  sm.opus_enabled = true;
  sm.opus_granted = true;
  sm.opus_check_pending = true;
  sm.opus_lock_pending = true;
  sm.opus_submission_needed = true;
  sm.opus_plan_sequence = 5;

  sm.reset_planner();

  EXPECT_FALSE(sm.opus_granted);
  EXPECT_FALSE(sm.opus_check_pending);
  EXPECT_FALSE(sm.opus_lock_pending);
  EXPECT_FALSE(sm.opus_submission_needed);
  EXPECT_EQ(sm.opus_plan_sequence, 6u);  // bumped by 1
}

TEST(PlannerStateMachine, ResetSkipsOpusWhenDisabled) {
  PlannerStateMachine sm;
  sm.opus_enabled = false;
  sm.opus_plan_sequence = 5;

  sm.reset_planner();

  // Should not have incremented
  EXPECT_EQ(sm.opus_plan_sequence, 5u);
}


// ============================================================================
// Full mission cycle: OFF → TAKING_OFF → ... → FINISHED → reinit → OFF
// ============================================================================

TEST(PlannerStateMachine, FullMissionCycleOmniDrones) {
  PlannerStateMachine sm;
  sm.runtime_mode = RuntimeModes::OMNIDRONES;
  sm.opus_enabled = false;
  sm.setup_waypoint_mission();
  sm.home = Eigen::Vector3d(0.0, 0.0, 0.0);
  sm.initial_heading = 0.0;

  // 1. Takeoff
  sm.set_state(PlanningStates::TAKING_OFF);
  sm.position = Eigen::Vector3d(0.0, 0.0, 1.5);
  sm.check_takeoff_complete();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::ALIGNING_HEADING);

  // 2. Align heading
  sm.current_yaw = sm.goal_heading;
  sm.check_heading_aligned();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::TRAJECTORY_CONTROL);

  // 3. Reach WP0 (distance < threshold triggers)
  sm.position = sm.world_waypoints[0];
  sm.check_goal_reached();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::HOLDING_WAYPOINT);

  // 4. Hold → advance to WP1
  sm.time_in_current_state = 1.0;
  sm.check_holding_waypoint();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::ALIGNING_HEADING);
  EXPECT_EQ(sm.current_waypoint_index, 1u);

  // 5. Align and fly to WP1
  sm.current_yaw = sm.goal_heading;
  sm.check_heading_aligned();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::TRAJECTORY_CONTROL);
  sm.position = sm.world_waypoints[1];
  sm.check_goal_reached();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::HOLDING_WAYPOINT);

  // 6. Hold → advance to WP2
  sm.time_in_current_state = 2.0;
  sm.check_holding_waypoint();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::ALIGNING_HEADING);

  // 7. Align and fly to last WP
  sm.current_yaw = sm.goal_heading;
  sm.check_heading_aligned();
  sm.position = sm.world_waypoints[2];
  sm.check_goal_reached();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::HOLDING_WAYPOINT);

  // 8. Hold at last WP → mission complete → GO_TO_GOAL
  sm.time_in_current_state = 1.5;
  sm.check_holding_waypoint();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::GO_TO_GOAL);
  EXPECT_FALSE(sm.waypoint_mission_active);

  // 9. Trajectory completes → FINISHED
  sm.had_reference_trajectory = true;
  sm.trajectory_duration = 2.0;
  sm.trajectory_elapsed = 2.5;
  sm.check_trajectory_complete();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::FINISHED);

  // 10. Reinitialise → fly home → OFF
  EXPECT_TRUE(sm.try_reinitialise());
  sm.position = sm.home;
  sm.current_yaw = sm.initial_heading;
  sm.check_reinitialise_complete();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::OFF);
}

TEST(PlannerStateMachine, BrakeToLandToOffCycle) {
  PlannerStateMachine sm;
  sm.runtime_mode = RuntimeModes::OMNIDRONES;
  sm.home = Eigen::Vector3d(0.0, 0.0, 0.2);

  // Flying → BRAKE
  sm.set_state(PlanningStates::TRAJECTORY_CONTROL);
  EXPECT_TRUE(sm.try_brake());
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::BRAKE);

  // BRAKE → LAND
  EXPECT_TRUE(sm.try_land());
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::LAND);

  // LAND → OFF when altitude reached
  sm.position = Eigen::Vector3d(1.0, 2.0, 0.25);
  sm.check_land_complete();
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::OFF);
}
