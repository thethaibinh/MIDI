/**
 * @file test_opus_state_machine.cpp
 * @brief Unit tests for the OPUS planning state transitions.
 *
 * Models the planner's OPUS flag logic and PlanningStates transitions
 * in isolation — no ROS2, no depth images, no Ruckig.
 *
 * Tests verify:
 *   1. ALIGNING_HEADING → WAITING_FOR_OPUS (when heading aligned + opus enabled)
 *   2. WAITING_FOR_OPUS flag cascade: submission_needed → lock_send → grant → submit → accept → TRAJECTORY_CONTROL
 *   3. WAITING_FOR_OPUS → abort path (resets all flags)
 *   4. TRAJECTORY_CONTROL → HOLDING_WAYPOINT on waypoint reached
 *   5. Stale plan_sequence rejection
 */

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>

#include "autopilot_states.h"

using autopilot::PlanningStates;

/**
 * Minimal reproduction of the planner's OPUS flag state.
 * Extracted from planner_node.hpp / planner_node.cpp so we can test
 * the transition logic without any ROS2/sensor dependencies.
 */
struct OpusStateMachine {
  // State
  std::atomic<PlanningStates> planner_state{PlanningStates::OFF};

  // OPUS flags (normally guarded by opus_mutex_)
  std::mutex opus_mutex;
  bool opus_enabled = true;
  bool opus_granted = false;
  bool opus_check_pending = false;
  bool opus_lock_pending = false;
  bool opus_submission_needed = false;
  uint32_t opus_plan_sequence = 0;

  // ----- Transitions extracted from planner_node.cpp -----

  /// ALIGNING_HEADING → WAITING_FOR_OPUS (from update_planner_state ~line 883)
  void try_transition_aligning_to_waiting(double yaw_error, double threshold) {
    if (planner_state != PlanningStates::ALIGNING_HEADING) return;

    if (std::abs(yaw_error) < threshold) {
      if (opus_enabled) {
        const std::lock_guard<std::mutex> lock(opus_mutex);
        opus_submission_needed = true;
        planner_state = PlanningStates::WAITING_FOR_OPUS;
      } else {
        planner_state = PlanningStates::TRAJECTORY_CONTROL;
      }
    }
  }

  /// img_callback OPUS management block (~line 1651): decide what to do
  enum class OpusAction { NONE, SEND_LOCK, READY_TO_SUBMIT, WAIT };

  OpusAction get_opus_action() {
    if (!opus_enabled || !opus_submission_needed) return OpusAction::NONE;
    const std::lock_guard<std::mutex> lock(opus_mutex);

    if (opus_check_pending) return OpusAction::WAIT;
    if (opus_granted) return OpusAction::READY_TO_SUBMIT;
    if (!opus_lock_pending) return OpusAction::SEND_LOCK;
    return OpusAction::WAIT;
  }

  /// opus_send_lock_goal (~line 1762)
  void send_lock_goal() {
    const std::lock_guard<std::mutex> lock(opus_mutex);
    ++opus_plan_sequence;
    opus_lock_pending = true;
  }

  /// opus_lock_result_callback (~line 1817): grant arrives with echoed sequence
  void lock_result(bool permitted, uint32_t echoed_seq) {
    const std::lock_guard<std::mutex> lock(opus_mutex);
    opus_lock_pending = false;

    if (permitted && echoed_seq == opus_plan_sequence) {
      opus_granted = true;
    } else {
      opus_granted = false;
    }
  }

  /// opus_submit_trajectory (~line 2003): submission sent, waiting for response
  void submit_trajectory() {
    const std::lock_guard<std::mutex> lock(opus_mutex);
    opus_check_pending = true;
  }

  /// opus_trajectory_check_response (~line 1861): coordinator responds
  void trajectory_check_response(bool accepted, uint32_t expected_seq) {
    const std::lock_guard<std::mutex> lock(opus_mutex);

    if (accepted && expected_seq == opus_plan_sequence) {
      opus_check_pending = false;
      opus_submission_needed = false;
      opus_granted = false;

      if (planner_state == PlanningStates::WAITING_FOR_OPUS) {
        planner_state = PlanningStates::TRAJECTORY_CONTROL;
      }
    } else {
      // Collision rejection: retain lock, allow retry
      opus_check_pending = false;
      // opus_granted stays true for retry path
    }
  }

  /// opus_abort_planning (~line 1909): abort all OPUS state
  void abort_planning() {
    const std::lock_guard<std::mutex> lock(opus_mutex);
    if (!opus_enabled) return;
    if (!opus_granted && !opus_lock_pending) return;

    opus_granted = false;
    opus_check_pending = false;
    opus_lock_pending = false;
    opus_submission_needed = false;
    // In real code: also cancels goal_handle and publishes abort message
  }

  /// TRAJECTORY_CONTROL → HOLDING_WAYPOINT or GO_TO_GOAL on goal reached
  void check_goal_reached(double distance, double threshold, bool waypoint_active) {
    if (planner_state != PlanningStates::TRAJECTORY_CONTROL) return;
    if (distance < threshold) {
      abort_planning();
      planner_state = waypoint_active
        ? PlanningStates::HOLDING_WAYPOINT
        : PlanningStates::GO_TO_GOAL;
    }
  }
};


// ============================================================================
// Test: Full OPUS Happy Path
// ============================================================================

TEST(OpusStateMachine, HappyPathFullCycle) {
  OpusStateMachine sm;
  sm.planner_state = PlanningStates::ALIGNING_HEADING;

  // 1. Heading aligns → WAITING_FOR_OPUS
  sm.try_transition_aligning_to_waiting(0.05, 0.17);  // error < threshold
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::WAITING_FOR_OPUS);
  EXPECT_TRUE(sm.opus_submission_needed);

  // 2. First img_callback: should send lock goal
  EXPECT_EQ(sm.get_opus_action(), OpusStateMachine::OpusAction::SEND_LOCK);
  sm.send_lock_goal();
  EXPECT_TRUE(sm.opus_lock_pending);
  EXPECT_EQ(sm.opus_plan_sequence, 1u);

  // 3. While lock pending, action is WAIT
  EXPECT_EQ(sm.get_opus_action(), OpusStateMachine::OpusAction::WAIT);

  // 4. Lock grant arrives
  sm.lock_result(true, 1);
  EXPECT_TRUE(sm.opus_granted);
  EXPECT_FALSE(sm.opus_lock_pending);

  // 5. Next img_callback: ready to submit
  EXPECT_EQ(sm.get_opus_action(), OpusStateMachine::OpusAction::READY_TO_SUBMIT);
  sm.submit_trajectory();
  EXPECT_TRUE(sm.opus_check_pending);

  // 6. While check pending, action is WAIT
  EXPECT_EQ(sm.get_opus_action(), OpusStateMachine::OpusAction::WAIT);

  // 7. Trajectory accepted → TRAJECTORY_CONTROL
  sm.trajectory_check_response(true, 1);
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::TRAJECTORY_CONTROL);
  EXPECT_FALSE(sm.opus_submission_needed);
  EXPECT_FALSE(sm.opus_check_pending);
  EXPECT_FALSE(sm.opus_granted);
}

// ============================================================================
// Test: Heading Not Aligned → Stay in ALIGNING_HEADING
// ============================================================================

TEST(OpusStateMachine, HeadingNotAligned) {
  OpusStateMachine sm;
  sm.planner_state = PlanningStates::ALIGNING_HEADING;

  sm.try_transition_aligning_to_waiting(0.5, 0.17);  // error > threshold
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::ALIGNING_HEADING);
  EXPECT_FALSE(sm.opus_submission_needed);
}

// ============================================================================
// Test: OPUS Disabled → Skip to TRAJECTORY_CONTROL
// ============================================================================

TEST(OpusStateMachine, OpusDisabledSkipsToTrajectoryControl) {
  OpusStateMachine sm;
  sm.planner_state = PlanningStates::ALIGNING_HEADING;
  sm.opus_enabled = false;

  sm.try_transition_aligning_to_waiting(0.05, 0.17);
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::TRAJECTORY_CONTROL);
  EXPECT_FALSE(sm.opus_submission_needed);
}

// ============================================================================
// Test: Abort Path — Resets All Flags
// ============================================================================

TEST(OpusStateMachine, AbortResetsAllFlags) {
  OpusStateMachine sm;
  sm.planner_state = PlanningStates::WAITING_FOR_OPUS;

  // Get to granted + check_pending state
  sm.opus_submission_needed = true;
  sm.send_lock_goal();
  sm.lock_result(true, 1);
  sm.submit_trajectory();

  // Verify pre-abort state
  EXPECT_TRUE(sm.opus_granted);
  EXPECT_TRUE(sm.opus_check_pending);
  EXPECT_TRUE(sm.opus_submission_needed);

  // Abort
  sm.abort_planning();

  EXPECT_FALSE(sm.opus_granted);
  EXPECT_FALSE(sm.opus_check_pending);
  EXPECT_FALSE(sm.opus_lock_pending);
  EXPECT_FALSE(sm.opus_submission_needed);
}

// ============================================================================
// Test: Abort When Not Granted → No-Op
// ============================================================================

TEST(OpusStateMachine, AbortWhenNotGrantedIsNoop) {
  OpusStateMachine sm;
  sm.planner_state = PlanningStates::WAITING_FOR_OPUS;
  sm.opus_submission_needed = true;
  // Not granted, not lock_pending → abort should be a no-op
  sm.abort_planning();
  EXPECT_TRUE(sm.opus_submission_needed);  // unchanged
}

// ============================================================================
// Test: Stale Plan Sequence Rejection
// ============================================================================

TEST(OpusStateMachine, StaleSequenceRejected) {
  OpusStateMachine sm;
  sm.planner_state = PlanningStates::WAITING_FOR_OPUS;
  sm.opus_submission_needed = true;

  // Send lock, then increment sequence (simulates a newer planning round)
  sm.send_lock_goal();  // seq=1
  uint32_t stale_seq = sm.opus_plan_sequence;

  // Before grant arrives, a new round starts
  sm.opus_lock_pending = false;  // old round cancelled
  sm.send_lock_goal();           // seq=2

  // Old grant arrives with stale seq=1
  sm.lock_result(true, stale_seq);
  EXPECT_FALSE(sm.opus_granted);  // rejected as stale
  EXPECT_EQ(sm.opus_plan_sequence, 2u);
}

// ============================================================================
// Test: Collision Rejection → Retry Path
// ============================================================================

TEST(OpusStateMachine, CollisionRejectionAllowsRetry) {
  OpusStateMachine sm;
  sm.planner_state = PlanningStates::WAITING_FOR_OPUS;
  sm.opus_submission_needed = true;

  sm.send_lock_goal();
  sm.lock_result(true, 1);
  sm.submit_trajectory();

  // Coordinator rejects: collision (accepted=false, seq matches)
  sm.trajectory_check_response(false, 1);

  // Should still be in WAITING_FOR_OPUS (not transitioned)
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::WAITING_FOR_OPUS);
  // Lock retained for retry
  EXPECT_FALSE(sm.opus_check_pending);  // Can submit again

  // Next img_callback should still allow submission
  EXPECT_EQ(sm.get_opus_action(), OpusStateMachine::OpusAction::READY_TO_SUBMIT);
}

// ============================================================================
// Test: Goal Reached → HOLDING_WAYPOINT / GO_TO_GOAL
// ============================================================================

TEST(OpusStateMachine, GoalReachedWaypointMission) {
  OpusStateMachine sm;
  sm.planner_state = PlanningStates::TRAJECTORY_CONTROL;
  sm.opus_granted = true;
  sm.opus_submission_needed = true;

  sm.check_goal_reached(0.1, 0.5, true);  // waypoint mission
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::HOLDING_WAYPOINT);
  EXPECT_FALSE(sm.opus_granted);
  EXPECT_FALSE(sm.opus_submission_needed);
}

TEST(OpusStateMachine, GoalReachedFinalGoal) {
  OpusStateMachine sm;
  sm.planner_state = PlanningStates::TRAJECTORY_CONTROL;
  sm.opus_granted = true;

  sm.check_goal_reached(0.1, 0.5, false);  // not a waypoint mission
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::GO_TO_GOAL);
}

// ============================================================================
// Test: Full Multi-Waypoint Cycle
// ============================================================================

TEST(OpusStateMachine, MultiWaypointCycle) {
  OpusStateMachine sm;

  // Waypoint 1: align → wait → plan → execute → reach
  sm.planner_state = PlanningStates::ALIGNING_HEADING;
  sm.try_transition_aligning_to_waiting(0.01, 0.17);
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::WAITING_FOR_OPUS);

  sm.send_lock_goal();
  sm.lock_result(true, 1);
  sm.submit_trajectory();
  sm.trajectory_check_response(true, 1);
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::TRAJECTORY_CONTROL);

  // Reach waypoint 1
  sm.check_goal_reached(0.05, 0.5, true);
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::HOLDING_WAYPOINT);

  // Waypoint 2: re-enter alignment
  sm.planner_state = PlanningStates::ALIGNING_HEADING;
  sm.try_transition_aligning_to_waiting(0.02, 0.17);
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::WAITING_FOR_OPUS);
  EXPECT_TRUE(sm.opus_submission_needed);

  // Plan again
  sm.send_lock_goal();
  sm.lock_result(true, 2);
  sm.submit_trajectory();
  sm.trajectory_check_response(true, 2);
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::TRAJECTORY_CONTROL);

  // Reach final goal
  sm.check_goal_reached(0.05, 0.5, false);
  EXPECT_EQ(sm.planner_state.load(), PlanningStates::GO_TO_GOAL);
}
