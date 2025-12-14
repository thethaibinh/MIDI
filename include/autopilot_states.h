#pragma once

namespace autopilot {

enum class PlanningStates {
  OFF,
  TAKING_OFF,
  LAND,
  GO_TO_GOAL,
  TRAJECTORY_CONTROL
};
enum class RuntimeModes { FLIGHTMARE = 1, MAVROS = 2, OMNIDRONES = 3 };
enum class MavrosControlModes { KINEMATIC = 1, ATTITUDE = 2 };
enum class TravelingCost { DIRECTION = 1, DISTANCE = 2 };
}  // namespace autopilot
