#pragma once

namespace autopilot {

enum class PlanningStates {
  OFF,
  TAKING_OFF,
  TRAJECTORY_CONTROL,
  GO_TO_GOAL,
  LAND
};
enum class RuntimeModes { OMNIDRONES = 1, MAVROS = 2 };
enum class MavrosControlModes { KINEMATIC = 1, ATTITUDE = 2 };
enum class TravelingCost { DIRECTION = 1, DISTANCE = 2 };
}  // namespace autopilot
