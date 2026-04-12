#pragma once

namespace autopilot {

enum class PlanningStates {
  OFF,
  TAKING_OFF,
  ALIGNING_HEADING,
  WAITING_FOR_OPUS,
  TRAJECTORY_CONTROL,
  GO_TO_GOAL,
  HOLDING_WAYPOINT,
  LAND,
  FINISHED
};
enum class RuntimeModes { OMNIDRONES = 1, MAVROS = 2 };
enum class MavrosControlModes { KINEMATIC = 1, ATTITUDE = 2 };
enum class SetpointTypes { POSITION_ONLY = 1, FULL_STATE = 2 };
enum class TravelingCost { DIRECTION = 1, DISTANCE = 2 };
}  // namespace autopilot
