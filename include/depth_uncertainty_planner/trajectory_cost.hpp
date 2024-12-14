#pragma once

#include <autopilot_states.h>

using namespace autopilot;
//! Class used to find trajectory that moves the fastest in the desired
//! exploration direction

class ExplorationCost {
 public:
  //! Constructor. Defines desired exploration direction as written in the
  //! camera-fixed frame. For example, a value of (0, 0, 1) would give
  //! trajectories that travel the fastest in the direction that the camera is
  //! pointing the lowest cost.
  ExplorationCost(Eigen::Vector3d explorationDirection, TravelingCost traveling_cost)
    : _explorationDirection(explorationDirection), _traveling_cost(traveling_cost) {}
  //! Returns the cost of a given trajectory. Note that because each candidate
  //! trajectory is written in the camera-fixed frame, the initial position of
  //! each candidate trajectory is always (0, 0, 0) because we fix the
  //! trajectories to originate at the focal point of the camera. Thus, we only
  //! need to evaluate the end point of the trajectory.
  double get_direction_cost(Eigen::Vector3d& endpoint_vector) {
    Eigen::Vector3d endpoint_unit_vector = endpoint_vector.normalized();
    return -_explorationDirection.normalized().dot(endpoint_unit_vector);
  }

  double get_distance_to_goal_cost(Eigen::Vector3d endpoint_vector) {
    return -endpoint_vector.dot(_explorationDirection.normalized());
  }

  TravelingCost get_traveling_cost() {
    return _traveling_cost;
  }

  //! We pass this wrapper function to the planner (see
  //! FindLowestCostTrajectory), and this function calls the related GetCost
  //! function to compute the cost of the given trajectory. We structure the
  //! code this way so that other custom cost functions can be used in a similar
  //! fashion in the future.
  static double get_cost_wrapper(void* ptr2obj,
                                      Eigen::Vector3d& endpoint_vector) {
    ExplorationCost* explorationCost = (ExplorationCost*)ptr2obj;
    switch (explorationCost->get_traveling_cost()) {
      case TravelingCost::DIRECTION:
        return explorationCost->get_direction_cost(endpoint_vector);
      case TravelingCost::DISTANCE:
        return explorationCost->get_distance_to_goal_cost(endpoint_vector);
      default:
        throw std::invalid_argument("Invalid traveling cost");
    }
  }
 private:
  Eigen::Vector3d _explorationDirection;
  TravelingCost _traveling_cost;
};
