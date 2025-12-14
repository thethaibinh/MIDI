#ifndef QUADROTOR_COMMON_TRAJECTORY_POINT_H
#define QUADROTOR_COMMON_TRAJECTORY_POINT_H

#include <Eigen/Dense>

namespace quadrotor_common {

struct TrajectoryPoint {
  Eigen::Vector3d position;
  Eigen::Vector3d velocity;
  Eigen::Vector3d acceleration;
  Eigen::Vector3d jerk;
  double heading;
  
  TrajectoryPoint() : 
    position(Eigen::Vector3d::Zero()),
    velocity(Eigen::Vector3d::Zero()),
    acceleration(Eigen::Vector3d::Zero()),
    jerk(Eigen::Vector3d::Zero()),
    heading(0.0) {}
};

} // namespace quadrotor_common

#endif // QUADROTOR_COMMON_TRAJECTORY_POINT_H

