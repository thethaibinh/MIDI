/*!
 * Copyright 2024 by Binh Nguyen <thethaibinh@gmail.com>
 *
 * This code is free software: you can redistribute
 * it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * This code is distributed in the hope that it will
 * be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with the code.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once
#include <memory>
#include <random>
#include <iostream>
#include <chrono>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "common_math/segment.hpp"
#include "common_math/segment3.hpp"
#include "common_math/segment2.hpp"
#include "common_math/monotonic_segment3.hpp"
#include "common_math/pinhole_camera_model.hpp"
#include "common_math/pyramid.hpp"
#include "depth_uncertainty_planner/trajectory_cost.hpp"
#include "depth_uncertainty_planner/sampling.hpp"

// CUDA
#include "common_math/cuda_segment.cuh"
#include "common_math/cuda_segment2.cuh"
#include "common_math/cuda_segment3.cuh"
#include "common_math/cuda_conversions.cuh"
#include "common_math/cuda_monotonic_segment3.cuh"
// Ruckig
#include <ruckig/ruckig.hpp>
#include <ruckig/profile.hpp>

using namespace common_math;

namespace depth_uncertainty_planner {

enum class CollisionCheckingMethod {
  MIDI,
  PYRAMID
};

class DuPlanner {
 public:

  //! Constructor. Requires a depth image and the related camera intrinsics
  /*!
   * @param depthImage The depth image stored as a 16 bit single channel image (CV_16UC1)
   * @param camera
   * @param physicalVehicleRadius The true radius of the vehicle. Any depth values closer than this distance to the camera
   * will be ignored. [meters]
   * @param vehicleRadiusForPlanning We plan as if the vehicle has this radius. This value should be slightly larger than
   * physicalVehicleRadius to account for pose estimation errors and trajectory tracking errors. [meters]
   * @param minimumCollisionDistance We do not perform collision checking on parts of the candidate trajectory closer than
   * this distance to the camera. Obstacles closer than this distance can still occlude other parts of the trajectory (causing
   * them to be labeled as in collision). This is used to enforce field of view constraints as well; we assume there is an
   * obstacle just outside of the field of view this distance away from the camera. [meters]
   */
  DuPlanner(const cv::Mat& depthImage,
                  const PinholeCamera& camera,
                  const CollisionCheckingMethod& collision_checking_method,
                  const double& checking_time_ratio,
                  const uint32_t& sampled_trajectories_threshold,
                  const uint32_t& checked_trajectories_threshold,
                  const bool& debug_num_trajectories,
                  const double& collision_probability_threshold,
                  const uint32_t& openmp_chunk_size);

  //! Finds the trajectory with the lowest user-provided cost using a user-provided
  //! trajectory generator.
  /*!
   * @param opt_trajectory A trajectory that defines the state of the vehicle when the depth image was taken
   * @param allocated_computation_time The planner will exit after this amount of time has passed [seconds]
   * @param cost_function_definition_object This object must define the cost function to be used to evaluate
   * each of the candidate trajectories. The member variables of this object can be used to evaluate the cost
   * function (see ExplorationCost, for example).
   * @param cost_function_wrapper This function is a wrapper that takes cost_function_definition_object
   * and a candidate trajectory as arguments. It is used to call the cost function defined in cost_function_definition_object
   * and return the result.
   * @param trajectory_sampler This object must define a function that returns new candidate trajectories (see
   * RandomTrajectorySampler, for example)
   * @return True if a feasible trajectory was found, false otherwise. If a feasible trajectory was found,
   * the argument trajectory will be updated to contain the lowest cost trajectory found.
   */
  bool find_lowest_cost_trajectory(
    ruckig::InputParameter<3>& initial_state,
    ruckig::Trajectory<3>& opt_trajectory,
    RandomTrajectorySampler& trajectory_sampler,
    double allocated_computation_time, void* cost_function_definition_object,
    double (*cost_function_wrapper)(void* cost_function_definition_object,
                                    Eigen::Vector3d& endpoint_vector));

  /*!
   * @return the steering amount
   */
  float get_steering() {
    return steering_direction;
  }

  /*!
   * @return A vector containing all of the generated pyramids
   */
  std::vector<Pyramid> get_pyramids() {
    return _pyramids;
  }
  /*!
   * @return The number of pyramids generated
   */
  int get_num_pyramids() {
    return _pyramids.size();
  }

  /*!
   * @return The number of candidate trajectories sampled
   */
  int get_num_trajectories_sampled() {
    return _num_trajectories_sampled;
  }

  /*!
   * @return The number of candidate trajectories generated by the planner
   */
  int get_num_trajectories_generated() {
    return _num_trajectories_generated;
  }
  /*!
   * @return The number of trajectories checked for collisions (i.e. not filtered by cost or dynamic feasibility first)
   */
  int GetNumCollisionChecks() {
    return _num_collision_checked;
  }
  /*!
   * @return The time spent checking for collisions [seconds]
   */
  double get_checking_time() {
    return _checking_time;
  }
  /*!
   * @return The total time spent generating trajectories [seconds]
   */
  double get_total_generation_time() {
    return _generating_time;
  }

 private:
  //! Scan the current depth image to decide which steering direction.
  int scan_depth();

  //! Collision checking algorithm as described by
  /*!
   * @param segment Candidate trajectory to be checked for collisions. We assume the
   * candidate trajectory is written in the camera-fixed frame and has an initial
   * position of (0, 0, 0).
   * @return True if the trajectory is found to be collision free, false otherwise
   */
  // bool is_segment3_collision_free(const ThirdOrderSegment* segment);

  bool is_cuda_segment3_collision_free(const CudaThirdOrderSegment* segment);

  bool is_segment2_collision_free(const SecondOrderSegment* segment, double& trajectory_collision_probability, double& mahalanobis_distance);

  bool is_cuda_segment2_collision_free(const CudaSecondOrderSegment* segment, double& trajectory_collision_probability, double& mahalanobis_distance);

  double segment_collision_cost(const Segment* segment);

  //! Splits the candidate trajectory into sections with monotonically changing depth
  //! using the methods described in Section II.B of the RAPPIDS paper.
  /*!
   * @param opt_trajectory The candidate trajectory to be split into its sections with monotonically changing depth
   * @return A vector of sections of the trajectory with monotonically changing depth
   */
  std::vector<MonotonicSegment3> get_monotonic_segments(const ThirdOrderSegment* segment);

  std::vector<CudaMonotonicSegment3> get_cuda_monotonic_segments(const CudaThirdOrderSegment* segment);

  std::vector<Segment*> get_segments(const ruckig::Trajectory<3> trajectory);

  std::vector<CudaSegment*> get_cuda_segments(const ruckig::Trajectory<3> trajectory);
  //! Tries to find an existing pyramid that contains the given sample point
  /*!
   * @param pixelX The x-coordinate of the sample point in pixel coordinates
   * @param pixelY The y-coordinate of the sample point in pixel coordinates
   * @param depth The depth of the sample point in meters
   * @param outPyramid A pyramid that contains the sample point, if one is found
   * @return True a pyramid containing the sample point was found, false otherwise
   */
  bool find_containing_pyramid(int pixel_x, int pixel_y, double depth, Pyramid &out_pyramid);

  //! Computes the time in seconds at which the given trajectory collides with the given pyramid (if it does at all)
  //! using the methods described in Section II.C of the RAPPIDS paper.
  /*!
   * @param monoTraj A trajectory with monotonically changing depth
   * @param pyramid The pyramid to be used for collision checking
   * @param outCollisionTime The time in seconds at which the trajectory collides with a lateral face of the pyramid.
   * If the trajectory collides with lateral faces of the pyramid multiple times, this is the time at which
   * the collision with the deepest depth occurs.
   * @return True if the trajectory collides with at least one lateral face of the pyramid
   */
  bool find_deepest_collision_time(CudaMonotonicSegment3 mono_traj, Pyramid pyramid,
                                double& out_collision_time);

  //! Attempts to generate a pyramid that contains a given point.
  /*!
   * @param x0 The sample point's horizontal pixel coordinate
   * @param y0 The sample point's vertical pixel coordinate
   * @param minimumDepth The minimum depth of the base plane of the pyramid (i.e. the depth of the sample point)
   * @param outPyramid The created pyramid (can be undefined if no pyramid could be created)
   * @return True if a pyramid was successfully created, false otherwise
   */
  bool inflate_pyramid(const int& x0, const int& y0, const double& minimumDepth,
                                     Pyramid& outPyramid);

  //! Pointer to the array where the depth image is stored
  const float* _depth_data;

  //! Pinhole camera model
  PinholeCamera _camera;

  //! The minimum allowed thrust (used for checking dynamic feasibility) [m/s^2]
  double _minimum_allowed_thrust;
  //! The maximum allowed thrust (used for checking dynamic feasibility) [m/s^2]
  double _maximum_allowed_thrust;
  //! The maximum allowed angular velocity (used for checking dynamic feasibility) [rad/s]
  double _maximum_allowed_angular_velocity;
  //! Minimum time section to test when checking for dynamic feasibility [seconds]
  double _minimum_section_time_dynamic_feasibility;

  //! After this amount of time has been spent generating pyramids, no more pyramids will be generated (i.e. instead of trying to
  //! generate a new pyramid, the trajectory will simply be labeled as in-collision). This limit is not set by default. [seconds]
  double _max_pyramid_gen_time;
  //! The amount of time spent generating pyramids [nanoseconds]
  double _pyramid_gen_time_nanoseconds;
  //! The maximum number of pyramids we allow the planner to generate. This limit is not set by default.
  int _max_num_pyramids;

  //! If the endpoint of a trajectory is within this many pixels of the lateral face of a pyramid, we will not use that pyramid
  //! for collision checking. This is to prevent the scenario where a trajectory intersects the lateral face of a pyramid, and
  //! then the same pyramid is repeatedly used for collision checking (entering an infinite loop).
  int _pyramid_search_pixel_buffer;

  //! Time allocated for running the planner in seconds. The planner checks whether it should exit after evaluating each trajectory.
  double _allocated_computation_time;
  //! The time at which the planner starts.
  std::chrono::high_resolution_clock::time_point start_time;

  //! Counter used to keep track of how many trajectories we have sampled
  int _num_trajectories_sampled;

  //! Counter used to keep track of how many trajectories we have generated/evaluated
  int _num_trajectories_generated;
  //! Counter used to keep track of how many trajectories we have checked for collisions
  int _num_collision_checked, _num_collision_free;
  //! The time spent checking for collisions [microseconds]
  double _checking_time;
  //! The total time spent generating trajectories [microseconds]
  double _generating_time;

  //! The steering amount in degree
  float steering_direction;

  // Store the maximum collision probability for each trajectory
  double max_collision_probability;

  //! The ratio of the traveling cost to the collision cost
  double _checking_time_ratio;

  //! The threshold for the number of trajectories checked
  uint32_t _checked_trajectories_threshold;
  //! The threshold for the number of trajectories sampled
  uint32_t _sampled_trajectories_threshold;

  //! The threshold for the collision probability
  double _collision_probability_threshold;

  //! The chunk size for OpenMP parallelization
  uint32_t _openmp_chunk_size;

  //! The list of all pyramids found by the planner. This list is ordered based on the depth of the base plane of each pyramid.
  std::vector<Pyramid> _pyramids;

  CollisionCheckingMethod collision_checking_method;

  //! Whether to print the number of trajectories sampled and checked
  bool _debug_num_trajectories;
};

}  // namespace depth_uncertainty_planner
