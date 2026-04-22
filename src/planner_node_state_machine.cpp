#include "planner_node.hpp"

using namespace quadrotor_common;

void PlannerNode::update_planner_state() {
  // Snapshot flight controller status under lock (written by ardupilot_status_callback
  // on state_callback_group_, read here on control_callback_group_)
  mavros_msgs::msg::State fc_status;
  if (_runtime_mode == RuntimeModes::MAVROS) {
    const std::lock_guard<std::mutex> lock(fc_status_mutex_);
    fc_status = flight_controller_status;
  }

  // For MAVROS mode: Handle FC startup sequence (GUIDED -> ARM -> TAKEOFF)
  // Trigger on either _goal_set (mission) or takeoff_requested_ (takeoff-only)
  if (_runtime_mode == RuntimeModes::MAVROS && _planner_state == PlanningStates::OFF && 
      (_goal_set || takeoff_requested_)) {
    
    // Note: _goal_heading is now computed in the goal callbacks (mission_upload_callback)
    // at the same time as goal position, using the same state snapshot.
    // For takeoff-only (no goal), use default heading of 0.
    if (!_goal_set) {
      _goal_heading = 0.0;  // Default heading for takeoff-only
    }
    // Log the heading that will be used
    RCLCPP_INFO_ONCE(this->get_logger(), "Using goal heading: %.1f deg (%.2f rad)",
                     _goal_heading * 180.0 / M_PI, _goal_heading);

    // Release pending latches either when the FC has already reached the
    // requested state (confirmed via /mavros/state) or after a 2s timeout
    // so we can retry if the request was silently dropped. This prevents
    // spamming the FC command queue with duplicate requests while the
    // /mavros/state topic lags behind the service response by ~100 ms.
    const auto now = std::chrono::steady_clock::now();
    constexpr auto kRetryCooldown = std::chrono::seconds(2);
    if (mode_switch_pending_ &&
        (fc_status.mode == "GUIDED" ||
         now - mode_switch_sent_time_ > kRetryCooldown)) {
      mode_switch_pending_ = false;
    }
    if (arming_pending_ &&
        (fc_status.armed ||
         now - arming_sent_time_ > kRetryCooldown)) {
      arming_pending_ = false;
    }
    // takeoff_pending_ is released by the service-response callback on
    // success (which also sets TAKING_OFF); here we only need the timeout
    // fallback in case the response is lost.
    if (takeoff_pending_ && now - takeoff_sent_time_ > kRetryCooldown) {
      takeoff_pending_ = false;
    }

    // Step 1: Switch to GUIDED mode if not already
    if (fc_status.mode != "GUIDED" && !mode_switch_pending_) {
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
        "Attempting GUIDED mode switch... current mode: '%s', service ready: %d",
        fc_status.mode.c_str(), mode_srv->service_is_ready());
      if (mode_srv->service_is_ready()) {
        auto request = std::make_shared<mavros_msgs::srv::SetMode::Request>();
        request->custom_mode = "GUIDED";
        mode_switch_pending_ = true;
        mode_switch_sent_time_ = now;

        mode_srv->async_send_request(request,
          [this](rclcpp::Client<mavros_msgs::srv::SetMode>::SharedFuture future) {
            // Do NOT clear mode_switch_pending_ here — let observed
            // fc_status.mode=="GUIDED" or the retry-cooldown timeout
            // clear it, to avoid re-sending before /mavros/state catches up.
            try {
              auto response = future.get();
              if (response->mode_sent) {
                RCLCPP_INFO(this->get_logger(), "GUIDED mode request sent");
              } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to send GUIDED mode request");
                mode_switch_pending_ = false;  // rejected → retry immediately
              }
            } catch (const std::exception& e) {
              RCLCPP_ERROR(this->get_logger(), "Mode switch service failed: %s", e.what());
              mode_switch_pending_ = false;
            }
          });
      }
      return;  // Wait for mode switch
    }
    
    // Step 2: Arm if in GUIDED but not armed
    if (fc_status.mode == "GUIDED" && !fc_status.armed && !arming_pending_) {
      if (arming_srv->service_is_ready()) {
        auto request = std::make_shared<mavros_msgs::srv::CommandBool::Request>();
        request->value = true;
        arming_pending_ = true;
        arming_sent_time_ = now;

        arming_srv->async_send_request(request,
          [this](rclcpp::Client<mavros_msgs::srv::CommandBool>::SharedFuture future) {
            // Do NOT clear arming_pending_ here — observed fc_status.armed
            // or retry-cooldown timeout clears it. ArduCopter silently
            // ignores arm-while-armed, so we don't want to re-send every
            // planning cycle while /mavros/state lags by ~100 ms.
            try {
              auto response = future.get();
              if (response->success) {
                RCLCPP_INFO(this->get_logger(), "Arming command accepted");
              } else {
                RCLCPP_ERROR(this->get_logger(), "Arming command rejected");
                arming_pending_ = false;  // rejected → retry immediately
              }
            } catch (const std::exception& e) {
              RCLCPP_ERROR(this->get_logger(), "Arming service failed: %s", e.what());
              arming_pending_ = false;
            }
          });
      }
      return;  // Wait for arming
    }
    
    // Step 3: Takeoff if armed
    if (fc_status.mode == "GUIDED" && fc_status.armed && !takeoff_pending_) {
      if (takeoff_srv->service_is_ready()) {
        auto request = std::make_shared<mavros_msgs::srv::CommandTOL::Request>();
        // Use goal altitude if set, otherwise use _goal_up_coordinate (from takeoff command)
        request->altitude = _goal_set ? _goal_in_world_frame.z : _goal_up_coordinate;
        takeoff_pending_ = true;
        takeoff_sent_time_ = now;

        takeoff_srv->async_send_request(request,
          [this](rclcpp::Client<mavros_msgs::srv::CommandTOL>::SharedFuture future) {
            try {
              auto response = future.get();
              if (response->success) {
                RCLCPP_WARN(this->get_logger(), "Takeoff command accepted!");
                set_auto_pilot_state_forced(PlanningStates::TAKING_OFF);
                // Leave takeoff_pending_ true — we've already moved out of
                // OFF, so the startup block won't re-enter anyway.
              } else {
                RCLCPP_ERROR(this->get_logger(), "Takeoff command rejected by FCU");
                takeoff_pending_ = false;  // rejected → retry immediately
              }
            } catch (const std::exception& e) {
              RCLCPP_ERROR(this->get_logger(), "Takeoff service failed: %s", e.what());
              takeoff_pending_ = false;
            }
          });
      }
    }
    return;  // Don't proceed with other state logic while waiting for FC
  }

  // BRAKE is a dead-end state — no transitions out except via LAND or RESET, only holds position
  if (_planner_state == PlanningStates::BRAKE) {
    // MAVROS: send BRAKE mode switch once
    if (_runtime_mode == RuntimeModes::MAVROS && !brake_mode_switch_sent_ && !mode_switch_pending_) {
      if (mode_srv->service_is_ready()) {
        auto request = std::make_shared<mavros_msgs::srv::SetMode::Request>();
        request->custom_mode = "BRAKE";
        mode_switch_pending_ = true;
        brake_mode_switch_sent_ = true;

        mode_srv->async_send_request(request,
          [this](rclcpp::Client<mavros_msgs::srv::SetMode>::SharedFuture future) {
            mode_switch_pending_ = false;
            try {
              auto response = future.get();
              if (response->mode_sent) {
                RCLCPP_WARN(this->get_logger(), "BRAKE mode request sent to FC");
              } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to send BRAKE mode request");
                brake_mode_switch_sent_ = false;  // Retry next cycle
              }
            } catch (const std::exception& e) {
              RCLCPP_ERROR(this->get_logger(), "BRAKE mode switch failed: %s", e.what());
              brake_mode_switch_sent_ = false;
            }
          });
      }
    }
    return;  // Nothing escapes BRAKE except via LAND or RESET
  }

  // LAND: for OmniDrones, check if drone reached initial altitude → reset
  // For MAVROS, disarm detection above handles the reset after FC completes landing
  if (_planner_state == PlanningStates::LAND) {
    if (_runtime_mode == RuntimeModes::OMNIDRONES) {
      double altitude_error = std::abs(_state.pose.position.z - _home_in_world_frame.z);
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
          "Landing: alt=%.2f m, target=%.2f m, error=%.2f m",
          _state.pose.position.z, _home_in_world_frame.z, altitude_error);
      if (altitude_error < 0.1) {
        RCLCPP_INFO(this->get_logger(), "Landing complete, resetting planner");
        reset_planner();
      }
    }
    // MAVROS: the disarm detection block above will call reset_planner()
    // when flight_controller_status.armed becomes false after FC lands
    return;
  }

  // Handle disarm detection (for real FC)
  // Only reset if we were actually flying (TRAJECTORY_CONTROL or later), not during startup
  if (_runtime_mode == RuntimeModes::MAVROS && 
      _planner_state != PlanningStates::OFF &&
      _planner_state != PlanningStates::TAKING_OFF &&
      !fc_status.armed) {
    RCLCPP_WARN(this->get_logger(), "Vehicle disarmed, resetting planner");
    opus_abort_planning("Vehicle disarmed");
    reset_planner();
    return;
  }

  if (!_goal_set) return;

  geometry_msgs::msg::Point goal_in_world_frame = _goal_in_world_frame;
  goal_in_world_frame.z = _state.pose.position.z;
  double distance_to_goal = (geometryToEigen(_state.pose.position) - geometryToEigen(goal_in_world_frame)).norm();

  // Half-space check: has the drone crossed the perpendicular line at the goal,
  // drawn normal to the segment from the previous waypoint (or home) to the goal?
  // (drone_xy - goal_xy) · (goal_xy - prev_xy) > 0 means the drone has passed the goal.
  Eigen::Vector2d segment_start_xy;
  if (_waypoint_mission_active && _current_waypoint_index > 0) {
    segment_start_xy << _world_waypoints[_current_waypoint_index - 1].x(),
                        _world_waypoints[_current_waypoint_index - 1].y();
  } else {
    segment_start_xy << _home_in_world_frame.x, _home_in_world_frame.y;
  }
  Eigen::Vector2d goal_xy(_goal_in_world_frame.x, _goal_in_world_frame.y);
  Eigen::Vector2d drone_xy(_state.pose.position.x, _state.pose.position.y);
  Eigen::Vector2d seg_dir = goal_xy - segment_start_xy;
  double seg_len_sq = seg_dir.squaredNorm();
  // Half-space for non-degenerate segments; fall back to distance for zero-length
  bool passed_goal_halfspace = (seg_len_sq > 1e-6)
      ? (drone_xy - goal_xy).dot(seg_dir) > 0.0
      : (distance_to_goal < _go_to_goal_threshold);

  // Transition from TAKING_OFF to ALIGNING_HEADING when takeoff altitude reached.
  // Use _takeoff_altitude (what the FC was actually commanded to fly to, set by
  // takeoff_callback for MAVROS or first mission upload for OmniDrones), NOT
  // _goal_up_coordinate: a mission upload with a higher WP[0].up would override
  // _goal_up_coordinate and push the threshold above the FC's actual hover
  // altitude, stranding the drone in TAKING_OFF.
  double takeoff_complete_altitude = _takeoff_altitude - 0.1;
  if (_state.pose.position.z >= takeoff_complete_altitude && _planner_state == PlanningStates::TAKING_OFF) {
    RCLCPP_INFO(this->get_logger(), "Takeoff complete at z=%.2f (threshold=%.2f), aligning heading to goal",
                _state.pose.position.z, takeoff_complete_altitude);
    if (_waypoint_mission_active && _current_waypoint_index < _waypoint_headings.size()) {
      _goal_heading = _waypoint_headings[_current_waypoint_index];
    } else {
      _goal_heading = compute_heading_to_goal();
    }
    set_auto_pilot_state_forced(PlanningStates::ALIGNING_HEADING);
  }
  // ALIGNING_HEADING → TRAJECTORY_CONTROL when heading is within threshold
  else if (_planner_state == PlanningStates::ALIGNING_HEADING) {
    // Get current yaw from quaternion
    double qw = _state.pose.orientation.w;
    double qx = _state.pose.orientation.x;
    double qy = _state.pose.orientation.y;
    double qz = _state.pose.orientation.z;
    double current_yaw = std::atan2(2.0 * (qw * qz + qx * qy),
                                     1.0 - 2.0 * (qy * qy + qz * qz));
    double yaw_error = _goal_heading - current_yaw;
    // Normalize to [-pi, pi]
    while (yaw_error > M_PI) yaw_error -= 2.0 * M_PI;
    while (yaw_error < -M_PI) yaw_error += 2.0 * M_PI;

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
      "Aligning heading: current=%.1f deg, target=%.1f deg, error=%.1f deg",
      current_yaw * 180.0 / M_PI, _goal_heading * 180.0 / M_PI,
      yaw_error * 180.0 / M_PI);

    if (std::abs(yaw_error) < kHeadingAlignThreshold_) {
      RCLCPP_INFO(this->get_logger(), "Heading aligned (error=%.1f deg), starting trajectory control",
                  yaw_error * 180.0 / M_PI);
      // Clear trajectory state so planner starts fresh
      had_reference_trajectory = false;
      {
        const std::lock_guard<std::mutex> tlock(trajectory_mutex_);
        trajectory_queue_.clear();
        pending_trajectory_start_time_override_.reset();
      }
      // OPUS: trigger immediate submission on the first planned trajectory.
      // Do NOT reset opus_pre_queue_ here — it was already cleared by
      // opus_abort_planning("Waypoint reached, advancing") on the
      // TRAJECTORY_CONTROL → HOLDING_WAYPOINT transition, and img_callback is
      // gated to TRAJECTORY_CONTROL/WAITING_FOR_OPUS so nothing repopulates
      // it during HOLDING_WAYPOINT or ALIGNING_HEADING.
      if (opus_enabled_) {
        const std::lock_guard<std::mutex> olock(opus_mutex_);
        opus_submission_needed_ = true;
        set_auto_pilot_state_forced(PlanningStates::WAITING_FOR_OPUS);
      } else {
        set_auto_pilot_state_forced(PlanningStates::TRAJECTORY_CONTROL);
      }
    }
  }
  // Transition to GO_TO_GOAL / HOLDING_WAYPOINT when the drone is near the goal
  // OR has crossed the perpendicular line at the goal (half-space check along
  // the segment from the previous waypoint to the current goal).
  // Also fires from WAITING_FOR_OPUS — the drone may be carried past the goal
  // by inertia while waiting for OPUS.
  else if ((_planner_state == PlanningStates::TRAJECTORY_CONTROL ||
            _planner_state == PlanningStates::WAITING_FOR_OPUS) &&
           (distance_to_goal < _go_to_goal_threshold || passed_goal_halfspace)) {
    if (_waypoint_mission_active) {
      // Abort any OPUS planning state since we're done with this segment
      opus_abort_planning("Waypoint reached, advancing");
      set_auto_pilot_state_forced(PlanningStates::HOLDING_WAYPOINT);
    } else {
      opus_abort_planning("Goal reached");
      set_auto_pilot_state_forced(PlanningStates::GO_TO_GOAL);
    }
  }
  // HOLDING_WAYPOINT: hold position for hold_time, then advance
  else if (_planner_state == PlanningStates::HOLDING_WAYPOINT) {
    double time_in_state = (this->now() - time_of_switch_to_current_state_).seconds();
    double hold_time = 1.0;  // default
    if (_current_waypoint_index < _waypoint_hold_times.size()) {
      hold_time = _waypoint_hold_times[_current_waypoint_index];
    }
    if (time_in_state >= hold_time) {
      if (advance_waypoint()) {
        // More waypoints → align heading then plan to next. The OPUS lock is
        // only requested once we enter WAITING_FOR_OPUS: during ALIGNING_HEADING
        // the drone has no goal-direction depth image, so any trajectory it
        // could speculatively submit would not be collision-checked against
        // the real obstacle set along the path. Holding the lock during
        // yaw alignment would also starve peers that could plan meanwhile.
        set_auto_pilot_state_forced(PlanningStates::ALIGNING_HEADING);
      } else {
        // Mission complete → go to goal (final position hold)
        set_auto_pilot_state_forced(PlanningStates::GO_TO_GOAL);
      }
    }
  }
  // GO_TO_GOAL → FINISHED when the last trajectory has been fully tracked
  else if (_planner_state == PlanningStates::GO_TO_GOAL && had_reference_trajectory) {
    rclcpp::Duration trajectory_point_time = this->now() - _reference_trajectory_start_time;
    double point_time = trajectory_point_time.seconds();
    if (point_time > reference_trajectory_.get_duration()) {
      RCLCPP_INFO(this->get_logger(),
          "Trajectory complete (%.2f s > %.2f s duration), mission finished",
          point_time, reference_trajectory_.get_duration());
      set_auto_pilot_state_forced(PlanningStates::FINISHED);
    }
  }
  // FINISHED + reinitialise requested: check if drone reached home → auto-reset
  else if (_planner_state == PlanningStates::FINISHED && _reinitialise_requested) {
    double dist_to_home = (geometryToEigen(_state.pose.position) -
                           geometryToEigen(_home_in_world_frame)).norm();

    double qw = _state.pose.orientation.w;
    double qx = _state.pose.orientation.x;
    double qy = _state.pose.orientation.y;
    double qz = _state.pose.orientation.z;
    double current_yaw = std::atan2(2.0 * (qw * qz + qx * qy),
                                     1.0 - 2.0 * (qy * qy + qz * qz));
    double yaw_error = _initial_heading - current_yaw;
    while (yaw_error > M_PI) yaw_error -= 2.0 * M_PI;
    while (yaw_error < -M_PI) yaw_error += 2.0 * M_PI;

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "Re-initialise: dist=%.2f m, heading_err=%.1f deg",
        dist_to_home, yaw_error * 180.0 / M_PI);

    if (dist_to_home < 0.1 && std::abs(yaw_error) < kHeadingAlignThreshold_) {
      RCLCPP_INFO(this->get_logger(),
          "Re-initialise complete: at home position, resetting planner");
      reset_planner();
    }
  }
  // Land when at goal (MAVROS only)
  // else if (_runtime_mode == RuntimeModes::MAVROS &&
  //          _planner_state == PlanningStates::GO_TO_GOAL &&
  //          distance_to_goal < _go_to_goal_threshold * 0.2 &&
  //          !land_pending_) {
  //   if (land_srv->service_is_ready()) {
  //     auto request = std::make_shared<mavros_msgs::srv::CommandTOL::Request>();
  //     land_pending_ = true;
      
  //     land_srv->async_send_request(request,
  //       [this](rclcpp::Client<mavros_msgs::srv::CommandTOL>::SharedFuture future) {
  //         land_pending_ = false;
  //         try {
  //           auto response = future.get();
  //           if (response->success) {
  //             set_auto_pilot_state_forced(PlanningStates::LAND);
  //             RCLCPP_WARN(this->get_logger(), "Land command accepted!");
  //           } else {
  //             RCLCPP_ERROR(this->get_logger(), "Land command rejected by FCU");
  //           }
  //         } catch (const std::exception& e) {
  //           RCLCPP_ERROR(this->get_logger(), "Land service failed: %s", e.what());
  //         }
  //       });
  //   }
  // }
  // Reset after landing complete
  // else if (_runtime_mode == RuntimeModes::MAVROS &&
  //          _planner_state == PlanningStates::LAND && !flight_controller_status.armed) {
  //   reset_callback(nullptr);
  // }
}

void PlannerNode::set_auto_pilot_state_forced(const PlanningStates& new_state) {
  const rclcpp::Time time_now = this->now();

  if (new_state != PlanningStates::TRAJECTORY_CONTROL &&
      new_state != PlanningStates::WAITING_FOR_OPUS) {
    const std::lock_guard<std::mutex> lock(trajectory_mutex_);
    if (!trajectory_queue_.empty()) {
      trajectory_queue_.clear();
    }
    pending_trajectory_start_time_override_.reset();
  }
  time_of_switch_to_current_state_ = time_now;
  _planner_state.store(new_state, std::memory_order_release);

  std::string state_name;
  switch (new_state) {
    case PlanningStates::OFF:               state_name = "OFF"; break;
    case PlanningStates::TAKING_OFF:        state_name = "TAKING_OFF"; break;
    case PlanningStates::ALIGNING_HEADING:  state_name = "ALIGNING_HEADING"; break;
    case PlanningStates::WAITING_FOR_OPUS:  state_name = "WAITING_FOR_OPUS"; break;
    case PlanningStates::TRAJECTORY_CONTROL: state_name = "TRAJECTORY_CONTROL"; break;
    case PlanningStates::GO_TO_GOAL:        state_name = "GO_TO_GOAL"; break;
    case PlanningStates::HOLDING_WAYPOINT:  state_name = "HOLDING_WAYPOINT"; break;
    case PlanningStates::LAND:              state_name = "LAND"; break;
    case PlanningStates::FINISHED:          state_name = "FINISHED"; break;
    case PlanningStates::BRAKE:             state_name = "BRAKE"; break;
  }
  RCLCPP_WARN(this->get_logger(), "Switched to %s state", state_name.c_str());
}
