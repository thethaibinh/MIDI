#include "planner_node.hpp"

using namespace quadrotor_common;

// Phase 1: Upload waypoints, convert FLU→world, log, ACK back to GCS
void PlannerNode::mission_upload_callback(const ground_system_msgs::msg::SwarmMissionUpload::SharedPtr msg) {
  RCLCPP_INFO(this->get_logger(), "Received mission upload: '%s' (%zu waypoints, loop_count=%u)",
              msg->mission_name.c_str(), msg->waypoints.size(), msg->loop_count);

  // Build ACK message
  auto ack = ground_system_msgs::msg::SwarmMissionAck();
  ack.drone_id = opus_drone_id_;
  ack.mission_name = msg->mission_name;

  if (mission_received_) {
    RCLCPP_WARN(this->get_logger(), "Mission already in flight, rejecting upload");
    ack.success = false;
    ack.message = "Mission already in flight";
    ack.num_waypoints = 0;
    mission_ack_pub_->publish(ack);
    return;
  }

  if (msg->waypoints.empty()) {
    RCLCPP_WARN(this->get_logger(), "Mission '%s' has no waypoints, rejecting", msg->mission_name.c_str());
    ack.success = false;
    ack.message = "No waypoints provided";
    ack.num_waypoints = 0;
    mission_ack_pub_->publish(ack);
    return;
  }

  _mission_name = msg->mission_name;

  // Store home position before computing goal (relative waypoints use this).
  // _state is written on state_callback_group_, so snapshot under the lock.
  double qw, qx, qy, qz;
  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    _home_in_world_frame = _state.pose.position;
    qw = _state.pose.orientation.w;
    qx = _state.pose.orientation.x;
    qy = _state.pose.orientation.y;
    qz = _state.pose.orientation.z;
  }
  double initial_yaw = std::atan2(2.0 * (qw * qz + qx * qy),
                                   1.0 - 2.0 * (qy * qy + qz * qz));
  _initial_heading = initial_yaw;  // Store for reinitialise

  _waypoint_list.assign(msg->waypoints.begin(), msg->waypoints.end());
  _current_waypoint_index = 0;
  _remaining_loops = msg->loop_count > 0 ? msg->loop_count - 1 : 0;
  _waypoint_mission_active = true;

  // Set takeoff altitude from first waypoint.
  // For MAVROS: a prior takeoff command may have already set this; the mission overrides it
  //   since the mission defines the actual flight altitude.
  // For OmniDrones: drones are already airborne, so this must match the mission altitude
  //   to allow the TAKING_OFF → ALIGNING_HEADING transition.
  _goal_up_coordinate = msg->waypoints[0].up;

  // Convert FLU waypoints to world frame ONCE using initial yaw.
  convert_waypoints_to_world(initial_yaw);

  RCLCPP_INFO(this->get_logger(), "Mission '%s' uploaded: %zu waypoints, %u total passes, initial_yaw=%.1f deg",
              msg->mission_name.c_str(), _waypoint_list.size(),
              msg->loop_count > 0 ? msg->loop_count : 1,
              initial_yaw * 180.0 / M_PI);

  set_goal_from_waypoint();
  _goal_heading = compute_heading_to_goal();
  _goal_set = true;
  mission_uploaded_ = true;
  log_mission_to_yaml(msg);

  // Publish ACK
  ack.success = true;
  ack.num_waypoints = static_cast<uint32_t>(msg->waypoints.size());
  ack.message = "OK";
  mission_ack_pub_->publish(ack);
  RCLCPP_INFO(this->get_logger(), "Mission ACK sent (success, %u waypoints)", ack.num_waypoints);
}

// Phase 2: Start swarming (triggered by GCS after all agents ACK)
void PlannerNode::mission_callback(const ground_system_msgs::msg::StartSwarmMission::SharedPtr msg) {
  RCLCPP_INFO(this->get_logger(), "Received START command: '%s'", msg->mission_name.c_str());

  if (mission_received_) {
    RCLCPP_WARN(this->get_logger(), "Mission already started, ignoring duplicate start");
    return;
  }

  if (!mission_uploaded_) {
    RCLCPP_ERROR(this->get_logger(), "No mission uploaded! Upload waypoints before starting.");
    return;
  }

  if (msg->mission_name != _mission_name) {
    RCLCPP_ERROR(this->get_logger(), "Mission name mismatch: uploaded='%s', start='%s'",
                 _mission_name.c_str(), msg->mission_name.c_str());
    return;
  }

  mission_received_ = true;

  RCLCPP_INFO(this->get_logger(), "Starting mission '%s': goal=(%.2f, %.2f, %.2f), heading=%.1f deg",
              _mission_name.c_str(),
              _goal_in_world_frame.x, _goal_in_world_frame.y, _goal_in_world_frame.z,
              _goal_heading * 180.0 / M_PI);

  if (_runtime_mode == RuntimeModes::OMNIDRONES) {
    RCLCPP_WARN(this->get_logger(), "[SIM] Starting navigation!");
    set_auto_pilot_state_forced(PlanningStates::TAKING_OFF);
    // Trajectory fields protected by trajectory_mutex_ (img_callback on default
    // group may still be mid-execution past the _planner_state gate).
    {
      const std::lock_guard<std::mutex> tlock(trajectory_mutex_);
      steering_value = 0.0f;
      _steered = false;
      trajectory_queue_.clear();
      pending_trajectory_start_time_override_.reset();
      reference_trajectory_ = ruckig::Trajectory<3>();
      had_reference_trajectory = false;
    }
  } else if (_runtime_mode == RuntimeModes::MAVROS) {
    RCLCPP_WARN(this->get_logger(), "[MAVROS] Initiating flight sequence...");
  }
}

void PlannerNode::takeoff_callback(const ground_system_msgs::msg::Takeoff::SharedPtr msg) {
  RCLCPP_INFO(this->get_logger(), "Received takeoff command: altitude = %.2f m", msg->altitude);
  
  if (_planner_state != PlanningStates::OFF) {
    RCLCPP_WARN(this->get_logger(), "Planner not in OFF state, ignoring takeoff command");
    return;
  }
  
  // Set takeoff altitude in goal_up_coordinate (used by update_planner_state for takeoff)
  _goal_up_coordinate = msg->altitude;
  
  // NOTE: For takeoff-only, we do NOT set _goal_set or mission_received_
  // This allows the planner to just takeoff and hover
  
  if (_runtime_mode == RuntimeModes::OMNIDRONES) {
    // Simulation mode: directly start takeoff
    RCLCPP_WARN(this->get_logger(), "[SIM] Starting takeoff to %.2f m!", msg->altitude);
    set_auto_pilot_state_forced(PlanningStates::TAKING_OFF);
    {
      const std::lock_guard<std::mutex> lock(state_mutex_);
      _home_in_world_frame = _state.pose.position;
    }
    {
      const std::lock_guard<std::mutex> tlock(trajectory_mutex_);
      steering_value = 0.0f;
      _steered = false;
      trajectory_queue_.clear();
      pending_trajectory_start_time_override_.reset();
      reference_trajectory_ = ruckig::Trajectory<3>();
      had_reference_trajectory = false;
    }
  } else if (_runtime_mode == RuntimeModes::MAVROS) {
    // Real FC mode: set flag to trigger GUIDED->ARM->TAKEOFF sequence in update_planner_state()
    // Do NOT immediately change state - let the FC sequence complete first
    RCLCPP_WARN(this->get_logger(), "[MAVROS] Takeoff command received, initiating takeoff to %.2f m...",
                msg->altitude);
    {
      // Snapshot current pose as takeoff origin — used by the observed-
      // altitude fallback in update_planner_state() to detect a real
      // takeoff when COMMAND_ACKs are lost on the multi-GCS MAVLink link.
      const std::lock_guard<std::mutex> lock(state_mutex_);
      _home_in_world_frame = _state.pose.position;
    }
    takeoff_requested_ = true;
  }
}

void PlannerNode::reset_planner() {
  RCLCPP_WARN(this->get_logger(), "Planner: Reset quadrotor!");
  set_auto_pilot_state_forced(PlanningStates::OFF);
  // Trajectory fields protected by trajectory_mutex_ (img_callback on default
  // group may still be mid-execution past the _planner_state gate).
  {
    const std::lock_guard<std::mutex> tlock(trajectory_mutex_);
    steering_value = 0.0f;
    _steered = false;
    trajectory_queue_.clear();
    pending_trajectory_start_time_override_.reset();
    reference_trajectory_ = ruckig::Trajectory<3>();
    had_reference_trajectory = false;
  }
  _goal_set = false;
  mission_received_ = false;
  mission_uploaded_ = false;
  mode_switch_pending_ = false;
  arming_pending_ = false;
  takeoff_pending_ = false;
  land_pending_ = false;
  takeoff_requested_ = false;
  _reinitialise_requested = false;
  brake_mode_switch_sent_ = false;
  // Reset fence breach recovery state
  _has_valid_setpoint = false;
  _last_valid_position = Eigen::Vector3d(0.0, 0.0, 0.0);
  _last_valid_heading = 0.0;
  // Reset takeoff altitude
  _goal_up_coordinate = 0.0;
  // Reset waypoint mission state
  _waypoint_list.clear();
  _world_waypoints.clear();
  _waypoint_headings.clear();
  _waypoint_hold_times.clear();
  _current_waypoint_index = 0;
  _remaining_loops = 0;
  _waypoint_mission_active = false;
  // Reset OPUS coordination state
  if (opus_enabled_) {
    const std::lock_guard<std::mutex> olock(opus_mutex_);
    opus_granted_ = false;
    opus_check_pending_ = false;
    opus_lock_pending_ = false;
    opus_submission_needed_ = false;
    opus_grant_time_ = std::chrono::steady_clock::time_point{};
    opus_pre_queue_.reset();
    opus_pending_trajectory_ = ruckig::Trajectory<3>();  // Clear stale trajectory
    // Bump sequence so any in-flight service-response / status callbacks for
    // the old round will be detected as stale and discarded.
    ++opus_plan_sequence_;
  }
  // Fire-and-forget CANCEL to the coordinator so any residual queue/lock
  // entry for this drone is cleared on the GCS side too.
  if (opus_enabled_ && opus_lock_req_pub_) {
    OpusPlanLockReqMsg req;
    req.drone_id = opus_drone_id_;
    req.plan_sequence = opus_plan_sequence_;
    req.action = OpusPlanLockReqMsg::ACTION_CANCEL;
    opus_lock_req_pub_->publish(req);
  }
}

// ============================================================================
// Waypoint Mission Helpers
// ============================================================================

void PlannerNode::set_goal_from_waypoint() {
  if (_current_waypoint_index >= _world_waypoints.size()) return;

  const auto& wp_world = _world_waypoints[_current_waypoint_index];
  _goal_in_world_frame.x = wp_world.x();
  _goal_in_world_frame.y = wp_world.y();
  _goal_in_world_frame.z = wp_world.z();

  // Use pre-computed heading (NaN was resolved during convert_waypoints_to_world)
  _goal_heading = _waypoint_headings[_current_waypoint_index];

  RCLCPP_INFO(this->get_logger(), "Waypoint %zu/%zu: world=(%.2f,%.2f,%.2f) heading=%.1f deg",
              _current_waypoint_index + 1, _world_waypoints.size(),
              _goal_in_world_frame.x, _goal_in_world_frame.y, _goal_in_world_frame.z,
              _goal_heading * 180.0 / M_PI);
}

bool PlannerNode::advance_waypoint() {
  _current_waypoint_index++;

  if (_current_waypoint_index >= _world_waypoints.size()) {
    if (_remaining_loops > 0) {
      _remaining_loops--;
      _current_waypoint_index = 0;
      RCLCPP_INFO(this->get_logger(), "Waypoint loop restart (%u loops remaining)",
                  _remaining_loops);
    } else {
      RCLCPP_INFO(this->get_logger(), "Waypoint mission complete");
      _waypoint_mission_active = false;
      return false;
    }
  }

  set_goal_from_waypoint();
  return true;
}

void PlannerNode::convert_waypoints_to_world(double initial_yaw) {
  _world_waypoints.clear();
  _waypoint_headings.clear();
  _waypoint_hold_times.clear();

  double cos_yaw = std::cos(initial_yaw);
  double sin_yaw = std::sin(initial_yaw);
  double home_x = _home_in_world_frame.x;
  double home_y = _home_in_world_frame.y;

  for (size_t i = 0; i < _waypoint_list.size(); i++) {
    const auto& wp = _waypoint_list[i];

    // Rotate FLU (Forward/Left/Up) by initial yaw to get world-frame offset
    // Body FLU: X=Forward, Y=Left, Z=Up
    // World ENU: X=East, Y=North, Z=Up (MAVROS)
    // World NWU: X=North, Y=West, Z=Up (OmniDrones)
    // The rotation is the same in both conventions — yaw rotates the horizontal plane
    double world_dx = cos_yaw * wp.forward - sin_yaw * wp.left;
    double world_dy = sin_yaw * wp.forward + cos_yaw * wp.left;

    Eigen::Vector3d wp_world(home_x + world_dx, home_y + world_dy, wp.up);
    _world_waypoints.push_back(wp_world);
    _waypoint_hold_times.push_back(wp.hold_time);

    // Pre-compute heading: if NaN, compute from home (or previous WP) toward this WP
    if (std::isnan(wp.heading)) {
      Eigen::Vector3d from;
      if (i == 0) {
        from = Eigen::Vector3d(home_x, home_y, wp.up);
      } else {
        from = _world_waypoints[i - 1];
      }
      double dx = wp_world.x() - from.x();
      double dy = wp_world.y() - from.y();
      double auto_heading = std::atan2(dy, dx);
      _waypoint_headings.push_back(auto_heading);
    } else {
      // Absolute heading from message (already in world frame)
      _waypoint_headings.push_back(wp.heading);
    }

    RCLCPP_INFO(this->get_logger(),
      "  WP%zu FLU(fwd=%.2f, left=%.2f, up=%.2f) -> world(%.2f, %.2f, %.2f) heading=%.1f deg",
      i + 1, wp.forward, wp.left, wp.up,
      wp_world.x(), wp_world.y(), wp_world.z(),
      _waypoint_headings.back() * 180.0 / M_PI);
  }
}

double PlannerNode::compute_heading_to_goal() const {
  double dx = _goal_in_world_frame.x - _state.pose.position.x;
  double dy = _goal_in_world_frame.y - _state.pose.position.y;
  // ENU: yaw = atan2(dy, dx) → 0 = East, π/2 = North
  // NWU: yaw = atan2(dy, dx) → 0 = North, π/2 = West
  // Both use the same atan2(y_diff, x_diff) in their respective frames
  return std::atan2(dy, dx);
}

void PlannerNode::log_mission_to_yaml(const ground_system_msgs::msg::SwarmMissionUpload::SharedPtr& msg) {
  // Build YAML mission log
  YAML::Emitter out;
  out << YAML::BeginMap;
  out << YAML::Key << "mission_name" << YAML::Value << msg->mission_name;
  out << YAML::Key << "drone_id" << YAML::Value << static_cast<int>(opus_drone_id_);
  out << YAML::Key << "timestamp" << YAML::Value << this->now().seconds();

  // Initial position and heading at mission receive time
  out << YAML::Key << "initial_position" << YAML::Value << YAML::BeginMap;
  out << YAML::Key << "x" << YAML::Value << _home_in_world_frame.x;
  out << YAML::Key << "y" << YAML::Value << _home_in_world_frame.y;
  out << YAML::Key << "z" << YAML::Value << _home_in_world_frame.z;
  out << YAML::EndMap;

  double qw = _state.pose.orientation.w;
  double qx = _state.pose.orientation.x;
  double qy = _state.pose.orientation.y;
  double qz = _state.pose.orientation.z;
  double initial_yaw = std::atan2(2.0 * (qw * qz + qx * qy),
                                   1.0 - 2.0 * (qy * qy + qz * qz));
  out << YAML::Key << "initial_yaw_rad" << YAML::Value << initial_yaw;
  out << YAML::Key << "initial_yaw_deg" << YAML::Value << initial_yaw * 180.0 / M_PI;

  out << YAML::Key << "loop_count" << YAML::Value << msg->loop_count;

  // Original FLU waypoints (as received from GCS)
  out << YAML::Key << "waypoints_flu" << YAML::Value << YAML::BeginSeq;
  for (const auto& wp : msg->waypoints) {
    out << YAML::BeginMap;
    out << YAML::Key << "forward" << YAML::Value << wp.forward;
    out << YAML::Key << "left" << YAML::Value << wp.left;
    out << YAML::Key << "up" << YAML::Value << wp.up;
    out << YAML::Key << "heading" << YAML::Value << wp.heading;
    out << YAML::Key << "hold_time" << YAML::Value << wp.hold_time;
    out << YAML::EndMap;
  }
  out << YAML::EndSeq;

  // Computed world-frame waypoints (what the agent will actually navigate to)
  out << YAML::Key << "waypoints_world" << YAML::Value << YAML::BeginSeq;
  for (size_t i = 0; i < _world_waypoints.size(); i++) {
    out << YAML::BeginMap;
    out << YAML::Key << "x" << YAML::Value << _world_waypoints[i].x();
    out << YAML::Key << "y" << YAML::Value << _world_waypoints[i].y();
    out << YAML::Key << "z" << YAML::Value << _world_waypoints[i].z();
    out << YAML::Key << "heading_rad" << YAML::Value << _waypoint_headings[i];
    out << YAML::Key << "hold_time" << YAML::Value << _waypoint_hold_times[i];
    out << YAML::EndMap;
  }
  out << YAML::EndSeq;
  out << YAML::EndMap;

  // Write to /tmp/mission_log_<drone_id>_<timestamp>.yaml
  auto time_now = std::chrono::system_clock::now();
  auto epoch = std::chrono::duration_cast<std::chrono::seconds>(
    time_now.time_since_epoch()).count();
  std::string filename = "/tmp/mission_log_drone" +
    std::to_string(opus_drone_id_) + "_" + std::to_string(epoch) + ".yaml";

  std::ofstream file(filename);
  if (file.is_open()) {
    file << out.c_str();
    file.close();
    RCLCPP_INFO(this->get_logger(), "Mission logged to %s", filename.c_str());
  } else {
    RCLCPP_WARN(this->get_logger(), "Failed to write mission log to %s", filename.c_str());
  }
}

void PlannerNode::reinitialise_callback(const std_msgs::msg::Empty::SharedPtr msg) {
  (void)msg;

  if (_planner_state != PlanningStates::FINISHED) {
    RCLCPP_WARN(this->get_logger(),
        "Re-initialise rejected: only allowed in FINISHED state (current: %d)",
        static_cast<int>(_planner_state.load()));
    return;
  }

  RCLCPP_INFO(this->get_logger(),
      "Re-initialise: returning to home (%.2f, %.2f, %.2f) heading=%.1f deg",
      _home_in_world_frame.x, _home_in_world_frame.y, _home_in_world_frame.z,
      _initial_heading * 180.0 / M_PI);

  // Set goal to initial position and heading — track_trajectory will fly there in FINISHED state
  _goal_in_world_frame = _home_in_world_frame;
  _goal_heading = _initial_heading;
  _reinitialise_requested = true;
}

void PlannerNode::brake_callback(const std_msgs::msg::Empty::SharedPtr msg) {
  (void)msg;
  if (_planner_state == PlanningStates::LAND || _planner_state == PlanningStates::FINISHED || _planner_state == PlanningStates::OFF || _planner_state == PlanningStates::BRAKE)
  {
    RCLCPP_WARN(this->get_logger(), "BRAKE: Emergency stop not allowed in this state!");
    return;
  }
  RCLCPP_WARN(this->get_logger(), "BRAKE: Emergency hold at current position!");

  // Abort any OPUS planning
  if (opus_enabled_) {
    opus_abort_planning("Brake");
  }

  // Clear trajectory state
  {
    const std::lock_guard<std::mutex> tlock(trajectory_mutex_);
    trajectory_queue_.clear();
    pending_trajectory_start_time_override_.reset();
    reference_trajectory_ = ruckig::Trajectory<3>();
    had_reference_trajectory = false;
  }

  // Switch to BRAKE (locks trajectory_mutex_ internally for non-TRAJECTORY_CONTROL)
  set_auto_pilot_state_forced(PlanningStates::BRAKE);
}

void PlannerNode::land_swarm_callback(const std_msgs::msg::Empty::SharedPtr msg) {
  (void)msg;

  if (_planner_state != PlanningStates::FINISHED &&
      _planner_state != PlanningStates::BRAKE) {
    RCLCPP_WARN(this->get_logger(),
        "Land rejected: only allowed in FINISHED or BRAKE state (current: %d)",
        static_cast<int>(_planner_state.load()));
    return;
  }

  RCLCPP_WARN(this->get_logger(),
      "LAND: descending to initial altitude %.2f m at current XY",
      _home_in_world_frame.z);

  // Abort any OPUS planning
  if (opus_enabled_) {
    opus_abort_planning("Land");
  }

  // Clear trajectory state
  {
    const std::lock_guard<std::mutex> tlock(trajectory_mutex_);
    trajectory_queue_.clear();
    pending_trajectory_start_time_override_.reset();
    reference_trajectory_ = ruckig::Trajectory<3>();
    had_reference_trajectory = false;
  }

  if (_runtime_mode == RuntimeModes::OMNIDRONES) {
    // OmniDrones: switch to LAND immediately, tracker will descend to home altitude
    set_auto_pilot_state_forced(PlanningStates::LAND);
  } else if (_runtime_mode == RuntimeModes::MAVROS) {
    // MAVROS: send LAND mode to ArduPilot
    if (!land_pending_ && !mode_switch_pending_) {
      if (mode_srv->service_is_ready()) {
        auto request = std::make_shared<mavros_msgs::srv::SetMode::Request>();
        request->custom_mode = "LAND";
        mode_switch_pending_ = true;
        land_pending_ = true;

        mode_srv->async_send_request(request,
          [this](rclcpp::Client<mavros_msgs::srv::SetMode>::SharedFuture future) {
            mode_switch_pending_ = false;
            try {
              auto response = future.get();
              if (response->mode_sent) {
                RCLCPP_WARN(this->get_logger(), "LAND mode request sent to FC");
                set_auto_pilot_state_forced(PlanningStates::LAND);
              } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to send LAND mode request");
                land_pending_ = false;
              }
            } catch (const std::exception& e) {
              RCLCPP_ERROR(this->get_logger(), "LAND mode switch failed: %s", e.what());
              land_pending_ = false;
            }
          });
      } else {
        RCLCPP_WARN(this->get_logger(), "SetMode service not ready, cannot send LAND");
      }
    }
  }
}
