#include "planner_node.hpp"

using namespace quadrotor_common;

// ============================================================================
// OPUS Coordination Functions
// ============================================================================

// ---- OPUS Lock Request Publisher + Status Subscription ----
//
// Fully topic-based protocol: publish a lock request, watch /opus/status for
// the grant. No ROS2 services or actions on the lock path — zenoh v1.7.1
// drops short service/action replies with a "less than 20 bytes" framing bug.
//   1. opus_send_lock_request() publishes OpusPlanLockRequest (REQUEST).
//   2. opus_status_callback() watches /opus/status; flips opus_granted_ when
//      planning_drone_id == my drone_id.
//   3. Agent publishes trajectory on /opus/trajectory_submit; coordinator
//      broadcasts result on /opus/trajectory_ack (filtered by drone_id).

void PlannerNode::opus_send_lock_request() {
  ++opus_plan_sequence_;  // New planning round

  OpusPlanLockReqMsg req;
  req.drone_id = opus_drone_id_;
  req.plan_sequence = opus_plan_sequence_;
  req.action = OpusPlanLockReqMsg::ACTION_REQUEST;

  opus_lock_req_pub_->publish(req);
  opus_lock_pending_ = true;
  RCLCPP_INFO(this->get_logger(), "OPUS: Sent lock request (drone_id=%d, seq=%u)",
              opus_drone_id_, opus_plan_sequence_);
}

void PlannerNode::opus_status_callback(
    const ground_system_msgs::msg::OpusStatus::SharedPtr msg) {
  const std::lock_guard<std::mutex> lock(opus_mutex_);

  // Detect grant transition: we requested a lock, and the coordinator now
  // reports *us* as the holder.
  if (opus_lock_pending_ && msg->planning_drone_id == opus_drone_id_) {
    opus_lock_pending_ = false;
    opus_granted_ = true;
    opus_grant_time_ = std::chrono::steady_clock::now();
    RCLCPP_INFO(this->get_logger(),
      "OPUS: Planning lock GRANTED via status (seq=%u)", opus_plan_sequence_);
    return;
  }

  // Detect involuntary loss of lock (coordinator reset / timeout). If we
  // thought we were granted but coordinator now shows someone else (or 0),
  // clear our state so the next replan cycle re-requests.
  if (opus_granted_ && msg->planning_drone_id != opus_drone_id_) {
    RCLCPP_WARN(this->get_logger(),
      "OPUS: Lost lock (coordinator now holds for drone %u) — clearing state",
      msg->planning_drone_id);
    opus_granted_ = false;
    opus_check_pending_ = false;
    // Do NOT clear opus_submission_needed_: the drone still needs to submit.
    // Clearing it here caused stuck-in-WAITING_FOR_OPUS when
    // had_reference_trajectory is false (first waypoint).
    opus_grant_time_ = std::chrono::steady_clock::time_point{};
    opus_pre_queue_.reset();
  }
}

// ---- OPUS Trajectory Ack Callback ----

void PlannerNode::opus_trajectory_ack_callback(
    const ground_system_msgs::msg::OpusTrajectoryAck::SharedPtr msg) {
  // Acks are broadcast — ignore those that aren't for us.
  if (msg->drone_id != opus_drone_id_) {
    return;
  }

  // Snapshot OPUS state under opus_mutex_, then release it BEFORE locking
  // trajectory_mutex_.  update_reference_trajectory() locks in the opposite
  // order (trajectory → opus), so nesting opus → trajectory here would
  // deadlock when both fire concurrently.
  bool accepted = false;
  bool structural_reject = false;
  bool collision_reject = false;
  rclcpp::Time submit_t{0, 0, RCL_ROS_TIME};
  ruckig::Trajectory<3> accepted_trajectory;
  std::string reject_reason;

  {
    const std::lock_guard<std::mutex> olock(opus_mutex_);

    // Stale-ack guard: plan_sequence has moved on (abort/reset) → discard silently.
    if (msg->plan_sequence != opus_plan_sequence_) {
      RCLCPP_WARN(this->get_logger(),
        "OPUS: Discarding stale trajectory ack (seq %u, current %u)",
        msg->plan_sequence, opus_plan_sequence_);
      return;
    }

    opus_check_pending_ = false;

    if (msg->accepted) {
      accepted = true;
      opus_granted_ = false;
      opus_grant_time_ = std::chrono::steady_clock::time_point{};

      RCLCPP_INFO(this->get_logger(), "OPUS: Trajectory ACCEPTED (seq=%u) — executing",
                  msg->plan_sequence);
      opus_submission_needed_ = false;
      opus_pre_queue_.reset();

      submit_t = opus_submission_time_;
      accepted_trajectory = opus_pending_trajectory_;
    } else {
      reject_reason = msg->reason;
      RCLCPP_WARN(this->get_logger(), "OPUS: Trajectory REJECTED by GCS: %s",
                  reject_reason.c_str());
      structural_reject = (reject_reason.find("Stale") != std::string::npos ||
                           reject_reason.find("Lock not held") != std::string::npos);
      collision_reject = !structural_reject;

      if (structural_reject) {
        opus_granted_ = false;
        opus_lock_pending_ = false;
        opus_submission_needed_ = false;
        opus_grant_time_ = std::chrono::steady_clock::time_point{};
        opus_pre_queue_.reset();
      } else {
        RCLCPP_WARN(this->get_logger(),
          "OPUS: Collision — releasing lock, will re-request (seq=%u)",
          opus_plan_sequence_);
        opus_granted_ = false;
        opus_lock_pending_ = false;
        opus_submission_needed_ = true;
        opus_grant_time_ = std::chrono::steady_clock::time_point{};
        opus_pre_queue_.reset();
      }
    }
  }  // opus_mutex_ released

  // Now safe to lock trajectory_mutex_ without risk of deadlock.
  if (accepted) {
    const std::lock_guard<std::mutex> tlock(trajectory_mutex_);
    steering_value = 0.0f;
    _steered = false;
    trajectory_queue_.push_back(accepted_trajectory);
    pending_trajectory_start_time_override_ = submit_t;
    if (_planner_state == PlanningStates::WAITING_FOR_OPUS) {
      set_auto_pilot_state_forced(PlanningStates::TRAJECTORY_CONTROL);
    }
  }
}

// ---- OPUS Helpers ----

bool PlannerNode::opus_should_abort_replanning(double* elapsed_sec) {
  const std::lock_guard<std::mutex> lock(opus_mutex_);
  if (!opus_enabled_ || !opus_granted_ || opus_check_pending_ ||
      opus_local_replan_timeout_ <= 0.0) {
    return false;
  }

  const double elapsed = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - opus_grant_time_).count();
  if (elapsed_sec != nullptr) {
    *elapsed_sec = elapsed;
  }

  return elapsed >= opus_local_replan_timeout_;
}

void PlannerNode::opus_abort_planning(const std::string& reason) {
  double elapsed = 0.0;
  bool was_queued_or_granted = false;
  {
    const std::lock_guard<std::mutex> lock(opus_mutex_);
    if (!opus_enabled_) {
      return;
    }

    was_queued_or_granted = opus_granted_ || opus_lock_pending_;
    if (opus_granted_) {
      elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - opus_grant_time_).count();
    }

    opus_granted_ = false;
    opus_check_pending_ = false;
    opus_lock_pending_ = false;
    opus_submission_needed_ = false;
    opus_grant_time_ = std::chrono::steady_clock::time_point{};
    opus_pre_queue_.reset();
  }

  // Always publish CANCEL so the coordinator removes our trajectory from
  // the database.  Past trajectory phases can never collide with future
  // ones (time non-overlap), so the only effect is clearing the virtual
  // hold — correct because the drone is leaving its endpoint.
  if (opus_lock_req_pub_) {
    OpusPlanLockReqMsg req;
    req.drone_id = opus_drone_id_;
    req.plan_sequence = opus_plan_sequence_;
    req.action = OpusPlanLockReqMsg::ACTION_CANCEL;
    opus_lock_req_pub_->publish(req);
  }

  // Also publish OpusPlanAbort — belt-and-braces + carries the reason string.
  auto msg = ground_system_msgs::msg::OpusPlanAbort();
  msg.header.stamp = this->now();
  msg.drone_id = opus_drone_id_;
  msg.plan_sequence = opus_plan_sequence_;
  msg.reason = reason;
  opus_plan_abort_pub_->publish(msg);

  if (elapsed > 0.0) {
    RCLCPP_WARN(this->get_logger(),
      "OPUS: Aborting planning after %.2fs: %s",
      elapsed, reason.c_str());
  } else {
    RCLCPP_WARN(this->get_logger(),
      "OPUS: Aborting planning: %s",
      reason.c_str());
  }
}

Eigen::Vector3d PlannerNode::transform_camera_to_world(
    const Eigen::Vector3d& camera_vec,
    const geometry_msgs::msg::TransformStamped& body_to_world,
    bool is_position,
    const geometry_msgs::msg::Point& world_position) {
  // Camera (RDF: Right-Down-Forward) to Body (FLU: Forward-Left-Up) rotation:
  // cam_x(right) -> body_-y(left, negated)
  // cam_y(down) -> body_-z(up, negated)
  // cam_z(forward) -> body_x(forward)
  Eigen::Vector3d body_vec;
  body_vec.x() =  camera_vec.z();  // forward
  body_vec.y() = -camera_vec.x();  // left (negated right)
  body_vec.z() = -camera_vec.y();  // up (negated down)

  // Body (FLU) to World (ENU) rotation via quaternion from TF
  tf2::Quaternion q(
    body_to_world.transform.rotation.x,
    body_to_world.transform.rotation.y,
    body_to_world.transform.rotation.z,
    body_to_world.transform.rotation.w);
  tf2::Vector3 body_tf(body_vec.x(), body_vec.y(), body_vec.z());
  tf2::Vector3 world_tf = tf2::quatRotate(q, body_tf);

  Eigen::Vector3d result(world_tf.x(), world_tf.y(), world_tf.z());

  // For positions, add current world position offset
  // (camera frame positions are relative to current drone position)
  if (is_position) {
    result.x() += world_position.x;
    result.y() += world_position.y;
    result.z() += world_position.z;
  }

  return result;
}

void PlannerNode::opus_submit_trajectory(
    const ruckig::Trajectory<3>& traj,
    const geometry_msgs::msg::TransformStamped& body_to_world,
    const geometry_msgs::msg::Point& world_position) {
  // Extract per-phase polynomial parameters from Ruckig profile and transform
  // each phase from camera frame (RDF) to world frame (ENU).
  //
  // Each constant-acceleration phase is described by (p0, v0, a, duration)
  // in local time [0, duration]. The affine transform p(t) = p0 + v0*t + 0.5*a*t^2
  // transforms linearly: R*p(t) + offset = (R*p0+offset) + (R*v0)*t + 0.5*(R*a)*t^2
  //
  // This eliminates the need for the GCS to reconstruct the Ruckig trajectory.

  ground_system_msgs::msg::OpusTrajectory msg;
  msg.header.stamp = this->now();
  msg.drone_id = opus_drone_id_;

  auto profile_array = traj.get_profiles();
  assert(profile_array.size() == 1);
  auto profiles = profile_array[0];
  assert(profiles.size() == 3);

  // Helper lambda: extract a single phase and transform to world frame
  auto add_phase = [&](double duration,
                       const std::array<double, 3>& p0_cam,
                       const std::array<double, 3>& v0_cam,
                       const std::array<double, 3>& a_cam) {
    if (duration < 1e-6) return;

    Eigen::Vector3d p0(p0_cam[0], p0_cam[1], p0_cam[2]);
    Eigen::Vector3d v0(v0_cam[0], v0_cam[1], v0_cam[2]);
    Eigen::Vector3d a(a_cam[0], a_cam[1], a_cam[2]);

    // Transform each vector to world frame (ENU)
    Eigen::Vector3d p0_world = transform_camera_to_world(p0, body_to_world, true, world_position);
    Eigen::Vector3d v0_world = transform_camera_to_world(v0, body_to_world, false, world_position);
    Eigen::Vector3d a_world  = transform_camera_to_world(a,  body_to_world, false, world_position);

    ground_system_msgs::msg::OpusPhaseSegment phase;
    phase.p0 = {p0_world.x(), p0_world.y(), p0_world.z()};
    phase.v0 = {v0_world.x(), v0_world.y(), v0_world.z()};
    phase.a  = {a_world.x(),  a_world.y(),  a_world.z()};
    phase.duration = duration;
    msg.phases.push_back(phase);
  };

  // Brake sub-profile
  double brake_dur = profiles[0].brake.duration;
  if (brake_dur > 1e-6) {
    add_phase(brake_dur,
              {profiles[0].brake.p[0], profiles[1].brake.p[0], profiles[2].brake.p[0]},
              {profiles[0].brake.v[0], profiles[1].brake.v[0], profiles[2].brake.v[0]},
              {profiles[0].brake.a[0], profiles[1].brake.a[0], profiles[2].brake.a[0]});
  }

  // Accel sub-profile
  double accel_dur = profiles[0].accel.duration;
  if (accel_dur > 1e-6) {
    add_phase(accel_dur,
              {profiles[0].accel.p[0], profiles[1].accel.p[0], profiles[2].accel.p[0]},
              {profiles[0].accel.v[0], profiles[1].accel.v[0], profiles[2].accel.v[0]},
              {profiles[0].accel.a[0], profiles[1].accel.a[0], profiles[2].accel.a[0]});
  }

  // 7 main phases
  for (uint8_t i = 0; i < 7; i++) {
    double dt = profiles[0].t[i];
    if (dt < 1e-6) continue;
    add_phase(dt,
              {profiles[0].p[i], profiles[1].p[i], profiles[2].p[i]},
              {profiles[0].v[i], profiles[1].v[i], profiles[2].v[i]},
              {profiles[0].a[i], profiles[1].a[i], profiles[2].a[i]});
  }

  // Compute target position in world frame (last phase endpoint)
  // Use the Ruckig output directly for the final position
  std::array<double, 3> final_pos, final_vel, final_acc;
  traj.at_time(traj.get_duration(), final_pos, final_vel, final_acc);
  Eigen::Vector3d final_pos_cam(final_pos[0], final_pos[1], final_pos[2]);
  Eigen::Vector3d tgt_pos_world = transform_camera_to_world(final_pos_cam, body_to_world, true, world_position);
  msg.target_position = {tgt_pos_world.x(), tgt_pos_world.y(), tgt_pos_world.z()};

  msg.duration = traj.get_duration();
  msg.start_time = 0.0;  // GCS stamps this on accept

  ground_system_msgs::msg::OpusTrajectorySubmit submit;
  submit.drone_id = opus_drone_id_;
  submit.plan_sequence = opus_plan_sequence_;
  submit.trajectory = msg;
  opus_traj_submit_pub_->publish(submit);

  RCLCPP_INFO(this->get_logger(),
    "OPUS: Submitted trajectory (target=[%.2f,%.2f,%.2f], dur=%.2fs, %zu phases)",
    tgt_pos_world.x(), tgt_pos_world.y(), tgt_pos_world.z(),
    traj.get_duration(), msg.phases.size());
}

// ruckig_input_to_msg and ruckig_to_global_segments removed:
// Agent now submits per-phase polynomial parameters (OpusTrajectory) directly.
// GCS performs collision checking using opus_math.py.
