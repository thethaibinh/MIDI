#include "planner_node.hpp"

using namespace quadrotor_common;

void PlannerNode::ardupilot_status_callback(const mavros_msgs::msg::State::SharedPtr msg) {
  flight_controller_status = *msg;
}

void PlannerNode::mav_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
  _latest_pose_stamp = rclcpp::Time(msg->header.stamp);

  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    _state.pose = msg->pose;
    // Use the most recent stamp — `_state.t` reflects "when state was last
    // observed" for the staleness check. Using min() would keep _state.t = 0
    // until both pose AND twist have been received at least once, making the
    // planner reject every frame during MAVROS warmup.
    _state.t = std::max(_latest_pose_stamp, _latest_twist_stamp).seconds();
  }

  // Publish initial position once for GCS 3D map (MAVROS mode)
  if (!initial_position_published_) {
    geometry_msgs::msg::PointStamped pt;
    pt.header.stamp = msg->header.stamp;
    pt.header.frame_id = _world_frame;
    pt.point.x = msg->pose.position.x;
    pt.point.y = msg->pose.position.y;
    pt.point.z = msg->pose.position.z;
    initial_position_pub_->publish(pt);
    initial_position_published_ = true;
    RCLCPP_INFO(this->get_logger(), "Published initial position (%.2f, %.2f, %.2f)",
      pt.point.x, pt.point.y, pt.point.z);
  }
}

void PlannerNode::mav_twist_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg) {
  _latest_twist_stamp = rclcpp::Time(msg->header.stamp);

  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    _state.velocity.linear = msg->twist.linear;
    _state.velocity.angular = msg->twist.angular;
    _state.t = std::max(_latest_pose_stamp, _latest_twist_stamp).seconds();
  }
}

void PlannerNode::mav_accel_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
  // Acceleration callback - only used for PYRAMID collision checking which is removed
  // For MIDI collision checking (CPU-only), we don't need acceleration
  (void)msg;  // Suppress unused parameter warning
  return;
}

void PlannerNode::odometry_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
  const std::lock_guard<std::mutex> lock(state_mutex_);

  _state.t = rclcpp::Time(msg->header.stamp).seconds();
  _state.pose = msg->pose.pose;
  _state.velocity = msg->twist.twist;

  // Publish initial position once for GCS 3D map
  if (!initial_position_published_) {
    geometry_msgs::msg::PointStamped pt;
    pt.header.stamp = msg->header.stamp;
    pt.header.frame_id = _world_frame;
    pt.point = msg->pose.pose.position;
    initial_position_pub_->publish(pt);
    initial_position_published_ = true;
    RCLCPP_INFO(this->get_logger(), "Published initial position (%.2f, %.2f, %.2f)",
      pt.point.x, pt.point.y, pt.point.z);
  }
  // Note: acceleration can be computed from velocity if needed, but not used in MIDI method

  // Debug: Log odometry with yaw periodically
  // Convert quaternion to yaw (rotation around Z axis)
  // double qw = msg->pose.pose.orientation.w;
  // double qx = msg->pose.pose.orientation.x;
  // double qy = msg->pose.pose.orientation.y;
  // double qz = msg->pose.pose.orientation.z;
  // double yaw_rad = atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
  // double yaw_deg = yaw_rad * 180.0 / M_PI;
  // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
  //   "Odometry: pos=(%.2f, %.2f, %.2f), yaw=%.1f deg (%.2f rad)",
  //   msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z,
  //   yaw_deg, yaw_rad);

  // Publish TF transform (world_frame -> vehicle_frame) from odometry
  // Only for OmniDrones mode - MAVROS handles TF publishing via tf.send: true
  if (_runtime_mode == RuntimeModes::OMNIDRONES) {
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = msg->header.stamp;
    tf_msg.header.frame_id = _world_frame;
    tf_msg.child_frame_id = _vehicle_frame;
    tf_msg.transform.translation.x = msg->pose.pose.position.x;
    tf_msg.transform.translation.y = msg->pose.pose.position.y;
    tf_msg.transform.translation.z = msg->pose.pose.position.z;
    tf_msg.transform.rotation = msg->pose.pose.orientation;
    tf_broadcaster_->sendTransform(tf_msg);
  }
}
