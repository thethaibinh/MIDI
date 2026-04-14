#include "planner_node.hpp"

using namespace quadrotor_common;

void PlannerNode::visualise(const sensor_msgs::msg::Image::SharedPtr depth_msg) {
  if (!rclcpp::ok()) {
    return;
  }
  
  // Debug: log that we received a depth image
  // static int depth_count = 0;
  // if (++depth_count % 100 == 1) {
  //   RCLCPP_INFO(this->get_logger(), "Received depth image #%d (%dx%d, encoding: %s)", 
  //               depth_count, depth_msg->width, depth_msg->height, depth_msg->encoding.c_str());
  // }
  
  // convert depth image to point cloud
  pointcloud_type* cloud = create_point_cloud(depth_msg);
  sensor_msgs::msg::PointCloud2 cloudMessage;
  pcl::toROSMsg(*cloud, cloudMessage);
  // Use node's current time for consistent TF lookups
  cloudMessage.header.stamp = this->now();
  cloudMessage.header.frame_id = _vehicle_frame;
  point_cloud_pub->publish(cloudMessage);
  delete cloud;  // Free memory

  // Helper lambda to create an arrow marker
  auto create_arrow_marker = [&](int id, const std::string& ns, 
                                  const geometry_msgs::msg::Point& start,
                                  const geometry_msgs::msg::Point& end,
                                  float r, float g, float b, float a,
                                  const std::string& frame_id) {
    visualization_msgs::msg::Marker arrow;
    arrow.header.frame_id = frame_id;
    arrow.header.stamp = this->now();
    arrow.ns = ns;
    arrow.id = id;
    arrow.type = visualization_msgs::msg::Marker::ARROW;
    arrow.action = visualization_msgs::msg::Marker::ADD;
    arrow.points.push_back(start);
    arrow.points.push_back(end);
    arrow.scale.x = 0.05;  // shaft diameter
    arrow.scale.y = 0.1;   // head diameter
    arrow.scale.z = 0.1;   // head length
    arrow.color.r = r;
    arrow.color.g = g;
    arrow.color.b = b;
    arrow.color.a = a;
    arrow.lifetime = rclcpp::Duration::from_seconds(0.5);
    return arrow;
  };
  
  // 1. VELOCITY FEEDBACK (CYAN) - in world frame, origin at drone position
  // This shows the raw velocity from OmniDrones odometry
  // {
  //   geometry_msgs::msg::Point start, end;
  //   start.x = _state.pose.position.x;
  //   start.y = _state.pose.position.y;
  //   start.z = _state.pose.position.z;
    
  //   // Scale velocity for visibility (1 m/s = 1 meter arrow)
  //   double vel_scale = 1.0;
  //   end.x = start.x + _state.velocity.linear.x * vel_scale;
  //   end.y = start.y + _state.velocity.linear.y * vel_scale;
  //   end.z = start.z + _state.velocity.linear.z * vel_scale;
    
  //   auto vel_arrow = create_arrow_marker(0, "velocity_world", start, end, 0.0, 1.0, 1.0, 1.0, _world_frame);
  //   debug_markers.markers.push_back(vel_arrow);
    
  //   Log velocity periodically
  //   double vel_mag = std::sqrt(_state.velocity.linear.x * _state.velocity.linear.x +
  //                              _state.velocity.linear.y * _state.velocity.linear.y +
  //                              _state.velocity.linear.z * _state.velocity.linear.z);
  //   RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
  //     "VELOCITY (world): [%.2f, %.2f, %.2f] mag=%.2f m/s",
  //     _state.velocity.linear.x, _state.velocity.linear.y, _state.velocity.linear.z, vel_mag);
  // }
  
  // 2. VELOCITY IN BODY FRAME (MAGENTA) - transformed to body frame, shown from origin in base_link
  // {
  //   geometry_msgs::msg::Vector3 velocity_world_frame, velocity_body_frame;
  //   velocity_world_frame = _state.velocity.linear;
  //   tf2::doTransform(velocity_world_frame, velocity_body_frame, world_to_body);
    
  //   geometry_msgs::msg::Point start, end;
  //   start.x = start.y = start.z = 0.0;
    
  //   double vel_scale = 1.0;
  //   end.x = velocity_body_frame.x * vel_scale;
  //   end.y = velocity_body_frame.y * vel_scale;
  //   end.z = velocity_body_frame.z * vel_scale;
    
  //   auto vel_body_arrow = create_arrow_marker(1, "velocity_body", start, end, 1.0, 0.0, 1.0, 1.0, _vehicle_frame);
  //   debug_markers.markers.push_back(vel_body_arrow);
    
  //   RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
  //     "VELOCITY (body FLU): [%.2f, %.2f, %.2f]",
  //     velocity_body_frame.x, velocity_body_frame.y, velocity_body_frame.z);
  // }
  
  // 3. GOAL VECTOR (GREEN) - direction from drone to goal in world frame
  // if (_goal_set) {
  //   geometry_msgs::msg::Point start, end;
  //   start.x = _state.pose.position.x;
  //   start.y = _state.pose.position.y;
  //   start.z = _state.pose.position.z;
    
  //   // Normalize and scale for visibility
  //   double dx = _goal_in_world_frame.x - start.x;
  //   double dy = _goal_in_world_frame.y - start.y;
  //   double dz = _goal_in_world_frame.z - start.z;
  //   double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
  //   double arrow_len = std::min(dist, 3.0);  // Cap at 3m for visibility
    
  //   if (dist > 0.1) {
  //     end.x = start.x + (dx / dist) * arrow_len;
  //     end.y = start.y + (dy / dist) * arrow_len;
  //     end.z = start.z + (dz / dist) * arrow_len;
      
  //     auto goal_arrow = create_arrow_marker(2, "goal_vector", start, end, 0.0, 1.0, 0.0, 1.0, _world_frame);
  //     debug_markers.markers.push_back(goal_arrow);
      
  //     RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
  //       "GOAL VECTOR (world): [%.2f, %.2f, %.2f] dist=%.2f",
  //       dx, dy, dz, dist);
  //   }
  // }
  // RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
  //   "Point cloud: %zu total points, %zu valid points, frame_id=%s",
  //   cloud->points.size(), valid_points, cloud->header.frame_id.c_str());

  
  // 4. GOAL IN BODY FRAME (YELLOW) - goal direction in body frame (FLU)
  // if (_goal_set) {
  //   geometry_msgs::msg::PointStamped goal_world, goal_body;
  //   goal_world.header.frame_id = _world_frame;
  //   goal_world.point = _goal_in_world_frame;
    
  //   try {
  //     tf2::doTransform(goal_world, goal_body, world_to_body);
      
  //     geometry_msgs::msg::Point start, end;
  //     start.x = start.y = start.z = 0.0;
      
  //     double dist = std::sqrt(goal_body.point.x * goal_body.point.x +
  //                             goal_body.point.y * goal_body.point.y +
  //                             goal_body.point.z * goal_body.point.z);
  //     double arrow_len = std::min(dist, 3.0);
      
  //     if (dist > 0.1) {
  //       end.x = (goal_body.point.x / dist) * arrow_len;
  //       end.y = (goal_body.point.y / dist) * arrow_len;
  //       end.z = (goal_body.point.z / dist) * arrow_len;
        
  //       auto goal_body_arrow = create_arrow_marker(3, "goal_body", start, end, 1.0, 1.0, 0.0, 1.0, _vehicle_frame);
  //       debug_markers.markers.push_back(goal_body_arrow);
        
  //       RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
  //         "GOAL (body FLU): [%.2f, %.2f, %.2f]",
  //         goal_body.point.x, goal_body.point.y, goal_body.point.z);
  //     }
  //   } catch (tf2::TransformException& ex) {
  //     // Ignore transform errors
  //   }
  // }
  
  // 5. EXPLORATION VECTOR / GOAL IN CAMERA FRAME (RED) - this is what the planner uses
  // if (_goal_set) {
  //   geometry_msgs::msg::PointStamped goal_world, goal_body;
  //   goal_world.header.frame_id = _world_frame;
  //   goal_world.point = _goal_in_world_frame;
    
  //   try {
  //     tf2::doTransform(goal_world, goal_body, world_to_body);
      
  //     // Transform from body (FLU) to camera (RDF)
  //     geometry_msgs::msg::Point goal_camera;
  //     frame_transform::transform_body_to_camera(goal_body.point, goal_camera);
      
  //     // The exploration vector in camera frame (RDF: X=Right, Y=Down, Z=Forward)
  //     // Visualize in vehicle frame by transforming back for display
  //     // Camera RDF to Body FLU: x_body = z_cam, y_body = -x_cam, z_body = -y_cam
  //     geometry_msgs::msg::Point start, end;
  //     start.x = start.y = start.z = 0.0;
      
  //     double dist = std::sqrt(goal_camera.x * goal_camera.x +
  //                             goal_camera.y * goal_camera.y +
  //                             goal_camera.z * goal_camera.z);
  //     double arrow_len = std::min(dist, 3.0);
      
  //     // Compute exploration cost values for debugging
  //     Eigen::Vector3d goal_vector_camera_frame(goal_camera.x, goal_camera.y, goal_camera.z);
  //     Eigen::Vector3d exploration_unit = goal_vector_camera_frame.normalized();
      
  //     // Simulate a sample endpoint (e.g., 1m forward in camera frame) to show cost calculation
  //     Eigen::Vector3d sample_endpoint(0.0, 0.0, 1.0);  // 1m forward in camera RDF
  //     double direction_cost = -exploration_unit.dot(sample_endpoint.normalized());
  //     double distance_cost = -sample_endpoint.dot(exploration_unit);
      
  //     if (dist > 0.1) {
  //       // Display the camera-frame vector transformed back to body frame for visualization
  //       // Camera RDF -> Body FLU: Forward=Z_cam, Left=-X_cam, Up=-Y_cam
  //       end.x = (goal_camera.z / dist) * arrow_len;  // Forward (body X) = Camera Z
  //       end.y = (-goal_camera.x / dist) * arrow_len; // Left (body Y) = -Camera X
  //       end.z = (-goal_camera.y / dist) * arrow_len; // Up (body Z) = -Camera Y
        
  //       auto explore_arrow = create_arrow_marker(4, "exploration_camera", start, end, 1.0, 0.0, 0.0, 1.0, _vehicle_frame);
  //       debug_markers.markers.push_back(explore_arrow);
        
  //       // Get traveling cost type as string
  //       std::string cost_type_str = (_traveling_cost == TravelingCost::DIRECTION) ? "DIRECTION" : "DISTANCE";
  //       double active_cost = (_traveling_cost == TravelingCost::DIRECTION) ? direction_cost : distance_cost;
        
  //       RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
  //         "EXPLORATION (camera RDF): [%.2f, %.2f, %.2f] | Cost type: %s | "
  //         "Direction cost: %.3f | Distance cost: %.3f | Active cost: %.3f",
  //         goal_camera.x, goal_camera.y, goal_camera.z,
  //         cost_type_str.c_str(), direction_cost, distance_cost, active_cost);
  //     }
  //   } catch (tf2::TransformException& ex) {
  //     // Ignore transform errors
  //   }
  // }
  
  // 6. DRONE FORWARD DIRECTION (WHITE) - shows drone heading
  // {
  //   geometry_msgs::msg::Point start, end;
  //   start.x = start.y = start.z = 0.0;
  //   end.x = 1.0;  // 1m forward in body frame
  //   end.y = 0.0;
  //   end.z = 0.0;
    
  //   auto forward_arrow = create_arrow_marker(5, "drone_forward", start, end, 1.0, 1.0, 1.0, 0.8, _vehicle_frame);
  //   debug_markers.markers.push_back(forward_arrow);
  // }
  
  // // Publish all debug markers
  // debug_vectors_pub_->publish(debug_markers);
  
  // ============================================================================
  // Original visualization code
  // ============================================================================
  visualization_msgs::msg::Marker pyramids_bases, pyramids_edges, polynomial_trajectory, goal_marker;
  pyramids_bases.header.frame_id = pyramids_edges.header.frame_id =
    polynomial_trajectory.header.frame_id = goal_marker.header.frame_id = _world_frame;
  pyramids_bases.header.stamp = pyramids_edges.header.stamp =
    polynomial_trajectory.header.stamp = goal_marker.header.stamp = this->now();
  pyramids_bases.ns = pyramids_edges.ns = polynomial_trajectory.ns = goal_marker.ns =
    "visualization";
  pyramids_bases.action = pyramids_edges.action = polynomial_trajectory.action = goal_marker.action =
    visualization_msgs::msg::Marker::ADD;
  goal_marker.action = visualization_msgs::msg::Marker::MODIFY;
  pyramids_bases.pose.orientation.w = pyramids_edges.pose.orientation.w =
    polynomial_trajectory.pose.orientation.w = goal_marker.pose.orientation.w = 1.0;
  pyramids_bases.id = 1;
  pyramids_edges.id = 2;
  polynomial_trajectory.id = 3;
  goal_marker.id = 4;
  polynomial_trajectory.type = visualization_msgs::msg::Marker::LINE_STRIP;
  goal_marker.type = visualization_msgs::msg::Marker::CUBE;
  // LINE_STRIP markers use only the x component of scale, for the line width
  polynomial_trajectory.scale.x = 0.05;
  goal_marker.scale.x = 0.2;
  goal_marker.scale.y = 0.2;
  goal_marker.scale.z = 0.2;
  // Trajectory is blue, goal is green
  polynomial_trajectory.color.b = 1.0;
  polynomial_trajectory.color.a = 1.0;
  goal_marker.color.g = 1.0;
  goal_marker.color.a = 1.0;

  // Publish goal marker
  if (_goal_set) {
    goal_marker.pose.position.x = _goal_in_world_frame.x;
    goal_marker.pose.position.y = _goal_in_world_frame.y;
    goal_marker.pose.position.z = _goal_in_world_frame.z;
    goal_marker.pose.orientation.w = 1.0;
    visual_pub->publish(goal_marker);
  }

  // Publish polynomial trajectory
  if (!had_reference_trajectory) {
    return;
  }
  
  double trajectory_duration = reference_trajectory_.get_duration();
  if (trajectory_duration < 0.01) {
    polynomial_trajectory.points.clear();
    visual_pub->publish(polynomial_trajectory);
    return;
  }
  geometry_msgs::msg::Point p;
  for (int i = 0; i <= 100; i++) {
    geometry_msgs::msg::Point position =
      reference_trajectory_.get_position_in_world_frame(trajectory_duration * i / 100);
    p.x = position.x;
    p.y = position.y;
    p.z = position.z;
    polynomial_trajectory.points.push_back(p);
  }

    // Change color to red when in GO_TO_GOAL state
    if (_planner_state == PlanningStates::GO_TO_GOAL) {
      polynomial_trajectory.color.b = 0.0;
      polynomial_trajectory.color.r = 1.0;
    }
    if (_planner_state == PlanningStates::TAKING_OFF) {
      polynomial_trajectory.points.clear();
    }
    visual_pub->publish(polynomial_trajectory);
}
