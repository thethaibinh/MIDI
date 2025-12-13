#include "planner_node.hpp"

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PlannerNode>();
    
    // Two-phase initialization: init() must be called after construction
    // because it uses shared_from_this() which requires the shared_ptr to exist
    if (!node->init()) {
        RCLCPP_ERROR(node->get_logger(), "Failed to initialize PlannerNode");
        rclcpp::shutdown();
        return 1;
    }
    
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
