#include "planner_node.hpp"
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PlannerNode>();
    // MultiThreadedExecutor so OPUS action/service callbacks can fire
    // concurrently with the high-frequency depth image planning loop.
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
