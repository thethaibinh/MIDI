#include "planner_node.hpp"

int main(int argc, char **argv) {
    ros::init(argc, argv, "planner");
    PlannerNode pn; //Construct class
    ros::spin();
    return 0;
}
