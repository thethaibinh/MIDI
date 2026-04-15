# Standalone CMake for CI testing of the header-only math library.
# This avoids pulling in the full ROS2/OpenCV/PCL dependency chain
# required by the main midi CMakeLists.txt.
cmake_minimum_required(VERSION 3.10)
project(midi_math_tests LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ament_cmake is needed for ament_target_dependencies()
find_package(ament_cmake QUIET)
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)
find_package(geometry_msgs REQUIRED)

set(MIDI_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
include_directories(${MIDI_ROOT}/include ${EIGEN3_INCLUDE_DIRS})

enable_testing()

# Polynomial root-finding (polynomial2/3/4)
add_executable(test_polynomials ${MIDI_ROOT}/test/test_polynomials.cpp)
target_link_libraries(test_polynomials GTest::gtest GTest::gtest_main)
add_test(NAME test_polynomials COMMAND test_polynomials)

# Segment distance functions (segment2/3)
add_executable(test_segments ${MIDI_ROOT}/test/test_segments.cpp)
target_link_libraries(test_segments GTest::gtest GTest::gtest_main)
add_test(NAME test_segments COMMAND test_segments)

# Frame transforms (FLU/RDF/RFU, Eigen/ROS conversions, quaternion math)
add_executable(test_frame_transforms ${MIDI_ROOT}/test/test_frame_transforms.cpp)
target_link_libraries(test_frame_transforms GTest::gtest GTest::gtest_main)
ament_target_dependencies(test_frame_transforms geometry_msgs)
add_test(NAME test_frame_transforms COMMAND test_frame_transforms)

# Trajectory cost (ExplorationCost direction/distance)
add_executable(test_trajectory_cost ${MIDI_ROOT}/test/test_trajectory_cost.cpp)
target_link_libraries(test_trajectory_cost GTest::gtest GTest::gtest_main)
add_test(NAME test_trajectory_cost COMMAND test_trajectory_cost)

# OPUS planner state machine transitions
add_executable(test_opus_state_machine ${MIDI_ROOT}/test/test_opus_state_machine.cpp)
target_link_libraries(test_opus_state_machine GTest::gtest GTest::gtest_main)
add_test(NAME test_opus_state_machine COMMAND test_opus_state_machine)

# Planner state machine transitions (non-OPUS)
add_executable(test_planner_state_machine ${MIDI_ROOT}/test/test_planner_state_machine.cpp)
target_link_libraries(test_planner_state_machine GTest::gtest GTest::gtest_main)
add_test(NAME test_planner_state_machine COMMAND test_planner_state_machine)
