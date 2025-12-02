cmake_minimum_required(VERSION 3.10)
project(midi LANGUAGES C CXX)

# Core settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
cmake_policy(SET CMP0079 NEW)

# Find ament packages
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(mavros_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)

# Find MPI with C component first (required by PCL/VTK)
# This must be done before pcl_conversions to make MPI::MPI_C target available
find_package(MPI REQUIRED COMPONENTS C CXX)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(Eigen3 REQUIRED)

# Find other required packages
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(yaml-cpp REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Include directories
include_directories(
  include
  modules/ruckig/include
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${MPI_CXX_INCLUDE_DIRS}
)

# Add submodules
# Disable ruckig examples and tests
set(BUILD_EXAMPLES OFF CACHE BOOL "Build ruckig examples" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "Build ruckig tests" FORCE)
set(BUILD_BENCHMARK OFF CACHE BOOL "Build ruckig benchmark" FORCE)
add_subdirectory(modules/ruckig)

# Main MIDI library (CPU-only)
add_library(${PROJECT_NAME} SHARED
  src/du_planner.cpp
  src/planner_node.cpp
  src/planner_node_parameters.cpp
)

target_link_libraries(${PROJECT_NAME}
  ruckig
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  ${MPI_CXX_LIBRARIES}
  yaml-cpp
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  ament_index_cpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  mavros_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_eigen
  cv_bridge
  image_transport
  pcl_conversions
)

# Executable
add_executable(midi_planner_node src/main.cpp)

target_link_libraries(midi_planner_node
  ${PROJECT_NAME}
  Boost::program_options
)

ament_target_dependencies(midi_planner_node
  rclcpp
)

# Install targets
install(TARGETS
  ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS
  midi_planner_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  include/
  DESTINATION include
)

install(DIRECTORY
  configs/
  DESTINATION share/${PROJECT_NAME}/configs
)

# Export dependencies
ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp
  ament_index_cpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  mavros_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_eigen
  cv_bridge
  image_transport
  pcl_conversions
)

ament_package()
