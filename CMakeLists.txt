cmake_minimum_required(VERSION 3.10)
project(midi)

# Core settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_policy(SET CMP0079 NEW)
set(CMAKE_BUILD_TYPE Release)

# Check for CUDA (optional)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CUDA_AVAILABLE TRUE)
    message(STATUS "CUDA found - enabling GPU acceleration")
    message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")
    set(CMAKE_CUDA_ARCHITECTURES 75)
else()
    set(CUDA_AVAILABLE FALSE)
    message(STATUS "CUDA not found - building without GPU acceleration")
endif()

# Find ament and ROS2 packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Custom workspace packages
find_package(dodgeros REQUIRED)
find_package(dodgeros_msgs REQUIRED)
find_package(dodgelib REQUIRED)
find_package(quadrotor_msgs REQUIRED)
find_package(quadrotor_common REQUIRED)
find_package(position_controller REQUIRED)
find_package(mavros_msgs REQUIRED)

# System packages
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(PCL REQUIRED)

# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Add Ruckig submodule
add_subdirectory(modules/ruckig)

# Include directories
include_directories(
    include
    modules/ruckig/include
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

# Add CUDA include dirs if available
if(CUDA_AVAILABLE)
    include_directories(/usr/local/cuda/include)
endif()

# Collect sources
set(MIDI_SOURCES
    src/du_planner.cpp
    src/pyramid_planner.cpp
    src/planner_node.cpp
    src/planner_node_parameters.cpp
)

# CUDA library (optional)
if(CUDA_AVAILABLE)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -fPIC")
    add_library(cuda_collision_checker STATIC
        src/cuda_collision_checker.cu
    )
    set_target_properties(cuda_collision_checker PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_include_directories(cuda_collision_checker PUBLIC
        include
        /usr/local/cuda/include
    )
    # Need to add ROS2 includes to CUDA library
    ament_target_dependencies(cuda_collision_checker
        rclcpp
        geometry_msgs
    )
    set(CUDA_LINK_LIBS cuda_collision_checker CUDA::cudart)
    add_definitions(-DCUDA_AVAILABLE)
else()
    set(CUDA_LINK_LIBS "")
endif()

# Create the main library
add_library(${PROJECT_NAME} SHARED ${MIDI_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(${PROJECT_NAME}
    rclcpp
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    visualization_msgs
    cv_bridge
    image_transport
    pcl_conversions
    pcl_ros
    tf2
    tf2_ros
    tf2_geometry_msgs
    dodgeros
    dodgeros_msgs
    dodgelib
    quadrotor_msgs
    quadrotor_common
    position_controller
    mavros_msgs
)
target_link_libraries(${PROJECT_NAME}
    ${CUDA_LINK_LIBS}
    ${OpenCV_LIBS}
    yaml-cpp
    ruckig
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()

# Create the executable
add_executable(main src/main.cpp)
target_link_libraries(main
    ${PROJECT_NAME}
    ruckig
    ${OpenCV_LIBS}
    Boost::program_options
    yaml-cpp
)
ament_target_dependencies(main rclcpp)

# Install targets
set(INSTALL_TARGETS ${PROJECT_NAME} main)
if(CUDA_AVAILABLE)
    list(APPEND INSTALL_TARGETS cuda_collision_checker)
endif()

install(TARGETS ${INSTALL_TARGETS}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Install headers
install(DIRECTORY include/
    DESTINATION include
)

# Install launch files
install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(DIRECTORY configs/
    DESTINATION share/${PROJECT_NAME}/configs
)

# Export dependencies
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(
    rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs visualization_msgs
    cv_bridge image_transport pcl_conversions pcl_ros tf2 tf2_ros tf2_geometry_msgs
    dodgeros dodgeros_msgs dodgelib quadrotor_msgs quadrotor_common position_controller
)

ament_package()
