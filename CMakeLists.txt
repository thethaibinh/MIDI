cmake_minimum_required(VERSION 3.10)
set(PROJECT_NAME midi)
project(${PROJECT_NAME} CXX)

find_package(catkin_simple REQUIRED)
catkin_simple()

# Core settings
set(CMAKE_CXX_STANDARD 17)
cmake_policy(SET CMP0079 NEW)

set(CMAKE_BUILD_TYPE Release)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)

find_package(OpenMP REQUIRED)
# OpenMP settings
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Add submodules
add_subdirectory(modules/ruckig)

# planner sources
include_directories(modules/ruckig/include include ${catkin_INCLUDE_DIRS})
add_subdirectory(test)

# Create the main library
cs_add_library(${PROJECT_NAME}
    src/du_planner.cpp
    src/pyramid_planner.cpp
    src/planner_node.cpp
    src/planner_node_parameters.cpp
)

# Create the executable
cs_add_executable(main src/main.cpp)

# Link libraries
target_link_libraries(
    main
    ${PROJECT_NAME}
    ruckig
    ${OpenCV_LIBS}
    Boost::program_options
    yaml-cpp
    ${OpenMP_CXX_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()

cs_install()
cs_export()
