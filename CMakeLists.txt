cmake_minimum_required(VERSION 3.10)
set(PROJECT_NAME midi)
project(${PROJECT_NAME} CUDA CXX)

find_package(catkin_simple REQUIRED)
catkin_simple()

# Core settings
set(CMAKE_CXX_STANDARD 17)
cmake_policy(SET CMP0079 NEW)

set(CMAKE_BUILD_TYPE Release)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)

find_package(CUDA REQUIRED)
enable_language(CUDA)
# Add CUDA include directories
include_directories(
    ${CUDA_INCLUDE_DIRS}
)

find_package(OpenMP REQUIRED)
# OpenMP settings
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# CUDA settings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")
set(CMAKE_CUDA_ARCHITECTURES 75)
# For Orin NX (uncomment if needed):
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_87")
# set(CMAKE_CUDA_ARCHITECTURES 87)

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

# Create CUDA library
cuda_add_library(cuda_collision_checker
    src/cuda_collision_checker.cu
)

# Create the executable
cs_add_executable(main src/main.cpp)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    cuda_collision_checker
    ${CUDA_LIBRARIES}
)

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
