# MIDI Planner Quick Start Guide

## What Was Done

The MIDI ROS1 package has been successfully converted to ROS2 and integrated into your aircraft system:

✅ **Copied** MIDI from `github_clones/MIDI` to `aircraft/aircraft_ws/src/midi`
✅ **Converted** package.xml to ROS2 format (format 3)
✅ **Converted** CMakeLists.txt to ament_cmake
✅ **Created** replacement headers for missing dependencies (dodgeros, quadrotor_common, etc.)
✅ **Converted** all ROS1 API calls to ROS2 (NodeHandle→Node, callbacks, timers, etc.)
✅ **Updated** main.cpp, planner_node.cpp, planner_node.hpp for ROS2
✅ **Created** Ardupilot-specific configuration file
✅ **Created** ROS2 Python launch file
✅ **Integrated** into aircraft.yml.erb tmux configuration

## Quick Build & Test

### 1. Build MIDI

```bash
cd /home/binh/Documents/repos/swarm-autonomy-stack/aircraft/aircraft_ws

# Build MIDI package
colcon build --packages-select midi --cmake-args -DCMAKE_BUILD_TYPE=Release

# Source the workspace
source install/setup.bash
```

**Note:** If you get CUDA architecture errors, edit `src/midi/CMakeLists.txt` line 51-52 and set the correct `sm_XX` for your GPU.

### 2. Configure for Your Setup

Edit `src/midi/configs/ardupilot.yaml`:

```yaml
# Update depth camera topic
topics:
  depth: "/camera/depth/image_raw"  # Change to match your camera

# Update camera intrinsics (get from camera_info topic)
depth_camera:
  focal_length: 386.0  # fx or fy
  cx: 320.0            # optical center x
  cy: 240.0            # optical center y

# Set goal position (relative to takeoff, in meters)
goal_coordinate:
  north: 20.0   # Forward (Y in ENU)
  west: 0.0     # Right is negative (X in ENU)
  up: 3.0       # Altitude

# Adjust drone size
true_vehicle_radius: 0.25      # Your drone's actual radius
planning_vehicle_radius: 0.45  # Safety margin
```

### 3. Test Standalone

```bash
# Terminal 1: Start MAVROS (if not already running)
ros2 launch mavros apm.launch fcu_url:=udp://127.0.0.1:14550@14550

# Terminal 2: Start depth camera (example for RealSense)
ros2 launch realsense2_camera rs_launch.py depth_module.profile:=640x480x30

# Terminal 3: Launch MIDI
ros2 launch midi midi_planner.launch.py scenario:=ardupilot

# Terminal 4: Arm drone in GUIDED mode, then trigger planning
ros2 topic pub --once /start_navigation std_msgs/msg/Empty
```

### 4. Run with Aircraft Container

```bash
# Set environment variables
export USE_MIDI=true
export CAMERA=true
export AUTOPILOT=ardupilot

# Launch aircraft container
cd /home/binh/Documents/repos/swarm-autonomy-stack
./scripts/deploy_run.sh  # or sim_run.sh for simulation
```

MIDI will run in the "planning" tmux window.

## Key Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/depth/image_raw` | sensor_msgs/Image | Input: Depth image |
| `mavros/local_position/pose` | geometry_msgs/PoseStamped | Input: Position |
| `mavros/setpoint_raw/local` | mavros_msgs/PositionTarget | Output: Setpoints |
| `/start_navigation` | std_msgs/Empty | Trigger: Start planning |
| `/reset_planner` | std_msgs/Empty | Trigger: Reset |
| `/cloud_out` | sensor_msgs/PointCloud2 | Debug: Point cloud |

## Operational States

1. **OFF** - Idle, waiting for arm+GUIDED
2. **START** - Taking off to target altitude
3. **TRAJECTORY_CONTROL** - Active planning and navigation
4. **GO_TO_GOAL** - Final approach to goal
5. **LAND** - Landing sequence

## Troubleshooting

### Build fails with CUDA error
```bash
# Check your GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Edit CMakeLists.txt line 51-52:
# GTX 10-series: sm_75
# RTX 20-series: sm_75
# RTX 30-series: sm_86
# Jetson Orin: sm_87
```

### No depth image received
```bash
# Check camera is publishing
ros2 topic list | grep depth
ros2 topic hz /camera/depth/image_raw

# Update config with correct topic name
```

### Planner doesn't start
```bash
# Check MAVROS connection
ros2 topic echo --once mavros/state

# Ensure drone is armed and in GUIDED mode
# Send start command:
ros2 topic pub --once /start_navigation std_msgs/msg/Empty
```

### Planning is too slow
Edit `configs/ardupilot.yaml`:
```yaml
planning_cycle_time: 0.1  # Increase (slower planning rate)
sampled_trajectories_threshold: 5000  # Reduce (fewer samples)
checked_trajectories_threshold: 100   # Reduce (fewer checks)
```

### Collisions or too conservative
Adjust safety margins in config:
```yaml
planning_vehicle_radius: 0.35  # Decrease for tighter paths
minimum_clear_distance: 0.8    # Decrease for closer to obstacles
```

## Important Files

- **Main node**: `src/planner_node.cpp`
- **Config**: `configs/ardupilot.yaml`
- **Launch**: `launch/midi_planner.launch.py`
- **CMake**: `CMakeLists.txt` (adjust CUDA arch here)

## Next Steps

1. **Test in simulation** with Gazebo + depth camera plugin
2. **Tune parameters** for your specific drone and environment
3. **Integrate with mission** planner for waypoint-based goals
4. **Add safety checks** and emergency stop mechanisms
5. **Validate** depth camera calibration and range

## Support

- See `README_ROS2.md` for detailed documentation
- Original MIDI: https://github.com/uzh-rpg/midi
- Check aircraft system docs for MAVROS/Ardupilot setup

## Known Limitations

- **CUDA Required** - Must have NVIDIA GPU
- **Depth Range** - Limited by camera (typically <10m)
- **Computational** - Intensive, may need parameter tuning
- **Lighting** - Depth cameras need adequate lighting
- **Static Obstacles** - Designed for static environment planning

ros2 run midi midi_planner_node --ros-args --remap scenario:=omnidrones