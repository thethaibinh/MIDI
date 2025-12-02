# MIDI Integration with OmniDrones

## Overview

MIDI has been updated to work with OmniDrones simulator in a CPU-only configuration. All CUDA dependencies have been removed, and the system now receives depth images and odometry from OmniDrones.

## Key Changes

### 1. Removed CUDA Dependencies
- **CMakeLists.txt**: Removed all CUDA compilation and linking
- **No GPU required**: MIDI now runs entirely on CPU using OpenMP for parallelization
- **Collision checking**: Uses MIDI method only (Pyramid method required CUDA)

### 2. Replaced FLIGHTMARE with OMNIDRONES
- **Runtime mode**: Changed from `FLIGHTMARE` to `OMNIDRONES`
- **Configuration**: New `configs/omnidrones.yaml` for OmniDrones-specific settings
- **Frame conventions**: Updated to handle ENU (East-North-Up) coordinate system

### 3. OmniDrones Integration

#### Depth Camera
The `spawn_n_drones.py` script now:
- Spawns a depth camera on each drone
- Uses `distance_to_camera` data type (depth in meters)
- Resolution: 640x480
- Max range: 10m
- Publishes to: `/Drone{i}/camera/depth/image_raw`

#### Odometry Publishing
Each drone publishes full state to `/Drone{i}/odometry`:
- Position (x, y, z in ENU)
- Orientation (quaternion)
- Linear velocity (world frame ENU)
- Angular velocity (body frame)

## Configuration

### MIDI Config (`configs/omnidrones.yaml`)

```yaml
runtime_mode: "omnidrones"
collision_checking_method: "midi"  # CPU-only method

topics:
  depth: "/camera/depth/image_raw"

# Goal in absolute ENU coordinates
goal_coordinate:
  north: 20.0   # Y-axis (North)
  west: 5.0     # X-axis (East) 
  up: 3.0       # Z-axis (Up)

# Camera parameters match OmniDrones depth camera
depth_camera:
  depth_scale: 1.0  # Already in meters
  omnidrones_fov: 90.0
```

### OmniDrones Depth Camera Config

The depth camera is configured in `spawn_n_drones.py`:

```python
depth_camera_cfg = PinholeCameraCfg(
    sensor_tick=0,
    resolution=(640, 480),
    data_types=["distance_to_camera"],  # Depth in meters
    usd_params=PinholeCameraCfg.UsdCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 10.0),  # 10m max range
    ),
)
```

Camera is mounted:
- **Position**: 10cm forward, 5cm up from drone center
- **Orientation**: Pointing forward along drone's X-axis

## Running the System

### 1. Start OmniDrones Simulator

```bash
cd github_clones/OmniDrones/examples
python spawn_n_drones.py n_drones=3
```

This will:
- Spawn 3 drones with depth cameras
- Publish odometry to `/Drone{1-3}/odometry`
- Publish depth images to `/Drone{1-3}/camera/depth/image_raw`
- Accept velocity commands on `/Drone{1-3}/mavros/setpoint_velocity/cmd_vel`

### 2. Start MIDI Planner

For Drone 1:
```bash
cd aircraft/aircraft_ws
source install/setup.bash
ros2 run midi midi_planner_node --ros-args \
  -r __ns:=/Drone1 \
  -p scenario:=omnidrones \
  -p use_sim_time:=false
```

Or use the aircraft container with:
```bash
export USE_MIDI=true
export AUTOPILOT=omnidrones
./scripts/sim_run.sh
```

### 3. Trigger Planning

```bash
# Start navigation
ros2 topic pub --once /Drone1/start_navigation std_msgs/msg/Empty

# Reset if needed
ros2 topic pub --once /Drone1/reset_planner std_msgs/msg/Empty
```

## Topic Structure

### Per Drone (N drones, indexed 1 to N)

**Inputs to MIDI:**
- `/DroneN/odometry` (nav_msgs/Odometry) - Full state from OmniDrones
- `/DroneN/camera/depth/image_raw` (sensor_msgs/Image) - Depth image

**Outputs from MIDI:**
- `/DroneN/mavros/setpoint_velocity/cmd_vel` (geometry_msgs/TwistStamped) - Velocity commands

**Control:**
- `/DroneN/start_navigation` (std_msgs/Empty) - Start planning
- `/DroneN/reset_planner` (std_msgs/Empty) - Reset planner

**Visualization:**
- `/DroneN/visualization` (visualization_msgs/Marker) - Goal and trajectory markers
- `/DroneN/cloud_out` (sensor_msgs/PointCloud2) - Depth point cloud

## Frame Conventions

### OmniDrones (ENU - East North Up)
- **X**: East
- **Y**: North  
- **Z**: Up
- **Origin**: World center

### MIDI Camera Frame (RDF - Right Down Forward)
- **X**: Right
- **Y**: Down
- **Z**: Forward
- Mounted on drone, pointing forward

### Body Frame (FLU - Forward Left Up)
- **X**: Forward
- **Y**: Left
- **Z**: Up
- Aligned with drone orientation

## Coordinate Transformations

MIDI handles frame transformations:
1. **Odometry (ENU)** → stored in world frame
2. **Goal (ENU)** → stored directly
3. **Depth camera (RDF)** → converted to body frame (FLU)
4. **Planning** → in camera frame
5. **Trajectory** → transformed back to world frame (ENU)

## Performance

**CPU-Only Operation:**
- Planning rate: ~10-15 Hz (vs 20 Hz with GPU)
- Sufficient for real-time obstacle avoidance
- Uses OpenMP for parallelization

**Recommended Settings:**
```yaml
planning_cycle_time: 0.05  # 20 Hz planning attempts
sampled_trajectories_threshold: 5000  # Reduce for faster planning
checked_trajectories_threshold: 100
```

## Troubleshooting

### No depth images received
```bash
# Check OmniDrones is publishing
ros2 topic list | grep depth
ros2 topic hz /Drone1/camera/depth/image_raw

# View depth image
ros2 run rqt_image_view rqt_image_view /Drone1/camera/depth/image_raw
```

### No odometry
```bash
# Check odometry publishing
ros2 topic echo /Drone1/odometry --once
```

### Slow planning
- Reduce `sampled_trajectories_threshold` in config
- Reduce `checked_trajectories_threshold`
- Increase `planning_cycle_time` (lower rate)

### Drone not moving
- Ensure planning state is `TRAJECTORY_CONTROL`
- Check logs for collision probability warnings
- Verify depth camera has valid data
- Increase safety margins if too conservative

## Architecture Diagram

```
┌─────────────────┐
│  OmniDrones     │
│  Simulator      │
└────────┬────────┘
         │
         ├─► Depth Images (/DroneN/camera/depth/image_raw)
         │   (640x480, distance_to_camera in meters)
         │
         └─► Odometry (/DroneN/odometry)
             (Position, velocity, orientation in ENU)
             
         ┌──────────────────┐
         │  MIDI Planner    │
         │  (CPU-only)      │
         └────────┬─────────┘
                  │
                  └─► Velocity Commands
                      (/DroneN/mavros/setpoint_velocity/cmd_vel)
                      
         ┌──────────────────┐
         │  OmniDrones      │
         │  Controller      │
         └──────────────────┘
```

## Differences from Original MIDI

1. **No CUDA**: All collision checking on CPU
2. **No MAVROS**: Direct odometry instead of MAVROS topics  
3. **Simplified control**: Velocity commands instead of attitude/thrust
4. **OmniDrones integration**: Custom depth camera and state publishers
5. **Removed launch files**: Direct node execution

## Future Improvements

- [ ] Add dynamic obstacle detection
- [ ] Implement trajectory tracking controller
- [ ] Add multi-drone coordination
- [ ] Optimize CPU performance
- [ ] Add re-planning triggers
- [ ] Integrate with mission planner

## References

- Original MIDI: https://github.com/uzh-rpg/midi
- OmniDrones: https://github.com/btx0424/OmniDrones

