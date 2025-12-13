#!/bin/bash

# Pass number of rollouts as argument
if [ $1 ]
then
  N="$1"
else
  N=10
fi

parent_path=$(dirname "$PWD")
# Set Flightmare Path if it is not set
if [ -z $FLIGHTMARE_PATH ]
then
  export FLIGHTMARE_PATH=$parent_path/flightmare
fi

# Set Planner Path if it is not set
if [ -z $PLANNER_PATH ]
then
  export PLANNER_PATH=$parent_path/midi
fi

# Source ROS 2 environment
source /opt/ros/humble/setup.bash
source $parent_path/../install/setup.bash

# Export OpenMP settings
export OMP_CANCELLATION=true
export OMP_NUM_THREADS=4

# Configuration
QUAD_NAME="kingfisher"
RENDER=true
RVIZ=true

# Launch the simulator components, unless already running
if [ -z "$(pgrep visionsim_node)" ]
then
  echo "Starting simulation environment..."
  
  # Start Flightmare Unity renderer
  if [ "$RENDER" = true ]; then
    echo "Starting Flightmare renderer..."
    # Unity needs to run from its directory to find RPG_Flightmare_Data
    cd $FLIGHTMARE_PATH/flightrender
    ./RPG_Flightmare.x86_64 &
    FLIGHTMARE_PID="$!"
    cd - > /dev/null
    echo "Flightmare PID: $FLIGHTMARE_PID"
    echo "Waiting for Flightmare to initialize..."
    sleep 10
  fi
  
  # Start envsim visionsim_node
  echo "Starting envsim visionsim_node..."
  ros2 run envsim visionsim_node \
    --ros-args \
    -r __ns:=/$QUAD_NAME \
    -p agi_param_dir:=$parent_path/dodgedrone_simulation/dodgelib/params \
    -p ros_param_dir:=$parent_path/agile_flight/envsim/parameters \
    -p use_bem_propeller_model:=false \
    -p pilot_config:=simple_sim_pilot.yaml \
    -p real_time_factor:=1.0 \
    -p low_level_controller:=Simple \
    -p camera_config:=$parent_path/agile_flight/envsim/parameters/camera_config.yaml \
    -p render:=$RENDER &
  ENVSIM_PID="$!"
  echo "Envsim PID: $ENVSIM_PID"
  
  # Start RViz2 for visualization
  if [ "$RVIZ" = true ]; then
    echo "Starting RViz2..."
    ros2 run rviz2 rviz2 -d $parent_path/agile_flight/envsim/resources/rviz/envsim.rviz &
    RVIZ_PID="$!"
    echo "RViz PID: $RVIZ_PID"
  fi
  
  sleep 5
else
  ENVSIM_PID=""
  FLIGHTMARE_PID=""
  RVIZ_PID=""
  echo "Simulator already running, skipping launch..."
fi

# Start MIDI planner node
echo "Starting MIDI planner node..."
ros2 run midi main --ros-args -p scenario:=sim &
MIDI_PID="$!"
echo "MIDI PID: $MIDI_PID"
sleep 3

SUMMARY_FILE="evaluation.yaml"

# Perform N evaluation runs
for i in $(eval echo {1..$N})
do
  echo "=== Run $i of $N ==="
  
  # Publish simulator reset commands via ROS 2 (with timeout to avoid hanging)
  timeout 2 ros2 topic pub --once /$QUAD_NAME/dodgeros_pilot/off std_msgs/msg/Empty "{}" 2>/dev/null || true
  timeout 2 ros2 topic pub --once /$QUAD_NAME/dodgeros_pilot/reset_sim std_msgs/msg/Empty "{}" 2>/dev/null || true
  timeout 2 ros2 topic pub --once /$QUAD_NAME/dodgeros_pilot/enable std_msgs/msg/Bool "data: true" 2>/dev/null || true
  
  # Run benchmarking node if it exists
  # if [ -f "$parent_path/agile_flight/envtest/ros/benchmarking_node.py" ]; then
  #   cd $parent_path/agile_flight/envtest/ros/
  #   python3 benchmarking_node.py --policy=midi &
  #   PY_PID="$!"
  #   cd -
  # else
  #   PY_PID=""
  # fi
  
  sleep 0.5
  timeout 2 ros2 topic pub --once /$QUAD_NAME/start_navigation std_msgs/msg/Empty "{}" 2>/dev/null || true
  
  # Wait until the benchmarking script has finished
  # while ps -p $PY_PID > /dev/null 2>&1
  while true
  do
    sleep 1
  done
  
  # Merge evaluation results
  # if [ -f "$SUMMARY_FILE" ] && [ -f "$parent_path/agile_flight/envtest/ros/summary.yaml" ]; then
  #   cat "$SUMMARY_FILE" "$parent_path/agile_flight/envtest/ros/summary.yaml" > "tmp.yaml"
  #   mv "tmp.yaml" "$SUMMARY_FILE"
  # fi
done

echo "Evaluation complete!"

# Cleanup
cleanup() {
  echo "Shutting down..."
  [ -n "$MIDI_PID" ] && kill -SIGINT "$MIDI_PID" 2>/dev/null
  [ -n "$ENVSIM_PID" ] && kill -SIGINT "$ENVSIM_PID" 2>/dev/null
  [ -n "$RVIZ_PID" ] && kill -SIGINT "$RVIZ_PID" 2>/dev/null
  [ -n "$FLIGHTMARE_PID" ] && kill -SIGINT "$FLIGHTMARE_PID" 2>/dev/null
  # [ -n "$PY_PID" ] && kill -SIGINT "$PY_PID" 2>/dev/null
}

trap cleanup EXIT
cleanup
