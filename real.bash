#!/bin/bash

# Launch the simulator, unless it is already running
if [ -z $(pgrep main) ]
then
  roslaunch midi real.launch &
  ROS_PID="$!"
  echo $ROS_PID
  sleep 1
else
  ROS_PID=""
fi

sleep 3
rosrun mavros mavsys rate --raw-controller 0
sleep 0.1
rosrun mavros mavsys rate --rc-channels 0
sleep 0.1
rosrun mavros mavsys rate --extra1 100
sleep 0.1
rosrun mavros mavsys rate --extra2 0
sleep 0.1
rosrun mavros mavsys rate --extra3 0
sleep 0.1
rosrun mavros mavsys rate --position 100
sleep 0.1
rosrun mavros mavsys rate --raw-sensors 100

while true
do
  sleep 3
done