#!/bin/bash
if [[ ! -f "$(pwd)/setup.bash" ]]
then
  echo "please launch from the midi folder!"
  exit
fi

parent_path=$(dirname "$PWD")

echo "export FLIGHTMARE_PATH=$parent_path/flightmare" >> ~/.bashrc
echo "export PLANNER_PATH=$parent_path/midi" >> ~/.bashrc
echo "export OMP_CANCELLATION=true" >> ~/.bashrc
echo "export OMP_NUM_THREADS=4" >> ~/.bashrc

sudo apt install python3-pip

echo "Making sure submodules are initialized and up-to-date"
git submodule update --init --recursive

echo "Using apt to install dependencies..."
echo "Will ask for sudo permissions:"
sudo apt update
sudo apt install -y --no-install-recommends build-essential cmake libzmqpp-dev libopencv-dev unzip gdown libyaml-cpp-dev
sudo apt install -y ros-humble-cv-bridge ros-humble-image-transport ros-humble-pcl-ros ros-humble-pcl-conversions \
    ros-humble-tf2-ros ros-humble-tf2-eigen ros-humble-tf2-geometry-msgs ros-humble-mavros-msgs \
    ros-humble-rqt ros-humble-rqt-common-plugins ros-humble-rqt-robot-plugins

echo "Ignoring unused Flightmare folders!"
touch $parent_path/flightmare/flightros/CATKIN_IGNORE

FLIGHTMARE_ZIP="$parent_path/flightmare/flightrender/RPG_Flightmare_Data.zip"
if [[ -f "$FLIGHTMARE_ZIP" ]]; then
    echo "Flightmare Unity standalone zip already exists, skipping download..."
else
    echo "Downloading Flightmare Unity standalone..."
    gdown https://drive.google.com/uc?id=1scWY4-PCGrZoO8HGgiUQ8arKGwiWt374 -O "$FLIGHTMARE_ZIP"
fi

echo "Unzipping Flightmare Unity Standalone... (this might take a while)"
unzip -o "$FLIGHTMARE_ZIP" -d $parent_path/flightmare/flightrender | awk 'BEGIN {ORS=" "} {if(NR%10==0)print "."}'
echo ""

echo "Removing Flightmare Unity Standalone zip file"
rm "$FLIGHTMARE_ZIP"

chmod +x $parent_path/flightmare/flightrender/RPG_Flightmare.x86_64

echo "Done!"
echo "Have a safe flight!"
echo ""
echo "To build the workspace:"
echo "  cd /home/binh/Documents/repos/ros2_ws"
echo "  source /opt/ros/humble/setup.bash"
echo "  colcon build --symlink-install"
echo ""
echo "To run the simulation:"
echo "  cd $parent_path/midi"
echo "  source /opt/ros/humble/setup.bash"
echo "  source /home/binh/Documents/repos/ros2_ws/install/setup.bash"
echo "  ./sim.bash N  # where N is number of trees"