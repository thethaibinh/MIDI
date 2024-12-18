#!/bin/bash
if [[ ! -f "$(pwd)/setup.bash" ]]
then
  echo "please launch from the agile_flight folder!"
  exit
fi

parent_path=$(dirname "$PWD")

echo "export FLIGHTMARE_PATH=$parent_path/flightmare" >> ~/.bashrc
echo "export PLANNER_PATH=$parent_path/midi" >> ~/.bashrc
echo "export OMP_CANCELLATION=true" >> ~/.bashrc
echo "export OMP_NUM_THREADS=4" >> ~/.bashrc

sudo apt install python3-pip

echo "Creating an conda environment from the environment.yaml file. Make sure you have anaconda installed"
conda env create -f environment.yaml

echo "Source the anaconda environment. If errors, change to the right anaconda path."
source ~/anaconda3/etc/profile.d/conda.sh

echo "Activating the environment"
conda activate agileflight

echo "Compiling the agile flight environment and install the environment as python package"
cd $parent_path/flightmare/flightlib/build
cmake ..
make -j10
pip install .

echo "Install RPG baseline"
cd $parent_path/flightmare/flightpy/flightrl
pip install .
pip install flightgym rpg_baselines

echo "Making sure submodules are initialized and up-to-date"
git submodule update --init --recursive

echo "Using apt to install dependencies..."
echo "Will ask for sudo permissions:"
sudo apt update
sudo apt install -y --no-install-recommends build-essential cmake libzmqpp-dev libopencv-dev unzip python3-catkin-tools
sudo apt-get install ros-noetic-rqt ros-noetic-rqt-common-plugins ros-noetic-rqt-robot-plugins ros-noetic-mavros ros-noetic-grid-map-rviz-plugin

echo "Ignoring unused Flightmare folders!"
touch $parent_path/flightmare/flightros/CATKIN_IGNORE

echo "Downloading Flightmare Unity standalone..."
gdown https://drive.google.com/uc?id=1scWY4-PCGrZoO8HGgiUQ8arKGwiWt374 -O $parent_path/flightmare/flightrender/RPG_Flightmare_Data.zip

echo "Unziping Flightmare Unity Standalone... (this might take a while)"
unzip -o $parent_path/flightmare/flightrender/RPG_Flightmare_Data.zip -d $parent_path/flightmare/flightrender | awk 'BEGIN {ORS=" "} {if(NR%10==0)print "."}'

echo "Removing Flightmare Unity Standalone zip file"
rm $parent_path/flightmare/flightrender/RPG_Flightmare_Data.zip

chmod +x $parent_path/flightmare/flightrender/RPG_Flightmare.x86_64

echo "conda activate agileflight" >> ~/.bashrc

echo "Done!"
echo "Have a save flight!"