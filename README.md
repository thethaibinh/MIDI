# Non-Conservative Efficient Collision Checking and Depth Noise-Awareness for Trajectory Planning

This repository contains the implementation code for the algorithm described in the manuscript "Non-Conservative Efficient Collision Checking and Depth Noise-Awareness for Trajectory Planning". Please don't hesitate to contact the corresponding author [Thai Binh Nguyen](mailto:thethaibinh@gmail.com) if you have any requests.

## Demonstration video
[![MIDI](https://img.youtube.com/vi/zv_CQVPB5Ls/0.jpg)](https://www.youtube.com/watch?v=zv_CQVPB5Ls)

## Update
Detailed instructions are coming soon!

### Prerequisite

We currently support Ubuntu 20.04 with ROS noetic and CUDA 12.7. Other setups are likely to work as well but not actively supported.

1. Before continuing, make sure to have g++ and gcc to version 9.3.0. You can check this by typing in a terminal `gcc --version` and `g++ --version`. Follow [this guide](https://linuxize.com/post/how-to-install-gcc-compiler-on-ubuntu-18-04/) if your compiler is not compatible.

2. In addition, make sure to have ROS installed. Follow [this guide](http://wiki.ros.org/noetic/Installation/Ubuntu) and install ROS Noetic if you don't already have it.

3. Install catkin tools, vcstool.
```
sudo apt install python3-catkin-tools python3-vcstool
```
4. Install [anaconda](https://www.anaconda.com/).
5. Install CUDA 12.7 following [this guide](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04).

### Installation
Start by creating a new catkin workspace.
```
cd     # or wherever you'd like to install this code
export ROS_VERSION=noetic
export CATKIN_WS=./ros_ws
mkdir -p $CATKIN_WS/src
cd $CATKIN_WS
echo "source $PWD/devel/setup.bash" >> ~/.bashrc
catkin init
catkin config --extend /opt/ros/$ROS_VERSION
catkin config --merge-devel
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-fdiagnostics-color

cd src
git clone https://github.com/thethaibinh/midi
vcs import < midi/midi.repos
cd midi
git submodule update --init --recursive
```
Check your GPU Compute Capability [here](https://developer.nvidia.com/cuda-gpus) and configure the CUDA build variables in the file `midi/CMakeLists.txt` corresponding to your hardware architecture.
```bash
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")
set(CMAKE_CUDA_ARCHITECTURES 75)
```
Run the `setup.bash` in the midi folder, it will ask for sudo permissions. Then build the packages.

```bash
./setup.bash
conda activate agileflight
source ~/.bashrc
catkin build
```

### Running the simulation

To run the the evaluation automatically, you can use the `./sim.bash N` script provided in this folder. It will automatically perform `N` rollouts and then create an `evaluation.yaml` file which summarizes the rollout statistics.
