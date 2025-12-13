"""ROS 2 launch file for MIDI planner in simulation mode."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get package directories
    midi_share_dir = get_package_share_directory('midi')
    
    # Declare launch arguments
    quad_name_arg = DeclareLaunchArgument(
        'quad_name',
        default_value='kingfisher',
        description='Name of the quadrotor'
    )
    
    scenario_arg = DeclareLaunchArgument(
        'scenario',
        default_value='sim',
        description='Scenario type: sim, sitl, or real'
    )

    # MIDI Planner Node
    planner_node = Node(
        package='midi',
        executable='main',
        name='planner',
        namespace=LaunchConfiguration('quad_name'),
        parameters=[{
            'scenario': LaunchConfiguration('scenario'),
        }],
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        quad_name_arg,
        scenario_arg,
        planner_node,
    ])
