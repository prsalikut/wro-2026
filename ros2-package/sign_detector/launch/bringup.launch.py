"""Bring up the whole stack: USB webcam + YDLIDAR X2 + sign detector + steering.
Used as the container's default command.

The steering pipeline (sign_steering -> steering_bridge -> Arduino) is included so
the WRO pass-rule behaviour comes up with the sensors: RED pillar -> keep RIGHT
(steer +deg), GREEN pillar -> keep LEFT (steer -deg).  See config/params.yaml.
"""
import glob
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def _resolve_lidar_port(default='/dev/ttyUSB0'):
    """Return the YDLIDAR X2's real /dev/ttyUSB* path, found via its stable
    /dev/serial/by-id symlink (Silicon Labs CP210x).  Both the lidar and the
    Arduino Nano enumerate as /dev/ttyUSB*, and a USB re-plug can renumber them,
    so we pin the lidar by its USB chip identity instead of a bare number.
    Falls back to `default` if the by-id link is missing."""
    for link in sorted(glob.glob('/dev/serial/by-id/*')):
        low = link.lower()
        if 'cp210' in low or 'silicon_labs' in low:
            try:
                return os.path.realpath(link)
            except OSError:
                return link
    return default


def generate_launch_description():
    cfg = os.path.join(
        get_package_share_directory('sign_detector'), 'config', 'params.yaml')
    lidar_port = _resolve_lidar_port()

    return LaunchDescription([
        Node(
            package='v4l2_camera', executable='v4l2_camera_node', name='camera',
            parameters=[{
                'video_device': '/dev/video0',
                'pixel_format': 'YUYV',
                'image_size': [640, 480],
                'white_balance_automatic': True,
                'brightness': 100,
                'saturation': 180,
                'backlight_compensation': 1,
                'power_line_frequency': 1,
                'contrast': 128,
                'sharpness': 128,
                'gain': 64,
                'hue': 128,
            }],
            output='screen',
        ),

        Node(
            package='ydlidar_ros2_driver', executable='ydlidar_ros2_driver_node',
            name='ydlidar', output='screen',
            parameters=[{
                'port': lidar_port,
                'frame_id': 'laser_frame',
                'baudrate': 115200,
                'lidar_type': 1,
                'device_type': 0,
                'sample_rate': 3,
                'abnormal_check_count': 4,
                'fixed_resolution': True,
                'reversion': False,
                'inverted': True,
                'auto_reconnect': True,
                'isSingleChannel': True,
                'intensity': False,
                'support_motor_dtr': True,
                'angle_max': 180.0,
                'angle_min': -180.0,
                'range_max': 8.0,
                'range_min': 0.1,
                'frequency': 7.0,
                'invalid_range_is_inf': False,
            }],
        ),

        Node(
            package='tf2_ros', executable='static_transform_publisher', name='base_to_laser',
            arguments=['0', '0', '0.1', '0', '0', '0', 'base_link', 'laser_frame'],
        ),

        Node(
            package='sign_detector', executable='sign_detector', name='sign_detector',
            parameters=[cfg],
            output='screen',
        ),

        Node(
            package='sign_detector', executable='sign_steering', name='sign_steering',
            parameters=[cfg],
            output='screen',
        ),

        Node(
            package='sign_detector', executable='steering_bridge', name='steering_bridge',
            parameters=[cfg],
            output='screen',
        ),
    ])
