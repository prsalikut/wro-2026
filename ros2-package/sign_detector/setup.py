import os
from glob import glob
from setuptools import setup

package_name = 'sign_detector'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='team',
    maintainer_email='prsalikuti@gmail.com',
    description='WRO FE red/green traffic-sign detector (camera + lidar fusion).',
    license='MIT',
    entry_points={
        'console_scripts': [
            'sign_detector = sign_detector.sign_detector_node:main',
            'steering_bridge = sign_detector.steering_node:main',
            'sign_steering = sign_detector.sign_steering_node:main',
        ],
    },
)
