import os
from glob import glob
from setuptools import setup

package_name = 'straight_line'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'odometry    = straight_line.odometry:main',
            'pd_controller = straight_line.pd_controller:main',
            'line_follower_cv = straight_line.line_follower_cv:main',
            'camera_node      = straight_line.camera_node:main',
            'semaforo         = straight_line.semaforo:main',
            'web_viz          = straight_line.web_viz:main',
            'sign_detector    = straight_line.sign_detector:main',
            'lidar_stop       = straight_line.lidar_stop:main',
        ],
    },
)