from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        # --- Camara ---
        Node(package='straight_line', executable='camera_node',
             name='camera_node', output='screen'),

        # --- Vision ---
        Node(package='straight_line', executable='line_follower_cv',
             name='line_follower_cv', output='screen'),

        Node(package='straight_line', executable='semaforo',
             name='semaforo', output='screen'),

        Node(package='straight_line', executable='sign_detector',
             name='sign_detector', output='screen'),

        # --- LiDAR obstacle stop ---
        Node(package='straight_line', executable='obstacle_stop',
             name='obstacle_stop', output='screen'),

        # --- RPLiDAR A1 ---
        Node(package='rplidar_ros', executable='rplidar_composition',
             name='rplidar_node', output='screen',
             parameters=[{
                 'serial_port':      '/dev/ttyUSB0',
                 'serial_baudrate':  115200,
                 'frame_id':         'laser',
                 'angle_compensate': True,
                 'scan_mode':        'Standard',
             }]),

        # --- Dashboard web ---
        Node(package='straight_line', executable='web_viz',
             name='web_viz', output='screen'),
    ])
