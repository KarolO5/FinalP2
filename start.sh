#!/bin/bash
source /opt/ros/jazzy/setup.bash
source ~/uros_ws/install/setup.bash
source ~/FinalP2/puzzlebot_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ROBOT_IP=$(hostname -I | awk '{print $1}')

# micro_ros_agent se corre manualmente desde otro SSH:
# ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/hackerboard -b 115200
#
# udev rules en /etc/udev/rules.d/99-puzzlebot.rules:
#   HackerBoard serial: 0001                             -> /dev/hackerboard
#   RPLiDAR serial:     26fe7efa28ffec1186596d508ce70331 -> /dev/rplidar

echo "=== Iniciando RPLiDAR A1 ==="
ros2 run rplidar_ros rplidar_composition \
    --ros-args \
    -p serial_port:=/dev/ttyUSB1 \
    -p serial_baudrate:=115200 \
    -p frame_id:=laser \
    -p angle_compensate:=true &
sleep 2

echo "=== Iniciando odometria ==="
ros2 run straight_line odometry &
sleep 2

echo "=== Iniciando camara ==="
ros2 run straight_line camera_node &
sleep 2

echo "=== Iniciando semaforo ==="
ros2 run straight_line semaforo &
sleep 1

echo "=== Iniciando sign_detector ==="
ros2 run straight_line sign_detector &
sleep 1

echo "=== Iniciando web_viz  http://${ROBOT_IP}:8080 ==="
ros2 run straight_line web_viz &
sleep 1

echo "=== Iniciando lidar_stop ==="
ros2 run straight_line lidar_stop &
sleep 1

echo "=== Iniciando seguidor de linea ==="
ros2 run straight_line line_follower_cv

kill %1 %2 %3 %4 %5 %6 %7 %8
