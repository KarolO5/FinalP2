#!/bin/bash
source /opt/ros/jazzy/setup.bash
source ~/uros_ws/install/setup.bash
source ~/FinalP2/puzzlebot_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ROBOT_IP=$(hostname -I | awk '{print $1}')

echo "=== Iniciando micro_ros_agent ==="
ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 -b 115200 &
sleep 3

echo "=== Iniciando odometria ==="
ros2 run straight_line odometry &
sleep 2

echo "=== Iniciando camara ==="
ros2 run straight_line camera_node &
sleep 2

echo "=== Iniciando semaforo ==="
ros2 run straight_line semaforo &
sleep 1

echo "=== Iniciando web_viz  http://${ROBOT_IP}:8080 ==="
ros2 run straight_line web_viz &
sleep 1

echo "=== Iniciando seguidor de linea ==="
ros2 run straight_line line_follower_cv

kill %1 %2 %3 %4 %5
