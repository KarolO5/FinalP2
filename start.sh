#!/bin/bash
source /opt/ros/jazzy/setup.bash
source ~/uros_ws/install/setup.bash
source ~/FinalP2/puzzlebot_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA   # evita uso excesivo de /dev/shm

ROBOT_IP=$(hostname -I | awk '{print $1}')

# Liberar puerto 8080 si quedó ocupado de sesion anterior
fuser -k 8080/tcp 2>/dev/null || true
sleep 0.5

# micro_ros_agent se corre manualmente desde otro SSH:
# ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 -b 115200

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

echo "=== Iniciando seguidor de linea ==="
ros2 run straight_line line_follower_cv

kill %1 %2 %3 %4 %5
