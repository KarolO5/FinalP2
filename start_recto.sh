#!/bin/bash
# Linea recta con corrección PD sobre odometría

source /opt/ros/jazzy/setup.bash
source ~/uros_ws/install/setup.bash
cd ~/puzzlebot_docker/puzzlebot_ws
source install/setup.bash

PIDS=()
cleanup() {
    echo "Deteniendo nodos..."
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null; done
    exit 0
}
trap cleanup SIGINT SIGTERM

echo "[1/3] micro_ros_agent"
ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 -b 115200 &
PIDS+=($!); sleep 4

echo "[2/3] odometry"
ros2 run straight_line odometry &
PIDS+=($!); sleep 4

echo "[3/3] pd_controller"
ros2 run straight_line pd_controller &
PIDS+=($!)

echo "Listo. Ctrl+C para detener."
wait
