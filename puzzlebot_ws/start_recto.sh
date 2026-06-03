#!/bin/bash
# start_recto.sh — Solo micro_ros_agent + ir_recto (avance recto)
set -e

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)] $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] $1${NC}"; }

source /opt/ros/jazzy/setup.bash
source ~/uros_ws/install/setup.bash
cd ~/puzzlebot_docker/puzzlebot_ws
source install/setup.bash

PIDS=()
cleanup() {
    echo ""
    warn "Deteniendo..."
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null; done
    log "Listo."
    exit 0
}
trap cleanup SIGINT SIGTERM

log "Lanzando micro_ros_agent..."
ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 -b 115200 &
PIDS+=($!)
sleep 4

log "Lanzando ir_recto..."
ros2 run straight_line ir_recto &
PIDS+=($!)

warn "Robot avanzando recto. Ctrl+C para detener."
wait
