#!/bin/bash
# =============================================================================
# start.sh  —  Lanza todos los nodos del PuzzleBot en una sola terminal
# =============================================================================
# Uso: ./start.sh
# Para detener todo: Ctrl+C
# =============================================================================

WS_DIR="$(cd "$(dirname "$0")/puzzlebot_ws" && pwd)"
SETUP="$WS_DIR/install/setup.bash"

# ── Verificar workspace compilado ─────────────────────────────────────────────
if [ ! -f "$SETUP" ]; then
    echo "[ERROR] No se encontró $SETUP"
    echo "        Compila primero con:  cd puzzlebot_ws && colcon build"
    exit 1
fi

source "$SETUP"

# ── Obtener IP del robot para mostrarla al usuario ────────────────────────────
ROBOT_IP=$(hostname -I | awk '{print $1}')
echo ""
echo "============================================================"
echo "  PuzzleBot — iniciando nodos"
echo "  Dashboard: http://${ROBOT_IP}:8080"
echo "============================================================"
echo ""

# ── Función: matar todos los nodos al salir ───────────────────────────────────
PIDS=()
cleanup() {
    echo ""
    echo "[INFO] Deteniendo nodos..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    wait 2>/dev/null
    echo "[INFO] Todo detenido."
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Lanzar nodos ──────────────────────────────────────────────────────────────
echo "[1/4] camera_node"
ros2 run straight_line camera_node &
PIDS+=($!)
sleep 1

echo "[2/4] semaforo"
ros2 run straight_line semaforo &
PIDS+=($!)
sleep 0.5

echo "[3/4] line_follower_cv"
ros2 run straight_line line_follower_cv &
PIDS+=($!)
sleep 0.5

echo "[4/4] web_viz  →  http://${ROBOT_IP}:8080"
ros2 run straight_line web_viz &
PIDS+=($!)

echo ""
echo "[OK] Todos los nodos activos. Ctrl+C para detener."
echo ""

# Esperar a que algún nodo muera y limpiar todo
wait -n "${PIDS[@]}" 2>/dev/null
echo "[WARN] Un nodo terminó inesperadamente — deteniendo todo."
cleanup
