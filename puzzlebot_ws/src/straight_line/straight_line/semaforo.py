#!/usr/bin/env python3
# =============================================================================
# semaforo.py
# =============================================================================
# Detecta el color del semáforo (rojo, amarillo, verde) en la esquina
# superior derecha del frame usando OpenCV (HSV + máscaras de color).
#
# ROI — esquina superior derecha:
#   - Vertical  : 0 .. ROI_H_FRAC  del alto total
#   - Horizontal: (1 - ROI_W_FRAC) .. 1  del ancho total
#
# Lógica de estado:
#   rojo    → robot se detiene; permanece detenido hasta recibir verde
#   amarillo → velocidad a la mitad; sale cuando deja de ver amarillo o ve verde
#   verde   → velocidad normal
#   ninguno → no cambia el estado anterior (sin semáforo visible)
#
# TÓPICOS
# ────────
#   Sub : /image/raw          [sensor_msgs/Image]
#   Pub : /semaforo/estado    [std_msgs/String]      "rojo"|"amarillo"|"verde"|"ninguno"
#   Pub : /semaforo/debug_img [sensor_msgs/Image]    frame con ROI anotado y máscaras
# =============================================================================

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS DE ROI  (esquina superior derecha)
# ─────────────────────────────────────────────────────────────────────────────
ROI_H_FRAC = 0.35   # fracción del alto desde arriba   (0–35 %)
ROI_W_FRAC = 0.35   # fracción del ancho desde la derecha (65–100 %)

# ─────────────────────────────────────────────────────────────────────────────
# RANGOS HSV
# ─────────────────────────────────────────────────────────────────────────────
# Rojo cruza el 0/180 en HSV → dos rangos
RED_LO1 = np.array([  0, 120,  80])
RED_HI1 = np.array([ 10, 255, 255])
RED_LO2 = np.array([165, 120,  80])
RED_HI2 = np.array([180, 255, 255])

YELLOW_LO = np.array([ 18, 100,  80])
YELLOW_HI = np.array([ 35, 255, 255])

GREEN_LO  = np.array([ 40,  80,  60])
GREEN_HI  = np.array([ 90, 255, 255])

MIN_PIXELS = 200   # píxeles mínimos para detección válida


# ─────────────────────────────────────────────────────────────────────────────
# NODO
# ─────────────────────────────────────────────────────────────────────────────

class SemaforoNode(Node):

    def __init__(self):
        super().__init__('semaforo')

        qos_be = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._bridge = CvBridge()

        # Estado persistente: el semáforo no cambia hasta que hay una señal clara
        self._estado_actual = 'ninguno'

        self._pub_estado = self.create_publisher(String, '/semaforo/estado',    10)
        self._pub_debug  = self.create_publisher(Image,  '/semaforo/debug_img', 10)

        self.create_subscription(Image, '/image/raw', self._image_cb, qos_be)

        self.get_logger().info(
            f'SemaforoNode listo | ROI esquina sup-der '
            f'{int(ROI_H_FRAC*100)}% alto x {int(ROI_W_FRAC*100)}% ancho'
        )

    # ── Callback ──────────────────────────────────────────────────────────

    def _image_cb(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        estado, debug_frame = self._detect(frame)

        state_msg      = String()
        state_msg.data = estado
        self._pub_estado.publish(state_msg)

        dbg_msg        = self._bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
        dbg_msg.header = msg.header
        self._pub_debug.publish(dbg_msg)

    # ── Detección ─────────────────────────────────────────────────────────

    def _detect(self, frame: np.ndarray):
        h, w = frame.shape[:2]

        # Coordenadas ROI esquina superior derecha
        roi_y0 = 0
        roi_y1 = int(h * ROI_H_FRAC)
        roi_x0 = int(w * (1.0 - ROI_W_FRAC))
        roi_x1 = w

        roi     = frame[roi_y0:roi_y1, roi_x0:roi_x1]
        blurred = cv2.GaussianBlur(roi, (7, 7), 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask_red    = (cv2.inRange(hsv, RED_LO1, RED_HI1) |
                       cv2.inRange(hsv, RED_LO2, RED_HI2))
        mask_yellow = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
        mask_green  = cv2.inRange(hsv, GREEN_LO,  GREEN_HI)

        px_red    = int(cv2.countNonZero(mask_red))
        px_yellow = int(cv2.countNonZero(mask_yellow))
        px_green  = int(cv2.countNonZero(mask_green))

        # Determinar color dominante (prioridad: rojo > amarillo > verde)
        color_detectado = 'ninguno'
        if px_red >= MIN_PIXELS and px_red >= px_yellow and px_red >= px_green:
            color_detectado = 'rojo'
        elif px_yellow >= MIN_PIXELS and px_yellow >= px_green:
            color_detectado = 'amarillo'
        elif px_green >= MIN_PIXELS:
            color_detectado = 'verde'

        # Lógica de transición de estado persistente:
        #   rojo   → se queda en rojo hasta que aparezca verde
        #   amarillo → se queda en amarillo hasta que desaparezca o aparezca verde
        #   verde / ninguno → transición directa
        estado_anterior = self._estado_actual

        if color_detectado == 'verde':
            self._estado_actual = 'verde'
        elif color_detectado == 'rojo':
            self._estado_actual = 'rojo'
        elif color_detectado == 'amarillo':
            # Solo entra en amarillo si no estaba en rojo
            if estado_anterior != 'rojo':
                self._estado_actual = 'amarillo'
        else:
            # Ningún color detectado
            if estado_anterior == 'amarillo':
                # Salir de amarillo cuando deja de verse
                self._estado_actual = 'ninguno'
            # Si era rojo, permanece rojo hasta que llegue verde
            # Si era verde/ninguno, permanece igual

        if self._estado_actual != estado_anterior:
            self.get_logger().info(
                f'Semáforo: {estado_anterior} → {self._estado_actual} '
                f'(R={px_red} A={px_yellow} V={px_green})'
            )

        # ── Frame de depuración ───────────────────────────────────────
        debug = frame.copy()

        COLORS = {
            'rojo':    (0,   0, 220),
            'amarillo':(0, 200, 220),
            'verde':   (0, 200,  60),
            'ninguno': (120, 120, 120),
        }
        box_color = COLORS[self._estado_actual]

        # Rectángulo ROI
        cv2.rectangle(debug, (roi_x0, roi_y0), (roi_x1, roi_y1), box_color, 2)

        # Overlay de la máscara ganadora sobre el ROI
        mask_map = {
            'rojo':    mask_red,
            'amarillo':mask_yellow,
            'verde':   mask_green,
        }
        if color_detectado in mask_map:
            colored = np.zeros_like(roi)
            colored[mask_map[color_detectado] > 0] = box_color
            blended = cv2.addWeighted(
                debug[roi_y0:roi_y1, roi_x0:roi_x1], 0.6, colored, 0.4, 0
            )
            debug[roi_y0:roi_y1, roi_x0:roi_x1] = blended

        # Etiqueta
        label = (f'{self._estado_actual.upper()} '
                 f'R={px_red} A={px_yellow} V={px_green}')
        cv2.putText(debug, label,
                    (roi_x0, roi_y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        return self._estado_actual, debug


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = SemaforoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
