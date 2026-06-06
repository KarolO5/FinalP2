#!/usr/bin/env python3
# =============================================================================
# semaforo.py  --  Detector de semaforo por color HSV
# =============================================================================
#
# ROI: esquina superior derecha del frame
#
#  +-----------------------------+----------+
#  |                             |          |  <- ROI_H_FRAC (50% superior)
#  |        ignorado             |  ROI     |
#  |                             | (28% der)|
#  +-----------------------------+----------+
#  |         ignorado                       |
#  +----------------------------------------+
#
# Logica de estado persistente:
#   rojo    -> robot para; se queda en rojo hasta ver verde
#   amarillo -> velocidad a la mitad; sale al dejar de verlo o al ver verde
#   verde   -> velocidad normal
#   ninguno -> mantiene estado anterior (parpadeos no cambian el estado)
#
# TOPICOS
# -------
#   Sub : /image/raw          [sensor_msgs/Image]
#   Pub : /semaforo/estado    [std_msgs/String]   rojo|amarillo|verde|ninguno
#   Pub : /semaforo/debug_img [sensor_msgs/Image] frame anotado con mascaras
# =============================================================================

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# -----------------------------------------------------------------------
# ROI  (esquina superior derecha, ajustado al rectangulo de la senal)
# -----------------------------------------------------------------------
ROI_H_FRAC = 0.50    # fraccion del alto desde arriba  (0 - 50%)
ROI_W_FRAC = 0.28    # fraccion del ancho desde la derecha (72% - 100%)

# -----------------------------------------------------------------------
# RANGOS HSV
# -----------------------------------------------------------------------
# Rojo cruza el 0/180 en HSV -> dos rangos
RED_LO1 = np.array([  0, 120,  80])
RED_HI1 = np.array([ 10, 255, 255])
RED_LO2 = np.array([165, 120,  80])
RED_HI2 = np.array([180, 255, 255])

YELLOW_LO = np.array([ 18, 100,  80])
YELLOW_HI = np.array([ 35, 255, 255])

GREEN_LO  = np.array([ 40,  80,  60])
GREEN_HI  = np.array([ 90, 255, 255])

# Pixeles minimos para considerar deteccion valida
MIN_PIXELS = 150


# -----------------------------------------------------------------------
# NODO
# -----------------------------------------------------------------------

class SemaforoNode(Node):

    def __init__(self):
        super().__init__('semaforo')

        qos_be = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._bridge = CvBridge()
        self._estado_actual = 'ninguno'

        self._pub_estado = self.create_publisher(String, '/semaforo/estado',    10)
        self._pub_debug  = self.create_publisher(Image,  '/semaforo/debug_img', 10)

        self.create_subscription(Image, '/image/raw', self._image_cb, qos_be)

        self.get_logger().info(
            'SemaforoNode listo | ROI esquina sup-der {}% alto x {}% ancho'.format(
                int(ROI_H_FRAC * 100), int(ROI_W_FRAC * 100))
        )

    def _image_cb(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn('cv_bridge: {}'.format(e))
            return

        estado, debug_frame = self._detect(frame)

        state_msg      = String()
        state_msg.data = estado
        self._pub_estado.publish(state_msg)

        dbg_msg        = self._bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
        dbg_msg.header = msg.header
        self._pub_debug.publish(dbg_msg)

    def _detect(self, frame: np.ndarray):
        h, w = frame.shape[:2]

        # Coordenadas del ROI
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

        # Color dominante
        color_det = 'ninguno'
        if px_red >= MIN_PIXELS and px_red >= px_yellow and px_red >= px_green:
            color_det = 'rojo'
        elif px_yellow >= MIN_PIXELS and px_yellow >= px_green:
            color_det = 'amarillo'
        elif px_green >= MIN_PIXELS:
            color_det = 'verde'

        # Transicion de estado persistente
        anterior = self._estado_actual

        if color_det == 'verde':
            self._estado_actual = 'verde'
        elif color_det == 'rojo':
            self._estado_actual = 'rojo'
        elif color_det == 'amarillo':
            if anterior != 'rojo':
                self._estado_actual = 'amarillo'
        else:
            if anterior == 'amarillo':
                self._estado_actual = 'ninguno'
            # rojo se mantiene hasta ver verde

        if self._estado_actual != anterior:
            self.get_logger().info(
                'Semaforo: {} -> {}  (R={} A={} V={})'.format(
                    anterior, self._estado_actual, px_red, px_yellow, px_green)
            )

        # ---------------------------------------------------------------
        # Debug: frame completo con ROI anotado
        # ---------------------------------------------------------------
        debug = frame.copy()

        COLORS = {
            'rojo':     (0,   0, 220),
            'amarillo': (0, 200, 220),
            'verde':    (0, 200,  60),
            'ninguno':  (120, 120, 120),
        }
        box_color = COLORS[self._estado_actual]

        # Oscurecer zona fuera del ROI
        ov = debug.copy()
        cv2.rectangle(ov, (0, 0), (roi_x0, h), (0, 0, 0), -1)
        cv2.rectangle(ov, (0, roi_y1), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, debug, 0.45, 0, debug)

        # Overlay de la mascara ganadora dentro del ROI
        mask_map = {
            'rojo':     mask_red,
            'amarillo': mask_yellow,
            'verde':    mask_green,
        }
        if color_det in mask_map:
            colored = np.zeros_like(roi)
            colored[mask_map[color_det] > 0] = box_color
            blended = cv2.addWeighted(
                debug[roi_y0:roi_y1, roi_x0:roi_x1], 0.5, colored, 0.5, 0
            )
            debug[roi_y0:roi_y1, roi_x0:roi_x1] = blended

        # Borde del ROI (grueso, siempre visible)
        cv2.rectangle(debug, (roi_x0, roi_y0), (roi_x1 - 1, roi_y1), box_color, 3)

        # Texto de estado
        label = '{} R={} A={} V={}'.format(
            self._estado_actual.upper(), px_red, px_yellow, px_green)
        cv2.putText(debug, label, (roi_x0, roi_y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(debug, label, (roi_x0, roi_y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

        # Mini-previews de las 3 mascaras (R / A / V) debajo del ROI
        roi_w  = roi_x1 - roi_x0
        roi_h_ = roi_y1 - roi_y0
        th = roi_h_ // 3
        tw = roi_w  // 3
        for i, (lbl, msk, col) in enumerate([
            ('R', mask_red,    (0, 0, 200)),
            ('A', mask_yellow, (0, 200, 220)),
            ('V', mask_green,  (0, 200, 60)),
        ]):
            t = cv2.resize(msk, (tw, th))
            t_bgr = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
            x0t = roi_x0 + i * tw
            y0t = roi_y1 + 30
            if y0t + th <= h and x0t + tw <= w:
                debug[y0t:y0t + th, x0t:x0t + tw] = t_bgr
                cv2.rectangle(debug, (x0t, y0t), (x0t + tw, y0t + th), col, 1)
                cv2.putText(debug, lbl, (x0t + 3, y0t + 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

        return self._estado_actual, debug


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

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
