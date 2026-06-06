#!/usr/bin/env python3
# =============================================================================
# line_follower_cv.py  --  Seguidor de linea por contorno + PID
# =============================================================================
#
# ROI activa: franja inferior + zona central del ancho
#
#  +---------+-------------------+---------+
#  |         |   zona ignorada   |         |  <- parte superior (ROI_TOP_FRAC)
#  +---------+-------------------+---------+
#  | ignor.  |   ZONA ACTIVA     | ignor.  |  <- 40% inferior
#  | 30% izq |   40% central     | 30% der |
#  +---------+-------------------+---------+
#
# Esto evita confundir la linea central con las lineas laterales del circuito
# cuando hay curvas.
#
# TOPICOS
# -------
#   Sub : /image/raw         [sensor_msgs/Image]
#   Sub : /semaforo/estado   [std_msgs/String]
#   Pub : /cmd_vel           [geometry_msgs/Twist]
#   Pub : /vision/debug_img  [sensor_msgs/Image]
#   Pub : /vision/error      [std_msgs/Float32]
# =============================================================================

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg   import Image
from std_msgs.msg      import Float32, String
from cv_bridge         import CvBridge
from rclpy.qos         import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# -----------------------------------------------------------------------
# PARAMETROS
# -----------------------------------------------------------------------

LINEAR_VEL  = 0.15
MAX_ANGULAR = 0.60

KP = 1.6
KI = 0.04
KD = 0.30
MAX_INTEGRAL = 0.40

# ROI vertical: el ROI empieza en este porcentaje del alto (ignora la parte superior)
ROI_TOP_FRAC = 0.55

# ROI horizontal: zona activa central (ignora los lados)
ROI_LEFT_FRAC  = 0.30   # ignora el 30% izquierdo
ROI_RIGHT_FRAC = 0.70   # ignora el 30% derecho (zona activa = 40% central)

# Umbral adaptativo
ADAPT_BLOCK = 31
ADAPT_C     = 8

# Morfologia
MORPH_KSIZE = (9, 9)

# Area minima del contorno para considerarlo linea valida (px^2)
MIN_CONTOUR_AREA = 80

# Recovery
RECOVERY_FRAMES = 25
RECOVERY_OMEGA  = 0.20


# -----------------------------------------------------------------------
# DETECTOR
# -----------------------------------------------------------------------

class ContourLineDetector:

    def __init__(self):
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KSIZE)

    def process(self, frame: np.ndarray):
        h, w = frame.shape[:2]

        # Coordenadas del ROI activo
        roi_y0 = int(h * ROI_TOP_FRAC)
        roi_x0 = int(w * ROI_LEFT_FRAC)
        roi_x1 = int(w * ROI_RIGHT_FRAC)
        roi_w  = roi_x1 - roi_x0

        # Recortar la zona activa
        roi = frame[roi_y0:h, roi_x0:roi_x1]

        # Umbral adaptativo
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            ADAPT_BLOCK, ADAPT_C,
        )

        # Cierre morfologico
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        # Contorno de mayor area dentro de la zona activa
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        found    = False
        # cx_line en coordenadas de la zona activa (0..roi_w)
        cx_roi   = roi_w // 2
        best_cnt = None

        if contours:
            best_cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(best_cnt) >= MIN_CONTOUR_AREA:
                M = cv2.moments(best_cnt)
                if M['m00'] > 0:
                    cx_roi = int(M['m10'] / M['m00'])
                    found  = True

        # Error relativo al centro de la zona activa, normalizado [-1, 1]
        error_norm = float((roi_w / 2 - cx_roi) / (roi_w / 2))

        # cx en coordenadas del frame completo (para dibujar)
        cx_frame = roi_x0 + cx_roi
        cx_center_frame = (roi_x0 + roi_x1) // 2

        # ---------------------------------------------------------------
        # Frame de debug
        # ---------------------------------------------------------------
        debug = frame.copy()

        # 1. Oscurecer zona fuera del ROI vertical (parte superior)
        ov = debug.copy()
        cv2.rectangle(ov, (0, 0), (w, roi_y0), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.65, debug, 0.35, 0, debug)

        # 2. Oscurecer franjas laterales ignoradas
        ov2 = debug.copy()
        cv2.rectangle(ov2, (0, roi_y0), (roi_x0, h), (0, 0, 60), -1)
        cv2.rectangle(ov2, (roi_x1, roi_y0), (w, h), (0, 0, 60), -1)
        cv2.addWeighted(ov2, 0.6, debug, 0.4, 0, debug)

        # 3. Overlay verde de la mascara binaria SOLO en zona activa
        mask_color = np.zeros((h - roi_y0, roi_w, 3), dtype=np.uint8)
        mask_color[mask > 0] = (0, 255, 0)
        debug[roi_y0:h, roi_x0:roi_x1] = cv2.addWeighted(
            debug[roi_y0:h, roi_x0:roi_x1], 0.45, mask_color, 0.55, 0
        )

        # 4. Contorno detectado en cian
        if best_cnt is not None and found:
            shifted = best_cnt.copy()
            shifted[:, :, 0] += roi_x0   # desplazar X
            shifted[:, :, 1] += roi_y0   # desplazar Y
            cv2.drawContours(debug, [shifted], -1, (0, 255, 255), 3)

        # 5. Bordes de la zona activa (cian grueso -- siempre visibles)
        cv2.line(debug, (roi_x0, roi_y0), (roi_x1, roi_y0), (0, 220, 220), 3)  # top
        cv2.line(debug, (roi_x0, roi_y0), (roi_x0, h),      (0, 220, 220), 2)  # izq
        cv2.line(debug, (roi_x1, roi_y0), (roi_x1, h),      (0, 220, 220), 2)  # der

        # 6. Centro de la zona activa (azul -- referencia de error=0)
        cv2.line(debug, (cx_center_frame, roi_y0), (cx_center_frame, h), (255, 100, 0), 2)

        # 7. Centroide detectado (rojo)
        cy_abs = roi_y0 + (h - roi_y0) // 2
        if found:
            cv2.line(debug, (cx_frame, roi_y0), (cx_frame, h), (0, 0, 255), 2)
            cv2.circle(debug, (cx_frame, cy_abs), 12, (0, 0, 255), -1)
            cv2.circle(debug, (cx_frame, cy_abs), 12, (255, 255, 255), 2)

        # 8. Flecha de error (centro -> centroide)
        arr_color = (0, 255, 0) if found else (80, 80, 80)
        cv2.arrowedLine(debug, (cx_center_frame, cy_abs), (cx_frame, cy_abs),
                        arr_color, 3, tipLength=0.2)

        # 9. Texto de estado con borde negro
        txt   = 'err={:+.3f}'.format(error_norm) if found else 'SIN LINEA'
        color = (0, 255, 120) if found else (0, 60, 255)
        cv2.putText(debug, txt, (roi_x0, roi_y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(debug, txt, (roi_x0, roi_y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 10. Mini-preview de la mascara en esquina inferior derecha
        thumb_h = (h - roi_y0) // 3
        thumb_w = roi_w // 3
        thumb   = cv2.resize(mask, (thumb_w, thumb_h))
        x0t = w - thumb_w - 4
        y0t = h - thumb_h - 4
        debug[y0t:y0t + thumb_h, x0t:x0t + thumb_w] = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(debug, (x0t, y0t), (x0t + thumb_w, y0t + thumb_h), (150, 150, 150), 1)
        cv2.putText(debug, 'MASK', (x0t + 3, y0t + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return error_norm, found, debug


# -----------------------------------------------------------------------
# NODO ROS 2
# -----------------------------------------------------------------------

class LineFollowerCV(Node):

    def __init__(self):
        super().__init__('line_follower_cv')

        qos_be = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._detector = ContourLineDetector()
        self._bridge   = CvBridge()

        self._prev_error  = 0.0
        self._integral    = 0.0
        self._prev_time   = None
        self._last_error  = 0.0
        self._frames_lost = 0

        self._semaforo = 'ninguno'

        self._pub_cmd = self.create_publisher(Twist,   '/cmd_vel',          qos_be)
        self._pub_dbg = self.create_publisher(Image,   '/vision/debug_img', 10)
        self._pub_err = self.create_publisher(Float32, '/vision/error',     10)

        self.create_subscription(Image,  '/image/raw',       self._image_cb,    qos_be)
        self.create_subscription(String, '/semaforo/estado', self._semaforo_cb, 10)

        self.get_logger().info(
            'LineFollowerCV listo | KP={} KI={} KD={} | v={} m/s | '
            'ROI top={}% | lateral {}%-{}%'.format(
                KP, KI, KD, LINEAR_VEL,
                int(ROI_TOP_FRAC * 100),
                int(ROI_LEFT_FRAC * 100), int(ROI_RIGHT_FRAC * 100))
        )

    def _semaforo_cb(self, msg: String):
        nuevo = msg.data
        if nuevo != self._semaforo:
            self.get_logger().info('Semaforo: {} -> {}'.format(self._semaforo, nuevo))
            if nuevo != 'rojo':
                self._integral = 0.0
        self._semaforo = nuevo

    def _image_cb(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn('cv_bridge: {}'.format(e))
            return

        error_norm, found, debug_frame = self._detector.process(frame)

        dbg_msg        = self._bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
        dbg_msg.header = msg.header
        self._pub_dbg.publish(dbg_msg)

        err_msg      = Float32()
        err_msg.data = float(error_norm)
        self._pub_err.publish(err_msg)

        self._control(error_norm, found)

    def _control(self, error: float, found: bool):
        if self._semaforo == 'rojo':
            self._pub_cmd.publish(Twist())
            self._prev_time = None
            return

        vel_factor = 0.5 if self._semaforo == 'amarillo' else 1.0

        now = self.get_clock().now().nanoseconds * 1e-9
        dt  = (now - self._prev_time) if self._prev_time is not None else 0.033
        dt  = max(dt, 1e-4)
        self._prev_time = now

        cmd = Twist()

        if found:
            self._frames_lost = 0
            self._integral   += error * dt
            self._integral    = max(-MAX_INTEGRAL, min(MAX_INTEGRAL, self._integral))

            derivative        = (error - self._prev_error) / dt
            u                 = KP * error + KI * self._integral + KD * derivative
            u                 = max(-MAX_ANGULAR, min(MAX_ANGULAR, u))

            self._prev_error = error
            self._last_error = error

            cmd.linear.x  = LINEAR_VEL * vel_factor
            cmd.angular.z = u

        else:
            self._frames_lost += 1
            self._integral     = 0.0

            if self._frames_lost < RECOVERY_FRAMES:
                u             = KP * self._last_error * 0.4
                cmd.linear.x  = LINEAR_VEL * vel_factor * 0.4
                cmd.angular.z = max(-MAX_ANGULAR, min(MAX_ANGULAR, u))
            else:
                self.get_logger().warn(
                    'Linea perdida ({} frames) -- buscando'.format(self._frames_lost)
                )
                cmd.linear.x  = 0.0
                cmd.angular.z = (RECOVERY_OMEGA
                                 if self._last_error >= 0
                                 else -RECOVERY_OMEGA)

        self._pub_cmd.publish(cmd)

    def stop(self):
        self._pub_cmd.publish(Twist())


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerCV()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.get_logger().info('Motores detenidos.')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
