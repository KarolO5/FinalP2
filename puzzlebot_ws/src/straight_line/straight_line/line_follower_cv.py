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
#  | ignor.  |   ZONA ACTIVA     | ignor.  |
#  | 30% izq |   40% central     | 30% der |
#  +---------+-------------------+---------+
#
# Seleccion de contorno:
#   Se elige el contorno cuyo centroide esta mas cerca del centro-inferior
#   del ROI (el punto del suelo mas proximo al robot), NO el de mayor area.
#   Esto evita que en curvas el robot salte a la linea exterior.
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
# PARAMETROS  -- ajusta estos sin tocar la logica
# -----------------------------------------------------------------------

LINEAR_VEL  = 0.13       # m/s, constante
MAX_ANGULAR = 0.60       # rad/s maximo

KP = 1.4
KI = 0.03
KD = 0.15
MAX_INTEGRAL = 0.35

# ROI vertical: ignorar el porcentaje superior del frame
ROI_TOP_FRAC = 0.72      # sube para mirar solo lo mas cercano al robot

# ROI horizontal: zona activa central (ignorar lados)
ROI_LEFT_FRAC  = 0.30
ROI_RIGHT_FRAC = 0.70

# Umbral adaptativo
ADAPT_BLOCK = 25
ADAPT_C     = 6

# Morfologia
MORPH_KSIZE = (7, 7)

# Area minima del contorno (px^2)
MIN_CONTOUR_AREA = 60

# Recovery
RECOVERY_FRAMES = 25
RECOVERY_OMEGA  = 0.20


# -----------------------------------------------------------------------
# DETECTOR
# -----------------------------------------------------------------------

class ContourLineDetector:

    def __init__(self):
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KSIZE)

    def _best_contour(self, contours, roi_w, roi_h):
        """
        Elige el contorno cuyo centroide esta mas cerca del centro-inferior
        del ROI. Eso equivale al punto del suelo mas proximo al robot,
        que es la linea que ya esta siguiendo.
        """
        ref_x = roi_w / 2.0
        ref_y = float(roi_h)

        best   = None
        best_d = float('inf')

        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            d  = ((cx - ref_x) ** 2 + (cy - ref_y) ** 2) ** 0.5
            if d < best_d:
                best_d = d
                best   = cnt

        return best

    def process(self, frame: np.ndarray):
        h, w = frame.shape[:2]

        roi_y0 = int(h * ROI_TOP_FRAC)
        roi_x0 = int(w * ROI_LEFT_FRAC)
        roi_x1 = int(w * ROI_RIGHT_FRAC)
        roi_w  = roi_x1 - roi_x0
        roi_h  = h - roi_y0

        roi = frame[roi_y0:h, roi_x0:roi_x1]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            ADAPT_BLOCK, ADAPT_C,
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        found    = False
        cx_roi   = roi_w // 2
        cy_roi   = roi_h // 2
        best_cnt = None

        if contours:
            best_cnt = self._best_contour(contours, roi_w, roi_h)
            if best_cnt is not None:
                M = cv2.moments(best_cnt)
                if M['m00'] > 0:
                    cx_roi = int(M['m10'] / M['m00'])
                    cy_roi = int(M['m01'] / M['m00'])
                    found  = True

        error_norm    = float((roi_w / 2 - cx_roi) / (roi_w / 2))
        cx_frame      = roi_x0 + cx_roi
        cy_frame      = roi_y0 + cy_roi
        cx_center_abs = (roi_x0 + roi_x1) // 2

        # ---------------------------------------------------------------
        # Debug
        # ---------------------------------------------------------------
        debug = frame.copy()

        # Zona superior oscurecida
        ov = debug.copy()
        cv2.rectangle(ov, (0, 0), (w, roi_y0), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.65, debug, 0.35, 0, debug)

        # Franjas laterales ignoradas (tinte morado)
        ov2 = debug.copy()
        cv2.rectangle(ov2, (0, roi_y0), (roi_x0, h), (30, 0, 80), -1)
        cv2.rectangle(ov2, (roi_x1, roi_y0), (w, h), (30, 0, 80), -1)
        cv2.addWeighted(ov2, 0.55, debug, 0.45, 0, debug)

        # Mascara binaria en verde sobre zona activa
        mask_color = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
        mask_color[mask > 0] = (0, 255, 0)
        debug[roi_y0:h, roi_x0:roi_x1] = cv2.addWeighted(
            debug[roi_y0:h, roi_x0:roi_x1], 0.45, mask_color, 0.55, 0
        )

        # Todos los contornos validos en gris (cuantas lineas ve)
        for cnt in contours:
            if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA:
                s = cnt.copy()
                s[:, :, 0] += roi_x0
                s[:, :, 1] += roi_y0
                cv2.drawContours(debug, [s], -1, (120, 120, 120), 1)

        # Contorno seleccionado en cian
        if best_cnt is not None and found:
            s = best_cnt.copy()
            s[:, :, 0] += roi_x0
            s[:, :, 1] += roi_y0
            cv2.drawContours(debug, [s], -1, (0, 255, 255), 3)

        # Bordes del ROI activo
        cv2.line(debug, (roi_x0, roi_y0), (roi_x1, roi_y0), (0, 220, 220), 3)
        cv2.line(debug, (roi_x0, roi_y0), (roi_x0, h),      (0, 220, 220), 2)
        cv2.line(debug, (roi_x1, roi_y0), (roi_x1, h),      (0, 220, 220), 2)

        # Linea de centro (referencia error=0)
        cv2.line(debug, (cx_center_abs, roi_y0), (cx_center_abs, h), (255, 100, 0), 2)

        # Centroide seleccionado
        if found:
            cv2.line(debug, (cx_frame, roi_y0), (cx_frame, h), (0, 0, 255), 2)
            cv2.circle(debug, (cx_frame, cy_frame), 12, (0, 0, 255), -1)
            cv2.circle(debug, (cx_frame, cy_frame), 12, (255, 255, 255), 2)

        # Flecha de error
        arr_y     = h - 20
        arr_color = (0, 255, 0) if found else (80, 80, 80)
        cv2.arrowedLine(debug,
                        (cx_center_abs, arr_y),
                        (cx_frame if found else cx_center_abs, arr_y),
                        arr_color, 3, tipLength=0.2)

        # Texto
        txt   = 'err={:+.3f}'.format(error_norm) if found else 'SIN LINEA'
        color = (0, 255, 120) if found else (0, 60, 255)
        cv2.putText(debug, txt, (roi_x0, roi_y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(debug, txt, (roi_x0, roi_y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Mini-preview mascara
        thumb_h = roi_h // 3
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
            'LineFollowerCV listo | KP={} KI={} KD={} | v={} m/s'.format(
                KP, KI, KD, LINEAR_VEL)
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

            derivative = (error - self._prev_error) / dt
            u          = KP * error + KI * self._integral + KD * derivative
            u          = max(-MAX_ANGULAR, min(MAX_ANGULAR, u))

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
