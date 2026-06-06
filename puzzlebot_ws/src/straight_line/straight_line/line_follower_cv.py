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
#  | ignor.  |   ZONA ACTIVA     | ignor.  |  <- 45% inferior
#  | 30% izq |   40% central     | 30% der |
#  +---------+-------------------+---------+
#
# Seleccion de contorno:
#   Se descarta el criterio de "mayor area" (confunde linea exterior con central).
#   En su lugar se elige el contorno cuyo centroide esta mas cerca del
#   centro-inferior del ROI (= el punto del suelo mas cercano al robot).
#   Esto hace que el robot "siga" la linea que tiene debajo, no la de enfrente.
#
# Velocidad dinamica:
#   v = LINEAR_VEL * (1 - SPEED_REDUCTION * |error|)
#   En curva (error alto) el robot frena automaticamente y puede girar mas.
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

# Velocidad base (m/s)
LINEAR_VEL    = 0.18
MIN_VEL       = 0.05      # velocidad minima en curva cerrada

# Reduccion de velocidad proporcional al error: v = v_base * (1 - k*|e|)
SPEED_REDUCTION = 0.75    # 0=sin reduccion, 1=para completamente en error=1

# PID
MAX_ANGULAR  = 1.2        # aumentado para poder girar mas rapido en curvas
KP = 2.0
KI = 0.03
KD = 0.25
MAX_INTEGRAL = 0.35

# ROI vertical
ROI_TOP_FRAC = 0.58

# ROI horizontal (zona activa central)
ROI_LEFT_FRAC  = 0.28
ROI_RIGHT_FRAC = 0.72

# Umbral adaptativo
ADAPT_BLOCK = 25
ADAPT_C     = 6

# Morfologia
MORPH_KSIZE = (7, 7)

# Area minima del contorno (px^2)
MIN_CONTOUR_AREA = 60

# Recovery
RECOVERY_FRAMES = 20
RECOVERY_OMEGA  = 0.25


# -----------------------------------------------------------------------
# DETECTOR
# -----------------------------------------------------------------------

class ContourLineDetector:

    def __init__(self):
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KSIZE)

    def _best_contour(self, contours, roi_w, roi_h):
        """
        Selecciona el contorno cuyo centroide esta mas cerca del
        punto de referencia: centro horizontal, fila inferior del ROI.
        Esto favorece la linea mas cercana al robot (la que pisa)
        sobre las lineas del fondo o laterales.
        """
        ref_x = roi_w / 2.0
        ref_y = float(roi_h)   # fila inferior = mas cercana al robot

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

        # Coordenadas del ROI activo
        roi_y0 = int(h * ROI_TOP_FRAC)
        roi_x0 = int(w * ROI_LEFT_FRAC)
        roi_x1 = int(w * ROI_RIGHT_FRAC)
        roi_w  = roi_x1 - roi_x0
        roi_h  = h - roi_y0

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

        # Contornos
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

        # Error relativo al centro de la zona activa [-1, 1]
        error_norm = float((roi_w / 2 - cx_roi) / (roi_w / 2))

        # Coordenadas absolutas para dibujar
        cx_frame      = roi_x0 + cx_roi
        cy_frame      = roi_y0 + cy_roi
        cx_center_abs = (roi_x0 + roi_x1) // 2

        # ---------------------------------------------------------------
        # Frame de debug
        # ---------------------------------------------------------------
        debug = frame.copy()

        # Oscurecer zona superior (fuera del ROI vertical)
        ov = debug.copy()
        cv2.rectangle(ov, (0, 0), (w, roi_y0), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.65, debug, 0.35, 0, debug)

        # Oscurecer franjas laterales ignoradas (tinte azul)
        ov2 = debug.copy()
        cv2.rectangle(ov2, (0, roi_y0), (roi_x0, h), (30, 0, 80), -1)
        cv2.rectangle(ov2, (roi_x1, roi_y0), (w, h), (30, 0, 80), -1)
        cv2.addWeighted(ov2, 0.55, debug, 0.45, 0, debug)

        # Overlay verde de la mascara binaria en zona activa
        mask_color = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
        mask_color[mask > 0] = (0, 255, 0)
        debug[roi_y0:h, roi_x0:roi_x1] = cv2.addWeighted(
            debug[roi_y0:h, roi_x0:roi_x1], 0.45, mask_color, 0.55, 0
        )

        # Todos los contornos validos en gris (para ver cuantas lineas detecta)
        for cnt in contours:
            if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA:
                shifted = cnt.copy()
                shifted[:, :, 0] += roi_x0
                shifted[:, :, 1] += roi_y0
                cv2.drawContours(debug, [shifted], -1, (120, 120, 120), 1)

        # Contorno seleccionado en cian grueso
        if best_cnt is not None and found:
            shifted = best_cnt.copy()
            shifted[:, :, 0] += roi_x0
            shifted[:, :, 1] += roi_y0
            cv2.drawContours(debug, [shifted], -1, (0, 255, 255), 3)

        # Bordes del ROI activo (cian -- siempre visibles)
        cv2.line(debug, (roi_x0, roi_y0), (roi_x1, roi_y0), (0, 220, 220), 3)
        cv2.line(debug, (roi_x0, roi_y0), (roi_x0, h),      (0, 220, 220), 2)
        cv2.line(debug, (roi_x1, roi_y0), (roi_x1, h),      (0, 220, 220), 2)

        # Centro de referencia (azul)
        cv2.line(debug, (cx_center_abs, roi_y0), (cx_center_abs, h), (255, 100, 0), 2)

        # Punto de referencia inferior (donde busca la linea)
        cv2.circle(debug, (cx_center_abs, h - 10), 6, (255, 200, 0), -1)

        # Centroide seleccionado (rojo)
        if found:
            cv2.line(debug, (cx_frame, roi_y0), (cx_frame, h), (0, 0, 255), 2)
            cv2.circle(debug, (cx_frame, cy_frame), 12, (0, 0, 255), -1)
            cv2.circle(debug, (cx_frame, cy_frame), 12, (255, 255, 255), 2)

        # Flecha de error
        arr_y     = h - 20
        arr_color = (0, 255, 0) if found else (80, 80, 80)
        cv2.arrowedLine(debug, (cx_center_abs, arr_y), (cx_frame if found else cx_center_abs, arr_y),
                        arr_color, 3, tipLength=0.2)

        # Texto
        txt   = 'err={:+.3f}'.format(error_norm) if found else 'SIN LINEA'
        color = (0, 255, 120) if found else (0, 60, 255)
        cv2.putText(debug, txt, (roi_x0, roi_y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(debug, txt, (roi_x0, roi_y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Mini-preview de la mascara
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
            'LineFollowerCV listo | KP={} KI={} KD={} | '
            'v_base={} min_v={} | ROI top={}% lat={}%-{}%'.format(
                KP, KI, KD, LINEAR_VEL, MIN_VEL,
                int(ROI_TOP_FRAC * 100),
                int(ROI_LEFT_FRAC * 100),
                int(ROI_RIGHT_FRAC * 100))
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

        sem_factor = 0.5 if self._semaforo == 'amarillo' else 1.0

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

            # Velocidad dinamica: frena en curva proporcional al error
            speed = LINEAR_VEL * (1.0 - SPEED_REDUCTION * abs(error))
            speed = max(MIN_VEL, speed) * sem_factor

            cmd.linear.x  = speed
            cmd.angular.z = u

        else:
            self._frames_lost += 1
            self._integral     = 0.0

            if self._frames_lost < RECOVERY_FRAMES:
                u             = KP * self._last_error * 0.4
                cmd.linear.x  = MIN_VEL * sem_factor
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
