#!/usr/bin/env python3
# =============================================================================
# line_follower_cv.py  --  Seguidor de linea con ROI trapezoidal
# =============================================================================
#
# ROI en forma de TRAPECIO perspectivo:
#
#      [TOP_L]-----------[TOP_R]       <- fila roi_y0  (estrecho = lejos)
#       /                       \
#      /                         \
#  [BOT_L]-------------------[BOT_R]  <- fila h       (ancho = cerca del robot)
#
# Fracciones del ancho total del frame:
#   Abajo : BOT_LEFT=0.22  ..  BOT_RIGHT=0.78  (56% del ancho)
#   Arriba: TOP_LEFT=0.35  ..  TOP_RIGHT=0.65  (30% del ancho)
#
# Esto compensa la perspectiva: las lineas laterales lejanas aparecen
# mas juntas en la imagen y quedan FUERA del trapecio superior,
# evitando confusion con la linea central.
#
# Estrategia de deteccion:
#   1. Mascara trapezoidal aplicada al frame
#   2. Umbral adaptativo gaussiano (robusto a cambios de luz)
#   3. Cierre morfologico para rellenar huecos
#   4. Contorno mas cercano al centro-inferior del trapecio
#   5. Centroide con momentos de imagen -> error normalizado [-1,1]
#   6. Controlador PID
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

LINEAR_VEL  = 0.11
MAX_ANGULAR = 0.60

KP = 1.4
KI = 0.03
KD = 0.15
MAX_INTEGRAL = 0.35

AMARILLO_FACTOR = 0.60   # velocidad al 60% con semaforo amarillo (reduccion 40%)

# ROI vertical: donde empieza el trapecio (parte superior)
ROI_TOP_FRAC = 0.58      # 42% inferior del frame analizado

# Trapecio horizontal (fracciones del ancho del frame)
# Fila inferior (cerca del robot) -- mas ancho
TRAP_BOT_LEFT  = 0.22
TRAP_BOT_RIGHT = 0.78
# Fila superior (lejos) -- mas estrecho para evitar lineas laterales
TRAP_TOP_LEFT  = 0.36
TRAP_TOP_RIGHT = 0.64

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

    def _best_contour(self, contours, ref_x, ref_y):
        """Contorno cuyo centroide esta mas cerca del punto de referencia."""
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
                best   = (cnt, int(cx), int(cy))

        return best

    def process(self, frame: np.ndarray):
        h, w = frame.shape[:2]

        roi_y0 = int(h * ROI_TOP_FRAC)

        # Vertices del trapecio en coordenadas del frame completo
        bot_l = (int(w * TRAP_BOT_LEFT),  h)
        bot_r = (int(w * TRAP_BOT_RIGHT), h)
        top_l = (int(w * TRAP_TOP_LEFT),  roi_y0)
        top_r = (int(w * TRAP_TOP_RIGHT), roi_y0)

        # Centro horizontal del trapecio (para calcular el error)
        cx_center = (top_l[0] + top_r[0] + bot_l[0] + bot_r[0]) // 4
        # Ancho de referencia en la fila inferior (para normalizar)
        ref_width = bot_r[0] - bot_l[0]

        # ---- Mascara trapezoidal ----
        trap_mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array([bot_l, bot_r, top_r, top_l], dtype=np.int32)
        cv2.fillPoly(trap_mask, [pts], 255)

        # ---- ROI recortado al bounding box del trapecio para procesar ----
        roi_x0 = top_l[0]
        roi_x1 = top_r[0]   # usamos el ancho superior como bbox izq/der
        # Para el bbox completo tomamos los extremos del trapecio
        bbox_x0 = bot_l[0]
        bbox_x1 = bot_r[0]

        roi = frame[roi_y0:h, bbox_x0:bbox_x1].copy()
        # Aplicar mascara trapezoidal recortada al bbox
        trap_local = trap_mask[roi_y0:h, bbox_x0:bbox_x1]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            ADAPT_BLOCK, ADAPT_C,
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        # Solo conservar pixeles dentro del trapecio
        mask = cv2.bitwise_and(mask, trap_local)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        found    = False
        cx_abs   = cx_center
        cy_abs   = roi_y0 + (h - roi_y0) // 2
        best_cnt = None

        roi_h_local = h - roi_y0
        # Referencia: centro horizontal del trapecio, fila inferior
        ref_x_local = (bot_l[0] + bot_r[0]) / 2.0 - bbox_x0
        ref_y_local = float(roi_h_local)

        if contours:
            res = self._best_contour(contours, ref_x_local, ref_y_local)
            if res is not None:
                cnt, cx_local, cy_local = res
                best_cnt = cnt
                cx_abs   = bbox_x0 + cx_local
                cy_abs   = roi_y0  + cy_local
                found    = True

        # Error: desplazamiento del centroide respecto al centro del trapecio
        # normalizado por el semi-ancho inferior
        error_norm = float((cx_center - cx_abs) / (ref_width / 2))
        error_norm = max(-1.0, min(1.0, error_norm))

        # ---------------------------------------------------------------
        # Debug
        # ---------------------------------------------------------------
        debug = frame.copy()

        # Oscurecer zona fuera del trapecio
        inv_mask = cv2.bitwise_not(trap_mask)
        # Zona superior (fuera del ROI vertical)
        inv_top = np.zeros_like(trap_mask)
        inv_top[:roi_y0, :] = 255
        outside = cv2.bitwise_or(inv_mask, inv_top)
        dark = debug.copy()
        dark[outside > 0] = (dark[outside > 0] * 0.25).astype(np.uint8)
        debug = dark

        # Overlay verde de la mascara binaria dentro del trapecio
        mask_full = np.zeros((h, w), dtype=np.uint8)
        mask_full[roi_y0:h, bbox_x0:bbox_x1] = mask
        mask_color = np.zeros_like(frame)
        mask_color[mask_full > 0] = (0, 255, 0)
        debug = cv2.addWeighted(debug, 0.5, mask_color, 0.5, 0)

        # Todos los contornos validos en gris
        for cnt in contours:
            if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA:
                s = cnt.copy()
                s[:, :, 0] += bbox_x0
                s[:, :, 1] += roi_y0
                cv2.drawContours(debug, [s], -1, (120, 120, 120), 1)

        # Contorno seleccionado en cian
        if best_cnt is not None and found:
            s = best_cnt.copy()
            s[:, :, 0] += bbox_x0
            s[:, :, 1] += roi_y0
            cv2.drawContours(debug, [s], -1, (0, 255, 255), 3)

        # Dibujar trapecio (borde cian)
        cv2.polylines(debug, [pts], True, (0, 220, 220), 2)

        # Linea central de referencia (azul)
        cv2.line(debug, (cx_center, roi_y0), (cx_center, h), (255, 100, 0), 2)

        # Centroide detectado (rojo)
        if found:
            cv2.line(debug, (cx_abs, roi_y0), (cx_abs, h), (0, 0, 255), 2)
            cv2.circle(debug, (cx_abs, cy_abs), 12, (0, 0, 255), -1)
            cv2.circle(debug, (cx_abs, cy_abs), 12, (255, 255, 255), 2)

        # Flecha de error
        arr_y     = h - 20
        arr_color = (0, 255, 0) if found else (80, 80, 80)
        cv2.arrowedLine(debug,
                        (cx_center, arr_y),
                        (cx_abs if found else cx_center, arr_y),
                        arr_color, 3, tipLength=0.2)

        # Texto
        txt   = 'err={:+.3f}'.format(error_norm) if found else 'SIN LINEA'
        color = (0, 255, 120) if found else (0, 60, 255)
        cv2.putText(debug, txt, (top_l[0], roi_y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(debug, txt, (top_l[0], roi_y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Mini-preview mascara
        roi_h_local2 = h - roi_y0
        roi_w_local2 = bbox_x1 - bbox_x0
        if roi_h_local2 > 0 and roi_w_local2 > 0:
            th = roi_h_local2 // 3
            tw = roi_w_local2 // 3
            thumb = cv2.resize(mask, (tw, th))
            x0t = w - tw - 4
            y0t = h - th - 4
            debug[y0t:y0t + th, x0t:x0t + tw] = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(debug, (x0t, y0t), (x0t + tw, y0t + th), (150, 150, 150), 1)
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
        self._semaforo    = 'ninguno'

        self._pub_cmd = self.create_publisher(Twist,   '/cmd_vel',          qos_be)
        self._pub_dbg = self.create_publisher(Image,   '/vision/debug_img', 10)
        self._pub_err = self.create_publisher(Float32, '/vision/error',     10)

        self.create_subscription(Image,  '/image/raw',       self._image_cb,    qos_be)
        self.create_subscription(String, '/semaforo/estado', self._semaforo_cb, 10)

        self.get_logger().info(
            'LineFollowerCV listo | KP={} KI={} KD={} | v={} m/s | '
            'trapecio BOT={}%-{}% TOP={}%-{}%'.format(
                KP, KI, KD, LINEAR_VEL,
                int(TRAP_BOT_LEFT*100), int(TRAP_BOT_RIGHT*100),
                int(TRAP_TOP_LEFT*100), int(TRAP_TOP_RIGHT*100))
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

        vel_factor = AMARILLO_FACTOR if self._semaforo == 'amarillo' else 1.0

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
