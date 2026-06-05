#!/usr/bin/env python3
# =============================================================================
# line_follower_cv.py  —  Seguidor de línea por contorno + PID
# =============================================================================
#
# ESTRATEGIA DE DETECCIÓN
# ───────────────────────
# 1. Se recorta una ROI del porcentaje inferior del frame.
# 2. Se convierte a escala de grises y se aplica umbral adaptativo gaussiano
#    (se adapta a cambios de iluminación en el piso).
# 3. Se aplica cierre morfológico para rellenar huecos en la línea.
# 4. Se buscan contornos externos y se selecciona el de mayor área.
# 5. Se calcula el centroide del contorno con momentos de imagen.
# 6. El error es la distancia normalizada del centroide al centro horizontal.
# 7. Un controlador PID convierte ese error en velocidad angular.
#
# TÓPICOS
# ────────
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

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS  —  ajusta sin tocar la lógica
# ─────────────────────────────────────────────────────────────────────────────

# Velocidad base (m/s)
LINEAR_VEL  = 0.15
MAX_ANGULAR = 0.60

# PID visual (error normalizado [-1, 1])
KP = 1.6
KI = 0.04   # integral pequeña para compensar deriva en curvas largas
KD = 0.30
MAX_INTEGRAL = 0.40   # anti-windup

# ROI: fracción inferior del frame que se analiza
ROI_TOP_FRAC = 0.55   # el ROI empieza en el 55 % del alto (toma el 45 % inferior)

# Umbral adaptativo
ADAPT_BLOCK = 31     # tamaño de bloque (impar, > 1)
ADAPT_C     = 8      # constante sustraída a la media local

# Morfología: cierre para unir trazos rotos de la línea
MORPH_KSIZE = (9, 9)

# Área mínima de contorno para considerarlo línea válida (px²)
MIN_CONTOUR_AREA = 400

# Recovery si la línea se pierde
RECOVERY_FRAMES = 25
RECOVERY_OMEGA  = 0.20   # rad/s girando hacia la última dirección conocida


# ─────────────────────────────────────────────────────────────────────────────
# DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class ContourLineDetector:
    """
    Detecta la línea negra mediante contorno + momentos.

    Devuelve:
        error_norm  : float [-1, 1]  (0 = centrada, + = línea a izquierda)
        found       : bool
        debug       : imagen BGR anotada
    """

    def __init__(self):
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KSIZE)

    def process(self, frame: np.ndarray):
        h, w = frame.shape[:2]

        # ── 1. ROI inferior ───────────────────────────────────────────
        roi_y0 = int(h * ROI_TOP_FRAC)
        roi    = frame[roi_y0:h, :]

        # ── 2. Umbral adaptativo sobre escala de grises ───────────────
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,   # línea oscura → blanco
            ADAPT_BLOCK, ADAPT_C,
        )

        # ── 3. Cierre morfológico para rellenar huecos ────────────────
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        # ── 4. Contorno de mayor área ─────────────────────────────────
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        found      = False
        cx_line    = w // 2   # fallback: centro del frame
        best_cnt   = None

        if contours:
            best_cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(best_cnt) >= MIN_CONTOUR_AREA:
                M = cv2.moments(best_cnt)
                if M['m00'] > 0:
                    cx_line = int(M['m10'] / M['m00'])
                    found   = True

        # Error normalizado: 0 = centrado, +1 = línea extremo izquierdo
        error_norm = float((w / 2 - cx_line) / (w / 2))

        # ── 5. Frame de depuración ────────────────────────────────────
        debug = frame.copy()

        # Sombrear zona fuera de la ROI
        ov = debug.copy()
        cv2.rectangle(ov, (0, 0), (w, roi_y0), (20, 20, 20), -1)
        cv2.addWeighted(ov, 0.45, debug, 0.55, 0, debug)

        # Proyectar máscara binaria en verde sobre la ROI
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_bgr[:, :, 0] = 0   # quitar canal R y B → solo verde
        mask_bgr[:, :, 2] = 0
        debug[roi_y0:h, :] = cv2.addWeighted(
            debug[roi_y0:h, :], 0.6, mask_bgr, 0.4, 0
        )

        # Contorno detectado
        if best_cnt is not None and found:
            best_cnt_shifted = best_cnt.copy()
            best_cnt_shifted[:, :, 1] += roi_y0
            cv2.drawContours(debug, [best_cnt_shifted], -1, (0, 255, 255), 2)

        # Líneas de referencia
        cv2.line(debug, (0, roi_y0), (w, roi_y0), (180, 180, 0), 1)
        cv2.line(debug, (w // 2, roi_y0), (w // 2, h), (255, 80, 0), 1)   # centro

        # Centroide detectado
        if found:
            cy_abs = roi_y0 + (h - roi_y0) // 2
            cv2.circle(debug, (cx_line, cy_abs), 8, (0, 0, 255), -1)
            cv2.line(debug, (cx_line, roi_y0), (cx_line, h), (0, 0, 255), 2)

        # Flecha de error
        arr_y = roi_y0 + (h - roi_y0) // 2
        cv2.arrowedLine(
            debug, (w // 2, arr_y), (cx_line, arr_y),
            (0, 255, 0) if found else (60, 60, 60), 2, tipLength=0.25,
        )

        # Texto
        txt   = f'err={error_norm:+.3f}' if found else 'SIN LINEA'
        color = (0, 255, 120) if found else (0, 60, 255)
        cv2.putText(debug, txt, (8, roi_y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return error_norm, found, debug


# ─────────────────────────────────────────────────────────────────────────────
# NODO ROS 2
# ─────────────────────────────────────────────────────────────────────────────

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

        # Estado PID
        self._prev_error  = 0.0
        self._integral    = 0.0
        self._prev_time   = None
        self._last_error  = 0.0
        self._frames_lost = 0

        # Semáforo
        self._semaforo = 'ninguno'

        self._pub_cmd = self.create_publisher(Twist,   '/cmd_vel',          qos_be)
        self._pub_dbg = self.create_publisher(Image,   '/vision/debug_img', 10)
        self._pub_err = self.create_publisher(Float32, '/vision/error',     10)

        self.create_subscription(Image,  '/image/raw',       self._image_cb,    qos_be)
        self.create_subscription(String, '/semaforo/estado', self._semaforo_cb, 10)

        self.get_logger().info(
            f'LineFollowerCV listo | '
            f'KP={KP} KI={KI} KD={KD} | v={LINEAR_VEL} m/s'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _semaforo_cb(self, msg: String):
        nuevo = msg.data
        if nuevo != self._semaforo:
            self.get_logger().info(f'Semáforo: {self._semaforo} → {nuevo}')
            if nuevo != 'rojo':
                # Al salir de rojo, reiniciar integral para evitar windup acumulado
                self._integral = 0.0
        self._semaforo = nuevo

    def _image_cb(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge: {e}')
            return

        error_norm, found, debug_frame = self._detector.process(frame)

        dbg_msg        = self._bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
        dbg_msg.header = msg.header
        self._pub_dbg.publish(dbg_msg)

        err_msg      = Float32()
        err_msg.data = float(error_norm)
        self._pub_err.publish(err_msg)

        self._control(error_norm, found)

    # ── Controlador PID ───────────────────────────────────────────────────

    def _control(self, error: float, found: bool):
        # Semáforo rojo → parado; la integral no acumula
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

            # Integral con anti-windup
            self._integral += error * dt
            self._integral  = max(-MAX_INTEGRAL,
                                  min(MAX_INTEGRAL, self._integral))

            derivative = (error - self._prev_error) / dt
            u = KP * error + KI * self._integral + KD * derivative
            u = max(-MAX_ANGULAR, min(MAX_ANGULAR, u))

            self._prev_error = error
            self._last_error = error

            cmd.linear.x  = LINEAR_VEL * vel_factor
            cmd.angular.z = u

        else:
            self._frames_lost += 1
            self._integral     = 0.0   # reiniciar integral si se pierde la línea

            if self._frames_lost < RECOVERY_FRAMES:
                # Mantener última corrección suavizada
                u             = KP * self._last_error * 0.4
                cmd.linear.x  = LINEAR_VEL * vel_factor * 0.4
                cmd.angular.z = max(-MAX_ANGULAR, min(MAX_ANGULAR, u))
            else:
                # Girar en el sitio hacia la última dirección conocida
                self.get_logger().warn(
                    f'Línea perdida ({self._frames_lost} frames) — buscando'
                )
                cmd.linear.x  = 0.0
                cmd.angular.z = (RECOVERY_OMEGA
                                 if self._last_error >= 0
                                 else -RECOVERY_OMEGA)

        self._pub_cmd.publish(cmd)

    def stop(self):
        self._pub_cmd.publish(Twist())


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

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
