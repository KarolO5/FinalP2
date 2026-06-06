#!/usr/bin/env python3
# =============================================================================
# line_follower_cv.py  --  Seguidor de linea con lookahead
# =============================================================================
#
# Con la camara mas alta el robot ve mas adelante. Se usan DOS bandas:
#
#  +----------------------------------------+
#  |           zona ignorada                |  <- ROI_TOP_FRAC
#  +--------+------------------+------------+
#  | ignor  |  BANDA FAR       | ignor      |  <- lookahead (ve la curva antes)
#  | 30%    |  (ROI_FAR_FRAC)  | 30%        |
#  +--------+------------------+------------+
#  | ignor  |  BANDA NEAR      | ignor      |  <- control principal (linea cercana)
#  | 30%    |  (resto)         | 30%        |
#  +--------+------------------+------------+
#
# Error combinado:
#   error = NEAR_W * error_near + FAR_W * error_far
#
# La banda FAR anticipa la curva y "pre-gira" antes de llegar.
# La banda NEAR corrige la posicion actual.
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

LINEAR_VEL  = 0.13
MAX_ANGULAR = 0.60

KP = 1.4
KI = 0.03
KD = 0.15
MAX_INTEGRAL = 0.35

# ROI vertical: ignorar parte superior del frame
ROI_TOP_FRAC = 0.45    # con camara mas alta, la linea ocupa mas frame

# ROI horizontal: zona activa central
ROI_LEFT_FRAC  = 0.30
ROI_RIGHT_FRAC = 0.70

# Lookahead: la banda FAR ocupa los primeros FAR_FRAC del ROI activo
# La banda NEAR ocupa el resto (la parte inferior)
FAR_FRAC = 0.45    # fraccion DENTRO del ROI que es "lejos"

# Pesos del error combinado (deben sumar 1.0)
NEAR_W = 0.65      # peso del error cercano (control principal)
FAR_W  = 0.35      # peso del error lejano  (lookahead / anticipacion)

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

    def _threshold(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            ADAPT_BLOCK, ADAPT_C,
        )
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

    def _closest_centroid(self, mask, roi_w, roi_h):
        """Contorno mas cercano al centro-inferior (la linea que pisa el robot)."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        best   = None
        best_d = float('inf')
        ref_x  = roi_w / 2.0
        ref_y  = float(roi_h)

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

        return best   # (contour, cx, cy) o None

    def process(self, frame: np.ndarray):
        h, w = frame.shape[:2]

        roi_y0  = int(h * ROI_TOP_FRAC)
        roi_x0  = int(w * ROI_LEFT_FRAC)
        roi_x1  = int(w * ROI_RIGHT_FRAC)
        roi_w   = roi_x1 - roi_x0
        roi_h   = h - roi_y0

        # Separacion entre banda FAR y NEAR
        split_y = roi_y0 + int(roi_h * FAR_FRAC)

        # ---- Banda FAR (lookahead) ----
        roi_far  = frame[roi_y0:split_y, roi_x0:roi_x1]
        mask_far = self._threshold(roi_far) if roi_far.size > 0 else None

        found_far = False
        cx_far    = roi_w // 2
        cnt_far   = None

        if mask_far is not None:
            res = self._closest_centroid(mask_far, roi_w, roi_far.shape[0])
            if res is not None:
                cnt_far, cx_far, _ = res
                found_far = True

        # ---- Banda NEAR (control) ----
        roi_near  = frame[split_y:h, roi_x0:roi_x1]
        mask_near = self._threshold(roi_near)

        found_near = False
        cx_near    = roi_w // 2
        cnt_near   = None

        if mask_near is not None:
            res = self._closest_centroid(mask_near, roi_w, roi_near.shape[0])
            if res is not None:
                cnt_near, cx_near, _ = res
                found_near = True

        # ---- Error combinado ----
        found = found_near or found_far

        err_near = float((roi_w / 2 - cx_near) / (roi_w / 2))
        err_far  = float((roi_w / 2 - cx_far)  / (roi_w / 2))

        if found_near and found_far:
            error_norm = NEAR_W * err_near + FAR_W * err_far
        elif found_near:
            error_norm = err_near
        elif found_far:
            error_norm = err_far * 0.6   # solo lookahead: reaccion suave
        else:
            error_norm = 0.0

        error_norm = max(-1.0, min(1.0, error_norm))

        # ---------------------------------------------------------------
        # Debug
        # ---------------------------------------------------------------
        debug = frame.copy()

        # Oscurecer zona superior ignorada
        ov = debug.copy()
        cv2.rectangle(ov, (0, 0), (w, roi_y0), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.65, debug, 0.35, 0, debug)

        # Oscurecer franjas laterales
        ov2 = debug.copy()
        cv2.rectangle(ov2, (0, roi_y0), (roi_x0, h), (30, 0, 80), -1)
        cv2.rectangle(ov2, (roi_x1, roi_y0), (w, h), (30, 0, 80), -1)
        cv2.addWeighted(ov2, 0.55, debug, 0.45, 0, debug)

        # Overlay verde mascara FAR
        if mask_far is not None:
            mc = np.zeros_like(roi_far)
            mc[mask_far > 0] = (0, 200, 80)
            debug[roi_y0:split_y, roi_x0:roi_x1] = cv2.addWeighted(
                debug[roi_y0:split_y, roi_x0:roi_x1], 0.5, mc, 0.5, 0
            )

        # Overlay verde mascara NEAR
        mc2 = np.zeros_like(roi_near)
        mc2[mask_near > 0] = (0, 255, 0)
        debug[split_y:h, roi_x0:roi_x1] = cv2.addWeighted(
            debug[split_y:h, roi_x0:roi_x1], 0.45, mc2, 0.55, 0
        )

        # Contorno FAR en verde oscuro
        if cnt_far is not None and found_far:
            s = cnt_far.copy()
            s[:, :, 0] += roi_x0
            s[:, :, 1] += roi_y0
            cv2.drawContours(debug, [s], -1, (0, 180, 60), 2)

        # Contorno NEAR en cian
        if cnt_near is not None and found_near:
            s = cnt_near.copy()
            s[:, :, 0] += roi_x0
            s[:, :, 1] += split_y
            cv2.drawContours(debug, [s], -1, (0, 255, 255), 3)

        # Bordes del ROI activo
        cv2.line(debug, (roi_x0, roi_y0), (roi_x1, roi_y0), (0, 220, 220), 2)   # top
        cv2.line(debug, (roi_x0, roi_y0), (roi_x0, h),      (0, 220, 220), 2)   # izq
        cv2.line(debug, (roi_x1, roi_y0), (roi_x1, h),      (0, 220, 220), 2)   # der
        # Division FAR / NEAR
        cv2.line(debug, (roi_x0, split_y), (roi_x1, split_y), (255, 200, 0), 1)

        # Etiquetas de banda
        cv2.putText(debug, 'FAR', (roi_x0 + 4, roi_y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 80), 1)
        cv2.putText(debug, 'NEAR', (roi_x0 + 4, split_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Centro de referencia
        cx_center_abs = (roi_x0 + roi_x1) // 2
        cv2.line(debug, (cx_center_abs, roi_y0), (cx_center_abs, h), (255, 100, 0), 2)

        # Centroides
        if found_far:
            cx_far_abs = roi_x0 + cx_far
            cy_far_abs = roi_y0 + (split_y - roi_y0) // 2
            cv2.circle(debug, (cx_far_abs, cy_far_abs), 8, (0, 180, 60), -1)

        if found_near:
            cx_near_abs = roi_x0 + cx_near
            cy_near_abs = split_y + (h - split_y) // 2
            cv2.circle(debug, (cx_near_abs, cy_near_abs), 10, (0, 0, 255), -1)
            cv2.circle(debug, (cx_near_abs, cy_near_abs), 10, (255, 255, 255), 2)

        # Flecha de error combinado
        arr_y     = h - 20
        cx_result = cx_center_abs + int(-error_norm * (roi_w / 2))
        arr_color = (0, 255, 0) if found else (80, 80, 80)
        cv2.arrowedLine(debug, (cx_center_abs, arr_y), (cx_result, arr_y),
                        arr_color, 3, tipLength=0.2)

        # Texto
        txt   = 'err={:+.3f}'.format(error_norm) if found else 'SIN LINEA'
        color = (0, 255, 120) if found else (0, 60, 255)
        cv2.putText(debug, txt, (roi_x0, roi_y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(debug, txt, (roi_x0, roi_y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Mini-preview NEAR en esquina inferior derecha
        near_h = h - split_y
        if near_h > 0:
            th = near_h // 3
            tw = roi_w  // 3
            thumb = cv2.resize(mask_near, (tw, th))
            x0t = w - tw - 4
            y0t = h - th - 4
            debug[y0t:y0t + th, x0t:x0t + tw] = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(debug, (x0t, y0t), (x0t + tw, y0t + th), (150, 150, 150), 1)
            cv2.putText(debug, 'NEAR', (x0t + 2, y0t + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

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
            'NEAR_W={} FAR_W={}'.format(KP, KI, KD, LINEAR_VEL, NEAR_W, FAR_W)
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
