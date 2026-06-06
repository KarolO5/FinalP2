#!/usr/bin/env python3
# =============================================================================
# line_follower_cv.py  --  Seguidor de linea con lookahead
# =============================================================================
#
# ROI rectangular + 2 bandas horizontales:
#
#  +--------+------------------+--------+
#  |        |   zona ignorada  |        |  <- ROI_TOP_FRAC
#  +--------+------------------+--------+
#  | ignor  |  BANDA FAR       | ignor  |  <- FAR_FRAC del ROI
#  | 30%    |  (lookahead)     | 30%    |
#  +--------+------------------+--------+
#  | ignor  |  BANDA NEAR      | ignor  |  <- resto del ROI
#  | 30%    |  (control)       | 30%    |
#  +--------+------------------+--------+
#
# Error = NEAR_W * error_near + FAR_W * error_far
# FAR_W pequeño: solo anticipa curvas sin dominar el control.
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
MAX_ANGULAR = 0.55

KP = 1.2
KI = 0.02
KD = 0.10
MAX_INTEGRAL = 0.30

AMARILLO_FACTOR = 0.60

# ROI vertical
ROI_TOP_FRAC = 0.58

# ROI horizontal (zona activa central)
ROI_LEFT_FRAC  = 0.30
ROI_RIGHT_FRAC = 0.70

# Lookahead: fraccion DENTRO del ROI que es banda FAR (la de arriba)
FAR_FRAC = 0.40

# Pesos -- FAR pequeño para no jalar a lineas laterales
NEAR_W = 0.85
FAR_W  = 0.15

# Umbral adaptativo
ADAPT_BLOCK = 25
ADAPT_C     = 6

# Morfologia
MORPH_KSIZE = (7, 7)

MIN_CONTOUR_AREA = 60

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

    def _best_contour(self, contours, roi_w, roi_h):
        """Contorno cuyo centroide esta mas cerca del centro-inferior."""
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

        return best

    def process(self, frame: np.ndarray):
        h, w = frame.shape[:2]

        roi_y0 = int(h * ROI_TOP_FRAC)
        roi_x0 = int(w * ROI_LEFT_FRAC)
        roi_x1 = int(w * ROI_RIGHT_FRAC)
        roi_w  = roi_x1 - roi_x0
        roi_h  = h - roi_y0
        split_y = roi_y0 + int(roi_h * FAR_FRAC)

        cx_center = w // 2

        # ---- Banda FAR ----
        roi_far   = frame[roi_y0:split_y, roi_x0:roi_x1]
        mask_far  = self._threshold(roi_far) if roi_far.size > 0 else np.zeros((1,1), dtype=np.uint8)
        far_h     = roi_far.shape[0] if roi_far.size > 0 else 1

        found_far = False
        cx_far    = roi_w // 2
        cnt_far   = None
        res = self._best_contour(
            cv2.findContours(mask_far, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
            roi_w, far_h
        )
        if res:
            cnt_far, cx_far, _ = res
            found_far = True

        # ---- Banda NEAR ----
        roi_near  = frame[split_y:h, roi_x0:roi_x1]
        mask_near = self._threshold(roi_near)
        near_h    = roi_near.shape[0]

        found_near = False
        cx_near    = roi_w // 2
        cnt_near   = None
        res2 = self._best_contour(
            cv2.findContours(mask_near, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
            roi_w, near_h
        )
        if res2:
            cnt_near, cx_near, _ = res2
            found_near = True

        # ---- Error combinado (referencia = centro del frame) ----
        half = roi_w / 2.0
        err_near = (half - cx_near) / half
        err_far  = (half - cx_far)  / half

        found = found_near or found_far

        if found_near and found_far:
            error_norm = NEAR_W * err_near + FAR_W * err_far
        elif found_near:
            error_norm = err_near
        elif found_far:
            error_norm = err_far * 0.5
        else:
            error_norm = 0.0

        error_norm = max(-1.0, min(1.0, error_norm))

        # ---------------------------------------------------------------
        # Debug
        # ---------------------------------------------------------------
        debug = frame.copy()

        # Zona superior oscurecida
        ov = debug.copy()
        cv2.rectangle(ov, (0, 0), (w, roi_y0), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.65, debug, 0.35, 0, debug)

        # Franjas laterales
        ov2 = debug.copy()
        cv2.rectangle(ov2, (0, roi_y0), (roi_x0, h), (30, 0, 80), -1)
        cv2.rectangle(ov2, (roi_x1, roi_y0), (w, h), (30, 0, 80), -1)
        cv2.addWeighted(ov2, 0.55, debug, 0.45, 0, debug)

        # Overlay mascara FAR (verde oscuro)
        if mask_far.size > 1:
            mc = np.zeros((split_y - roi_y0, roi_w, 3), dtype=np.uint8)
            mc[mask_far > 0] = (0, 180, 60)
            debug[roi_y0:split_y, roi_x0:roi_x1] = cv2.addWeighted(
                debug[roi_y0:split_y, roi_x0:roi_x1], 0.5, mc, 0.5, 0
            )

        # Overlay mascara NEAR (verde brillante)
        mc2 = np.zeros((near_h, roi_w, 3), dtype=np.uint8)
        mc2[mask_near > 0] = (0, 255, 0)
        debug[split_y:h, roi_x0:roi_x1] = cv2.addWeighted(
            debug[split_y:h, roi_x0:roi_x1], 0.45, mc2, 0.55, 0
        )

        # Contorno FAR
        if cnt_far is not None and found_far:
            s = cnt_far.copy()
            s[:, :, 0] += roi_x0
            s[:, :, 1] += roi_y0
            cv2.drawContours(debug, [s], -1, (0, 180, 60), 2)

        # Contorno NEAR
        if cnt_near is not None and found_near:
            s = cnt_near.copy()
            s[:, :, 0] += roi_x0
            s[:, :, 1] += split_y
            cv2.drawContours(debug, [s], -1, (0, 255, 255), 3)

        # Bordes del ROI
        cv2.line(debug, (roi_x0, roi_y0), (roi_x1, roi_y0), (0, 220, 220), 2)
        cv2.line(debug, (roi_x0, roi_y0), (roi_x0, h),      (0, 220, 220), 2)
        cv2.line(debug, (roi_x1, roi_y0), (roi_x1, h),      (0, 220, 220), 2)
        # Division FAR/NEAR
        cv2.line(debug, (roi_x0, split_y), (roi_x1, split_y), (200, 180, 0), 1)
        cv2.putText(debug, 'FAR',  (roi_x0 + 4, roi_y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 80), 1)
        cv2.putText(debug, 'NEAR', (roi_x0 + 4, split_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Centro de referencia (azul = w//2)
        cv2.line(debug, (cx_center, roi_y0), (cx_center, h), (255, 100, 0), 2)

        # Centroide FAR
        if found_far:
            cv2.circle(debug, (roi_x0 + cx_far, roi_y0 + far_h // 2), 7, (0, 180, 60), -1)

        # Centroide NEAR
        if found_near:
            cx_near_abs = roi_x0 + cx_near
            cy_near_abs = split_y + near_h // 2
            cv2.circle(debug, (cx_near_abs, cy_near_abs), 11, (0, 0, 255), -1)
            cv2.circle(debug, (cx_near_abs, cy_near_abs), 11, (255, 255, 255), 2)

        # Flecha de error
        cx_result = cx_center + int(-error_norm * (roi_w / 2))
        arr_color = (0, 255, 0) if found else (80, 80, 80)
        cv2.arrowedLine(debug, (cx_center, h - 20),
                        (cx_result if found else cx_center, h - 20),
                        arr_color, 3, tipLength=0.2)

        # Texto
        txt   = 'err={:+.3f}'.format(error_norm) if found else 'SIN LINEA'
        color = (0, 255, 120) if found else (0, 60, 255)
        cv2.putText(debug, txt, (roi_x0, roi_y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(debug, txt, (roi_x0, roi_y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Mini-preview NEAR
        if near_h > 0:
            th = near_h // 3
            tw = roi_w  // 3
            thumb = cv2.resize(mask_near, (tw, th))
            x0t, y0t = w - tw - 4, h - th - 4
            debug[y0t:y0t+th, x0t:x0t+tw] = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(debug, (x0t, y0t), (x0t+tw, y0t+th), (150,150,150), 1)
            cv2.putText(debug, 'NEAR', (x0t+2, y0t+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)

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

        self._detector    = ContourLineDetector()
        self._bridge      = CvBridge()
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
            'LineFollowerCV | KP={} KI={} KD={} v={} | NEAR={} FAR={}'.format(
                KP, KI, KD, LINEAR_VEL, NEAR_W, FAR_W)
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

        dbg_msg = self._bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
        dbg_msg.header = msg.header
        self._pub_dbg.publish(dbg_msg)

        err_msg = Float32()
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

            d  = (error - self._prev_error) / dt
            u  = KP * error + KI * self._integral + KD * d
            u  = max(-MAX_ANGULAR, min(MAX_ANGULAR, u))

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
                    'Linea perdida ({} frames)'.format(self._frames_lost))
                cmd.linear.x  = 0.0
                cmd.angular.z = (RECOVERY_OMEGA if self._last_error >= 0
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
