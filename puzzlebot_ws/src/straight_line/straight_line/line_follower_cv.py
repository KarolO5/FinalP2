#!/usr/bin/env python3
# =============================================================================
# line_follower_cv.py  --  Seguidor de linea con tracking de posicion
# =============================================================================
#
# ROI rectangular + 2 bandas + tracking de centroide:
#
#  +--------+------------------+--------+
#  |        |   zona ignorada  |        |  <- ROI_TOP_FRAC
#  +--------+------------------+--------+
#  | ignor  |  BANDA FAR       | ignor  |  <- FAR_FRAC del ROI
#  +--------+------------------+--------+
#  | ignor  |  BANDA NEAR      | ignor  |  <- resto del ROI
#  +--------+------------------+--------+
#
# TRACKING:
#   En vez de buscar siempre desde el centro, el detector usa la posicion
#   del ultimo centroide valido como punto de referencia.
#   Esto evita que en curvas el robot salte a la linea lateral:
#   si la linea se mueve a la izquierda, el detector la sigue alla,
#   en lugar de volver al centro y encontrar la linea derecha.
#
# FAR se usa solo como lookahead suave (FAR_W=0.12).
# FAR tiene MIN_CONTOUR_AREA mayor para ignorar ruido del piso.
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

ROI_TOP_FRAC   = 0.58
ROI_LEFT_FRAC  = 0.30
ROI_RIGHT_FRAC = 0.70

FAR_FRAC = 0.40
NEAR_W   = 0.88
FAR_W    = 0.12

ADAPT_BLOCK = 25
ADAPT_C     = 6
MORPH_KSIZE = (7, 7)

MIN_CONTOUR_AREA_NEAR = 60    # banda cercana: admite contornos pequenos
MIN_CONTOUR_AREA_FAR  = 400   # banda lejana: solo contornos grandes (ignora ruido)

RECOVERY_FRAMES = 25
RECOVERY_OMEGA  = 0.20

# Radio maximo de salto del centroide entre frames (px).
# Si el nuevo centroide esta mas lejos, se rechaza (posible linea erronea).
MAX_JUMP_PX = 45   # reducido: rechaza saltos bruscos a lineas laterales


# -----------------------------------------------------------------------
# DETECTOR
# -----------------------------------------------------------------------

class ContourLineDetector:

    def __init__(self, roi_w):
        self._kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KSIZE)
        # Ultimo centroide conocido en coordenadas del ROI (x absoluto en frame)
        self._last_cx = None
        self._roi_w   = roi_w

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

    def _best_contour(self, contours, ref_x, ref_y, min_area):
        """
        Contorno cuyo centroide esta mas cerca de (ref_x, ref_y).
        ref_x es la posicion anterior del centroide o el centro del ROI.
        """
        best   = None
        best_d = float('inf')

        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
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

        roi_y0  = int(h * ROI_TOP_FRAC)
        roi_x0  = int(w * ROI_LEFT_FRAC)
        roi_x1  = int(w * ROI_RIGHT_FRAC)
        roi_w   = roi_x1 - roi_x0
        roi_h   = h - roi_y0
        split_y = roi_y0 + int(roi_h * FAR_FRAC)

        cx_center = w // 2

        # Referencia para la banda NEAR:
        # si tenemos posicion previa, usarla; si no, centro del ROI
        if self._last_cx is not None:
            ref_near_x = self._last_cx - roi_x0
        else:
            ref_near_x = roi_w / 2.0
        ref_near_x = max(0.0, min(float(roi_w), ref_near_x))

        # ---- Banda NEAR ----
        near_h    = h - split_y
        roi_near  = frame[split_y:h, roi_x0:roi_x1]
        mask_near = self._threshold(roi_near)

        found_near = False
        cx_near    = int(ref_near_x)
        cnt_near   = None

        cnts_near, _ = cv2.findContours(
            mask_near, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        res = self._best_contour(cnts_near, ref_near_x, float(near_h),
                                 MIN_CONTOUR_AREA_NEAR)
        if res:
            cnt_near_cand, cx_cand, _ = res
            cx_abs_cand = roi_x0 + cx_cand
            # Validar que no sea un salto demasiado grande
            if (self._last_cx is None or
                    abs(cx_abs_cand - self._last_cx) <= MAX_JUMP_PX):
                cnt_near    = cnt_near_cand
                cx_near     = cx_cand
                found_near  = True
                self._last_cx = cx_abs_cand

        # ---- Banda FAR ----
        far_h     = split_y - roi_y0
        roi_far   = frame[roi_y0:split_y, roi_x0:roi_x1]
        mask_far  = self._threshold(roi_far) if roi_far.size > 0 \
                    else np.zeros((1, 1), dtype=np.uint8)

        found_far = False
        cx_far    = int(ref_near_x)
        cnt_far   = None

        cnts_far, _ = cv2.findContours(
            mask_far, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        res2 = self._best_contour(cnts_far, ref_near_x, float(far_h),
                                  MIN_CONTOUR_AREA_FAR)
        if res2:
            cnt_far, cx_far, _ = res2
            found_far = True

        # ---- Error combinado ----
        found    = found_near or found_far
        half     = roi_w / 2.0
        err_near = (half - cx_near) / half
        err_far  = (half - cx_far)  / half

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

        ov = debug.copy()
        cv2.rectangle(ov, (0, 0), (w, roi_y0), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.65, debug, 0.35, 0, debug)

        ov2 = debug.copy()
        cv2.rectangle(ov2, (0, roi_y0), (roi_x0, h), (30, 0, 80), -1)
        cv2.rectangle(ov2, (roi_x1, roi_y0), (w, h), (30, 0, 80), -1)
        cv2.addWeighted(ov2, 0.55, debug, 0.45, 0, debug)

        # Mascaras
        if mask_far.size > 1 and far_h > 0:
            mc = np.zeros((far_h, roi_w, 3), dtype=np.uint8)
            mc[mask_far > 0] = (0, 160, 50)
            debug[roi_y0:split_y, roi_x0:roi_x1] = cv2.addWeighted(
                debug[roi_y0:split_y, roi_x0:roi_x1], 0.55, mc, 0.45, 0
            )
        mc2 = np.zeros((near_h, roi_w, 3), dtype=np.uint8)
        mc2[mask_near > 0] = (0, 255, 0)
        debug[split_y:h, roi_x0:roi_x1] = cv2.addWeighted(
            debug[split_y:h, roi_x0:roi_x1], 0.45, mc2, 0.55, 0
        )

        # Contornos
        if cnt_far is not None and found_far:
            s = cnt_far.copy(); s[:,:,0] += roi_x0; s[:,:,1] += roi_y0
            cv2.drawContours(debug, [s], -1, (0, 160, 50), 2)
        if cnt_near is not None and found_near:
            s = cnt_near.copy(); s[:,:,0] += roi_x0; s[:,:,1] += split_y
            cv2.drawContours(debug, [s], -1, (0, 255, 255), 3)

        # Bordes ROI
        cv2.line(debug, (roi_x0, roi_y0), (roi_x1, roi_y0), (0, 220, 220), 2)
        cv2.line(debug, (roi_x0, roi_y0), (roi_x0, h),      (0, 220, 220), 2)
        cv2.line(debug, (roi_x1, roi_y0), (roi_x1, h),      (0, 220, 220), 2)
        cv2.line(debug, (roi_x0, split_y), (roi_x1, split_y), (180, 160, 0), 1)
        cv2.putText(debug, 'FAR',  (roi_x0+4, roi_y0+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,160,50), 1)
        cv2.putText(debug, 'NEAR', (roi_x0+4, split_y+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

        # Centro (azul = w//2)
        cv2.line(debug, (cx_center, roi_y0), (cx_center, h), (255, 100, 0), 2)

        # Linea de tracking (donde buscamos la linea)
        track_abs = roi_x0 + int(ref_near_x)
        cv2.line(debug, (track_abs, split_y), (track_abs, h), (200, 200, 0), 1)

        # Centroides
        if found_far:
            cv2.circle(debug, (roi_x0+cx_far, roi_y0+far_h//2), 7, (0,160,50), -1)
        if found_near:
            cx_na = roi_x0 + cx_near
            cv2.circle(debug, (cx_na, split_y+near_h//2), 11, (0,0,255), -1)
            cv2.circle(debug, (cx_na, split_y+near_h//2), 11, (255,255,255), 2)

        # Flecha de error
        cx_result = cx_center + int(-error_norm * (roi_w / 2))
        arr_color = (0, 255, 0) if found else (80, 80, 80)
        cv2.arrowedLine(debug, (cx_center, h-20),
                        (cx_result if found else cx_center, h-20),
                        arr_color, 3, tipLength=0.2)

        txt   = 'err={:+.3f}'.format(error_norm) if found else 'SIN LINEA'
        color = (0, 255, 120) if found else (0, 60, 255)
        cv2.putText(debug, txt, (roi_x0, roi_y0-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 5)
        cv2.putText(debug, txt, (roi_x0, roi_y0-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Mini-preview
        if near_h > 0:
            th = near_h//3; tw = roi_w//3
            thumb = cv2.resize(mask_near, (tw, th))
            x0t, y0t = w-tw-4, h-th-4
            debug[y0t:y0t+th, x0t:x0t+tw] = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(debug,(x0t,y0t),(x0t+tw,y0t+th),(150,150,150),1)
            cv2.putText(debug,'NEAR',(x0t+2,y0t+12),
                        cv2.FONT_HERSHEY_SIMPLEX,0.35,(200,200,200),1)

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

        # Calcular roi_w para pasarlo al detector
        # Se calcula de manera aproximada (se actualiza con el primer frame)
        self._detector    = ContourLineDetector(roi_w=200)
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
            'LineFollowerCV | KP={} KI={} KD={} v={} | NEAR={} FAR={} jump={}px'.format(
                KP, KI, KD, LINEAR_VEL, NEAR_W, FAR_W, MAX_JUMP_PX)
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

            d = (error - self._prev_error) / dt
            u = KP * error + KI * self._integral + KD * d
            u = max(-MAX_ANGULAR, min(MAX_ANGULAR, u))

            self._prev_error = error
            self._last_error = error

            cmd.linear.x  = LINEAR_VEL * vel_factor
            cmd.angular.z = u

        else:
            self._frames_lost += 1
            self._integral     = 0.0
            # Resetear tracking al perder la linea para no quedar enganchado
            if self._frames_lost == 1:
                self._detector._last_cx = None

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
