#!/usr/bin/env python3
# =============================================================================
# line_follower_cv.py  --  Seguidor de linea + maquina de estados para senales
# =============================================================================
#
# Estados:
#   FOLLOWING    : PID normal siguiendo la linea
#   STOP_WAIT    : detenido 3 seg (senal STOP)
#   EXEC_LEFT    : girando izquierda en interseccion (TurnL)
#   EXEC_RIGHT   : girando derecha en interseccion (TurnR)
#   EXEC_AHEAD   : recto en interseccion (AOnly / Round)
#
# Modificadores de velocidad (no cambian estado):
#   semaforo rojo    -> parado hasta verde
#   semaforo amarillo -> vel * 0.60
#   Crossing visible  -> vel * 0.55
#   Give visible      -> vel * 0.55
#
# Interseccion detectada cuando found=False por >= INTERSECT_FRAMES frames
# y hay una accion pendiente (TurnL/TurnR/AOnly/Round).
#
# TOPICOS
# -------
#   Sub : /image/raw         [sensor_msgs/Image]
#   Sub : /semaforo/estado   [std_msgs/String]
#   Sub : /sign/deteccion    [std_msgs/String]
#   Pub : /cmd_vel           [geometry_msgs/Twist]
#   Pub : /vision/debug_img  [sensor_msgs/Image]
#   Pub : /vision/error      [std_msgs/Float32]
# =============================================================================

import time
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
# PARAMETROS PID y ROI
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

MIN_CONTOUR_AREA_NEAR = 60
MIN_CONTOUR_AREA_FAR  = 400

RECOVERY_FRAMES  = 25
RECOVERY_OMEGA   = 0.20
MAX_JUMP_PX      = 45

# -----------------------------------------------------------------------
# PARAMETROS DE SENALES / INTERSECCION
# -----------------------------------------------------------------------

# Frames sin linea para detectar interseccion
INTERSECT_FRAMES = 12

# STOP: segundos detenido
STOP_DURATION = 3.0

# Ejecucion de giro en interseccion
TURN_LINEAR  = 0.08    # vel lineal durante el giro
TURN_OMEGA_L = +0.50   # vel angular izquierda
TURN_OMEGA_R = -0.50   # vel angular derecha
EXEC_TIMEOUT = 6.0     # segundos max antes de volver a buscar

# Factor de velocidad para Crossing y Give
SLOW_SIGN_FACTOR = 0.55

# Estados
ST_FOLLOWING = 'following'
ST_STOP_WAIT = 'stop_wait'
ST_EXEC_L    = 'exec_left'
ST_EXEC_R    = 'exec_right'
ST_EXEC_FWD  = 'exec_ahead'


# -----------------------------------------------------------------------
# DETECTOR (igual que antes con tracking)
# -----------------------------------------------------------------------

class ContourLineDetector:

    def __init__(self):
        self._kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KSIZE)
        self._last_cx = None

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

        ref_near_x = (self._last_cx - roi_x0) if self._last_cx is not None else roi_w / 2.0
        ref_near_x = max(0.0, min(float(roi_w), ref_near_x))

        # Banda NEAR
        near_h   = h - split_y
        roi_near = frame[split_y:h, roi_x0:roi_x1]
        mask_near = self._threshold(roi_near)
        found_near = False
        cx_near    = int(ref_near_x)
        cnt_near   = None

        cnts_near, _ = cv2.findContours(mask_near, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        res = self._best_contour(cnts_near, ref_near_x, float(near_h), MIN_CONTOUR_AREA_NEAR)
        if res:
            cnt_c, cx_c, _ = res
            cx_abs_c = roi_x0 + cx_c
            if self._last_cx is None or abs(cx_abs_c - self._last_cx) <= MAX_JUMP_PX:
                cnt_near    = cnt_c
                cx_near     = cx_c
                found_near  = True
                self._last_cx = cx_abs_c

        # Banda FAR
        far_h    = split_y - roi_y0
        roi_far  = frame[roi_y0:split_y, roi_x0:roi_x1]
        mask_far = self._threshold(roi_far) if roi_far.size > 0 else np.zeros((1, 1), np.uint8)
        found_far = False
        cx_far    = int(ref_near_x)
        cnt_far   = None

        cnts_far, _ = cv2.findContours(mask_far, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        res2 = self._best_contour(cnts_far, ref_near_x, float(far_h), MIN_CONTOUR_AREA_FAR)
        if res2:
            cnt_far, cx_far, _ = res2
            found_far = True

        # Error combinado
        found = found_near or found_far
        half  = roi_w / 2.0
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

        # Debug
        debug = frame.copy()
        ov = debug.copy()
        cv2.rectangle(ov, (0, 0), (w, roi_y0), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.65, debug, 0.35, 0, debug)
        ov2 = debug.copy()
        cv2.rectangle(ov2, (0, roi_y0), (roi_x0, h), (30, 0, 80), -1)
        cv2.rectangle(ov2, (roi_x1, roi_y0), (w, h), (30, 0, 80), -1)
        cv2.addWeighted(ov2, 0.55, debug, 0.45, 0, debug)

        if mask_far.size > 1:
            mc = np.zeros((far_h, roi_w, 3), np.uint8)
            mc[mask_far > 0] = (0, 180, 60)
            debug[roi_y0:split_y, roi_x0:roi_x1] = cv2.addWeighted(
                debug[roi_y0:split_y, roi_x0:roi_x1], 0.5, mc, 0.5, 0)

        mc2 = np.zeros((near_h, roi_w, 3), np.uint8)
        mc2[mask_near > 0] = (0, 255, 0)
        debug[split_y:h, roi_x0:roi_x1] = cv2.addWeighted(
            debug[split_y:h, roi_x0:roi_x1], 0.45, mc2, 0.55, 0)

        if cnt_far is not None and found_far:
            s = cnt_far.copy(); s[:,:,0]+=roi_x0; s[:,:,1]+=roi_y0
            cv2.drawContours(debug, [s], -1, (0, 180, 60), 2)
        if cnt_near is not None and found_near:
            s = cnt_near.copy(); s[:,:,0]+=roi_x0; s[:,:,1]+=split_y
            cv2.drawContours(debug, [s], -1, (0, 255, 255), 3)

        cv2.line(debug, (roi_x0, roi_y0), (roi_x1, roi_y0), (0, 220, 220), 2)
        cv2.line(debug, (roi_x0, roi_y0), (roi_x0, h),      (0, 220, 220), 2)
        cv2.line(debug, (roi_x1, roi_y0), (roi_x1, h),      (0, 220, 220), 2)
        cv2.line(debug, (roi_x0, split_y), (roi_x1, split_y), (200, 180, 0), 1)
        cv2.line(debug, (cx_center, roi_y0), (cx_center, h), (255, 100, 0), 2)

        if found_near:
            cv2.circle(debug, (roi_x0 + cx_near, split_y + near_h // 2), 11, (0, 0, 255), -1)
            cv2.circle(debug, (roi_x0 + cx_near, split_y + near_h // 2), 11, (255, 255, 255), 2)
        if found_far:
            cv2.circle(debug, (roi_x0 + cx_far,  roi_y0 + far_h // 2),   7, (0, 180, 60), -1)

        cx_result = cx_center + int(-error_norm * (roi_w / 2))
        arr_color = (0, 255, 0) if found else (80, 80, 80)
        cv2.arrowedLine(debug, (cx_center, h - 20),
                        (cx_result if found else cx_center, h - 20),
                        arr_color, 3, tipLength=0.2)

        txt   = 'err={:+.3f}'.format(error_norm) if found else 'SIN LINEA'
        color = (0, 255, 120) if found else (0, 60, 255)
        cv2.putText(debug, txt, (roi_x0, roi_y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(debug, txt, (roi_x0, roi_y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if near_h > 0:
            th = near_h // 3; tw = roi_w // 3
            thumb = cv2.resize(mask_near, (tw, th))
            x0t, y0t = w - tw - 4, h - th - 4
            debug[y0t:y0t+th, x0t:x0t+tw] = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(debug, (x0t, y0t), (x0t+tw, y0t+th), (150,150,150), 1)

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

        # PID
        self._prev_error  = 0.0
        self._integral    = 0.0
        self._prev_time   = None
        self._last_error  = 0.0
        self._frames_lost = 0

        # Semaforo
        self._semaforo = 'ninguno'

        # Senal de trafico
        self._sign         = 'ninguno'
        self._pending      = None   # 'left' | 'right' | 'ahead' | None
        self._slow_sign    = False  # Crossing o Give visibles

        # Maquina de estados
        self._state      = ST_FOLLOWING
        self._state_t0   = 0.0     # tiempo de entrada al estado

        self._pub_cmd = self.create_publisher(Twist,   '/cmd_vel',          qos_be)
        self._pub_dbg = self.create_publisher(Image,   '/vision/debug_img', 10)
        self._pub_err = self.create_publisher(Float32, '/vision/error',     10)

        self.create_subscription(Image,  '/image/raw',       self._image_cb,    qos_be)
        self.create_subscription(String, '/semaforo/estado', self._semaforo_cb, 10)
        self.create_subscription(String, '/sign/deteccion',  self._sign_cb,     10)

        self.get_logger().info(
            'LineFollowerCV listo | KP={} KI={} KD={} v={}'.format(KP, KI, KD, LINEAR_VEL)
        )

    # ---- Callbacks -------------------------------------------------------

    def _semaforo_cb(self, msg: String):
        nuevo = msg.data
        if nuevo != self._semaforo:
            self.get_logger().info('Semaforo: {} -> {}'.format(self._semaforo, nuevo))
            if nuevo != 'rojo':
                self._integral = 0.0
        self._semaforo = nuevo

    def _sign_cb(self, msg: String):
        sign = msg.data
        if sign == self._sign:
            return

        self.get_logger().info('Senal recibida: {} -> {}'.format(self._sign, sign))
        self._sign = sign

        if sign == 'STOP' and self._state == ST_FOLLOWING:
            self._enter_state(ST_STOP_WAIT)

        elif sign == 'TurnL':
            self._pending   = 'left'
            self._slow_sign = False

        elif sign == 'TurnR':
            self._pending   = 'right'
            self._slow_sign = False

        elif sign in ('AOnly', 'Round'):
            self._pending   = 'ahead'
            self._slow_sign = False

        elif sign in ('Crossing', 'Give'):
            self._slow_sign = True

        elif sign == 'ninguno':
            self._slow_sign = False

    def _image_cb(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn('cv_bridge: {}'.format(e))
            return

        error_norm, found, debug_frame = self._detector.process(frame)

        # Anotar estado en debug
        state_color = {
            ST_FOLLOWING: (0, 255, 120),
            ST_STOP_WAIT: (0, 0, 220),
            ST_EXEC_L:    (255, 200, 0),
            ST_EXEC_R:    (255, 100, 0),
            ST_EXEC_FWD:  (0, 200, 255),
        }.get(self._state, (200, 200, 200))
        state_txt = 'STATE: {}  SIGN: {}  PEND: {}'.format(
            self._state.upper(), self._sign, self._pending or '-')
        cv2.putText(debug_frame, state_txt, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(debug_frame, state_txt, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1)

        dbg_msg        = self._bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
        dbg_msg.header = msg.header
        self._pub_dbg.publish(dbg_msg)

        err_msg      = Float32()
        err_msg.data = float(error_norm)
        self._pub_err.publish(err_msg)

        self._run(error_norm, found)

    # ---- Maquina de estados ----------------------------------------------

    def _enter_state(self, state: str):
        self._state   = state
        self._state_t0 = time.monotonic()
        self.get_logger().info('Estado: {}'.format(state))

    def _run(self, error: float, found: bool):
        now = time.monotonic()

        # --- STOP_WAIT ---
        if self._state == ST_STOP_WAIT:
            self._pub_cmd.publish(Twist())
            if now - self._state_t0 >= STOP_DURATION:
                self.get_logger().info('STOP terminado, reanudando')
                self._enter_state(ST_FOLLOWING)
            return

        # --- EXEC_LEFT ---
        if self._state == ST_EXEC_L:
            if found or (now - self._state_t0 >= EXEC_TIMEOUT):
                self.get_logger().info('Giro izq terminado')
                self._pending = None
                self._enter_state(ST_FOLLOWING)
            else:
                cmd = Twist()
                cmd.linear.x  = TURN_LINEAR
                cmd.angular.z = TURN_OMEGA_L
                self._pub_cmd.publish(cmd)
            return

        # --- EXEC_RIGHT ---
        if self._state == ST_EXEC_R:
            if found or (now - self._state_t0 >= EXEC_TIMEOUT):
                self.get_logger().info('Giro der terminado')
                self._pending = None
                self._enter_state(ST_FOLLOWING)
            else:
                cmd = Twist()
                cmd.linear.x  = TURN_LINEAR
                cmd.angular.z = TURN_OMEGA_R
                self._pub_cmd.publish(cmd)
            return

        # --- EXEC_AHEAD ---
        if self._state == ST_EXEC_FWD:
            if found or (now - self._state_t0 >= EXEC_TIMEOUT):
                self.get_logger().info('Recto interseccion terminado')
                self._pending = None
                self._enter_state(ST_FOLLOWING)
            else:
                cmd = Twist()
                cmd.linear.x  = LINEAR_VEL
                cmd.angular.z = 0.0
                self._pub_cmd.publish(cmd)
            return

        # --- FOLLOWING ---
        # Deteccion de interseccion
        if not found:
            self._frames_lost += 1
            if self._frames_lost >= INTERSECT_FRAMES and self._pending is not None:
                self.get_logger().info(
                    'Interseccion! Ejecutando: {}'.format(self._pending))
                if self._pending == 'left':
                    self._enter_state(ST_EXEC_L)
                elif self._pending == 'right':
                    self._enter_state(ST_EXEC_R)
                else:
                    self._enter_state(ST_EXEC_FWD)
                return
        else:
            self._frames_lost = 0

        self._control_pid(error, found)

    def _control_pid(self, error: float, found: bool):
        # Prioridad: semaforo rojo
        if self._semaforo == 'rojo':
            self._pub_cmd.publish(Twist())
            self._prev_time = None
            return

        # Factor de velocidad
        vel_factor = 1.0
        if self._semaforo == 'amarillo':
            vel_factor = AMARILLO_FACTOR
        elif self._slow_sign:
            vel_factor = SLOW_SIGN_FACTOR

        now = time.monotonic()
        dt  = (now - self._prev_time) if self._prev_time is not None else 0.033
        dt  = max(dt, 1e-4)
        self._prev_time = now

        cmd = Twist()

        if found:
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
            self._integral = 0.0
            if self._frames_lost < RECOVERY_FRAMES:
                u             = KP * self._last_error * 0.4
                cmd.linear.x  = LINEAR_VEL * vel_factor * 0.4
                cmd.angular.z = max(-MAX_ANGULAR, min(MAX_ANGULAR, u))
            else:
                self.get_logger().warn('Linea perdida -- buscando')
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
