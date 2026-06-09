#!/usr/bin/env python3
# =============================================================================
# line_follower_cv.py  --  Seguidor de linea + maquina de estados (senales)
# =============================================================================
# Estrategia de vision: ROI rectangular banda unica, contorno mas cercano
# al centro-inferior (seleccion por proximidad, no por area).
#
# Maquina de estados para senales:
#   FOLLOWING  : PID normal
#   STOP_WAIT  : parado 3 seg (senal STOP)
#   EXEC_LEFT  : giro izquierda en interseccion
#   EXEC_RIGHT : giro derecha en interseccion
#   EXEC_AHEAD : recto en interseccion (AOnly/Round)
#
# TOPICOS
#   Sub : /image/raw        [sensor_msgs/Image]
#   Sub : /semaforo/estado  [std_msgs/String]
#   Sub : /sign/deteccion   [std_msgs/String]
#   Pub : /cmd_vel          [geometry_msgs/Twist]
#   Pub : /vision/debug_img [sensor_msgs/Image]
#   Pub : /vision/error     [std_msgs/Float32]
# =============================================================================

import time
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg   import Image
from std_msgs.msg      import Bool, Float32, String
from cv_bridge         import CvBridge
from rclpy.qos         import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# -----------------------------------------------------------------------
# PARAMETROS PID
# -----------------------------------------------------------------------
LINEAR_VEL  = 0.09
MAX_ANGULAR = 0.65       # subido para girar mas rapido en curvas

KP = 0.90               # un poco menos agresivo para reducir latigazos
KI = 0.02
KD = 0.06
MAX_INTEGRAL    = 0.30
AMARILLO_FACTOR = 0.60
SLOW_SIGN_FACTOR = 0.55

# Zona muerta mas grande: en recta ignora errores pequeños -> va mas recto
ERROR_DEADBAND  = 0.09   # era 0.05 -> mas estabilidad en recta

# ROI seguidor de linea
ROI_TOP_FRAC   = 0.78
ROI_LEFT_FRAC  = 0.25
ROI_RIGHT_FRAC = 0.68

# ROI deteccion de amarillo (banda encima del ROI del seguidor)
YELLOW_ROI_TOP_FRAC    = 0.55   # donde empieza el ROI amarillo
YELLOW_ROI_BOTTOM_FRAC = 0.75   # donde termina  (justo encima del seguidor)
YELLOW_ROI_LEFT_FRAC   = 0.25
YELLOW_ROI_RIGHT_FRAC  = 0.68
YELLOW_FILL_THRESH     = 0.15   # 40 %: si mas del 40% del ROI es amarillo -> detener

# Vision
ADAPT_BLOCK = 25
ADAPT_C     = 8
MORPH_KSIZE = (7, 7)
OPEN_KSIZE  = (3, 3)
MIN_CONTOUR_AREA  = 200   # un poco mayor para ignorar punteados pequenos
MAX_ASPECT_RATIO  = 2.5   # ancho/alto maximo: >2.5 = horizontal = junta o punteado

# Recovery
RECOVERY_FRAMES = 25
RECOVERY_OMEGA  = 0.20

# Deteccion lateral (Hough en zonas no usadas por el seguidor)
LATERAL_BIAS_MAX = 0.06   # bias maximo que se suma al error (muy pequeño)
LATERAL_HOUGH_TH = 25     # votos minimos HoughLinesP
LATERAL_MIN_LEN  = 18     # longitud minima de segmento (px)
LATERAL_MAX_GAP  = 10     # gap maximo entre segmentos (px)

# Kalman 1D para suavizar el error antes del PID
KALMAN_Q = 0.005   # ruido del proceso  (bajo = mas suave)
KALMAN_R = 0.08    # ruido de medicion  (alto = mas suave)

# Interseccion y senales
INTERSECT_FRAMES    = 20          # frames sin linea para recovery normal (sin pending)
INTERSECT_PREP_TURN = 3.0         # segundos recto antes de ejecutar giro (TurnL/TurnR)
INTERSECT_PREP_FWD  = 4.0         # segundos recto antes de ejecutar recto (AOnly/Round)
STOP_DURATION    = 3.0
TURN_LINEAR      = 0.0            # spin puro: rueda interior va para atras
TURN_OMEGA_L     = +0.60
TURN_OMEGA_R     = -0.30
TURN_OMEGA_L1     = -0.30
TURN_OMEGA_R1     = +0.60
POST_TURN_TIME   = 0.9          # segundos recto despues del giro para pasar punteados
EXEC_TIMEOUT     = 3.5            # segundos ejecutando el giro ignorando seguidor

# Cooldown: segundos que deben pasar antes de reaccionar a la MISMA senal
SIGN_COOLDOWN = 8.0

ST_FOLLOWING      = 'following'
ST_STOP_WAIT      = 'stop_wait'
ST_INTERSECT_PREP = 'intersect_prep'  # avanza recto antes de ejecutar
ST_EXEC_L         = 'exec_left'
ST_EXEC_R         = 'exec_right'
ST_EXEC_FWD       = 'exec_ahead'
ST_POST_TURN      = 'post_turn'       # avanza recto despues del giro (pasa punteados)


# -----------------------------------------------------------------------
# KALMAN 1D  --  suaviza el error escalar antes del PID
# -----------------------------------------------------------------------
class Kalman1D:
    def __init__(self, q=KALMAN_Q, r=KALMAN_R):
        self._x = 0.0
        self._p = 1.0
        self._q = q
        self._r = r

    def update(self, z):
        p_pred  = self._p + self._q
        k       = p_pred / (p_pred + self._r)
        self._x = self._x + k * (z - self._x)
        self._p = (1.0 - k) * p_pred
        return self._x

    def reset(self, z=0.0):
        self._x = z; self._p = 1.0


# -----------------------------------------------------------------------
# DETECTOR (banda unica, seleccion por proximidad)
# -----------------------------------------------------------------------
class ContourLineDetector:

    def __init__(self):
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KSIZE)
        self._kernel_open  = cv2.getStructuringElement(cv2.MORPH_RECT, OPEN_KSIZE)
        self._kalman       = Kalman1D()

    def _best_contour(self, contours, roi_w, roi_h):
        best = None; best_d = float('inf')
        ref_x = roi_w / 2.0; ref_y = float(roi_h)
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                continue
            # Filtro de aspecto: rechazar contornos muy horizontales
            # (juntas de piezas de pista y linea punteada son mas anchos que altos)
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bh == 0 or bw / float(bh) > MAX_ASPECT_RATIO:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = M['m10'] / M['m00']; cy = M['m01'] / M['m00']
            d  = ((cx - ref_x)**2 + (cy - ref_y)**2)**0.5
            if d < best_d:
                best_d = d; best = (cnt, int(cx), int(cy))
        return best

    def _yellow_ratio(self, frame, h, w):
        """Calcula la fraccion de pixeles amarillos en el ROI superior."""
        y0 = int(h * YELLOW_ROI_TOP_FRAC)
        y1 = int(h * YELLOW_ROI_BOTTOM_FRAC)
        x0 = int(w * YELLOW_ROI_LEFT_FRAC)
        x1 = int(w * YELLOW_ROI_RIGHT_FRAC)
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return 0.0, np.zeros((1, 1), dtype=np.uint8), (y0, y1, x0, x1)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Rango HSV para amarillo
        mask = cv2.inRange(hsv,
                           np.array([18, 80, 80],  dtype=np.uint8),
                           np.array([35, 255, 255], dtype=np.uint8))
        ratio = float(np.count_nonzero(mask)) / float(mask.size)
        return ratio, mask, (y0, y1, x0, x1)

    def _detect_lateral(self, frame, roi_y0, roi_x0, roi_x1, h, w):
        """Detecta lineas en las zonas laterales no usadas por el seguidor.
        Retorna (bias, left_det, right_det).
        bias > 0 -> empuja derecha (pared izquierda detectada)
        bias < 0 -> empuja izquierda (pared derecha detectada)"""

        def _has_lines(zone):
            if zone.size == 0 or zone.shape[1] < 5:
                return False
            gray  = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                    threshold=LATERAL_HOUGH_TH,
                                    minLineLength=LATERAL_MIN_LEN,
                                    maxLineGap=LATERAL_MAX_GAP)
            return lines is not None

        left_det  = _has_lines(frame[roi_y0:h, 0:roi_x0])
        right_det = _has_lines(frame[roi_y0:h, roi_x1:w])

        if left_det and not right_det:
            bias = +LATERAL_BIAS_MAX
        elif right_det and not left_det:
            bias = -LATERAL_BIAS_MAX
        else:
            bias = 0.0

        return bias, left_det, right_det

    def process(self, frame):
        h, w = frame.shape[:2]

        # --- Deteccion de amarillo en ROI superior ---
        yellow_ratio, yellow_mask, (yy0, yy1, yx0, yx1) = self._yellow_ratio(frame, h, w)

        roi_y0 = int(h * ROI_TOP_FRAC)
        roi_x0 = int(w * ROI_LEFT_FRAC)
        roi_x1 = int(w * ROI_RIGHT_FRAC)
        roi_w  = roi_x1 - roi_x0
        roi_h  = h - roi_y0

        roi  = frame[roi_y0:h, roi_x0:roi_x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)   # blur mayor suaviza mas el ruido
        mask = cv2.adaptiveThreshold(gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            ADAPT_BLOCK, ADAPT_C)
        # 1. Apertura: elimina manchas pequenas (ruido del piso, grietas, polvo)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel_open)
        # 2. Cierre: une los huecos dentro de la linea
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel_close)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found = False; cx_roi = roi_w//2; cy_roi = roi_h//2; best_cnt = None
        res = self._best_contour(cnts, roi_w, roi_h)
        if res:
            best_cnt, cx_roi, cy_roi = res; found = True

        error_norm    = float((roi_w/2 - cx_roi) / (roi_w/2))
        cx_frame      = roi_x0 + cx_roi
        cy_frame      = roi_y0 + cy_roi
        cx_center_abs = (roi_x0 + roi_x1) // 2

        # Debug
        debug = frame.copy()
        ov = debug.copy()
        cv2.rectangle(ov, (0,0), (w, roi_y0), (0,0,0), -1)
        cv2.addWeighted(ov, 0.65, debug, 0.35, 0, debug)
        ov2 = debug.copy()
        cv2.rectangle(ov2, (0,roi_y0), (roi_x0,h), (30,0,80), -1)
        cv2.rectangle(ov2, (roi_x1,roi_y0), (w,h), (30,0,80), -1)
        cv2.addWeighted(ov2, 0.55, debug, 0.45, 0, debug)

        mask_color = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
        mask_color[mask > 0] = (0, 255, 0)
        debug[roi_y0:h, roi_x0:roi_x1] = cv2.addWeighted(
            debug[roi_y0:h, roi_x0:roi_x1], 0.45, mask_color, 0.55, 0)

        for cnt in cnts:
            if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA:
                s = cnt.copy(); s[:,:,0]+=roi_x0; s[:,:,1]+=roi_y0
                cv2.drawContours(debug, [s], -1, (120,120,120), 1)
        if best_cnt is not None and found:
            s = best_cnt.copy(); s[:,:,0]+=roi_x0; s[:,:,1]+=roi_y0
            cv2.drawContours(debug, [s], -1, (0,255,255), 3)

        cv2.line(debug, (roi_x0,roi_y0), (roi_x1,roi_y0), (0,220,220), 3)
        cv2.line(debug, (roi_x0,roi_y0), (roi_x0,h),      (0,220,220), 2)
        cv2.line(debug, (roi_x1,roi_y0), (roi_x1,h),      (0,220,220), 2)
        cv2.line(debug, (cx_center_abs,roi_y0), (cx_center_abs,h), (255,100,0), 2)

        if found:
            cv2.line(debug, (cx_frame,roi_y0), (cx_frame,h), (0,0,255), 2)
            cv2.circle(debug, (cx_frame,cy_frame), 12, (0,0,255), -1)
            cv2.circle(debug, (cx_frame,cy_frame), 12, (255,255,255), 2)

        arr_color = (0,255,0) if found else (80,80,80)
        cv2.arrowedLine(debug, (cx_center_abs, h-20),
            (cx_frame if found else cx_center_abs, h-20), arr_color, 3, tipLength=0.2)

        txt   = 'err={:+.3f}'.format(error_norm) if found else 'SIN LINEA'
        color = (0,255,120) if found else (0,60,255)
        cv2.putText(debug, txt, (roi_x0, roi_y0-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 5)
        cv2.putText(debug, txt, (roi_x0, roi_y0-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        th2 = roi_h//3; tw2 = roi_w//3
        if th2>0 and tw2>0:
            thumb = cv2.resize(mask, (tw2, th2))
            x0t,y0t = w-tw2-4, h-th2-4
            debug[y0t:y0t+th2, x0t:x0t+tw2] = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(debug,(x0t,y0t),(x0t+tw2,y0t+th2),(150,150,150),1)
            cv2.putText(debug,'MASK',(x0t+3,y0t+14),cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,200),1)

        # --- Debug: dibujar ROI amarillo sobre la imagen ---
        roi_color = (0, 200, 255) if yellow_ratio >= YELLOW_FILL_THRESH else (0, 255, 200)
        cv2.rectangle(debug, (yx0, yy0), (yx1, yy1), roi_color, 2)
        ytxt = 'YEL:{:.0f}%{}'.format(
            yellow_ratio * 100,
            ' STOP' if yellow_ratio >= YELLOW_FILL_THRESH else '')
        cv2.putText(debug, ytxt, (yx0, yy0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(debug, ytxt, (yx0, yy0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, roi_color, 1)

        # --- Kalman + deteccion lateral ---
        lateral_bias, left_det, right_det = self._detect_lateral(
            frame, roi_y0, roi_x0, roi_x1, h, w)

        # Kalman: si no hay linea reseteamos para no arrastrar error viejo
        if not found:
            self._kalman.reset(error_norm)
        error_filtered = self._kalman.update(error_norm)
        error_out = float(np.clip(error_filtered + lateral_bias, -1.0, 1.0))

        # Debug lateral: borde de las zonas laterales
        lc = (0, 80, 255) if left_det  else (40, 0, 60)
        rc = (0, 80, 255) if right_det else (40, 0, 60)
        cv2.rectangle(debug, (0, roi_y0),      (roi_x0, h), lc, 2)
        cv2.rectangle(debug, (roi_x1, roi_y0), (w, h),      rc, 2)
        if left_det:
            cv2.putText(debug, 'L', (4, roi_y0 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, lc, 2)
        if right_det:
            cv2.putText(debug, 'R', (roi_x1 + 4, roi_y0 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, rc, 2)

        return error_out, found, yellow_ratio, debug


# -----------------------------------------------------------------------
# NODO
# -----------------------------------------------------------------------
class LineFollowerCV(Node):

    def __init__(self):
        super().__init__('line_follower_cv')

        qos_be = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                            history=QoSHistoryPolicy.KEEP_LAST, depth=1)

        self._detector    = ContourLineDetector()
        self._bridge      = CvBridge()
        self._prev_error  = 0.0
        self._integral    = 0.0
        self._prev_time   = None
        self._last_error  = 0.0
        self._frames_lost = 0
        self._semaforo    = 'ninguno'
        self._sign        = 'ninguno'
        self._pending     = None
        self._slow_sign   = False
        self._give_slow   = False          # CAMBIO 1: flag persistente para Give Way
        self._state       = ST_FOLLOWING
        self._state_t0    = 0.0
        self._yellow_stop = False
        self._lidar_stop  = False
        self._sign_last_t = {}

        self._pub_cmd = self.create_publisher(Twist,   '/cmd_vel',          qos_be)
        self._pub_dbg = self.create_publisher(Image,   '/vision/debug_img', 10)
        self._pub_err = self.create_publisher(Float32, '/vision/error',     10)

        self.create_subscription(Image,  '/image/raw',       self._image_cb,    qos_be)
        self.create_subscription(String, '/semaforo/estado', self._semaforo_cb, 10)
        self.create_subscription(String, '/sign/deteccion',  self._sign_cb,     10)
        self.create_subscription(Bool,   '/lidar/parar',     self._lidar_cb,    10)

        self.get_logger().info('LineFollowerCV listo | KP={} KI={} KD={} v={}'.format(
            KP, KI, KD, LINEAR_VEL))

    def _lidar_cb(self, msg: Bool):
        if msg.data != self._lidar_stop:
            self._lidar_stop = msg.data
            self.get_logger().warn(
                'Lidar: {}'.format('OBSTACULO — frenado' if msg.data else 'libre'))

    def _semaforo_cb(self, msg):
        nuevo = msg.data
        if nuevo != self._semaforo:
            self.get_logger().info('Semaforo: {} -> {}'.format(self._semaforo, nuevo))
            if nuevo != 'rojo':
                self._integral = 0.0
        self._semaforo = nuevo

    def _sign_cb(self, msg):
        sign = msg.data
        if sign == self._sign:
            return
        self._sign = sign

        if sign == 'ninguno':
            # CAMBIO 2: Crossing se apaga con ninguno, Give no (persiste hasta interseccion)
            if not self._give_slow:
                self._slow_sign = False
            return

        # Cooldown: ignorar la misma senal si paso hace menos de SIGN_COOLDOWN seg
        now = time.monotonic()
        last = self._sign_last_t.get(sign, 0.0)
        if now - last < SIGN_COOLDOWN:
            self.get_logger().info(
                'Senal {} ignorada (cooldown {:.1f}s restantes)'.format(
                    sign, SIGN_COOLDOWN - (now - last)))
            return

        self.get_logger().info('Senal activada: {}'.format(sign))
        self._sign_last_t[sign] = now

        if sign == 'STOP' and self._state == ST_FOLLOWING:
            self._enter(ST_STOP_WAIT)
        elif sign == 'TurnL':
            self._pending = 'left';  self._slow_sign = False
        elif sign == 'TurnR':
            self._pending = 'right'; self._slow_sign = False
        elif sign in ('AOnly', 'Round'):
            self._pending = 'ahead'; self._slow_sign = False
        elif sign == 'Crossing':
            self._slow_sign = True
        elif sign == 'Give':
            self._slow_sign = True
            self._give_slow = True   # CAMBIO 2: marcar Give como persistente

    def _image_cb(self, msg):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn('cv_bridge: {}'.format(e)); return

        error_norm, found, yellow_ratio, debug = self._detector.process(frame)

        # Actualizar flag de parada por amarillo
        prev_yellow = self._yellow_stop
        self._yellow_stop = yellow_ratio >= YELLOW_FILL_THRESH
        if self._yellow_stop != prev_yellow:
            self.get_logger().info(
                'Amarillo ROI: {:.0f}% -> robot {}'.format(
                    yellow_ratio * 100,
                    'DETENIDO' if self._yellow_stop else 'reanudado'))

        # Anotar estado en debug
        sc = {ST_FOLLOWING:(0,255,120), ST_STOP_WAIT:(0,0,220),
              ST_INTERSECT_PREP:(0,180,255),
              ST_EXEC_L:(255,200,0), ST_EXEC_R:(255,100,0), ST_EXEC_FWD:(0,200,255),
              ST_POST_TURN:(180,255,100)}
        txt = 'ST:{} SGN:{} PND:{}'.format(self._state.upper()[:4],
              self._sign[:4], self._pending or '-')
        cv2.putText(debug, txt, (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
        cv2.putText(debug, txt, (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    sc.get(self._state,(200,200,200)), 1)

        dbg_msg = self._bridge.cv2_to_imgmsg(debug, encoding='bgr8')
        dbg_msg.header = msg.header
        self._pub_dbg.publish(dbg_msg)

        err_msg = Float32(); err_msg.data = float(error_norm)
        self._pub_err.publish(err_msg)

        self._run(error_norm, found)

    def _enter(self, state):
        self._state = state; self._state_t0 = time.monotonic()
        self.get_logger().info('Estado: {}'.format(state))

    def _run(self, error, found):
        # Parada por deteccion de amarillo: tiene prioridad sobre todo excepto
        # sobre ST_STOP_WAIT (para no interferir con la logica de la senal STOP)
        if self._yellow_stop and self._state not in (ST_STOP_WAIT,):
            self._pub_cmd.publish(Twist())
            return

        if self._lidar_stop and self._state not in (ST_STOP_WAIT,):
            self._pub_cmd.publish(Twist())
            return

        now = time.monotonic()

        if self._state == ST_STOP_WAIT:
            self._pub_cmd.publish(Twist())
            if now - self._state_t0 >= STOP_DURATION:
                self._enter(ST_FOLLOWING)
            return

        if self._state == ST_INTERSECT_PREP:
            prep_t = INTERSECT_PREP_TURN if self._pending in ('left', 'right') else INTERSECT_PREP_FWD
            if now - self._state_t0 >= prep_t:
                self.get_logger().info(
                    'Prep terminado, ejecutando: {}'.format(self._pending))
                if self._pending == 'left':    self._enter(ST_EXEC_L)
                elif self._pending == 'right': self._enter(ST_EXEC_R)
                else:                          self._enter(ST_EXEC_FWD)
            else:
                cmd = Twist(); cmd.linear.x = LINEAR_VEL; cmd.angular.z = 0.0
                self._pub_cmd.publish(cmd)
            return

        if self._state == ST_EXEC_L:
            if now - self._state_t0 >= EXEC_TIMEOUT:
                self._pending = None; self._enter(ST_POST_TURN)
            else:
                cmd = Twist(); cmd.linear.x = TURN_LINEAR; cmd.angular.z = -TURN_OMEGA_R
                self._pub_cmd.publish(cmd)
            return

        if self._state == ST_EXEC_R:
            if now - self._state_t0 >= EXEC_TIMEOUT:
                self._pending = None; self._enter(ST_POST_TURN)
            else:
                cmd = Twist(); cmd.linear.x = TURN_LINEAR; cmd.angular.z = TURN_OMEGA_R
                self._pub_cmd.publish(cmd)
            return

        if self._state == ST_EXEC_FWD:
            if found or now - self._state_t0 >= EXEC_TIMEOUT:
                self._pending = None; self._enter(ST_FOLLOWING)
            else:
                cmd = Twist(); cmd.linear.x = LINEAR_VEL; cmd.angular.z = 0.0
                self._pub_cmd.publish(cmd)
            return

        if self._state == ST_POST_TURN:
            if now - self._state_t0 >= POST_TURN_TIME:
                self._enter(ST_FOLLOWING)
            else:
                cmd = Twist(); cmd.linear.x = LINEAR_VEL; cmd.angular.z = 0.0
                self._pub_cmd.publish(cmd)
            return

        # FOLLOWING
        if not found:
            self._frames_lost += 1
            if self._pending and self._frames_lost >= 1:
                prep_t = INTERSECT_PREP_TURN if self._pending in ('left', 'right') else INTERSECT_PREP_FWD
                self.get_logger().info(
                    'Interseccion detectada! prep recto {:.1f}s -> {}'.format(
                        prep_t, self._pending))
                self._give_slow   = False   # CAMBIO 3: Give Way termina en interseccion
                self._slow_sign   = False   # CAMBIO 3: recupera velocidad normal
                self._enter(ST_INTERSECT_PREP)
                return
        else:
            self._frames_lost = 0

        self._pid(error, found)

    def _pid(self, error, found):
        if self._semaforo == 'rojo':
            self._pub_cmd.publish(Twist()); self._prev_time = None; return

        vel = 1.0
        if self._semaforo == 'amarillo': vel = AMARILLO_FACTOR
        elif self._slow_sign:            vel = SLOW_SIGN_FACTOR

        now = time.monotonic()
        dt  = (now - self._prev_time) if self._prev_time else 0.033
        dt  = max(dt, 1e-4); self._prev_time = now

        cmd = Twist()
        if found:
            self._frames_lost = 0
            eff_error = 0.0 if abs(error) < ERROR_DEADBAND else error
            self._integral   += eff_error * dt
            self._integral    = max(-MAX_INTEGRAL, min(MAX_INTEGRAL, self._integral))
            d = (eff_error - self._prev_error) / dt
            u = KP*eff_error + KI*self._integral + KD*d
            u = max(-MAX_ANGULAR, min(MAX_ANGULAR, u))
            self._prev_error = eff_error; self._last_error = error
            cmd.linear.x = LINEAR_VEL * vel; cmd.angular.z = u
        else:
            self._integral = 0.0
            if self._frames_lost < RECOVERY_FRAMES:
                u = KP * self._last_error * 0.4
                cmd.linear.x  = LINEAR_VEL * vel * 0.4
                cmd.angular.z = max(-MAX_ANGULAR, min(MAX_ANGULAR, u))
            else:
                cmd.linear.x  = 0.0
                cmd.angular.z = RECOVERY_OMEGA if self._last_error >= 0 else -RECOVERY_OMEGA

        self._pub_cmd.publish(cmd)

    def stop(self):
        self._pub_cmd.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerCV()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()