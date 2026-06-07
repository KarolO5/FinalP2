#!/usr/bin/env python3
# =============================================================================
# line_follower_cv.py  --  Seguidor de linea + deteccion de interseccion por patron punteado
# =============================================================================
#
# ROIs del frame:
#
#  +------------------------------------------+
#  |           zona ignorada                  |  <- ROI_LINE_TOP
#  +--------+-----------------+--------+------+  <- DOT_ROI_TOP
#  |        |  ROI PUNTEADO   |        |      |  Detecta el patron de linea
#  |        |  (ancho total)  |        |      |  punteada antes de interseccion
#  +--------+-----------------+--------+------+  <- DOT_ROI_BOT = ROI_LINE_TOP
#  | ignor  |   ROI LINEA     | ignor  |      |  Seguidor de linea principal
#  | 33%    |   34% central   | 33%    |      |
#  +--------+-----------------+--------+------+
#
# Patron punteado: multiples contornos separados horizontalmente en el ROI punteado.
# Cuando el patron es visible → estado APPROACHING (sigue la linea normal).
# Cuando el patron desaparece (ya paso la interseccion) → ejecuta la señal.
#
# Maquina de estados:
#   FOLLOWING   : PID normal siguiendo la linea
#   APPROACHING : patron punteado visible, sigue la linea, se prepara
#   EXEC_LEFT   : gira izquierda hasta encontrar linea o timeout
#   EXEC_RIGHT  : gira derecha
#   EXEC_AHEAD  : avanza recto
#   STOP_WAIT   : parado 3 seg (STOP)
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
from std_msgs.msg      import Float32, String
from cv_bridge         import CvBridge
from rclpy.qos         import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# -----------------------------------------------------------------------
# PARAMETROS PID
# -----------------------------------------------------------------------
LINEAR_VEL       = 0.09
MAX_ANGULAR      = 0.50
KP               = 1.0
KI               = 0.02
KD               = 0.06
MAX_INTEGRAL     = 0.30
AMARILLO_FACTOR  = 0.60
SLOW_SIGN_FACTOR = 0.55
ERROR_DEADBAND   = 0.05

# -----------------------------------------------------------------------
# ROI SEGUIDOR DE LINEA (parte inferior)
# -----------------------------------------------------------------------
ROI_LINE_TOP   = 0.78    # el ROI de linea empieza aqui
ROI_LEFT_FRAC  = 0.33
ROI_RIGHT_FRAC = 0.67

# -----------------------------------------------------------------------
# ROI DETECTOR DE PATRON PUNTEADO (banda horizontal encima del ROI de linea)
# -----------------------------------------------------------------------
DOT_ROI_TOP  = 0.55    # el ROI punteado va de aqui...
DOT_ROI_BOT  = 0.75    # ...hasta aqui (justo encima del ROI de linea)
DOT_ROI_LEFT = 0.10    # ancho casi completo para capturar todos los puntos
DOT_ROI_RIGHT= 0.90

# Deteccion del patron: necesita al menos DOT_MIN_COUNT contornos separados
# que esten distribuidos en al menos DOT_MIN_SPREAD fraccion del ancho del ROI
DOT_MIN_COUNT    = 3      # minimo de segmentos/puntos separados
DOT_MAX_AREA     = 600    # area maxima de cada punto (no la linea solida)
DOT_MIN_AREA     = 60     # area minima para no contar ruido de textura
DOT_MIN_SPREAD   = 0.40   # los puntos deben cubrir al menos el 40% del ancho

# Frames consecutivos para confirmar patron o su desaparicion
DOT_CONFIRM_ON   = 10   # frames con patron para entrar a APPROACHING (mas robusto)
DOT_CONFIRM_OFF  = 8    # frames sin patron (estando en APPROACHING) para ejecutar

# Cooldown tras ejecutar una accion en interseccion
# Evita que el detector dispare de nuevo inmediatamente al salir
DOT_EXEC_COOLDOWN = 5.0   # segundos sin deteccion tras EXEC_*

# -----------------------------------------------------------------------
# VISION LINEA
# -----------------------------------------------------------------------
ADAPT_BLOCK      = 25
ADAPT_C          = 8
MORPH_KSIZE      = (7, 7)
OPEN_KSIZE       = (3, 3)
MIN_CONTOUR_AREA = 200
MAX_ASPECT_RATIO = 2.5

# -----------------------------------------------------------------------
# RECOVERY
# -----------------------------------------------------------------------
RECOVERY_FRAMES  = 25
RECOVERY_OMEGA   = 0.20

# -----------------------------------------------------------------------
# SENALES / INTERSECCION
# -----------------------------------------------------------------------
STOP_DURATION    = 3.0
TURN_LINEAR      = 0.07
TURN_OMEGA_L     = +0.50
TURN_OMEGA_R     = -0.65
EXEC_TIMEOUT     = 6.0
SIGN_COOLDOWN    = 8.0

ST_FOLLOWING   = 'following'
ST_APPROACHING = 'approaching'   # patron punteado visible, sigue la linea
ST_STOP_WAIT   = 'stop_wait'
ST_EXEC_L      = 'exec_left'
ST_EXEC_R      = 'exec_right'
ST_EXEC_FWD    = 'exec_ahead'


# -----------------------------------------------------------------------
# DETECTOR DE PATRON PUNTEADO
# -----------------------------------------------------------------------
class DotPatternDetector:
    """
    Detecta el patron de linea punteada en un ROI horizontal.
    Retorna True si ve el patron (multiples segmentos separados horizontalmente).
    """
    def __init__(self):
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._kernel = k

    def detect(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        y0 = int(h * DOT_ROI_TOP)
        y1 = int(h * DOT_ROI_BOT)
        x0 = int(w * DOT_ROI_LEFT)
        x1 = int(w * DOT_ROI_RIGHT)
        roi_w = x1 - x0

        roi  = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = cv2.adaptiveThreshold(gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrar contornos por area (ni muy pequenos ni muy grandes)
        dots = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if DOT_MIN_AREA < area < DOT_MAX_AREA:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = M['m10'] / M['m00']
                    dots.append(cx)

        # Verificar patron: suficientes puntos Y distribuidos horizontalmente
        detected = False
        if len(dots) >= DOT_MIN_COUNT:
            spread = (max(dots) - min(dots)) / roi_w
            if spread >= DOT_MIN_SPREAD:
                detected = True

        return detected, (y0, y1, x0, x1, dots, mask)


# -----------------------------------------------------------------------
# DETECTOR DE LINEA PRINCIPAL
# -----------------------------------------------------------------------
class ContourLineDetector:

    def __init__(self):
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KSIZE)
        self._kernel_open  = cv2.getStructuringElement(cv2.MORPH_RECT, OPEN_KSIZE)

    def _best_contour(self, contours, roi_w, roi_h):
        best = None; best_d = float('inf')
        ref_x = roi_w / 2.0; ref_y = float(roi_h)
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                continue
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

    def process(self, frame):
        h, w = frame.shape[:2]
        roi_y0 = int(h * ROI_LINE_TOP)
        roi_x0 = int(w * ROI_LEFT_FRAC)
        roi_x1 = int(w * ROI_RIGHT_FRAC)
        roi_w  = roi_x1 - roi_x0
        roi_h  = h - roi_y0

        roi  = frame[roi_y0:h, roi_x0:roi_x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        mask = cv2.adaptiveThreshold(gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            ADAPT_BLOCK, ADAPT_C)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel_open)
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

        return error_norm, found, cx_frame, cy_frame, cx_center_abs, roi_y0, roi_x0, roi_x1, roi_w, roi_h, mask, cnts, best_cnt


# -----------------------------------------------------------------------
# NODO
# -----------------------------------------------------------------------
class LineFollowerCV(Node):

    def __init__(self):
        super().__init__('line_follower_cv')

        qos_be = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                            history=QoSHistoryPolicy.KEEP_LAST, depth=1)

        self._line_det = ContourLineDetector()
        self._dot_det  = DotPatternDetector()
        self._bridge   = CvBridge()

        self._prev_error  = 0.0
        self._integral    = 0.0
        self._prev_time   = None
        self._last_error  = 0.0
        self._frames_lost = 0
        self._semaforo    = 'ninguno'
        self._sign        = 'ninguno'
        self._pending     = None
        self._slow_sign   = False
        self._state       = ST_FOLLOWING
        self._state_t0    = 0.0
        self._sign_last_t = {}

        # Contadores de confirmacion del patron punteado
        self._dot_on_count  = 0
        self._dot_off_count = 0
        self._dot_cooldown_t = 0.0  # tiempo en que termino el ultimo EXEC_*

        self._pub_cmd = self.create_publisher(Twist,   '/cmd_vel',          qos_be)
        self._pub_dbg = self.create_publisher(Image,   '/vision/debug_img', 10)
        self._pub_err = self.create_publisher(Float32, '/vision/error',     10)

        self.create_subscription(Image,  '/image/raw',       self._image_cb,    qos_be)
        self.create_subscription(String, '/semaforo/estado', self._semaforo_cb, 10)
        self.create_subscription(String, '/sign/deteccion',  self._sign_cb,     10)

        self.get_logger().info('LineFollowerCV listo | v={} KP={} KI={} KD={}'.format(
            LINEAR_VEL, KP, KI, KD))

    # ---- Callbacks -------------------------------------------------------

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
            self._slow_sign = False
            return

        now = time.monotonic()
        last = self._sign_last_t.get(sign, 0.0)
        if now - last < SIGN_COOLDOWN:
            self.get_logger().info('Senal {} en cooldown ({:.1f}s)'.format(
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
        elif sign in ('Crossing', 'Give'):
            self._slow_sign = True

    def _image_cb(self, msg):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn('cv_bridge: {}'.format(e)); return

        # Detectar patron punteado
        dot_visible, dot_info = self._dot_det.detect(frame)

        # Detectar linea principal
        (error_norm, found, cx_frame, cy_frame, cx_center_abs,
         roi_y0, roi_x0, roi_x1, roi_w, roi_h, mask, cnts, best_cnt) = \
            self._line_det.process(frame)

        # Actualizar maquina de estados con el patron
        self._update_dot_state(dot_visible)

        # Debug
        debug = self._draw_debug(frame, error_norm, found, cx_frame, cy_frame,
                                 cx_center_abs, roi_y0, roi_x0, roi_x1,
                                 roi_w, roi_h, mask, cnts, best_cnt,
                                 dot_visible, dot_info)

        dbg_msg = self._bridge.cv2_to_imgmsg(debug, encoding='bgr8')
        dbg_msg.header = msg.header
        self._pub_dbg.publish(dbg_msg)

        err_msg = Float32(); err_msg.data = float(error_norm)
        self._pub_err.publish(err_msg)

        self._run(error_norm, found)

    # ---- Deteccion patron punteado --------------------------------------

    def _update_dot_state(self, dot_visible: bool):
        if self._state in (ST_EXEC_L, ST_EXEC_R, ST_EXEC_FWD, ST_STOP_WAIT):
            return  # no cambiar estado durante ejecucion

        # Cooldown: ignorar el detector justo despues de ejecutar una accion
        if time.monotonic() - self._dot_cooldown_t < DOT_EXEC_COOLDOWN:
            self._dot_on_count  = 0
            self._dot_off_count = 0
            return

        if self._state == ST_FOLLOWING:
            if dot_visible:
                self._dot_on_count += 1
                self._dot_off_count = 0
                if self._dot_on_count >= DOT_CONFIRM_ON:
                    self._enter(ST_APPROACHING)
                    self._dot_on_count  = 0
                    self._dot_off_count = 0
                    self.get_logger().info(
                        'Patron punteado detectado -> APPROACHING | pendiente: {}'.format(
                            self._pending or 'ninguno'))
            else:
                self._dot_on_count = 0

        elif self._state == ST_APPROACHING:
            if not dot_visible:
                self._dot_off_count += 1
                self._dot_on_count  = 0
                if self._dot_off_count >= DOT_CONFIRM_OFF:
                    # Patron desaparecio → ejecutar accion
                    self._dot_off_count = 0
                    if self._pending == 'left':
                        self._enter(ST_EXEC_L)
                    elif self._pending == 'right':
                        self._enter(ST_EXEC_R)
                    elif self._pending == 'ahead':
                        self._enter(ST_EXEC_FWD)
                    else:
                        # Sin señal pendiente: seguir recto
                        self._enter(ST_EXEC_FWD)
                    self.get_logger().info(
                        'Patron desaparecio -> ejecutando: {}'.format(self._state))
            else:
                self._dot_off_count = 0

    # ---- Maquina de estados principal -----------------------------------

    def _enter(self, state):
        self._state = state; self._state_t0 = time.monotonic()
        self.get_logger().info('Estado: {}'.format(state))

    def _run(self, error, found):
        now = time.monotonic()

        if self._state == ST_STOP_WAIT:
            self._pub_cmd.publish(Twist())
            if now - self._state_t0 >= STOP_DURATION:
                self._enter(ST_FOLLOWING)
            return

        if self._state == ST_EXEC_L:
            if found or now - self._state_t0 >= EXEC_TIMEOUT:
                self._pending = None
                self._dot_cooldown_t = time.monotonic()
                self._enter(ST_FOLLOWING)
            else:
                cmd = Twist(); cmd.linear.x = TURN_LINEAR; cmd.angular.z = TURN_OMEGA_L
                self._pub_cmd.publish(cmd)
            return

        if self._state == ST_EXEC_R:
            if found or now - self._state_t0 >= EXEC_TIMEOUT:
                self._pending = None
                self._dot_cooldown_t = time.monotonic()
                self._enter(ST_FOLLOWING)
            else:
                cmd = Twist(); cmd.linear.x = TURN_LINEAR; cmd.angular.z = TURN_OMEGA_R
                self._pub_cmd.publish(cmd)
            return

        if self._state == ST_EXEC_FWD:
            if found or now - self._state_t0 >= EXEC_TIMEOUT:
                self._pending = None
                self._dot_cooldown_t = time.monotonic()
                self._enter(ST_FOLLOWING)
            else:
                cmd = Twist(); cmd.linear.x = LINEAR_VEL; cmd.angular.z = 0.0
                self._pub_cmd.publish(cmd)
            return

        # FOLLOWING o APPROACHING: PID normal
        self._pid(error, found)

    def _pid(self, error, found):
        if self._semaforo == 'rojo':
            self._pub_cmd.publish(Twist()); self._prev_time = None; return

        vel = 1.0
        if self._semaforo == 'amarillo':  vel = AMARILLO_FACTOR
        elif self._slow_sign:             vel = SLOW_SIGN_FACTOR

        now = time.monotonic()
        dt  = (now - self._prev_time) if self._prev_time else 0.033
        dt  = max(dt, 1e-4); self._prev_time = now

        cmd = Twist()
        if found:
            self._frames_lost = 0
            eff = 0.0 if abs(error) < ERROR_DEADBAND else error
            self._integral   += eff * dt
            self._integral    = max(-MAX_INTEGRAL, min(MAX_INTEGRAL, self._integral))
            d = (eff - self._prev_error) / dt
            u = KP*eff + KI*self._integral + KD*d
            u = max(-MAX_ANGULAR, min(MAX_ANGULAR, u))
            self._prev_error = eff; self._last_error = error
            cmd.linear.x = LINEAR_VEL * vel; cmd.angular.z = u
        else:
            self._frames_lost += 1
            self._integral = 0.0
            if self._frames_lost < RECOVERY_FRAMES:
                u = KP * self._last_error * 0.4
                cmd.linear.x  = LINEAR_VEL * vel * 0.4
                cmd.angular.z = max(-MAX_ANGULAR, min(MAX_ANGULAR, u))
            else:
                cmd.linear.x  = 0.0
                cmd.angular.z = RECOVERY_OMEGA if self._last_error >= 0 else -RECOVERY_OMEGA

        self._pub_cmd.publish(cmd)

    # ---- Debug ----------------------------------------------------------

    def _draw_debug(self, frame, error_norm, found, cx_frame, cy_frame,
                    cx_center_abs, roi_y0, roi_x0, roi_x1,
                    roi_w, roi_h, mask, cnts, best_cnt,
                    dot_visible, dot_info):
        debug = frame.copy()
        h, w  = frame.shape[:2]

        # Oscurecer zona superior
        ov = debug.copy()
        cv2.rectangle(ov, (0,0), (w, roi_y0), (0,0,0), -1)
        cv2.addWeighted(ov, 0.65, debug, 0.35, 0, debug)

        # Franjas laterales
        ov2 = debug.copy()
        cv2.rectangle(ov2, (0,roi_y0), (roi_x0,h), (30,0,80), -1)
        cv2.rectangle(ov2, (roi_x1,roi_y0), (w,h), (30,0,80), -1)
        cv2.addWeighted(ov2, 0.55, debug, 0.45, 0, debug)

        # Mascara linea en verde
        mask_color = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
        mask_color[mask > 0] = (0, 255, 0)
        debug[roi_y0:h, roi_x0:roi_x1] = cv2.addWeighted(
            debug[roi_y0:h, roi_x0:roi_x1], 0.45, mask_color, 0.55, 0)

        # Contornos
        for cnt in cnts:
            if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA:
                s = cnt.copy(); s[:,:,0]+=roi_x0; s[:,:,1]+=roi_y0
                cv2.drawContours(debug, [s], -1, (120,120,120), 1)
        if best_cnt is not None and found:
            s = best_cnt.copy(); s[:,:,0]+=roi_x0; s[:,:,1]+=roi_y0
            cv2.drawContours(debug, [s], -1, (0,255,255), 3)

        # Bordes ROI linea
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

        # ROI punteado
        dy0, dy1, dx0, dx1, dots, dot_mask = dot_info
        dot_color = (0, 255, 180) if dot_visible else (80, 80, 80)
        cv2.rectangle(debug, (dx0,dy0), (dx1,dy1), dot_color, 2)
        dot_lbl = 'DOTS:{} SPREAD:{:.0f}%'.format(
            len(dots),
            (max(dots)-min(dots))/(dx1-dx0)*100 if len(dots)>=2 else 0)
        cv2.putText(debug, dot_lbl, (dx0+4, dy0+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3)
        cv2.putText(debug, dot_lbl, (dx0+4, dy0+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, dot_color, 1)

        # Texto de estado
        st_colors = {ST_FOLLOWING:(0,255,120), ST_APPROACHING:(0,220,255),
                     ST_STOP_WAIT:(0,0,220), ST_EXEC_L:(255,200,0),
                     ST_EXEC_R:(255,100,0), ST_EXEC_FWD:(0,200,255)}
        txt = 'ST:{} SGN:{} PND:{}'.format(
            self._state[:4].upper(), self._sign[:4], self._pending or '-')
        cv2.putText(debug, txt, (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
        cv2.putText(debug, txt, (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    st_colors.get(self._state,(200,200,200)), 1)

        err_txt = 'err={:+.3f}'.format(error_norm) if found else 'SIN LINEA'
        err_col = (0,255,120) if found else (0,60,255)
        cv2.putText(debug, err_txt, (roi_x0, roi_y0-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 5)
        cv2.putText(debug, err_txt, (roi_x0, roi_y0-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, err_col, 2)

        # Mini MASK
        if roi_h > 0:
            th = roi_h//3; tw = (roi_x1-roi_x0)//3
            if th > 0 and tw > 0:
                thumb = cv2.resize(mask, (tw, th))
                x0t,y0t = w-tw-4, h-th-4
                debug[y0t:y0t+th, x0t:x0t+tw] = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(debug,(x0t,y0t),(x0t+tw,y0t+th),(150,150,150),1)

        return debug

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
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
