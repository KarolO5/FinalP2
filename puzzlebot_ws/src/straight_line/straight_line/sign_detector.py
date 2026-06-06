#!/usr/bin/env python3
# =============================================================================
# sign_detector.py  --  Detector de senales de trafico con YOLO ONNX
# =============================================================================
#
# Mismo ROI que semaforo (esquina superior derecha).
# Usa YOLO (forma/objeto), el semaforo usa HSV (color) -> no se confunden.
#
# Clases del modelo:
#   0=STOP  1=Crossing  2=TurnL  3=TurnR  4=AOnly  5=Give  6=Round
#
# TOPICOS
#   Sub : /image/raw        [sensor_msgs/Image]
#   Pub : /sign/deteccion   [std_msgs/String]  ninguno|STOP|Crossing|TurnL|TurnR|AOnly|Give|Round
#   Pub : /sign/debug_img   [sensor_msgs/Image]
#
# PARAMETROS ROS
#   model_path : ruta al .onnx  (default: ~/FinalP2/best_copy.onnx)
#   confidence : umbral         (default: 0.50)
# =============================================================================

import os
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg    import String
from cv_bridge       import CvBridge
from rclpy.qos       import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

try:
    from ultralytics import YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False

# -----------------------------------------------------------------------
# PARAMETROS
# -----------------------------------------------------------------------

SIGN_CLASSES = ['STOP', 'Crossing', 'TurnL', 'TurnR', 'AOnly', 'Give', 'Round']

# ROI identico al semaforo
ROI_H_FRAC = 0.50
ROI_W_FRAC = 0.35

CLASS_COLORS = {
    'STOP':     (0,   0, 220),
    'Crossing': (0, 180, 255),
    'TurnL':    (255, 200,  0),
    'TurnR':    (255, 100,  0),
    'AOnly':    (0,  255, 80),
    'Give':     (180,  0, 255),
    'Round':    (0,  220, 220),
    'ninguno':  (80,  80,  80),
}


# -----------------------------------------------------------------------
# NODO
# -----------------------------------------------------------------------

class SignDetectorNode(Node):

    def __init__(self):
        super().__init__('sign_detector')

        self.declare_parameter('model_path',
                               os.path.expanduser('~/FinalP2/best_copy.onnx'))
        self.declare_parameter('confidence', 0.50)

        model_path = self.get_parameter('model_path').value
        self._conf = float(self.get_parameter('confidence').value)

        if not _YOLO_OK:
            self.get_logger().error('ultralytics no instalado: pip install ultralytics')
            raise RuntimeError('ultralytics no disponible')

        if not os.path.isfile(model_path):
            self.get_logger().error('Modelo no encontrado: {}'.format(model_path))
            raise RuntimeError('Modelo ONNX no encontrado')

        self.get_logger().info('Cargando modelo: {}'.format(model_path))
        self._model = YOLO(model_path, task='detect')
        self.get_logger().info('Modelo listo | conf={}'.format(self._conf))

        qos_be = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._bridge    = CvBridge()
        self._last_sign = 'ninguno'

        self._pub_sign  = self.create_publisher(String, '/sign/deteccion', 10)
        self._pub_debug = self.create_publisher(Image,  '/sign/debug_img', 10)

        self.create_subscription(Image, '/image/raw', self._image_cb, qos_be)

        self.get_logger().info(
            'SignDetector listo | ROI sup-der {}%x{}%'.format(
                int(ROI_H_FRAC * 100), int(ROI_W_FRAC * 100))
        )

    def _image_cb(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn('cv_bridge: {}'.format(e))
            return

        sign, debug = self._detect(frame)

        sign_msg      = String()
        sign_msg.data = sign
        self._pub_sign.publish(sign_msg)

        dbg_msg        = self._bridge.cv2_to_imgmsg(debug, encoding='bgr8')
        dbg_msg.header = msg.header
        self._pub_debug.publish(dbg_msg)

    def _detect(self, frame: np.ndarray):
        h, w = frame.shape[:2]

        roi_y0 = 0
        roi_y1 = int(h * ROI_H_FRAC)
        roi_x0 = int(w * (1.0 - ROI_W_FRAC))
        roi_x1 = w

        roi = frame[roi_y0:roi_y1, roi_x0:roi_x1]

        results  = self._model(roi, conf=self._conf, verbose=False)
        best_sign = 'ninguno'
        best_conf = 0.0

        debug = frame.copy()
        ov = debug.copy()
        cv2.rectangle(ov, (0, 0),    (roi_x0, h), (0, 0, 0), -1)
        cv2.rectangle(ov, (0, roi_y1), (w, h),    (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, debug, 0.45, 0, debug)

        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                if cls_id >= len(SIGN_CLASSES):
                    continue
                label = SIGN_CLASSES[cls_id]
                color = CLASS_COLORS.get(label, (200, 200, 200))

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                ax1 = int(x1) + roi_x0
                ay1 = int(y1) + roi_y0
                ax2 = int(x2) + roi_x0
                ay2 = int(y2) + roi_y0

                cv2.rectangle(debug, (ax1, ay1), (ax2, ay2), color, 2)
                txt = '{} {:.0f}%'.format(label, conf * 100)
                cv2.putText(debug, txt, (ax1, ay1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                cv2.putText(debug, txt, (ax1, ay1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

                if conf > best_conf:
                    best_conf = conf
                    best_sign = label

        if best_sign != self._last_sign:
            self.get_logger().info('Senal: {} -> {} (conf={:.2f})'.format(
                self._last_sign, best_sign, best_conf))
            self._last_sign = best_sign

        border = CLASS_COLORS.get(best_sign, (80, 80, 80))
        cv2.rectangle(debug, (roi_x0, roi_y0), (roi_x1 - 1, roi_y1), border, 2)
        lbl = 'SENAL: {}'.format(best_sign.upper())
        cv2.putText(debug, lbl, (roi_x0, roi_y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(debug, lbl, (roi_x0, roi_y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, border, 2)

        return best_sign, debug


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = SignDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
