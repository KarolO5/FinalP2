#!/usr/bin/env python3
# =============================================================================
# web_viz.py  --  Dashboard 2x2 con todos los streams simultaneos
# =============================================================================
# Muestra los 4 streams en una sola pagina sin necesidad de cambiar pestanas.
# Usa ThreadingHTTPServer para manejar 4 conexiones MJPEG en paralelo.
#
#  +------------------+------------------+
#  |  RAW             |  LINEA           |
#  +------------------+------------------+
#  |  SEMAFORO        |  SENAL           |
#  +------------------+------------------+
# =============================================================================

import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

PORT = 8080

STREAMS = {
    'raw':      '/image/raw',
    'linea':    '/vision/debug_img',
    'semaforo': '/semaforo/debug_img',
    'senal':    '/sign/debug_img',
}

QOS_BY_KEY = {
    'raw':      'best_effort',
    'linea':    'reliable',
    'semaforo': 'reliable',
    'senal':    'reliable',
}

LABELS = {
    'raw':      'RAW',
    'linea':    'LINEA',
    'semaforo': 'SEMAFORO',
    'senal':    'SENAL',
}

HTML = b"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>PuzzleBot Vision</title>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{background:#0d0d1a;color:#eee;font-family:monospace}
    header{padding:8px 16px;background:#12122a;
           border-bottom:2px solid #00ffe7;display:flex;align-items:center}
    header h1{font-size:0.95rem;color:#00ffe7}
    .grid{display:grid;grid-template-columns:1fr 1fr;
          grid-template-rows:1fr 1fr;gap:6px;padding:6px;
          height:calc(100vh - 38px)}
    .cell{background:#111;border:1px solid #222;
          border-radius:4px;overflow:hidden;display:flex;
          flex-direction:column}
    .cell-title{padding:3px 8px;background:#1a1a2e;
                font-size:0.7rem;color:#00ffe7;flex-shrink:0}
    .cell img{width:100%;flex:1;display:block;object-fit:contain;min-height:0}
  </style>
</head>
<body>
  <header><h1>PuzzleBot &mdash; Vision Dashboard</h1></header>
  <div class="grid">
    <div class="cell">
      <div class="cell-title">RAW &mdash; Camara cruda</div>
      <img src="/stream/raw" alt="raw">
    </div>
    <div class="cell">
      <div class="cell-title">LINEA &mdash; Seguidor</div>
      <img src="/stream/linea" alt="linea">
    </div>
    <div class="cell">
      <div class="cell-title">SEMAFORO &mdash; Color HSV</div>
      <img src="/stream/semaforo" alt="semaforo">
    </div>
    <div class="cell">
      <div class="cell-title">SENAL &mdash; YOLO detector</div>
      <img src="/stream/senal" alt="senal">
    </div>
  </div>
</body>
</html>"""


# -----------------------------------------------------------------------
# Buffer thread-safe
# -----------------------------------------------------------------------
class FrameBuffer:
    def __init__(self, name: str):
        self._lock = threading.Lock()
        self._data = None
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(img, 'Esperando: {}'.format(name), (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
        ok, buf = cv2.imencode('.jpg', img)
        self._placeholder = buf.tobytes() if ok else b''

    def update(self, frame: np.ndarray):
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with self._lock:
                self._data = buf.tobytes()

    def read(self):
        with self._lock:
            return self._data if self._data is not None else self._placeholder


# -----------------------------------------------------------------------
# HTTP handler
# -----------------------------------------------------------------------
_buffers = {}


class Handler(BaseHTTPRequestHandler):

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML)
            return

        if self.path.startswith('/stream/'):
            name = self.path[len('/stream/'):]
            if name not in _buffers:
                self.send_response(404); self.end_headers(); return

            self.send_response(200)
            self.send_header('Content-Type',
                             'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            buf = _buffers[name]
            try:
                while True:
                    data = buf.read()
                    self.wfile.write(
                        b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
                        str(len(data)).encode() + b'\r\n\r\n' +
                        data + b'\r\n')
                    time.sleep(0.05)   # ~20 fps
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        self.send_response(404); self.end_headers()


# -----------------------------------------------------------------------
# Nodo ROS 2
# -----------------------------------------------------------------------
class WebVizNode(Node):

    def __init__(self):
        super().__init__('web_viz')

        qos_be = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                            history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        qos_rel = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                             history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        qos_map = {'best_effort': qos_be, 'reliable': qos_rel}

        self._bridge  = CvBridge()
        self._buffers = {k: FrameBuffer(k) for k in STREAMS}

        for key, topic in STREAMS.items():
            buf = self._buffers[key]
            qos = qos_map[QOS_BY_KEY[key]]
            self.create_subscription(
                Image, topic,
                lambda msg, b=buf: self._cb(msg, b),
                qos)

        self.get_logger().info(
            'WebVizNode listo -> http://0.0.0.0:{} | grid 2x2'.format(PORT))

    def _cb(self, msg: Image, buf: FrameBuffer):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            buf.update(frame)
        except Exception as e:
            self.get_logger().warn('cv_bridge: {}'.format(e))


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main(args=None):
    global _buffers
    rclpy.init(args=args)
    node = WebVizNode()
    _buffers = node._buffers

    server = ThreadingHTTPServer(('0.0.0.0', PORT), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
