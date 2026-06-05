#!/usr/bin/env python3
# =============================================================================
# web_viz.py  —  Servidor MJPEG sin dependencias externas (solo stdlib)
# =============================================================================
# Abre http://<IP_ROBOT>:8080 desde cualquier navegador en la misma red.
#
# Streams disponibles:
#   /stream/raw      ←  /image/raw
#   /stream/linea    ←  /vision/debug_img
#   /stream/semaforo ←  /semaforo/debug_img
# =============================================================================

import threading
import time
import cv2
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer

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
}

LABELS = {
    'raw':      'Cámara cruda',
    'linea':    'Seguidor de línea',
    'semaforo': 'Semáforo',
}

HTML = b"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>PuzzleBot Vision</title>
  <style>
    body{background:#111;color:#eee;font-family:monospace;margin:0}
    h1{padding:12px 20px;background:#1a1a2e;color:#00ffe7;margin:0;font-size:1.1rem}
    .grid{display:flex;flex-wrap:wrap;gap:14px;padding:14px}
    .card{background:#1e1e2e;border:1px solid #333;border-radius:6px;
          overflow:hidden;flex:1 1 380px;min-width:300px}
    .label{padding:6px 10px;background:#252540;font-size:0.8rem;color:#aaa}
    .label span{color:#00ffe7}
    img{width:100%;display:block}
  </style>
</head>
<body>
  <h1>PuzzleBot &mdash; Vision Dashboard</h1>
  <div class="grid">
    <div class="card">
      <div class="label"><span>raw</span> &mdash; C&aacute;mara cruda</div>
      <img src="/stream/raw">
    </div>
    <div class="card">
      <div class="label"><span>linea</span> &mdash; Seguidor de l&iacute;nea</div>
      <img src="/stream/linea">
    </div>
    <div class="card">
      <div class="label"><span>semaforo</span> &mdash; Sem&aacute;foro</div>
      <img src="/stream/semaforo">
    </div>
  </div>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Buffer de frames thread-safe
# ─────────────────────────────────────────────────────────────────────────────

class FrameBuffer:
    def __init__(self):
        self._lock  = threading.Lock()
        self._data  = None

    def update(self, frame_bgr: np.ndarray):
        ok, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with self._lock:
                self._data = buf.tobytes()

    def read(self):
        with self._lock:
            return self._data


# ─────────────────────────────────────────────────────────────────────────────
# HTTP handler
# ─────────────────────────────────────────────────────────────────────────────

_buffers: dict = {}

# Placeholder negro para cuando aún no hay frame
_placeholder: bytes = b''
def _make_placeholder():
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(img, 'Esperando...', (60, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()


class Handler(BaseHTTPRequestHandler):

    def log_message(self, *args):
        pass  # silenciar logs de acceso

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML)

        elif self.path.startswith('/stream/'):
            name = self.path[len('/stream/'):]
            if name not in _buffers:
                self.send_response(404)
                self.end_headers()
                return

            self.send_response(200)
            self.send_header('Content-Type',
                             'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()

            buf = _buffers[name]
            try:
                while True:
                    data = buf.read() or _placeholder
                    self.wfile.write(
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' +
                        data + b'\r\n'
                    )
                    time.sleep(0.05)   # ~20 fps
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()


# ─────────────────────────────────────────────────────────────────────────────
# Nodo ROS 2
# ─────────────────────────────────────────────────────────────────────────────

class WebVizNode(Node):

    def __init__(self):
        super().__init__('web_viz')

        qos_be = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._bridge  = CvBridge()
        self._buffers = {key: FrameBuffer() for key in STREAMS}

        for key, topic in STREAMS.items():
            buf = self._buffers[key]
            self.create_subscription(
                Image, topic,
                lambda msg, b=buf: self._cb(msg, b),
                qos_be,
            )

        self.get_logger().info(
            f'WebVizNode listo → http://0.0.0.0:{PORT}\n' +
            '\n'.join(f'  /stream/{k}  ←  {t}' for k, t in STREAMS.items())
        )

    def _cb(self, msg: Image, buf: FrameBuffer):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            buf.update(frame)
        except Exception as e:
            self.get_logger().warn(f'cv_bridge: {e}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    global _buffers, _placeholder
    _placeholder = _make_placeholder()

    rclpy.init(args=args)
    node = WebVizNode()

    _buffers = node._buffers

    server = HTTPServer(('0.0.0.0', PORT), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

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
