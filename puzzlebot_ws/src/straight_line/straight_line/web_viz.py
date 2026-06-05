#!/usr/bin/env python3
# =============================================================================
# web_viz.py  —  Servidor MJPEG sin dependencias externas (solo stdlib)
# =============================================================================
# Abre http://<IP_ROBOT>:8080 desde cualquier navegador en la misma red.
# Una sola vista con botones para cambiar entre streams.
#
# Streams:
#   raw      ←  /image/raw            (cámara cruda)
#   linea    ←  /vision/debug_img     (seguidor + ROI + contorno)
#   semaforo ←  /semaforo/debug_img   (semáforo + máscara de color)
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

# QoS por topic
QOS_BY_KEY = {
    'raw':      'best_effort',
    'linea':    'reliable',
    'semaforo': 'reliable',
}

HTML = b"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>PuzzleBot Vision</title>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{background:#0d0d1a;color:#eee;font-family:monospace;
         display:flex;flex-direction:column;height:100vh}
    header{padding:10px 18px;background:#12122a;
           border-bottom:2px solid #00ffe7;
           display:flex;align-items:center;gap:16px;flex-shrink:0}
    header h1{font-size:1rem;color:#00ffe7}
    .btns{display:flex;gap:8px}
    button{
      padding:6px 16px;border:1px solid #444;border-radius:4px;
      background:#1e1e3a;color:#aaa;cursor:pointer;font-family:monospace;
      font-size:0.85rem;transition:all .15s
    }
    button:hover{border-color:#00ffe7;color:#00ffe7}
    button.active{background:#00ffe7;color:#0d0d1a;border-color:#00ffe7;font-weight:bold}
    .badge{
      margin-left:auto;padding:4px 10px;border-radius:4px;font-size:0.8rem;
      background:#1e1e3a;border:1px solid #333
    }
    #label{color:#00ffe7}
    .viewer{flex:1;display:flex;align-items:center;justify-content:center;
            background:#000;overflow:hidden}
    #feed{max-width:100%;max-height:100%;display:block;object-fit:contain}
  </style>
</head>
<body>
  <header>
    <h1>PuzzleBot &mdash; Vision</h1>
    <div class="btns">
      <button onclick="setStream('raw')"      id="btn-raw">RAW</button>
      <button onclick="setStream('linea')"    id="btn-linea">LÍNEA</button>
      <button onclick="setStream('semaforo')" id="btn-semaforo">SEMÁFORO</button>
    </div>
    <div class="badge">viendo: <span id="label">raw</span></div>
  </header>
  <div class="viewer">
    <img id="feed" src="/stream/raw" alt="stream">
  </div>
  <script>
    var current = 'raw';
    function setStream(name){
      if(name === current) return;
      current = name;
      document.getElementById('feed').src = '/stream/' + name;
      document.getElementById('label').textContent = name;
      ['raw','linea','semaforo'].forEach(function(k){
        document.getElementById('btn-'+k).classList.toggle('active', k===name);
      });
    }
    document.getElementById('btn-raw').classList.add('active');
  </script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Buffer de frames thread-safe
# ─────────────────────────────────────────────────────────────────────────────

class FrameBuffer:
    def __init__(self, label: str):
        self._lock  = threading.Lock()
        self._data  = None
        self._label = label
        # Placeholder con el nombre del stream
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(img, f'Esperando: {label}', (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1)
        _, buf = cv2.imencode('.jpg', img)
        self._placeholder = buf.tobytes()

    def update(self, frame_bgr: np.ndarray):
        ok, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._lock:
                self._data = buf.tobytes()

    def read(self):
        with self._lock:
            return self._data if self._data is not None else self._placeholder


# ─────────────────────────────────────────────────────────────────────────────
# HTTP handler
# ─────────────────────────────────────────────────────────────────────────────

_buffers: dict = {}


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
                self.send_response(404)
                self.end_headers()
                return

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
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' +
                        data + b'\r\n'
                    )
                    time.sleep(0.04)   # ~25 fps
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

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
        qos_rel = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        qos_map = {
            'best_effort': qos_be,
            'reliable':    qos_rel,
        }

        self._bridge  = CvBridge()
        self._buffers = {key: FrameBuffer(key) for key in STREAMS}

        for key, topic in STREAMS.items():
            buf = self._buffers[key]
            qos = qos_map[QOS_BY_KEY[key]]
            self.create_subscription(
                Image, topic,
                lambda msg, b=buf: self._cb(msg, b),
                qos,
            )

        self.get_logger().info(
            f'WebVizNode listo → http://0.0.0.0:{PORT}  '
            f'| streams: {list(STREAMS.keys())}'
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
    global _buffers

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
