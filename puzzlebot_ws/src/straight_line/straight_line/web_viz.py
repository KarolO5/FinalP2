#!/usr/bin/env python3
# =============================================================================
# web_viz.py
# =============================================================================
# Servidor MJPEG en localhost:8080.
# Sirve los topics de imagen del sistema como streams de video en vivo.
#
# URL principal : http://localhost:8080
# Streams disponibles como MJPEG en /stream/<nombre>:
#   raw      → /image/raw            (cámara cruda)
#   linea    → /vision/debug_img     (seguidor con grilla y ROI)
#   semaforo → /semaforo/debug_img   (semáforo con máscaras de color)
#
# Requiere: flask  (pip install flask)
# =============================================================================

import threading
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from flask import Flask, Response, render_template_string

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
HOST = '0.0.0.0'
PORT = 8080

STREAMS = {
    'raw':      '/image/raw',
    'linea':    '/vision/debug_img',
    'semaforo': '/semaforo/debug_img',
}

STREAM_LABELS = {
    'raw':      'Cámara cruda',
    'linea':    'Seguidor de línea (ROI + grilla)',
    'semaforo': 'Semáforo (máscaras de color)',
}

# ─────────────────────────────────────────────────────────────────────────────
# HTML
# ─────────────────────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PuzzleBot Vision</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #111; color: #eee; font-family: monospace; }
    header { padding: 12px 20px; background: #1a1a2e;
             border-bottom: 2px solid #00ffe7; }
    header h1 { font-size: 1.2rem; color: #00ffe7; }
    .grid { display: flex; flex-wrap: wrap; gap: 16px; padding: 16px; }
    .card { background: #1e1e2e; border: 1px solid #333; border-radius: 8px;
            overflow: hidden; flex: 1 1 420px; min-width: 320px; }
    .card-title { padding: 8px 12px; background: #252540;
                  font-size: 0.85rem; color: #aaa; }
    .card-title span { color: #00ffe7; }
    .card img { width: 100%; display: block; }
    .card .no-signal { padding: 40px; text-align: center;
                       color: #555; font-size: 0.9rem; }
  </style>
</head>
<body>
  <header><h1>PuzzleBot &mdash; Vision Dashboard</h1></header>
  <div class="grid">
    {% for key, label in streams.items() %}
    <div class="card">
      <div class="card-title"><span>{{ key }}</span> &mdash; {{ label }}</div>
      <img src="/stream/{{ key }}" alt="{{ label }}"
           onerror="this.outerHTML='<div class=no-signal>Sin señal: {{ key }}</div>'">
    </div>
    {% endfor %}
  </div>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# BUFFER DE FRAMES (thread-safe)
# ─────────────────────────────────────────────────────────────────────────────

class FrameBuffer:
    def __init__(self):
        self._lock  = threading.Lock()
        self._frame = None   # JPEG bytes

    def update(self, frame_bgr: np.ndarray):
        ok, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._lock:
                self._frame = buf.tobytes()

    def read(self):
        with self._lock:
            return self._frame


# ─────────────────────────────────────────────────────────────────────────────
# NODO ROS 2
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
                lambda msg, b=buf: self._img_cb(msg, b),
                qos_be,
            )

        self.get_logger().info(
            f'WebVizNode listo — http://localhost:{PORT}\n' +
            '\n'.join(f'  /stream/{k}  ←  {t}' for k, t in STREAMS.items())
        )

    def _img_cb(self, msg: Image, buf: FrameBuffer):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            buf.update(frame)
        except Exception as e:
            self.get_logger().warn(f'cv_bridge: {e}')


# ─────────────────────────────────────────────────────────────────────────────
# SERVIDOR FLASK
# ─────────────────────────────────────────────────────────────────────────────

app     = Flask(__name__)
_buffers: dict[str, FrameBuffer] = {}


@app.route('/')
def index():
    return render_template_string(HTML, streams=STREAM_LABELS)


def _mjpeg_generator(buf: FrameBuffer):
    BOUNDARY = b'--frame'
    HEADER   = b'Content-Type: image/jpeg\r\n\r\n'

    # Placeholder negro mientras llega el primer frame
    placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(placeholder, 'Esperando stream...', (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
    _, ph_buf = cv2.imencode('.jpg', placeholder)
    placeholder_bytes = ph_buf.tobytes()

    while True:
        data = buf.read() or placeholder_bytes
        yield BOUNDARY + b'\r\n' + HEADER + data + b'\r\n'


@app.route('/stream/<name>')
def stream(name: str):
    if name not in _buffers:
        return Response('Stream no encontrado', status=404)
    return Response(
        _mjpeg_generator(_buffers[name]),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = WebVizNode()

    # Compartir buffers con Flask
    global _buffers
    _buffers = node._buffers

    # Flask en hilo daemon (muere cuando muere el proceso principal)
    flask_thread = threading.Thread(
        target=lambda: app.run(host=HOST, port=PORT, threaded=True),
        daemon=True,
    )
    flask_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
