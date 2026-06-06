#!/usr/bin/env python3
# =============================================================================
# web_viz.py  --  Snapshots periodicos en localhost:8080  (sin MJPEG)
# =============================================================================
# En vez de streaming de video, guarda fotos JPEG en /tmp/puzzlebot_viz/
# sobreescribiendo siempre los mismos 3 archivos (sin acumular memoria).
# La pagina HTML las recarga con JS cada REFRESH_MS milisegundos.
#
# Streams:
#   raw.jpg      <- /image/raw
#   linea.jpg    <- /vision/debug_img
#   semaforo.jpg <- /semaforo/debug_img
# =============================================================================

import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

PORT        = 8080
SAVE_DIR    = '/tmp/puzzlebot_viz'
REFRESH_MS  = 1500    # milisegundos entre recargas de imagen en el browser
JPEG_Q      = 80      # calidad JPEG (0-100)

STREAMS = {
    'raw':      '/image/raw',
    'linea':    '/vision/debug_img',
    'semaforo': '/semaforo/debug_img',
}
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
         display:flex;flex-direction:column;min-height:100vh}
    header{padding:10px 18px;background:#12122a;
           border-bottom:2px solid #00ffe7;
           display:flex;align-items:center;gap:12px;flex-shrink:0}
    header h1{font-size:1rem;color:#00ffe7;margin-right:auto}
    button{padding:6px 14px;border:1px solid #444;border-radius:4px;
           background:#1e1e3a;color:#aaa;cursor:pointer;
           font-family:monospace;font-size:0.85rem}
    button:hover{border-color:#00ffe7;color:#00ffe7}
    button.active{background:#00ffe7;color:#0d0d1a;border-color:#00ffe7;font-weight:bold}
    .badge{font-size:0.8rem;color:#555}
    .viewer{flex:1;display:flex;align-items:center;
            justify-content:center;background:#000;padding:8px}
    #feed{max-width:100%;max-height:calc(100vh - 60px);
          display:block;object-fit:contain}
    #ts{font-size:0.7rem;color:#444;padding:4px 18px;text-align:right}
  </style>
</head>
<body>
  <header>
    <h1>PuzzleBot Vision</h1>
    <button onclick="setView('raw')"      id="btn-raw">RAW</button>
    <button onclick="setView('linea')"    id="btn-linea">L&Iacute;NEA</button>
    <button onclick="setView('semaforo')" id="btn-semaforo">SEM&Aacute;FORO</button>
    <span class="badge" id="fps-badge"></span>
  </header>
  <div class="viewer">
    <img id="feed" src="/img/raw.jpg" alt="snapshot">
  </div>
  <div id="ts">--</div>
  <script>
    var current = 'raw';
    var lastLoad = 0;
    document.getElementById('btn-raw').classList.add('active');

    function setView(name){
      current = name;
      ['raw','linea','semaforo'].forEach(function(k){
        document.getElementById('btn-'+k).classList.toggle('active', k===name);
      });
      refresh();
    }

    function refresh(){
      var t = Date.now();
      var img = document.getElementById('feed');
      var newSrc = '/img/' + current + '.jpg?t=' + t;
      var tmp = new Image();
      tmp.onload = function(){
        img.src = newSrc;
        var dt = t - lastLoad;
        if(lastLoad > 0){
          document.getElementById('fps-badge').textContent =
            'ultima foto: ' + new Date(t).toLocaleTimeString();
        }
        lastLoad = t;
        document.getElementById('ts').textContent = new Date(t).toLocaleTimeString();
      };
      tmp.onerror = function(){};
      tmp.src = newSrc;
    }

    setInterval(refresh, """ + str(REFRESH_MS).encode() + b""");
  </script>
</body>
</html>"""


# -----------------------------------------------------------------------
# Escritor de snapshots thread-safe
# -----------------------------------------------------------------------

class SnapshotWriter:
    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()
        # Escribir placeholder negro
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        name = os.path.splitext(os.path.basename(path))[0]
        cv2.putText(img, 'Esperando: {}'.format(name), (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
        self._write(img)

    def update(self, frame: np.ndarray):
        with self._lock:
            self._write(frame)

    def _write(self, frame: np.ndarray):
        # Codificar en memoria para evitar dependencia del codec JPEG del sistema
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        if not ok:
            return
        tmp = self._path + '.tmp'
        with open(tmp, 'wb') as f:
            f.write(buf.tobytes())
        os.replace(tmp, self._path)


# -----------------------------------------------------------------------
# HTTP handler
# -----------------------------------------------------------------------

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

        if self.path.startswith('/img/'):
            filename = self.path.split('?')[0][len('/img/'):]
            filepath = os.path.join(SAVE_DIR, filename)
            if not os.path.isfile(filepath):
                self.send_response(404)
                self.end_headers()
                return
            with open(filepath, 'rb') as f:
                data = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Cache-Control', 'no-cache, no-store')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_response(404)
        self.end_headers()


# -----------------------------------------------------------------------
# Nodo ROS 2
# -----------------------------------------------------------------------

class WebVizNode(Node):

    def __init__(self):
        super().__init__('web_viz')

        os.makedirs(SAVE_DIR, exist_ok=True)

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
        qos_map = {'best_effort': qos_be, 'reliable': qos_rel}

        self._bridge   = CvBridge()
        self._writers  = {}

        for key in STREAMS:
            path = os.path.join(SAVE_DIR, '{}.jpg'.format(key))
            self._writers[key] = SnapshotWriter(path)

        for key, topic in STREAMS.items():
            w   = self._writers[key]
            qos = qos_map[QOS_BY_KEY[key]]
            self.create_subscription(
                Image, topic,
                lambda msg, wr=w: self._cb(msg, wr),
                qos,
            )

        self.get_logger().info(
            'WebVizNode listo -> http://0.0.0.0:{} | '
            'snapshots en {} cada ~{}ms'.format(PORT, SAVE_DIR, REFRESH_MS)
        )

    def _cb(self, msg: Image, writer: SnapshotWriter):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            writer.update(frame)
        except Exception as e:
            self.get_logger().warn('cv_bridge: {}'.format(e))


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = WebVizNode()

    server = HTTPServer(('0.0.0.0', PORT), Handler)
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
