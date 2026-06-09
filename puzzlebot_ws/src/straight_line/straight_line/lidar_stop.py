#!/usr/bin/env python3
# =============================================================================
# lidar_stop.py  --  Detector de obstaculos por LiDAR
# =============================================================================
# Publica /lidar/parar (Bool) True cuando hay un obstaculo en el cono frontal.
# El line_follower_cv suscribe ese flag y frena; este nodo nunca toca /cmd_vel.
#
# TOPICOS
#   Sub : /scan         [sensor_msgs/LaserScan]
#   Pub : /lidar/parar  [std_msgs/Bool]
# =============================================================================

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

DISTANCIA_PELIGRO = 0.35   # metros
ANGULO_FRONTAL    = 30.0   # grados a cada lado del frente
FRAMES_PARA_FRENAR  = 3    # frames consecutivos con obstaculo para activar freno
FRAMES_PARA_LIBERAR = 5    # frames consecutivos sin obstaculo para liberar


class LidarStop(Node):

    def __init__(self):
        super().__init__('lidar_stop')

        self._parar          = False
        self._frames_obst    = 0
        self._frames_libre   = 0

        self._pub = self.create_publisher(Bool, '/lidar/parar', 10)

        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)

        self.get_logger().info(
            'LidarStop listo | dist={:.2f}m angulo=+-{}deg'.format(
                DISTANCIA_PELIGRO, ANGULO_FRONTAL))

    def _scan_cb(self, msg: LaserScan):
        angulo_rad = math.radians(ANGULO_FRONTAL)
        total = len(msg.ranges)
        inc   = msg.angle_increment

        idx_centro = int(round((math.pi - msg.angle_min) / inc))
        idx_delta  = int(round(angulo_rad / inc))

        hay_obstaculo = False
        for i in range(idx_centro - idx_delta, idx_centro + idx_delta + 1):
            r = msg.ranges[i % total]
            if msg.range_min < r < DISTANCIA_PELIGRO:
                hay_obstaculo = True
                break

        if hay_obstaculo:
            self._frames_obst  += 1
            self._frames_libre  = 0
        else:
            self._frames_libre += 1
            self._frames_obst   = 0

        if not self._parar and self._frames_obst >= FRAMES_PARA_FRENAR:
            self._parar = True
            self.get_logger().warn('OBSTACULO DETECTADO — frenando')
        elif self._parar and self._frames_libre >= FRAMES_PARA_LIBERAR:
            self._parar = False
            self.get_logger().info('Camino libre — reanudando')

        out = Bool()
        out.data = self._parar
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = LidarStop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        out = Bool(); out.data = False
        node._pub.publish(out)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
