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


class LidarStop(Node):

    def __init__(self):
        super().__init__('lidar_stop')

        self._parar = False

        self._pub = self.create_publisher(Bool, '/lidar/parar', 10)

        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)

        self.get_logger().info(
            'LidarStop listo | dist={:.2f}m angulo=+-{}deg'.format(
                DISTANCIA_PELIGRO, ANGULO_FRONTAL))

    def _scan_cb(self, msg: LaserScan):
        angulo_rad = math.radians(ANGULO_FRONTAL)
        total = len(msg.ranges)
        inc   = msg.angle_increment

        idx_centro = int(round((0.0 - msg.angle_min) / inc))
        idx_delta  = int(round(angulo_rad / inc))

        hay_obstaculo = False
        for i in range(idx_centro - idx_delta, idx_centro + idx_delta + 1):
            r = msg.ranges[i % total]
            if msg.range_min < r < DISTANCIA_PELIGRO:
                hay_obstaculo = True
                break

        if hay_obstaculo != self._parar:
            self._parar = hay_obstaculo
            if hay_obstaculo:
                self.get_logger().warn('OBSTACULO DETECTADO — frenando')
            else:
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
