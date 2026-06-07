#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

DISTANCIA_PELIGRO = 0.5   # metros
ANGULO_FRONTAL    = 30.0  # grados a cada lado del frente


class ObstacleStop(Node):

    def __init__(self):
        super().__init__('obstacle_stop')

        self._obstaculo = False

        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # Timer a 20 Hz: mientras hay obstáculo pisa cualquier cmd_vel del line follower
        self.create_timer(0.05, self._timer_cb)

        self.get_logger().info(
            'obstacle_stop listo | d<{:.2f}m en +-{}°'.format(
                DISTANCIA_PELIGRO, ANGULO_FRONTAL))

    def _scan_cb(self, msg: LaserScan):
        angulo_rad = math.radians(ANGULO_FRONTAL)
        total = len(msg.ranges)
        inc   = msg.angle_increment
        if inc == 0 or total == 0:
            return

        idx_centro = int(round((0.0 - msg.angle_min) / inc))
        idx_delta  = int(round(angulo_rad / inc))

        hay_obstaculo = False
        for i in range(idx_centro - idx_delta, idx_centro + idx_delta + 1):
            r = msg.ranges[i % total]
            if msg.range_min < r < DISTANCIA_PELIGRO:
                hay_obstaculo = True
                break

        if hay_obstaculo != self._obstaculo:
            self._obstaculo = hay_obstaculo
            if hay_obstaculo:
                self.get_logger().warn('OBSTACULO! Frenando.')
            else:
                self.get_logger().info('Camino libre.')

    def _timer_cb(self):
        if self._obstaculo:
            self._cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleStop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
