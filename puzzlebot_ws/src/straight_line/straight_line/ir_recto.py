#!/usr/bin/env python3
"""
ir_recto.py — El robot avanza en línea recta a velocidad fija.
Nada más. Sin cámara, sin odometría, sin PD.

Ajusta LINEAR_VEL para cambiar la velocidad.
Ctrl+C para detener.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

LINEAR_VEL = 0.10   # m/s — cambia este valor para ir más rápido o lento

class IrRectoNode(Node):
    def __init__(self):
        super().__init__('ir_recto')
        self._pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._timer = self.create_timer(0.05, self._cb)   # 20 Hz
        self.get_logger().info(f'IrRecto: avanzando a {LINEAR_VEL} m/s — Ctrl+C para detener')

    def _cb(self):
        cmd = Twist()
        cmd.linear.x  = LINEAR_VEL
        cmd.angular.z = 0.0
        self._pub.publish(cmd)

    def stop(self):
        self._pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = IrRectoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.get_logger().info('Detenido.')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
