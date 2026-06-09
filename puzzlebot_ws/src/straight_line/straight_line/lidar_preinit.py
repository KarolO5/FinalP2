#!/usr/bin/env python3
"""
lidar_preinit.py
Drena el buffer del RPLiDAR A1 antes de que rplidar_composition intente
comunicarse. Necesario cuando el nodo anterior murió sin mandar STOP
y el LiDAR quedó transmitiendo datos en loop.
"""
import sys
import time
import serial

PORT = '/dev/rplidar'
BAUD = 115200


def main():
    try:
        p = serial.Serial(PORT, BAUD, timeout=0.1)
    except serial.SerialException as e:
        print(f'[lidar_preinit] No se pudo abrir {PORT}: {e}', file=sys.stderr)
        sys.exit(1)

    print('[lidar_preinit] Drenando buffer del LiDAR...')

    # Manda STOP varias veces
    for _ in range(5):
        p.write(b'\xa5\x25')
        time.sleep(0.15)

    # Drena hasta que no lleguen datos por 1.5 s
    p.timeout = 0.1
    deadline = time.time() + 5.0
    total = 0
    while time.time() < deadline:
        chunk = p.read(512)
        if chunk:
            total += len(chunk)
            deadline = time.time() + 1.5  # reinicia si sigue llegando

    p.reset_input_buffer()
    p.reset_output_buffer()
    time.sleep(0.3)

    # Verifica que el LiDAR responda a GET_DEVICE_INFO
    p.timeout = 3.0
    p.write(b'\xa5\x50')
    time.sleep(0.3)
    resp = p.read(27)
    p.close()

    if resp[:2] == b'\xa5\x5a':
        print(f'[lidar_preinit] OK — LiDAR responde, {total} bytes drenados')
        sys.exit(0)
    else:
        # Aunque no responda, el buffer ya está limpio; el nodo puede intentarlo
        print(f'[lidar_preinit] Buffer drenado ({total} bytes). '
              f'Respuesta: {resp.hex()[:20]} — el nodo intentará init')
        sys.exit(0)


if __name__ == '__main__':
    main()
