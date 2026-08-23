# Electromechanical schematics

Required by the WRO Future Engineers documentation rules: one or more schematic
diagrams (JPEG/PNG/PDF) showing the electromechanical components and how they are
wired together.

Must document the actual build:

| Component | Connects to |
|---|---|
| Raspberry Pi 5 (4 GB) | host controller; runs ROS 2 Humble in Docker |
| USB webcam | Pi USB — `/image_raw` |
| YDLIDAR X2 | Pi USB serial — `/scan` |
| Arduino Nano (FTDI) | Pi USB serial — steering + drive commands |
| Steering servo | Nano PWM |
| BTS7960 motor driver | Nano PWM — drive motor |
| MPU6050 IMU | Pi I2C — SDA pin 3, SCL pin 5, VCC pin 17 (3.3 V), GND pin 9 |

Not yet supplied — the wiring diagram must be drawn and added.
