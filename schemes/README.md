# Electromechanical schematics

Required by the WRO Future Engineers documentation rules: one or more schematic
diagrams (JPEG/PNG/PDF) showing the electromechanical components and how they
are wired together.

**Still to add:** the drawn diagram itself. What follows is the authoritative
parts list and connection table it must depict, kept in step with the firmware
and the ROS nodes.

---

## Parts list

### Compute and perception

| # | Part | Notes |
|---|---|---|
| 1 | Raspberry Pi 5, 4 GB | host controller; ROS 2 Humble in Docker |
| 1 | Pi 5 active cooler / fan | on header pin 4 (5 V) and pin 6 (GND) |
| 1 | microSD card | Raspberry Pi OS / Debian 13 |
| 1 | USB webcam | `/image_raw` via `v4l2_camera` |
| 1 | YDLIDAR X2 | USB serial (CP210x) -> `/scan` |
| 1 | BNO055 IMU, WCMCU-055 board | Pi I2C bus 1, address 0x28 |
| 4 | HC-SR04 ultrasonic | front, right, rear, left |
| 2 | momentary push button | START/STOP and MODE |

### Actuation

| # | Part | Notes |
|---|---|---|
| 1 | Arduino Nano, ATmega328P, FTDI FT232R | USB serial from the Pi |
| 1 | MG995 servo | rack-pivot front steering |
| 1 | BTS7960 half-bridge driver | traction motor |
| 1 | Johnson 500 motor | needs >=40-50% duty to break away |

### Power

| # | Part | Notes |
|---|---|---|
| 1 | 3S LiPo | motor power only |
| 1 | 20 A fuse | in the 3S positive line, before the switch |
| 1 | main power switch | between fuse and BTS7960 B+ |
| 1 | 5-6 V supply for the servo | **separate from the Nano's 5 V rail** |
| 1 | Pi power source | USB-C, independent of the motor pack |

> **Three power domains, one ground.** Motor (3S), servo (5-6 V) and Pi are
> separately supplied, but **every ground ties to one common bus**. An MG995
> stalls at a couple of amps and will brown out the Nano's regulator, resetting
> the board mid-run, so it never draws from the Nano 5 V pin.

---

## Connections

### Raspberry Pi 5 — 40-pin header

| Pin | Signal | Goes to |
|---|---|---|
| 1 | 3V3 | BNO055 `VCC` |
| 3 | GPIO2 / SDA1 | BNO055 `ATX` (SDA) |
| 4 | 5V | cooling fan + |
| 5 | GPIO3 / SCL1 | BNO055 `LRX` (SCL) |
| 6 | GND | cooling fan − |
| 9 | GND | BNO055 `GND` |
| 11 | GPIO17 | START/STOP button |
| 13 | GPIO27 | MODE button |
| 14 | GND | both buttons, common leg |
| 25 | GND | BNO055 `I2C` pin — selects address 0x28 |

Buttons are active-low against an internal pull-up: each is a bare switch to
ground, no resistor. On the BNO055, **S0 and S1 are solder-bridged to their −
pads**, which selects I2C mode (PS0 = PS1 = low).

If the IMU does not appear at 0x28, swap `ATX` and `LRX`: sources disagree on
which is SDA, and the lines are open-drain so a swap cannot damage anything.

### Pi — USB

| Port | Device |
|---|---|
| USB | webcam |
| USB | YDLIDAR X2 (CP210x — `flash.sh` skips this port) |
| USB | Arduino Nano (FTDI FT232R, 115200 8N1) |

### Arduino Nano

| Pin | Signal | Goes to |
|---|---|---|
| D2 | servo PWM | MG995 signal (orange) |
| D5 | RPWM forward | BTS7960 `RPWM` (Timer0 OC0B) |
| D6 | LPWM reverse | BTS7960 `LPWM` (Timer0 OC0A) |
| D7 | R_EN | BTS7960 `R_EN` — raised at boot, stays high |
| D8 | L_EN | BTS7960 `L_EN` — raised at boot, stays high |
| D12 | sonar trigger | all four HC-SR04 `TRIG`, shared |
| A0 | echo | HC-SR04 **front** |
| A1 | echo | HC-SR04 **right** |
| A2 | echo | HC-SR04 **rear** |
| A3 | echo | HC-SR04 **left** |
| 5V | logic | BTS7960 `VCC`, HC-SR04 `VCC` |
| GND | — | common ground bus |

> **D9/D10 are unusable for PWM.** `Servo.h` owns Timer1. Motor PWM must stay on
> D5/D6.

> **The echoes must stay on A0–A3.** All four share PORTC, and one PCINT1 vector
> times them together. The trigger is a plain output and can move; the echoes
> cannot.

> **Sonar order is silent when wrong.** front=A0, right=A1, rear=A2, left=A3. A
> swapped pair produces valid-looking readings and breaks lane centring.

### Power wiring

```
3S LiPo (+) ──[20 A fuse]──[switch]──> BTS7960 B+
3S LiPo (−) ─────────────────────────> BTS7960 B− ──┐
                                                     │
5–6 V PSU (+) ──> MG995 V+ (red)                     ├── COMMON GROUND BUS
5–6 V PSU (−) ──> MG995 GND (brown) ─────────────────┤
                                                     │
Arduino GND ─────────────────────────────────────────┘
```

Motor and servo grounds must both reach the Nano's ground, or the PWM
references float.

---

## Sensor roles

Complementary, not a fallback chain — each degrades without disabling the others.

| Sensor | Role | Degrades to |
|---|---|---|
| Ultrasonic L/R | trusted side range, larger fusion weight | lidar side, if credible |
| Lidar | front distance, corner detection, obstacle round | sonar front guard |
| IMU | heading reference: 90° corner exit, heading hold | error-derivative damping |

The lidar cannot read the glossy black side walls at grazing incidence — it
returns long, never short — which is why the ultrasonics carry the side range.

---

## Bring-up order

1. Solder the BNO055 header; bridge **S0 → −** and **S1 → −**
2. Wire the Pi header per the table above
3. Wire the Nano: servo, BTS7960, then the four sonars
4. Common ground bus last — check continuity before applying power
5. `./src/tools/deploy.sh --flash` (or `/car --flash` in Claude Code)
6. Confirm `i2cdetect -y 1` shows `0x28`
7. Wave a hand 20 cm from one sonar at a time; confirm the matching topic moves
