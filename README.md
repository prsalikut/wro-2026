# WRO 2026 — Future Engineers — Self-Driving Car

Autonomous 1:10-scale vehicle for the WRO 2026 Future Engineers challenge. The car drives
the Open Challenge using LiDAR, and the Obstacle Challenge by detecting the red and green
traffic-sign pillars with a camera, fusing each detection's bearing with a LiDAR range, and
steering to the correct side of the lane.

Everything runs on a **Raspberry Pi 5 (4 GB)** as **ROS 2 Humble inside Docker** (host is
Raspberry Pi OS / Debian 13). A **USB webcam** and a **YDLIDAR X2** provide perception; an
**Arduino Nano** drives the steering servo and the traction motor over a serial link.

**Pass rule (rules §9.19):** a **red** pillar means keep to the lane's **RIGHT** (steer
right, so the pillar passes on the car's left); a **green** pillar means keep **LEFT**.

---

## Repository layout

This repository follows the WRO Future Engineers template structure.

| Folder | Contents |
|---|---|
| `src/ros2-package/` | The ROS 2 package deployed to the car — detectors, sensor fusion, driving nodes, the serial bridge, launch files, and the Docker build. **This is the deliverable.** |
| `src/arduino/` | Firmware for the Arduino Nano (steering servo + BTS7960 traction motor), plus a flashing script. |
| `src/tools/` | Host- and Pi-side utilities: the web dashboard / RC console, calibration helpers, frame grabbers, smoke tests. |
| `schemes/` | Electromechanical wiring diagrams. |
| `models/` | 3D-printed / laser-cut part files. |
| `t-photos/`, `v-photos/`, `video/` | Team photos, vehicle photos, driving demonstration video. |
| `other/dataset/` | YOLO-format dataset auto-labelled from practice photos (164 labelled boxes: 86 red, 78 green). |
| `other/prototyping/` | Colab notebook and standalone experiments used while developing the detector. |
| `other/screenshots/` | Live detection results captured from the car. |
| `other/rules/` | The official WRO 2026 FE rules PDF. |

`training-images/` (~2 GB of raw practice photos) is intentionally **not** committed; it is
shared separately. The derived, labelled dataset in `other/dataset/` is committed instead.

---

## Hardware and how the software maps onto it

| Component | Interface | Software that owns it |
|---|---|---|
| Raspberry Pi 5 (4 GB) | — | Runs the whole ROS 2 graph inside the `signstack` container |
| USB webcam | USB (V4L2) | `v4l2_camera` node → `/image_raw` |
| YDLIDAR X2 | USB serial | `ydlidar_ros2_driver` → `/scan` |
| Arduino Nano | USB serial (FTDI) | `steering_bridge` → `S <deg>` / `M <pct>` commands |
| Steering servo | Nano PWM | Firmware, commanded by `steering_bridge` |
| BTS7960 + traction motor | Nano PWM | Firmware, commanded by `steering_bridge` |
| IMU (BNO055 or MPU6050) | Pi I²C bus 1 | `imu` node, optional — auto-detected |
| 4× HC-SR04 | Nano D4 trigger, A0–A3 echo | `sonar` node via the serial bridge |
| Start button | Pi GPIO17 (pin 11) | `start_button` node |

The split is deliberate: the Pi does all perception and decision-making, and the Nano does
nothing but convert two numbers (steering angle, drive percent) into PWM. That keeps the
safety-critical actuation path small enough to audit, and lets the firmware run an
independent watchdog that stops the motor if the Pi stops talking.

---

## Software modules

### Perception

- **`sign_detector_node.py`** — the fusion node. Takes camera detections and, for each one,
  searches `/scan` around the expected bearing for a matching range, then publishes a
  `vision_msgs/Detection3DArray` on `/traffic_signs` plus RViz markers. The camera gives a
  precise bearing but a poor distance; the LiDAR gives a precise distance but cannot tell
  red from green. Fusing them yields a 3D pose with both.
- **`cnn_block_detector.py`** — the active detector. Localises saturated, block-shaped blobs
  in LAB colour space, then classifies each crop with a 64×64 2-class CNN
  (`index 0 = green, index 1 = red`) running on TFLite.
- **`block_detector.py`** — the original HSV detector. **Superseded**: it triggers on the
  orange track mat and on skin tones, because plain colour thresholds cannot distinguish a
  red pillar from anything else reddish in frame.
- **`ml_block_detector.py`** — thin wrapper for the optional YOLO path.

### Driving

- **`open_round_node.py`** — Open Challenge driver. A drive/turn/halt state machine that
  centres in the lane on left-right range difference, takes corners on the mat's colour
  lines (wall distance as backup), and stops after 12 corners. Optional gyro gives 90°
  corner exit and heading hold; both disable themselves when no IMU is present.
- **`sonar_node.py`** — republishes the Nano's four HC-SR04 ranges. These are the primary
  source for lane centring, since they read the glossy walls the LiDAR cannot.
- **`line_detector_node.py`** — finds the mat's orange and blue corner lines.
- **`imu_node.py`** — BNO055 or MPU6050 over I²C, with hot-plug probing.
- **`start_button_node.py`** — implements the WRO start procedure (rules 9.11/9.14):
  boot into a waiting state, first press starts the round, a later press stops it.
- **`sign_steering_node.py`** — Obstacle Challenge steering. Picks the nearest in-range
  pillar and applies the pass rule. Note it consumes **only** `/traffic_signs` and has no
  wall avoidance, so it is not used for the Open Challenge.
- **`follow_block_node.py`** — development helper that drives toward a chosen block.

### Actuation and safety

- **`steering_node.py`** (`steering_bridge`) — the only node that talks to the Nano. It
  applies a direction flip and a mechanical limit, then forwards `S <deg>` / `M <pct>`. It
  re-sends the current angle at least every 0.5 s so the firmware watchdog never trips
  during normal operation, and it zeroes the motor if commands go stale for longer than
  `cmd_timeout_s`. A manual window suspends autonomous forwarding while the RC console is
  in use, so the two cannot fight over the actuators.

### Tools

- **`tools/viz_server.py`** — web dashboard on port 8080: live camera and LiDAR views, a
  pre-flight checklist, stack start/stop, an E-stop, and a gamepad RC console (`/rc`) used
  for manual driving.
- **`tools/auto_calib.py`**, **`grab_frame.py`**, **`smoke_test.py`**, **`proxy.py`** —
  calibration, single-frame capture, end-to-end checks, and a TCP proxy for remote access.

### Firmware

`src/arduino/steering_firmware` — accepts a small command set over serial (`S` steering,
`M` motor, `C` centre, `ARM`, `E` e-stop, `WD`/`DWD` watchdogs, `GET` state, `PING`), with
an independent watchdog that stops the motor if the Pi goes quiet.

---

## Build and run

### Prerequisites

- Raspberry Pi 5 (4 GB), Raspberry Pi OS / Debian 13, Docker installed
- USB webcam, YDLIDAR X2, Arduino Nano — all connected to the Pi
- On a workstation: `git`, and `arduino-cli` or the Arduino IDE to flash the Nano

### 1. Deploy the code

```bash
git clone <this-repo-url> ~/sign_detector-repo
cp -r ~/sign_detector-repo/src/ros2-package/sign_detector ~/sign_detector
cd ~/sign_detector
```

### 2. Build the image

```bash
docker build -f docker/Dockerfile -t sign_detector:humble .
```

### 3. Run the stack

```bash
docker run -d --name signstack --network host --ipc host --privileged \
  -v /dev:/dev -v ~/sign_detector:/ros2_ws/src/sign_detector \
  --restart unless-stopped \
  sign_detector:humble \
  bash -lc 'source /opt/ros/humble/setup.bash && \
            source /ros2_ws/install/setup.bash && \
            exec ros2 launch sign_detector bringup.launch.py'
```

`~/sign_detector` is bind-mounted into the workspace, so edits on the Pi take effect after
a rebuild without rebuilding the image.

### 4. Rebuild after changing Python nodes

```bash
docker exec signstack bash -lc \
  'source /opt/ros/humble/setup.bash && cd /ros2_ws && \
   colcon build --packages-select sign_detector --symlink-install'
docker restart signstack
```

### 5. Verify

```bash
docker exec signstack bash -lc 'source /opt/ros/humble/setup.bash && ros2 node list'
```

Expect: `/camera`, `/ydlidar`, `/base_to_laser`, `/sign_detector`, `/steering_bridge`,
`/viz_server`. Then open `http://<pi-address>:8080` for the dashboard.

### 6. Start the web dashboard

The dashboard is launched by the stack, but it can be run standalone:

```bash
docker exec -d signstack bash -lc \
  'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && \
   python3 /ros2_ws/src/sign_detector/tools/viz_server.py'
```

Open **http://\<pi\>:8080** — three modes: desktop, assembly, final. The gamepad RC
console is at `/rc`.

### 7. Flash the Nano

```bash
cd src/arduino && ./flash.sh
```

### 8. Other checks

`src/tools/smoke_test.py` is the quick "is anything publishing?" check.
`src/tools/auto_calib.py` re-derives the camera-LiDAR angle offset.
`src/tools/grab_frame.py` grabs a single camera frame for offline inspection.

---

## Configuration

All runtime tuning lives in `src/ros2-package/sign_detector/config/params.yaml`, applied
per node. The values that matter most:

| Parameter | Meaning |
|---|---|
| `detector` | `cnn` (active) or `hsv` (superseded) |
| `weights` | Path to the `.tflite` model **as seen inside the container** |
| `angle_offset_deg` | Rotation between the LiDAR's zero and the car's forward axis |
| `steer_dir`, `max_deg` | Steering direction flip and mechanical limit |
| `drive_max_pct` | Hard ceiling on motor duty |
| `cmd_timeout_s` | How long a stale command is tolerated before the motor stops |

`angle_offset_deg` is the one to check first after any change to the LiDAR mounting: it
defines where "forward" is, and every steering decision depends on it.

---

## Status and known issues

Recorded honestly, because these affect how the car behaves:

- **The LiDAR cannot see the side walls reliably.** They are glossy black, which at 905 nm
  reflects almost specularly: a wall straight ahead returns rock-steady, a wall alongside
  returns little or nothing. Measured on the mat, bearings that did return read up to
  7.08 m on a 3 m field. Ultrasonics were added for exactly this reason — sound does not
  care about surface gloss, and a side-mounted sensor faces its wall square on.
- **The drivetrain stalls from rest.** 12% duty does not move the car, 30% lurches then
  stalls, ~100% for about a second breaks it free. The driver kicks then drops to a low
  sustain duty; a kick is only started with clear room ahead, because it travels ~0.6 m.
- **Open Challenge is not yet verified over three full laps on hardware.** The controller
  passes closed-loop simulation with zero collisions.
- **`sign_steering` is not started by `bringup.launch.py`.** It zeroes `/drive_cmd` whenever
  vision is quiet, which fights the RC console, so it is started deliberately from the
  dashboard instead of automatically.

---

## Documentation checklist

Per the WRO Future Engineers rules, still to be added:

- [ ] `t-photos/` — official and funny team photos
- [ ] `v-photos/` — six vehicle photos (front, back, left, right, top, bottom)
- [ ] `video/video.md` — public link to the driving demonstration
- [ ] `schemes/` — electromechanical wiring diagram
- [ ] `models/` — 3D-printed / laser-cut part files
