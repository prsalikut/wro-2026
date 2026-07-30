# WRO 2026 — Future Engineers — Red/Green Sign Detection

Perception stack for the self-driving car: detect **red and green traffic-sign pillars**
from a webcam, fuse each detection's bearing with LiDAR range, and publish a 3D pose the
navigation layer can act on. Pass rule (rules §9.19) — **red → keep to the lane's RIGHT
(steer right; the pillar passes on the car's left); green → keep LEFT (steer left)**.

Runs live on a **Raspberry Pi 5 (4 GB)** as **ROS 2 Humble inside Docker** (host is Raspberry
Pi OS / Debian 13), with a **USB webcam** + **YDLIDAR X2**.

## Folder guide

| Folder | What's in it |
|---|---|
| `ros2-package/sign_detector/` | The ROS 2 package that runs on the robot — HSV + YOLO detectors, the fusion node (`vision_msgs/Detection3DArray` + RViz markers), launch files, and the `docker/` build (Dockerfile, compose, bringup launch with the X2 driver). **This is the deliverable that's deployed to the Pi.** |
| `dataset/` | `wro_dataset/` — auto-labeled YOLO dataset from the training photos (102 labeled, 86 red + 78 green boxes), plus `wro_dataset.zip` for Roboflow. |
| `training-images/` | 142 real photos of the practice blocks (+ original zip). **Not included here** — ~2 GB, shared separately. |
| `prototyping/` | `WRO_sign_detector.ipynb` (Colab: YOLO11n on synthetic data → NCNN/TFLite), `red_green_test.py` (minimal standalone HSV tester), and dev/diagnostic images. |
| `screenshots/` | Live detection results off the Pi, plus `detection_gallery.html` (open in a browser). |
| `rules/` | The official WRO 2026 FE rules PDF. |

## Key facts

- **Official sign colors:** red RGB (238,39,55), green RGB (68,214,44); pillars 50×50×100 mm.
- **Practice blocks ≠ official pillars:** the training photos are warm-lit cardboard, so red
  reads orange (OpenCV hue ~4–16) and green yellow-green (hue ~32–40). The detector uses a
  high **saturation** floor to separate red from the floor/wood. Switch `PROFILE`/`profile` to
  `"official"` and recalibrate at the venue.

## Setup — start here

### What you need

- Raspberry Pi 5 (4 GB) running Raspberry Pi OS / Debian 13, with Docker installed
- USB webcam, YDLIDAR X2, Arduino Nano (steering), all plugged into the Pi
- On your laptop: `git`, plus the Arduino IDE or `arduino-cli` if you plan to reflash the Nano

### 1. Get the code onto the Pi

```bash
git clone <this-repo-url> ~/sign_detector-repo
cp -r ~/sign_detector-repo/ros2-package/sign_detector ~/sign_detector
cd ~/sign_detector
```

### 2. Build and run the stack

```bash
docker build -f docker/Dockerfile -t sign_detector:humble .

docker run -d --name signstack --network host --ipc host --privileged \
  -v /dev:/dev -v ~/sign_detector:/ros2_ws/src/sign_detector \
  sign_detector:humble ros2 launch sign_detector bringup.launch.py
```

`bringup.launch.py` starts everything: the X2 LiDAR driver, the camera, the detector, the
fusion node, `sign_steering` (pass rule) and `steering_bridge` (serial to the Nano).

Check it came up:

```bash
docker logs -f signstack
docker exec -it signstack bash -lc 'source /opt/ros/humble/setup.bash && ros2 topic list'
```

Topics you should see: `/image_raw`, `/scan`, `/traffic_signs`, `/sign_debug`, `/steering_cmd`.

Stop it with `docker stop signstack` (and `docker rm signstack` before re-running).

### 3. Flash the Arduino (only if the firmware changed)

```bash
cd arduino
./flash.sh              # see the script for the port it expects
```

The Nano enumerates as an FTDI device. If `flash.sh` can't find it, list ports with
`ls /dev/serial/by-id/` and pass the right one.

### 4. Start the web dashboard (test console)

```bash
mkdir -p ~/sign_detector/tools
cp ~/sign_detector-repo/tools/viz_server.py ~/sign_detector/tools/

docker exec -d signstack bash -lc \
  'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && python3 /ros2_ws/src/sign_detector/tools/viz_server.py'
```

Open **http://\<pi\>:8080** — three modes: desktop, assembly, final.

### 5. Check it works

`tools/smoke_test.py` is the quick "is anything publishing?" check.
`tools/auto_calib.py` re-derives the camera↔LiDAR angle offset.
`tools/grab_frame.py` grabs a single camera frame for offline inspection.

### Tuning

All the knobs live in `ros2-package/sign_detector/config/params.yaml` — HSV bands, the
detector backend (`hsv` / `cnn`), `sign_height_m`, and the steering limits. **You will need
to recalibrate the HSV values at the venue**, because the practice blocks and the official
pillars are different colors (see Key facts above).

### A note on the training images

`training-images/` and the dataset zip are **not in this repo** — they're ~2 GB, well past
what GitHub accepts. Ask for them separately if you need to retrain; nothing in the runtime
stack depends on them.

## Status

- ✅ Camera + X2 + detector up in one container; red & green classified with bearings;
  `/traffic_signs` publishing 3D poses; camera color cast fixed.
- ✅ Camera↔LiDAR angle calibrated (`angle_sign`/`angle_offset_deg`, auto_calib 2026-07-11);
  fusion matcher fixed to take the nearest plausible cluster, so the range reads the
  pillar, not the wall behind it.
- ✅ Steering live end-to-end (2026-07-11): `bringup.launch.py` also starts `sign_steering`
  (pass rule → `/steering_cmd`) and `steering_bridge` (serial → Arduino Nano, `arduino/`
  firmware). Servo throw calibrated to ±600 µs (DEG2US 24, `S ±25` = full lock); layered
  failsafes: vision-quiet centering, stale-command centering, firmware watchdog.
- ⏳ Next: drive motor (BTS7960, phase 2); optionally fine-tune the YOLO model on real
  images; recalibrate HSV + `sign_height_m` at the venue on the official pillars.
