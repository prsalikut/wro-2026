# WRO 2026 — Future Engineers — Self-Driving Car

Autonomous 1:10-scale vehicle for the WRO 2026 Future Engineers challenge. The car drives
the Open Challenge **from the camera**, using the lidar and the ultrasonics as corroboration
rather than as the primary sense, and the Obstacle Challenge by detecting the red and green
traffic-sign pillars with the same camera, fusing each detection's bearing with a LiDAR
range, and steering to the correct side of the lane.

The Open Challenge was rebuilt around the camera for a specific reason: the two range
sensors on this car have each been caught lying. The lidar reads nothing or nonsense off the
glossy black side walls at grazing incidence, and an HC-SR04 was recorded latched at exactly
0.43 m for 213 consecutive samples while the wall was 2.94 m away. A camera does not share
either failure — a black wall on a white mat is about the easiest thing there is to segment,
and the line where it meets the floor converts directly into a distance.

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
| 4× HC-SR04 | Nano D12 trigger, A0–A3 echo | `sonar` node via the serial bridge |
| START/STOP button | Pi GPIO17 (pin 11) | `start_button` node |
| MODE button | Pi GPIO27 (pin 13) | `start_button` node |

The split is deliberate: the Pi does all perception and decision-making, and the Nano does
nothing but convert two numbers (steering angle, drive percent) into PWM. That keeps the
safety-critical actuation path small enough to audit, and lets the firmware run an
independent watchdog that stops the motor if the Pi stops talking.

---

## Software modules

### Perception

- **`wall_vision.py`** + **`vision_node.py`** — the Open Challenge's primary sense. It
  learns what the floor looks like from a patch immediately in front of the car, marks
  everything darker than the local floor brightness, and walks up each image column to the
  first sustained dark run — the pixel where the floor ends. Projecting those pixels onto
  the ground plane gives a metric free-space profile, and fitting straight lines to its
  left, right and front groups yields perpendicular wall distances **and the car's yaw
  relative to the corridor**, which is a heading reference the car otherwise only has when
  an IMU is fitted. It also finds the orange and blue corner lines and reports them with a
  *range*, so a corner can be timed from a mark painted on the mat instead of guessed from a
  wall distance that depends on how wide this round's corridors happen to be.

  Nothing is tuned to one mat. The floor model is relearned every frame and a quadratic
  brightness surface is fitted over the floor pixels, which cancels both lens vignetting
  (this lens darkens the image corners by about 40%, enough for plain mat to read as a black
  wall — and those corners project to about 20 cm in front of the car, so the car brakes for
  its own optics) and any lighting gradient across the venue.

- **`ground_geometry.py`** — the pixel ↔ ground-plane mapping everything above rests on.
  Pure maths, no ROS, with a self-test: `python3 ground_geometry.py`.

- **`range_fusion.py`** — combines camera, lidar and sonar per axis and, more importantly,
  notices when one of them is lying. Absence is not the test, because a latched sonar never
  goes silent: a channel is demoted when its value stops changing while the car is moving,
  when it leaves the physical range, or when it sits well outside what the other sensors
  agree on for over a second. Demotions are reported in `open_status` and are reversible.

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

- **`open_round_core.py`** + **`open_round_node.py`** — the Open Challenge driver. All the
  decisions live in the core, which has no ROS, no OpenCV and no hardware dependency, so it
  can be run against a simulator on a laptop; the node is only the plumbing. It centres in
  the lane on fused wall distances plus the camera's heading, eases toward the outside of
  the corridor on a corner approach, counts twelve corners, and then runs out to where the
  round started and stops there (rule 9.24.2).

  Corner geometry is the hard part and it is worth stating plainly. This chassis steers 25°
  on a 0.15 m wheelbase, so its tightest arc is **0.32 m**. A right-angle corner between two
  1.0 m corridors needs 0.42 m or less and is comfortable; the rules also allow 0.6 m
  corridors, which need **0.19 m** and are therefore *impossible* in a single arc for this
  car. So a corner that runs out of room reverses and takes a second bite — a three-point
  turn, which rule 9.21 permits within the section where it happened. In a tight corridor
  the drive is also pulsed, because this drivetrain does not move at all below about 30%
  duty and so has no slow speed to drop into; short bursts separated by coasting roughly
  halve the average speed and double the number of decisions the controller gets per metre.

  The round's direction is worked out, not configured — rule 9.9 forbids entering data by
  switch. Turning is always *away from the outer wall*, and the outer wall is the one the
  camera keeps seeing, because it runs the whole three metres while the inner block is a
  metre long and drops out of a 60° field of view from mid-corridor. The mat's own colour
  convention (clockwise crosses **orange** first at every corner) is used only as a
  cross-check, never as the source of truth, because it is a property of the printed
  artwork rather than of the rulebook.
- **`sonar_node.py`** — republishes the Nano's four HC-SR04 ranges. These are the primary
  source for lane centring, since they read the glossy walls the LiDAR cannot.
- **`line_detector_node.py`** — the original corner-line detector. **Superseded** by
  `wall_vision`, which reports a distance to each line rather than a bare "seen" flag and
  applies a saturation and chroma test the old value-only threshold did not. It is no longer
  started by `bringup.launch.py`; two publishers on `line_event` would double-count corners.
- **`imu_node.py`** — BNO055 or MPU6050 over I²C, with hot-plug probing.
- **`start_button_node.py`** — implements the WRO start procedure (rules 9.11/9.14):
  boot into a waiting state, first press starts the round, a later press stops it.
  A second **MODE** button picks which round START launches — Open or Obstacle — so
  the car needs no dashboard on the field. MODE is refused while a round is running,
  and a button held at boot is ignored until released, so a stuck switch cannot
  launch the car. The selected mode is published in `/start_status`.
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

- **`tools/vision_calib.py`** — measures the camera mounting the whole open round
  depends on. `--fit` solves height and pitch from the lidar (no tape measure), `--hfov`
  solves the field of view from one measured pair of side distances, and `--check` compares
  the camera's ranges against the lidar's and the sonars' live. **Run `--check` before
  trusting a round.**
- **`tools/sim_offline.py`** + **`tools/sim_field.py`** — the closed-loop simulator. It
  renders the WRO field in 3D and runs the *real* perception and control code against the
  pixels, so a change can be checked on a laptop before it goes near the car. It also models
  the failures this car has actually had: the lidar's specular dropout, a latched sonar, a
  sonar reading a constant offset short, and a drivetrain that will not run below 30% duty.
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

### 8. Calibrate the camera — before the first driven round

Everything the open round steers on is a distance derived from where the wall meets the
mat, so it is only as good as the camera's height, pitch and field of view. Those have never
been measured on this car; the values in `params.yaml` are placeholders.

```bash
# solve height and pitch -- the lidar supplies truth, so no tape measure needed.
# point the car at a wall, capture, move it, capture again: 5-6 distances
# between about 0.3 m and 1.5 m.
python3 tools/vision_calib.py --fit

# solve the field of view from one tape-measured pair of side distances
python3 tools/vision_calib.py --hfov --left 0.42 --right 0.55

# then confirm the camera agrees with the lidar and the sonars.
# under 5 cm is good; over 15 cm means the mounting numbers are still wrong.
python3 tools/vision_calib.py --check
```

Paste the printed blocks into the `wall_vision:` section of `config/params.yaml` and
redeploy.

### 9. Simulate before you drive

```bash
python3 src/tests/test_open_round.py            # unit tests, no ROS needed
python3 src/tools/sim_offline.py --sweep        # every named configuration
python3 src/tools/sim_offline.py --random 200   # randomised rounds
python3 src/tools/sim_offline.py --video run.mp4  # watch what the camera saw
```

The sweep covers both directions, all four starting sections, the starting zones, corridor
widths from 0.5 m to 1.1 m, a frozen sonar, a sonar reading 0.9 m short, all four sonars
dead, a lidar returning almost nothing, dim and bright and colour-cast lighting, three
camera mountings, and a mat printed with the line colours swapped.

### 10. Other checks

`src/tools/smoke_test.py` is the quick "is anything publishing?" check.
`src/tools/auto_calib.py` re-derives the camera-LiDAR angle offset.
`src/tools/grab_frame.py` grabs a single camera frame for offline inspection.

---

## Configuration

All runtime tuning lives in `src/ros2-package/sign_detector/config/params.yaml`, applied
per node. The values that matter most:

| Parameter | Meaning |
|---|---|
| `cam_height_m`, `cam_pitch_deg`, `hfov_deg` | Camera mounting. **Measure these** — every vision range scales with them |
| `turn_at_m` | Front distance at which a corner starts |
| `corner_bias_m` | How far to ease toward the outside before a corner |
| `drive_pct` | Open-round motor duty; **0 makes it a steering-only dry run** |
| `require_start` | Wait for the START button (rules 9.11/9.14) |
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

Recorded honestly, because these affect how the car behaves.

**Fixed in the camera-first rewrite:**

- **The LiDAR cannot see the side walls reliably.** They are glossy black, and at grazing
  incidence a side wall returns little or nothing while a wall straight ahead returns
  rock-steady. Measured on the mat, bearings that did return read up to 7.08 m on a 3 m
  field. The camera does not share this failure and is now the primary side sensor; the
  lidar still supplies the front distance, where it is good.
- **A latched ultrasonic used to drive the car into a wall.** The right sonar sat at exactly
  0.43 m for 213 consecutive samples while the lidar put the nearest object on that side at
  2.94 m, and the front read ~1 m short, which put the car permanently "at a corner".
  `range_fusion` now demotes a channel whose value stops changing while the car is moving,
  or that disagrees with the other sensors for over a second, and reports it in
  `open_status`. Both faults are in the simulated test matrix.
- **`open_round` had no start gate.** It now subscribes to `start_status` and holds still
  until START is pressed (rules 9.11/9.14), while still self-arming if no `start_button`
  node exists at all, so bench runs work.
- **Resuming from a halt fired a phantom corner**, because the "front has been close for
  long enough" timer kept running across the stop.

**Open, and they matter:**

- **The camera mounting has never been measured.** `cam_height_m`, `cam_pitch_deg` and
  `hfov_deg` are placeholders, and the bench frames in `other/screenshots/` suggest the lens
  may sit near level rather than pitched down. Every vision range scales with them. Run
  `tools/vision_calib.py --fit` and `--hfov` before the first driven round, and `--check`
  before trusting any round. **This is the single largest risk to a real run.**
- **A 600 mm corridor cannot be cornered in one arc by this car.** 25° of steering on a
  0.15 m wheelbase gives a tightest arc of 0.32 m; the corner needs 0.19 m. The driver
  handles it with a three-point turn, which works but costs time. Opening the steering up
  would remove the need — the firmware envelope allows about 27° today (±650 µs at
  24 µs/deg), and reaching the ~38° that a 600 mm corner really wants needs a linkage change
  as well as a firmware one.
- **`angle_offset_deg` (170.0) is still unverified on the track.** It defines where the
  lidar thinks "forward" is. The camera no longer depends on it, so a wrong value is much
  less dangerous than it was, but the lidar's contribution will be wrong until it is checked.
- **The IMU does not respond** (`Errno 121` on I²C). Corners fall back to the camera's own
  alignment cue plus dead reckoning, which the simulator covers, but a working BNO055 would
  make corner exits tighter. Bridge S0 and S1 each to GND for address 0x28.
- **Nothing here has run on the car yet.** It passes 31 of 31 named simulated configurations
  and the unit tests, and the ROS parameter declarations type-check statically, but the ROS
  graph itself has never been brought up.

## Where the numbers come from

The simulated field is built from the official 2026 playfield artwork rather than from the
rulebook's figures, which are internally inconsistent about the corner lines. Measured from
the print file:

- Mat is **white**; the 100 mm surround band outside the walls is dark navy and is visible
  over the wall top to any camera mounted above 100 mm.
- Corner lines are **20 mm** bands, PANTONE 151 C (orange) and 2728 C (blue). They are
  **radial spokes, not chords**: each runs from the corner of the *nominal* 1000 × 1000 mm
  inner square out to the outer wall, meeting it **420 mm** from the field corner at about
  30° and 60°. They are printed at fixed positions and do **not** move when the inner block
  is resized, so in a 600 mm corridor the apex end is hidden under the block.
- Driving **clockwise** the car crosses **orange first** at every corner; counter-clockwise,
  blue first. This is a property of the artwork, not of the rules — rule 13.9 specifies only
  colour and thickness — so the driver uses it as a cross-check and never as the source of
  truth. The simulator can render a mat with the colours swapped, and the driver still
  passes.
- The printed hazards a vision system must not mistake for walls: a **yellow-green 3 mm
  ring 50 mm inside every wall**, right where the code looks for the floor/wall join, and a
  grey 4 mm dotted section grid. Both are in the simulated mat.

## Documentation checklist

Per the WRO Future Engineers rules, still to be added:

- [ ] `t-photos/` — official and funny team photos
- [ ] `v-photos/` — six vehicle photos (front, back, left, right, top, bottom)
- [ ] `video/video.md` — public link to the driving demonstration
- [ ] `schemes/` — electromechanical wiring diagram
- [ ] `models/` — 3D-printed / laser-cut part files
