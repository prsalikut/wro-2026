# Meet-up checklist

Everything below is written, validated and committed to the working tree. None of
it is on the Pi yet — it went offline mid-session.

## 1. Hardware to fit

| Part | Where | Notes |
|---|---|---|
| BNO055 (optional) | VIN→pin 17, GND→pin 9, SDA→pin 3, SCL→pin 5 | solder header first; **bridge S0 and S1 each to GND** (selects I2C at 0x28); skip the loose crystal; never bridge S0/S1 to VCC |
| START/STOP button | either leg→pin 11 (GPIO17), other leg→pin 14 (GND) | no resistor, internal pull-up |
| MODE button | either leg→pin 13 (GPIO27), other leg→pin 14 (GND) | shares the pin 14 ground with START |
| 4× HC-SR04 | TRIG(all)→Nano D12, ECHO→Nano A0/A1/A2/A3 | 5 V rail, Nano is 5 V native so no dividers |

Sonar order is **front, right, rear, left** = A0, A1, A2, A3. Wiring them in a
different order silently swaps the readings.

## 2. Flash the Nano

    cd src/arduino && ./flash.sh

Adds `US` (one-shot), `USON`/`USOFF` (streaming). Interrupt-driven on PCINT1 so
the drive watchdog and serial parser never block.

## 3. Push to the Pi

From a machine that can reach the Pi (not a cloud session), one command does the
whole cycle -- copy, rebuild, restart, then check every sensor:

    ./src/tools/deploy.sh            # deploy + build + check
    ./src/tools/deploy.sh --flash    # also reflash the Nano first
    ./src/tools/deploy.sh --check-only
    ./src/tools/deploy.sh --logs     # add container logs

In Claude Code, the same thing is `/car` (see `.claude/commands/car.md`), which
also reads the sensor table back and says what to fix.

Override the address with `--host user@addr` or `PI_HOST`.

### By hand

    scp -r src/ros2-package/sign_detector/* pi@100.115.88.108:/home/pi/sign_detector/
    ssh pi@100.115.88.108 "docker exec signstack bash -lc \
      'source /opt/ros/humble/setup.bash && cd /ros2_ws && \
       colcon build --packages-select sign_detector --symlink-install' && \
      docker restart signstack"

## 4. Verify, in this order

    # IMU present
    ssh pi@100.115.88.108 "docker logs --tail 40 signstack 2>&1 | grep -i imu"
    # expect: BNO055 up ... NDOF

    # button reads
    ssh pi@100.115.88.108 "docker exec signstack bash -lc \
      'source /opt/ros/humble/setup.bash && ros2 topic echo /start_status --once'"
    # expect: {"state": "waiting", "mode": "open"}
    # press MODE, echo again: mode flips to "obstacle". If it never changes,
    # the MODE button is on the wrong line -- START is line 17, MODE is 27.

    # sonar streaming
    ssh pi@100.115.88.108 "docker exec signstack bash -lc \
      'source /opt/ros/humble/setup.bash && ros2 topic echo /sonar/left --once'"

## 5. Known follow-ups

- **Right and front sonar read wrong (2026-08-23, measured on hardware).**
  Right sat at exactly 0.43 m for 213 consecutive samples while the lidar put
  the nearest object on that side at 2.94 m. Front read ~0.93-1.47 m against a
  lidar front of 2.27 m -- under `turn_m` (1.10), so the car believes it is at a
  corner permanently, turns into the wall, and `reverse_enable` backs it into
  the wall behind. Reseat ECHO on A1 (right) and A0 (front) with 5V/GND, or swap
  the sensors. Order is front/right/rear/left = A0/A1/A2/A3.
- **`open_round` has no start gate.** It does not subscribe to `start_status`,
  so it drives the instant it launches and the START button does nothing for the
  open round. WRO requires starting on a button press; this needs fixing before
  a competition run.
- **`angle_offset_deg` (170.0) is unverified on the track.** If it is wrong the
  chassis frame is rotated, "front" is not front, and the car halts and reverses
  against a wall it thinks is ahead. Confirm before trusting a round: sweep
  candidate offsets and pick the one where front is the largest open gap and
  left/right are roughly equal.
- IMU does not respond (`Errno 121` on I2C). Check S0 and S1 are each bridged to
  GND for address 0x28. Heading hold and gyro corners stay off without it.

- `start_button` defaults to `gpiochip4`; if the button never reads, try
  `gpiochip0`. Needs `lgpio` or `python3-libgpiod` in the container.
- `sonar` sends one `USON` at startup. Every `arduino_cmd` opens a manual window
  in the bridge which suspends autonomy, so it must stay one-shot, never polled.
- The bridge's `RAW_ALLOWED` whitelist lives in the Pi's newer `steering_node.py`
  (not the repo copy). `USON`/`USOFF` may need adding there, and read-only verbs
  (`PING`, `GET`, `US`) should be exempted from opening a manual window.
### Sensor roles (complementary, not a fallback chain)

| Sensor | Role | Degrades to |
|---|---|---|
| Ultrasonic L/R | trusted side range, larger fusion weight | lidar side, if credible |
| Lidar | front distance, corner detection, obstacle round | sonar front guard |
| IMU | heading reference: 90 deg corner exit, heading hold | error-derivative damping |

`_fuse_side` weights sonar 0.7 against lidar 0.3 when the two agree within
`fuse_tol_m` (0.22). Beyond that it flags a disagreement and takes the sonar,
because grazing incidence on glossy black is the known lidar failure and it
reads long, never short. Disagreements are published in `open_status` — watch
that field on the first runs, it is the fastest way to spot a mis-wired sensor.

Front uses the NEAREST of lidar and sonar so either can stop the car.

Each degrades independently: no IMU loses heading hold but keeps derivative
damping from the ranges; no sonar falls back to lidar sides; no lidar keeps
sonar centring and the sonar front guard.

## What changed since the last working state

- open_round: gyro 90° corners + heading hold (auto-disables with no IMU),
  mat-line corner triggers, trigger-colour lock, 12-corner completion
- imu: BNO055 + MPU6050 with hot-plug probing
- line_detector: HSV thresholds measured off this camera, not defaults
- start_button, sonar: new
- sim: models mat lines and publishes real line events; viewer renders them
- params.yaml: stale `open_round` section removed (it was reverting the
  calibrated lidar offset on any params-file launch)
