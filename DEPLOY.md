# Meet-up checklist

Everything below is written, validated and committed to the working tree. None of
it is on the Pi yet — it went offline mid-session.

## 1. Hardware to fit

| Part | Where | Notes |
|---|---|---|
| BNO055 (optional) | VIN→pin 17, GND→pin 9, SDA→pin 3, SCL→pin 5 | solder header first; **bridge S0 and S1 each to GND** (selects I2C at 0x28); skip the loose crystal; never bridge S0/S1 to VCC |
| Start button | either leg→pin 11, other leg→pin 14 | no resistor, internal pull-up |
| 4× HC-SR04 | TRIG(all)→Nano D4, ECHO→Nano A0/A1/A2/A3 | 5 V rail, Nano is 5 V native so no dividers |

Sonar order is **front, right, rear, left** = A0, A1, A2, A3. Wiring them in a
different order silently swaps the readings.

## 2. Flash the Nano

    cd src/arduino && ./flash.sh

Adds `US` (one-shot), `USON`/`USOFF` (streaming). Interrupt-driven on PCINT1 so
the drive watchdog and serial parser never block.

## 3. Push to the Pi

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
    # expect: {"state": "waiting"}

    # sonar streaming
    ssh pi@100.115.88.108 "docker exec signstack bash -lc \
      'source /opt/ros/humble/setup.bash && ros2 topic echo /sonar/left --once'"

## 5. Known follow-ups

- `start_button` defaults to `gpiochip4`; if the button never reads, try
  `gpiochip0`. Needs `lgpio` or `python3-libgpiod` in the container.
- `sonar` sends one `USON` at startup. Every `arduino_cmd` opens a manual window
  in the bridge which suspends autonomy, so it must stay one-shot, never polled.
- The bridge's `RAW_ALLOWED` whitelist lives in the Pi's newer `steering_node.py`
  (not the repo copy). `USON`/`USOFF` may need adding there, and read-only verbs
  (`PING`, `GET`, `US`) should be exempted from opening a manual window.
- Sonar is now the PRIMARY source for lane centring, front guard and backout,
  with lidar as per-side fallback. Disable via `sonar_centre:=false` /
  `sonar_front_guard:=false` if a sensor misbehaves.
- The car drives without an IMU: no gyro simply disables heading hold and
  angle-based corner exit. Sonar left/right carry lane keeping on their own.

## What changed since the last working state

- open_round: gyro 90° corners + heading hold (auto-disables with no IMU),
  mat-line corner triggers, trigger-colour lock, 12-corner completion
- imu: BNO055 + MPU6050 with hot-plug probing
- line_detector: HSV thresholds measured off this camera, not defaults
- start_button, sonar: new
- sim: models mat lines and publishes real line events; viewer renders them
- params.yaml: stale `open_round` section removed (it was reverting the
  calibrated lidar offset on any params-file launch)
