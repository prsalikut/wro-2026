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

## 5. Calibrate the camera FIRST

The open round now drives on the camera. Everything it reports is a distance derived from
where the wall meets the mat, so it is only as good as three numbers that have never been
measured on this car -- lens height, pitch and field of view. The values in `params.yaml`
are placeholders, and the bench frames suggest the lens may sit near level rather than
pitched down.

    # inside the container, on the Pi
    python3 /ros2_ws/src/sign_detector/tools/vision_calib.py --fit
    python3 /ros2_ws/src/sign_detector/tools/vision_calib.py --hfov --left 0.42 --right 0.55
    python3 /ros2_ws/src/sign_detector/tools/vision_calib.py --check

`--fit` needs no tape measure: point the car at a wall, capture, move it, capture again,
five or six distances between 0.3 m and 1.5 m, and the lidar supplies the truth. `--hfov`
needs one tape-measured pair of side distances. `--check` is read-only and compares the
camera against the lidar and the sonars: **under 5 cm is good, over 15 cm means do not
drive on it yet.**

Paste the printed blocks into the `wall_vision:` section of `config/params.yaml` and
redeploy.

## 6. First runs, in this order

1. **Dry run.** `drive_pct:=0` steers the servo but never turns the motor, so the car can be
   pushed round the mat by hand while it thinks. The dashboard's "OPEN dry run" button does
   this. Watch `open_status`:
   - `direction` should lock within a metre or so, and match the round
   - `turns` should tick exactly once per corner
   - `health` should stay empty -- anything in it names a sensor that is lying
   - `vision_ok` should stay true; `tight` should appear only in narrow corridors
2. **Driven, reduced duty.** `drive_pct:=35`, one lap, ready to hit the e-stop.
3. **Full round** on the button.

## 7. Known follow-ups

- **The camera mounting is unmeasured** -- see section 5. Highest priority.
- **A 600 mm corridor cannot be cornered in one arc.** 25 deg of steering on a 0.15 m
  wheelbase gives 0.32 m; the corner needs 0.19 m. The driver reverses and takes a second
  bite, which works and is legal (9.21), but costs a few seconds per corner. Opening the
  steering up would remove the need: the firmware envelope allows about 27 deg today
  (+/-650 us at 24 us/deg), and ~38 deg would need a linkage change too.
- **Right and front sonar read wrong (2026-08-23, measured on hardware).**
  Right sat at exactly 0.43 m for 213 consecutive samples while the lidar put
  the nearest object on that side at 2.94 m. Front read ~0.93-1.47 m against a
  lidar front of 2.27 m. Reseat ECHO on A1 (right) and A0 (front) with 5V/GND,
  or swap the sensors. Order is front/right/rear/left = A0/A1/A2/A3.
  *The driver now survives both faults* -- `range_fusion` demotes a channel that stops
  changing or that disagrees with the others, and says so in `open_status.health` -- but
  the car is better off with four working sonars than three.
- **`angle_offset_deg` (170.0) is unverified on the track.** It defines where the lidar
  thinks forward is. The camera no longer depends on it, so a wrong value is far less
  dangerous than it was, but the lidar's contribution stays wrong until it is checked:
  sweep candidate offsets and pick the one where front is the largest open gap and
  left/right are roughly equal.
- IMU does not respond (`Errno 121` on I2C). Check S0 and S1 are each bridged to
  GND for address 0x28. Corners fall back to the camera's alignment cue plus dead
  reckoning without it, which the simulator covers.
- `start_button` defaults to `gpiochip4`; if the button never reads, try
  `gpiochip0`. Needs `lgpio` or `python3-libgpiod` in the container.
- `sonar` sends one `USON` at startup. Every `arduino_cmd` opens a manual window
  in the bridge which suspends autonomy, so it must stay one-shot, never polled.
- **`sign_detector_node` publishes camera-frame poses labelled `base_link`** (its own
  `_fuse` docstring says camera frame; `frame_id` is set to `base_link`). That is an
  obstacle-round bug, untouched by this work, and worth fixing before the obstacle rounds.

### Sensor roles (three witnesses, not a fallback chain)

| Sensor | Role | Trust | Fails by |
|---|---|---|---|
| Camera | side distances, heading, corner lines, front | 1.3 | losing a wall outside a 60 deg lens; darkness |
| Ultrasonic | side and front range, rear for reversing | 1.0 | latching at a constant value; reading short |
| Lidar | front distance, corner confirmation, open-space vote | 0.7 | grazing incidence on glossy black |

A fallback chain trusts a broken sensor until it goes silent, and a latched sonar never goes
silent -- which is exactly how this car drove into a wall. So each source is tracked
separately and demoted on *implausible behaviour* rather than absence: a value that never
changes while the wheels turn, a value outside the physical range, or a value sitting
outside what the others agree on for more than a second. What survives is combined by trust
weight; disagreements are reported rather than averaged away.

Decisions use the consensus, so one pessimistic sensor cannot stop the round. The brake uses
the *nearest* healthy reading, so a wall only one sensor can see still stops the car.

Watch `open_status.health` on the first runs. It names any channel that has been demoted and
why, and it is the fastest way to spot a mis-wired sensor.

## What changed since the last working state

- open_round: gyro 90° corners + heading hold (auto-disables with no IMU),
  mat-line corner triggers, trigger-colour lock, 12-corner completion
- imu: BNO055 + MPU6050 with hot-plug probing
- line_detector: HSV thresholds measured off this camera, not defaults
- start_button, sonar: new
- sim: models mat lines and publishes real line events; viewer renders them
- params.yaml: stale `open_round` section removed (it was reverting the
  calibrated lidar offset on any params-file launch)
