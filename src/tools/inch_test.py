#!/usr/bin/env python3
"""Inch the car in small steps and see whether the sensors agree about it.

Two things this settles that a stationary check cannot.

First, calibration. A static comparison only ever tests one distance, and a
camera mounting error trades off against range in a way one distance cannot
separate. Moving a known-but-unmeasured amount fixes that: whatever the car
actually travelled, the lidar and the camera must report the SAME change. If
the camera consistently reports 80% of the lidar's change, its assumed height
is 80% of the truth -- a scale factor that falls straight out, with no tape
measure and no faith in either absolute reading.

Second, the drivetrain. This motor does not move below about 30% duty and needs
a burst of full duty to break stiction, so "how far does one pulse travel"
is a number the open round needs and nobody has measured.

The IMU would be the natural reference for the turning half of this, but it is
not responding on this car (Errno 121 on I2C), so everything here leans on the
lidar and the camera instead.

Safe by construction: it only pulses with clear space ahead, each pulse is a
fraction of a second, and it stops the moment anything gets close. The bridge
zeroes the motor if commands go stale, so even a crash here stops the car.

    python3 inch_test.py                # 6 forward pulses, then 6 back
    python3 inch_test.py --steps 4 --pulse 0.20
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu, LaserScan, Range
from std_msgs.msg import Float32, String
from cv_bridge import CvBridge

sys.path.insert(0, "/ros2_ws/src/sign_detector")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "ros2-package", "sign_detector"))
from sign_detector.ground_geometry import CameraGeometry, pixel_to_ground
from sign_detector.wall_vision import WallVision, WallVisionParams

try:
    import json
except ImportError:                                   # pragma: no cover
    json = None


class Incher(Node):

    def __init__(self, angle_offset, angle_sign):
        super().__init__("inch_test")
        self.angle_offset = angle_offset
        self.angle_sign = angle_sign
        self.scan = None
        self.lane = None
        self.sonar = {}
        self.yaw = None
        self.img = None
        self.vision = None
        self.br = CvBridge()
        self.cam_args = None
        self.create_subscription(Image, "image_raw", self._image,
                                 qos_profile_sensor_data)
        self.pub_drive = self.create_publisher(Float32, "drive_cmd", 10)
        self.pub_steer = self.create_publisher(Float32, "steering_cmd", 10)
        self.create_subscription(LaserScan, "scan", self._scan,
                                 qos_profile_sensor_data)
        self.create_subscription(String, "vision/lane", self._lane, 10)
        self.create_subscription(Float32, "imu/yaw",
                                 lambda m: setattr(self, "yaw", float(m.data)), 10)
        for s in ("front", "right", "rear", "left"):
            self.create_subscription(
                Range, "sonar/{}".format(s),
                lambda m, k=s: self.sonar.__setitem__(k, float(m.range)), 10)

    def _scan(self, m):
        self.scan = m

    def _image(self, m):
        try:
            self.img = self.br.imgmsg_to_cv2(m, "bgr8")
        except Exception:
            return
        if self.vision is None and self.cam_args:
            hfov, h, pitch, cam_x = self.cam_args
            geom = CameraGeometry(self.img.shape[1], self.img.shape[0],
                                  hfov, h, pitch, cam_x)
            self.vision = WallVision(geom, WallVisionParams())

    def _lane(self, m):
        try:
            self.lane = json.loads(m.data)
        except (ValueError, AttributeError):
            pass

    def spin(self, secs):
        t0 = time.time()
        while time.time() - t0 < secs:
            rclpy.spin_once(self, timeout_sec=0.05)

    def lidar(self, half=8.0):
        m = self.scan
        if m is None:
            return None
        vals = []
        for i, r in enumerate(m.ranges):
            if not math.isfinite(r) or r <= m.range_min:
                continue
            a = math.degrees(m.angle_min + i * m.angle_increment)
            phi = (((a - self.angle_offset) / self.angle_sign) + 180.0) % 360.0 - 180.0
            if abs(phi) <= half:
                vals.append(r)
        return float(np.median(vals)) if len(vals) >= 3 else None

    def camera(self):
        return None if not self.lane else self.lane.get("front")

    def boundary_row(self):
        """Row where the floor ends in the middle of the frame, full-res.

        The raw observable, before any mounting assumption is applied -- which
        is what makes it usable for SOLVING the mounting rather than merely
        checking it.
        """
        import cv2
        if self.img is None or self.vision is None:
            return None
        p = self.vision.p
        work = cv2.resize(self.img, (p.work_width, p.work_height),
                          interpolation=cv2.INTER_AREA)
        blur = cv2.GaussianBlur(work, (3, 3), 0)
        lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        self.vision._update_floor(lab)
        if self.vision._floor is None:
            return None
        _f, blocked = self.vision._masks(lab, hsv)
        blocked = cv2.morphologyEx(blocked.astype(np.uint8), cv2.MORPH_OPEN,
                                   np.ones((3, 3), np.uint8)).astype(bool)
        rows = []
        h_w, w_w = blocked.shape
        for c in range(int(w_w * 0.44), int(w_w * 0.56)):
            idx = np.where(blocked[:, c])[0]
            if idx.size == 0:
                continue
            runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
            runs = [r for r in runs if r.size >= p.run_px]
            if runs:
                rows.append(float(runs[-1][-1]) + 0.5)
        if len(rows) < 4:
            return None
        return float(np.median(rows)) * (self.img.shape[0] / float(p.work_height))

    def sample(self, secs=0.8):
        """Settle, then take a median of each sensor."""
        lid, cam, son = [], [], []
        t0 = time.time()
        while time.time() - t0 < secs:
            rclpy.spin_once(self, timeout_sec=0.05)
            v = self.lidar()
            if v is not None:
                lid.append(v)
            v = self.camera()
            if v is not None:
                cam.append(v)
            v = self.sonar.get("front")
            if v is not None:
                son.append(v)
        med = lambda a: float(np.median(a)) if a else None   # noqa: E731
        return med(lid), med(cam), med(son)

    def hold(self, pct, secs, steer=0.0):
        """Command a duty for `secs`, refreshing so the bridge does not time out."""
        t0 = time.time()
        while time.time() - t0 < secs:
            self.pub_steer.publish(Float32(data=float(steer)))
            self.pub_drive.publish(Float32(data=float(pct)))
            rclpy.spin_once(self, timeout_sec=0.02)
            time.sleep(0.02)

    def stop(self):
        for _ in range(6):
            self.pub_drive.publish(Float32(data=0.0))
            self.pub_steer.publish(Float32(data=0.0))
            rclpy.spin_once(self, timeout_sec=0.02)
            time.sleep(0.02)


def measure_trim(n, args):
    """Find the steering offset that makes the car actually go straight.

    Command zero steering, pulse forward, and let the IMU say what the car did.
    A car that tracks straight holds its heading; one that drifts turns at a
    steady rate per metre travelled, and for a bicycle model that rate is
    tan(delta)/L for a standing steering error delta. So the correction is
    -atan(L * dpsi/ds) -- measured, not dialled in by eye.
    """
    print("\nMeasuring steering trim: %d forward pulses at zero steering."
          % args.steps)
    print("The IMU is the reference. Keep the path ahead clear.\n")
    print("%5s %9s %9s %9s" % ("step", "yaw", "d_yaw", "d_dist"))
    n.spin(1.0)
    yaw0 = n.yaw
    if yaw0 is None:
        print("No /imu/yaw -- cannot measure trim without a heading reference.")
        return None
    lid0 = n.sample(0.8)[0]
    total_yaw, total_dist = 0.0, 0.0
    prev_yaw, prev_lid = yaw0, lid0
    for i in range(args.steps):
        clear = n.sample(0.4)[0]
        if clear is not None and clear < args.min_clear:
            print("  only %.2f m ahead, stopping" % clear)
            break
        n.hold(args.duty, args.pulse, steer=args.trim_steer)
        n.stop()
        n.spin(0.6)
        lid, _cam, _son = n.sample(0.9)
        yaw = n.yaw
        if yaw is None or lid is None or prev_lid is None:
            prev_lid = lid
            continue
        d_yaw = ((yaw - prev_yaw) + 180.0) % 360.0 - 180.0
        d_dist = prev_lid - lid
        print("%5d %9.2f %9.2f %9.3f" % (i + 1, yaw, d_yaw, d_dist))
        if d_dist > 0.01:
            total_yaw += d_yaw
            total_dist += d_dist
        prev_yaw, prev_lid = yaw, lid
    n.stop()
    if total_dist < 0.08:
        print("\nThe car barely moved (%.3f m); cannot measure trim."
              % total_dist)
        return None
    rate = total_yaw / total_dist                       # deg per metre
    trim = -math.degrees(math.atan(args.wheelbase * math.radians(rate)))
    print("\n  travelled %.3f m, heading changed %+.1f deg  (%+.1f deg/m)"
          % (total_dist, total_yaw, rate))
    print("  -> straight_trim_deg: %+.2f" % trim)
    if abs(trim) > 8.0:
        print("  That is a large mechanical offset. Worth centring the servo")
        print("  horn mechanically as well, so the trim is not eating the")
        print("  steering range on one side.")
    print("\n  paste into params.yaml under open_round:")
    print("      trim_deg: %+.2f" % trim)
    return trim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--pulse", type=float, default=0.22,
                    help="seconds of full duty per step")
    ap.add_argument("--duty", type=float, default=100.0)
    ap.add_argument("--min-clear", dest="min_clear", type=float, default=0.55,
                    help="refuse to pulse forward with less room than this")
    ap.add_argument("--angle-offset", dest="angle_offset", type=float,
                    default=176.0)
    ap.add_argument("--angle-sign", dest="angle_sign", type=float, default=1.0)
    ap.add_argument("--cam-height", dest="cam_height", type=float, default=0.130)
    ap.add_argument("--cam-pitch", dest="cam_pitch", type=float, default=12.6)
    ap.add_argument("--cam-x", dest="cam_x", type=float, default=0.05)
    ap.add_argument("--hfov", type=float, default=60.0)
    ap.add_argument("--trim", action="store_true",
                    help="measure the steering trim instead of inching")
    ap.add_argument("--trim-steer", dest="trim_steer", type=float, default=0.0,
                    help="steering to hold while measuring (to verify a trim)")
    ap.add_argument("--wheelbase", type=float, default=0.15)
    args = ap.parse_args()

    rclpy.init()
    n = Incher(args.angle_offset, args.angle_sign)
    n.cam_args = (args.hfov, args.cam_height, args.cam_pitch, args.cam_x)
    # ROS 2 discovery can take a few seconds on this Pi; a short wait here made
    # the tool report "no lidar" against a lidar that was publishing fine.
    t0 = time.time()
    while n.scan is None and time.time() - t0 < 12.0:
        rclpy.spin_once(n, timeout_sec=0.2)
    if n.scan is None:
        print("No /scan after 12 s. Is the stack running?")
        return 1
    n.spin(1.0)
    if n.lane is None:
        print("No /vision/lane -- wall_vision is not publishing; the camera "
              "column will be empty.")

    if args.trim:
        try:
            measure_trim(n, args)
        finally:
            n.stop()
            n.destroy_node()
            rclpy.shutdown()
        return 0

    print("\nInching the car in %d steps of %.2f s at %.0f%% duty, then back."
          % (args.steps, args.pulse, args.duty))
    print("The lidar is the reference. Ctrl-C stops immediately.\n")
    print("%5s %9s %9s %9s   %9s %9s %9s"
          % ("step", "lid", "cam", "son", "d_lid", "d_cam", "cam/lid"))

    rows = []
    fit_pairs = []
    try:
        for direction, label in ((1.0, "forward"), (-1.0, "back")):
            print("  -- %s --" % label)
            prev = n.sample(1.0)
            row0 = n.boundary_row()
            if row0 is not None and prev[0] is not None:
                fit_pairs.append((row0, prev[0]))
            for i in range(args.steps):
                clear = prev[0]
                if direction > 0 and clear is not None and clear < args.min_clear:
                    print("  only %.2f m ahead, not pulsing forward" % clear)
                    break
                n.hold(direction * args.duty, args.pulse)
                n.stop()
                n.spin(0.5)
                cur = n.sample(0.9)
                row = n.boundary_row()
                if row is not None and cur[0] is not None:
                    fit_pairs.append((row, cur[0]))
                d_lid = (None if (prev[0] is None or cur[0] is None)
                         else prev[0] - cur[0])
                d_cam = (None if (prev[1] is None or cur[1] is None)
                         else prev[1] - cur[1])
                ratio = (None if (not d_lid or d_cam is None or abs(d_lid) < 0.01)
                         else d_cam / d_lid)
                f = lambda v: "   --   " if v is None else "%8.3f" % v  # noqa: E731
                print("%5d %9s %9s %9s   %9s %9s %9s"
                      % (i + 1, f(cur[0]), f(cur[1]), f(cur[2]),
                         f(d_lid), f(d_cam),
                         "   --   " if ratio is None else "%8.2f" % ratio))
                if d_lid is not None and d_cam is not None and abs(d_lid) > 0.02:
                    rows.append((d_lid, d_cam))
                prev = cur
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        n.stop()

    print()
    moved = [abs(d) for d, _c in rows]
    if moved:
        print("travel per pulse: %.3f m median (%.3f - %.3f)"
              % (float(np.median(moved)), min(moved), max(moved)))
        print("  -> about %.2f m/s while the pulse is on"
              % (float(np.median(moved)) / max(1e-3, args.pulse)))
    else:
        print("The car did not move measurably. Either the pulse is too short "
              "to break stiction, or the drive is not connected.")

    good = [(d, c) for d, c in rows if abs(d) > 0.03]
    if len(good) >= 3:
        num = sum(d * c for d, c in good)
        den = sum(d * d for d, c in good)
        scale = num / den if den else float("nan")
        print("\ncamera-vs-lidar scale over %d moves: %.3f" % (len(good), scale))
        if abs(scale - 1.0) <= 0.08:
            print("  The camera tracks the lidar. The mounting numbers are good.")
        else:
            print("  The camera reports %.0f%% of the true change, so its assumed"
                  % (100.0 * scale))
            print("  height is off by about the same factor. Multiply")
            print("  cam_height_m by %.3f and re-run --check." % (1.0 / scale))
    else:
        print("\nNot enough clean moves to judge the camera scale.")

    # --- solve the mounting from the motion --------------------------------
    # Each stop gives the raw wall-base row and the lidar's true distance to
    # that wall. Several stops at different distances is exactly what separates
    # camera height from camera pitch, which a single stationary view cannot.
    if len(fit_pairs) >= 4 and n.vision is not None:
        geom = n.vision.geom
        spread = max(r for _v, r in fit_pairs) - min(r for _v, r in fit_pairs)
        print("\nSolving the camera mounting from %d stops spanning %.2f m:"
              % (len(fit_pairs), spread))
        if spread < 0.15:
            print("  (only %.2f m of range covered -- height and pitch stay"
                  " entangled below about 0.3 m)" % spread)
        best = None
        for pitch in np.arange(-5.0, 40.0, 0.2):
            for hh in np.arange(0.04, 0.32, 0.002):
                err, ok = 0.0, True
                for v, truth in fit_pairs:
                    g = pixel_to_ground(geom.cx, v, hh, pitch, geom.fx, geom.fy,
                                        geom.cx, geom.cy, args.cam_x,
                                        max_range=1e9)
                    if g is None:
                        ok = False
                        break
                    err += (g[0] - truth) ** 2
                if ok and (best is None or err < best[0]):
                    best = (err, hh, pitch)
        if best is not None:
            err, hh, pitch = best
            rms = math.sqrt(err / len(fit_pairs))
            print("  cam_height_m: %.3f    cam_pitch_deg: %.1f   (residual"
                  " %.0f mm)" % (hh, pitch, 1000 * rms))
            if rms > 0.05:
                print("  Residual over 50 mm: the camera and the lidar may be")
                print("  looking at different things. Point the car at a plain")
                print("  wall with nothing in front of it and repeat.")
        else:
            print("  no mounting fits these stops")
    else:
        print("\nNot enough stops with both a wall base and a lidar range to "
              "solve the mounting.")

    n.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
