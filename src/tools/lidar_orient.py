#!/usr/bin/env python3
"""Find the lidar's true angular offset by matching it to the camera.

`angle_offset_deg` decides where the scan's zero sits relative to the car, and
every steering decision downstream depends on it. params.yaml has carried 170.0
with a comment admitting it was never checked on the track; if it is 180 out,
"front" is behind the car and every corner is taken the wrong way.

An earlier version of this compared four sector medians against the ultrasonics
and could not tell -- the sonars disagree with each other on this car, and four
numbers is not enough to pin an angle. This matches the SHAPE of the two range
profiles instead: the camera, now calibrated against a tape measure, produces a
metric free-space profile over its field of view, and the offset is the rotation
that lines the lidar's profile up with it. That is hundreds of constraints
instead of four, and it does not involve the sonars at all.

Read-only: it never commands the motor or the servo.

    python3 lidar_orient.py               # sweep and report
    python3 lidar_orient.py --plot        # add an ASCII overlay of the fit
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
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge

sys.path.insert(0, "/ros2_ws/src/sign_detector")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "ros2-package", "sign_detector"))
from sign_detector.ground_geometry import CameraGeometry      # noqa: E402
from sign_detector.wall_vision import WallVision, WallVisionParams  # noqa: E402


class Orient(Node):

    def __init__(self, hfov, height, pitch, cam_x):
        super().__init__("lidar_orient")
        self.br = CvBridge()
        self.img = None
        self.scan = None
        self.args = (hfov, height, pitch, cam_x)
        self.vision = None
        self.create_subscription(Image, "image_raw", self._img,
                                 qos_profile_sensor_data)
        self.create_subscription(LaserScan, "scan", self._scan,
                                 qos_profile_sensor_data)

    def _img(self, m):
        try:
            self.img = self.br.imgmsg_to_cv2(m, "bgr8")
        except Exception:
            return
        if self.vision is None:
            hfov, h, pitch, cam_x = self.args
            geom = CameraGeometry(self.img.shape[1], self.img.shape[0],
                                  hfov, h, pitch, cam_x)
            self.vision = WallVision(geom, WallVisionParams())

    def _scan(self, m):
        self.scan = m

    def spin(self, secs):
        t0 = time.time()
        while time.time() - t0 < secs:
            rclpy.spin_once(self, timeout_sec=0.05)

    def camera_profile(self, frames=5):
        """{bearing_deg: range_m} from the calibrated camera."""
        acc = {}
        for _ in range(frames):
            self.spin(0.25)
            if self.img is None or self.vision is None:
                continue
            res = self.vision.process(self.img)
            for bearing, x, _y in res.profile:
                b = round(bearing)
                acc.setdefault(b, []).append(x)
        return {b: float(np.median(v)) for b, v in acc.items() if len(v) >= 2}

    def lidar_profile(self):
        """{raw_scan_angle_deg: range_m}, before any offset is applied."""
        m = self.scan
        if m is None:
            return {}
        acc = {}
        for i, rng in enumerate(m.ranges):
            if not math.isfinite(rng) or rng <= m.range_min:
                continue
            a = math.degrees(m.angle_min + i * m.angle_increment)
            acc.setdefault(round(a), []).append(rng)
        return {a: float(np.median(v)) for a, v in acc.items()}


def score(cam, lid, offset, sign, max_range):
    """Mean |error| between the two profiles under a candidate offset."""
    errs = []
    for raw, r in lid.items():
        phi = (((raw - offset) / sign) + 180.0) % 360.0 - 180.0
        b = round(phi)
        if b not in cam:
            continue
        c = cam[b]
        # The camera saturates; "at least this far" cannot contradict a longer
        # lidar reading, so those pairs carry no information either way.
        if c >= max_range - 0.15 and r >= c:
            continue
        errs.append(abs(r - c))
    if len(errs) < 12:
        return None, len(errs)
    return float(np.mean(errs)), len(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sign", type=float, default=1.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--cam-height", dest="cam_height", type=float, default=0.130)
    ap.add_argument("--cam-pitch", dest="cam_pitch", type=float, default=12.6)
    ap.add_argument("--cam-x", dest="cam_x", type=float, default=0.05)
    ap.add_argument("--hfov", type=float, default=60.0)
    ap.add_argument("--max-range", dest="max_range", type=float, default=2.2)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    rclpy.init()
    n = Orient(args.hfov, args.cam_height, args.cam_pitch, args.cam_x)
    n.spin(3.0)
    if n.scan is None or n.img is None:
        print("Need both /scan and /image_raw.")
        return 1

    cam = n.camera_profile()
    lid = n.lidar_profile()
    print("\ncamera profile: %d bearings (%.0f to %.0f deg)"
          % (len(cam), min(cam) if cam else 0, max(cam) if cam else 0))
    print("lidar profile : %d bearings" % len(lid))
    if len(cam) < 15:
        print("\nToo little camera profile. Park the car facing walls with the")
        print("mat well lit, and make sure wall_vision reports ok.")
        return 1

    results = []
    for sign in ((1.0, -1.0) if args.sign == 0 else (args.sign,)):
        for off in np.arange(-180.0, 180.0, args.step):
            s, npts = score(cam, lid, float(off), sign, args.max_range)
            if s is not None:
                results.append((s, float(off), sign, npts))
    if not results:
        print("\nNo overlap between the two profiles.")
        return 1
    results.sort()

    print("\nBest offsets (mean |lidar - camera| over matched bearings):")
    print("%9s %6s %10s %8s" % ("offset", "sign", "mean_err", "points"))
    for s, off, sign, npts in results[:8]:
        print("%9.1f %6.0f %10.3f %8d" % (off, sign, s, npts))

    best_err, best_off, best_sign, _ = results[0]
    cur = min((r for r in results if abs(r[1] - 170.0) < 1.0 and r[2] == 1.0),
              default=None)
    print("\nBest fit: angle_offset_deg = %.1f, angle_sign = %.0f  (mean error "
          "%.3f m)" % (best_off, best_sign, best_err))
    if cur:
        print("Configured 170.0 / sign 1 scores %.3f m" % cur[0])
        delta = abs(((best_off - 170.0) + 180) % 360 - 180)
        if best_err < cur[0] - 0.03:
            print("  -> configured value is %.0f degrees out." % delta)
            if abs(delta - 180.0) < 25.0:
                print("     That is an INVERSION: the lidar's front is the")
                print("     car's rear.")
        else:
            print("  -> configured value is as good as the best; leave it.")

    if args.plot:
        print("\nbearing   camera   lidar@best")
        for b in sorted(cam):
            if b % 3:
                continue
            raw = b * best_sign + best_off
            key = round(((raw + 180) % 360) - 180)
            r = lid.get(key)
            bar = "#" * int(min(40, cam[b] * 12))
            print("%7d %8.2f %10s  %s"
                  % (b, cam[b], "--" if r is None else "%.2f" % r, bar))

    n.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
