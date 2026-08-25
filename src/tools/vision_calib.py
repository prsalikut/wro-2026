#!/usr/bin/env python3
"""Measure and check the camera mounting that wall_vision depends on.

Every distance the camera reports is derived from where the wall meets the mat,
so it is only as good as three numbers: how high the lens is, how far it is
pitched down, and its horizontal field of view.  None of those has ever been
measured on this car -- 0.12 m / 15 deg / 60 deg are placeholders -- and an
error in any of them is an error in every range the open round drives on.

Three things this does, all from inside the container on the Pi:

    python3 vision_calib.py --check
        Read-only.  Compares the camera's ranges against the lidar's and the
        sonars' live, and says whether they agree.  Run this first, and run it
        again after any change to the mount.

    python3 vision_calib.py --fit
        Solves height and pitch.  Point the car at a wall, capture a sample,
        move it, capture again -- five or six distances between 0.3 m and 1.5 m.
        The lidar supplies the true distance, so no tape measure is needed.

    python3 vision_calib.py --solve-hfov --left 0.42 --right 0.55
        Solves the field of view from one measured pair of side distances.
        Park the car in a corridor, tape-measure both sides to the wall, run it.

Nothing here commands the motor or the servo, so it is safe with the car on the
mat.  Output is a params.yaml block to paste into config/params.yaml.
"""

import argparse
import math
import os
import sys
import threading

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan, Range
from std_msgs.msg import String

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "ros2-package", "sign_detector"))
try:
    from sign_detector.ground_geometry import (CameraGeometry, focal_from_fov,
                                               pixel_to_ground)
    from sign_detector.wall_vision import WallVision, WallVisionParams
except ImportError:                       # deployed layout: tools/ inside pkg
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".."))
    from sign_detector.ground_geometry import (CameraGeometry, focal_from_fov,
                                               pixel_to_ground)
    from sign_detector.wall_vision import WallVision, WallVisionParams


class Collector(Node):

    def __init__(self, hfov, height, pitch, cam_x):
        super().__init__("vision_calib")
        self.bridge = CvBridge()
        self.frame = None
        self.scan = None
        self.sonar = {}
        self.lane = None
        self.hfov = hfov
        self.geom_args = (hfov, height, pitch, cam_x)
        self.vision = None
        self.lock = threading.Lock()
        self.create_subscription(Image, "image_raw", self._on_image,
                                 qos_profile_sensor_data)
        self.create_subscription(LaserScan, "scan", self._on_scan,
                                 qos_profile_sensor_data)
        self.create_subscription(String, "vision/lane", self._on_lane, 10)
        for side in ("front", "right", "rear", "left"):
            self.create_subscription(
                Range, "sonar/{}".format(side),
                lambda m, s=side: self.sonar.__setitem__(s, float(m.range)), 10)

    def _on_image(self, msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return
        with self.lock:
            self.frame = bgr
            if self.vision is None:
                hfov, h, pitch, cam_x = self.geom_args
                geom = CameraGeometry(bgr.shape[1], bgr.shape[0], hfov, h,
                                      pitch, cam_x)
                self.vision = WallVision(geom, WallVisionParams())

    def _on_scan(self, msg):
        self.scan = msg

    def _on_lane(self, msg):
        self.lane = msg.data

    # ------------------------------------------------------------------ read

    def lane_dict(self):
        """Latest /vision/lane payload, as published by the running node."""
        import json
        if not self.lane:
            return None
        try:
            return json.loads(self.lane)
        except ValueError:
            return None

    def lidar_front(self, angle_offset, angle_sign, half=8.0):
        """Median lidar range within `half` degrees of straight ahead."""
        msg = self.scan
        if msg is None:
            return None
        vals = []
        for i, rng in enumerate(msg.ranges):
            if not math.isfinite(rng) or rng <= msg.range_min:
                continue
            a = math.degrees(msg.angle_min + i * msg.angle_increment)
            phi = (((a - angle_offset) / angle_sign) + 180.0) % 360.0 - 180.0
            if abs(phi) <= half:
                vals.append(rng)
        return float(np.median(vals)) if len(vals) >= 3 else None

    def column_pairs(self, angle_offset, angle_sign, step=8):
        """(u, v, lidar_range) for each image column with a wall base in it.

        The column gives a bearing, the bearing gives a lidar range, and the
        boundary row is where that same wall meets the floor -- which is the
        constraint the mounting has to satisfy.
        """
        import cv2
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
            vision = self.vision
        msg = self.scan
        if frame is None or vision is None or msg is None:
            return None
        p = vision.p
        work = cv2.resize(frame, (p.work_width, p.work_height),
                          interpolation=cv2.INTER_AREA)
        blur = cv2.GaussianBlur(work, (3, 3), 0)
        lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        vision._update_floor(lab)
        if vision._floor is None:
            return None
        _floor, blocked = vision._masks(lab, hsv)
        blocked = cv2.morphologyEx(blocked.astype(np.uint8), cv2.MORPH_OPEN,
                                   np.ones((3, 3), np.uint8)).astype(bool)

        # lidar range by bearing, in the car frame
        by_bearing = []
        for i, rng in enumerate(msg.ranges):
            if not math.isfinite(rng) or rng <= msg.range_min:
                continue
            a = math.degrees(msg.angle_min + i * msg.angle_increment)
            phi = (((a - angle_offset) / angle_sign) + 180.0) % 360.0 - 180.0
            by_bearing.append((phi, rng))
        if not by_bearing:
            return None

        scale = float(frame.shape[1]) / p.work_width
        out = []
        h_work = blocked.shape[0]
        for c in range(0, blocked.shape[1], step):
            col = blocked[:, c]
            idx = np.where(col)[0]
            if idx.size == 0:
                continue
            runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
            runs = [r for r in runs if r.size >= p.run_px]
            if not runs:
                continue
            v_work = float(runs[-1][-1]) + 0.5
            if v_work >= h_work - 1:
                continue
            u_full = (c + 0.5) * scale
            v_full = v_work * scale
            bearing = vision.full_geom.bearing_deg(u_full) \
                if hasattr(vision, "full_geom") else 0.0
            near = [r for phi, r in by_bearing if abs(phi - bearing) <= 1.5]
            if not near:
                continue
            out.append((u_full, v_full, float(np.median(near))))
        return out

    def boundary_row(self):
        """Lowest row of the sustained dark run in the middle of the frame.

        This is the raw observable the geometry converts into a distance, so
        fitting against it keeps the fit independent of the very parameters
        being solved for.
        """
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
            vision = self.vision
        if frame is None or vision is None:
            return None, None
        res = vision.process(frame)
        if not res.profile:
            return None, res
        p = vision.p
        small = vision.geom
        import cv2
        work = cv2.resize(frame, (p.work_width, p.work_height),
                          interpolation=cv2.INTER_AREA)
        blur = cv2.GaussianBlur(work, (3, 3), 0)
        lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        vision._update_floor(lab)
        if vision._floor is None:
            return None, res
        _floor, blocked = vision._masks(lab, hsv)
        blocked = cv2.morphologyEx(blocked.astype(np.uint8), cv2.MORPH_OPEN,
                                   np.ones((3, 3), np.uint8)).astype(bool)
        h, w = blocked.shape
        c0, c1 = int(w * 0.42), int(w * 0.58)
        rows = []
        for c in range(c0, c1):
            col = blocked[:, c]
            idx = np.where(col)[0]
            if idx.size == 0:
                continue
            runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
            runs = [r for r in runs if r.size >= p.run_px]
            if runs:
                rows.append(float(runs[-1][-1]) + 0.5)
        if len(rows) < 4:
            return None, res
        return float(np.median(rows)), (small, res)


def _row_to_range(v, h, pitch_deg, fy, cy, cam_x):
    t = math.radians(pitch_deg)
    st, ct = math.sin(t), math.cos(t)
    b = (v - cy) / fy
    den = st + b * ct
    if den <= 1e-6:
        return None
    return h * (ct - b * st) / den + cam_x


def fit_height_pitch(samples, fy, cy, cam_x):
    """Least squares over (boundary row, true range) for height and pitch."""
    best = None
    for pitch in np.arange(-5.0, 45.0, 0.25):
        for h in np.arange(0.04, 0.30, 0.002):
            err = 0.0
            ok = True
            for v, truth in samples:
                got = _row_to_range(v, h, pitch, fy, cy, cam_x)
                if got is None:
                    ok = False
                    break
                err += (got - truth) ** 2
            if ok and (best is None or err < best[0]):
                best = (err, h, pitch)
    if best is None:
        return None
    err, h, pitch = best
    return h, pitch, math.sqrt(err / len(samples))


def cmd_auto(node, args):
    """Solve height and pitch from one frame, against the lidar profile.

    The interactive fit needs the car moved between captures because a single
    forward distance cannot separate height from pitch.  A wall seen at an
    ANGLE does separate them: every image column looks at a different distance,
    and the lidar measures all of them at once.  So one stationary frame with a
    wall across the view is enough, and nobody has to touch the car.
    """
    import time
    print("\nSolving camera height and pitch from one frame.")
    print("Point the car at a wall -- ideally at a slight angle, so the wall")
    print("spans a range of distances across the image -- and hold still.\n")

    t0 = time.time()
    while time.time() - t0 < 1.5:
        rclpy.spin_once(node, timeout_sec=0.1)

    pairs = node.column_pairs(args.angle_offset, args.angle_sign)
    if pairs is None or len(pairs) < 12:
        print("Not enough matched columns (%s). Is a wall in view, and does the"
              % (0 if pairs is None else len(pairs)))
        print("lidar return anything at those bearings?")
        return 1
    print("  matched %d image columns against lidar bearings" % len(pairs))
    spread = max(r for _u, _v, r in pairs) - min(r for _u, _v, r in pairs)
    print("  lidar ranges span %.2f m across the frame" % spread)
    if spread < 0.25:
        print("  That is nearly flat-on. Turn the car 20-30 degrees to the wall")
        print("  and re-run: height and pitch are hard to separate otherwise.")

    with node.lock:
        geom = node.vision.geom

    # Trimmed fit. In a room, plenty of columns have the camera looking at one
    # object and the lidar at another -- a chair leg, a doorway, someone's foot
    # -- and those pairs are not wrong measurements so much as measurements of
    # different things. Scoring on the best-agreeing KEEP fraction lets the
    # coherent subset (the actual wall base) decide the answer.
    keep = max(8, int(args.inliers * len(pairs)))
    best = None
    for pitch in np.arange(-12.0, 40.0, 0.25):
        for h in np.arange(0.04, 0.30, 0.002):
            res = []
            for u, v, rng in pairs:
                g = pixel_to_ground(u, v, h, pitch, geom.fx, geom.fy,
                                    geom.cx, geom.cy, args.cam_x,
                                    max_range=1e9)
                if g is None:
                    continue
                res.append((math.hypot(g[0], g[1]) - rng) ** 2)
            if len(res) < keep:
                continue
            res.sort()
            err = sum(res[:keep])
            if best is None or err < best[0]:
                best = (err, h, pitch)
    if best is None:
        print("No mounting fits. Check that the boundary really is a wall base.")
        return 1
    err, h, pitch = best
    rms = math.sqrt(err / keep)
    print("  fitted on the %d best-agreeing of %d columns" % (keep, len(pairs)))
    print("\n  camera height %.3f m, pitch %.1f deg down, residual %.0f mm"
          % (h, pitch, 1000.0 * rms))
    if rms > 0.06:
        print("  Residual over 60 mm -- treat this as a starting point and")
        print("  confirm with --check, or use --fit with the car moved.")
    print("\nPaste into config/params.yaml under wall_vision:\n")
    print("    cam_height_m: %.3f" % h)
    print("    cam_pitch_deg: %.1f" % pitch)
    return 0


def cmd_check(node, args):
    """Compare what the RUNNING node reports against the other sensors.

    This reads /vision/lane rather than re-running the perception here, because
    the point is to check the calibration the car is actually driving on. An
    earlier version built its own WallVision from this tool's defaults and so
    cheerfully reported disagreement for a node that was correctly configured.
    """
    import time
    print("\nComparing the running wall_vision node against the lidar and the")
    print("sonars. Park the car with walls around it and watch the agreement")
    print("column: under 5 cm is good, over 15 cm means the mounting numbers")
    print("in params.yaml are wrong.\n")
    print("%8s %8s %8s %8s %8s %8s %8s" %
          ("cam_L", "son_L", "cam_R", "son_R", "cam_F", "lid_F", "worst"))
    worst_seen = 0.0
    seen_any = False
    for _ in range(args.samples):
        t0 = time.time()
        while time.time() - t0 < 0.6:
            rclpy.spin_once(node, timeout_sec=0.1)
        lane = node.lane_dict()
        if lane is None:
            print("  (waiting for /vision/lane -- is wall_vision running?)")
            continue
        seen_any = True
        lid = node.lidar_front(args.angle_offset, args.angle_sign)
        cam = {k: lane.get(k) for k in ("left", "right", "front")}
        pairs = [(cam["left"], node.sonar.get("left")),
                 (cam["right"], node.sonar.get("right")),
                 (cam["front"], lid)]
        # A camera reading pinned at max_range_m is not a measurement that
        # disagrees, it is the camera declining to claim a distance it cannot
        # resolve.  Counting that as error made a correctly calibrated mount
        # look 30 cm out.
        sat = args.max_range - 0.20
        diffs = [abs(a - b) for a, b in pairs
                 if a is not None and b is not None
                 and not (a >= sat and b >= a)]
        worst = max(diffs) if diffs else None
        if worst is not None:
            worst_seen = max(worst_seen, worst)

        def f(v):
            return "   --  " if v is None else "%7.3f" % v
        print("%8s %8s %8s %8s %8s %8s %8s" % (
            f(cam["left"]), f(node.sonar.get("left")),
            f(cam["right"]), f(node.sonar.get("right")),
            f(cam["front"]), f(lid),
            "  --   " if worst is None else "%7.3f" % worst))
        if cam["front"] is not None and cam["front"] >= args.max_range - 0.20:
            note = " (camera front saturated at its %.1f m limit)" % args.max_range
            if not getattr(cmd_check, "_noted", False):
                print(note)
                cmd_check._noted = True

    if not seen_any:
        print("\nThe wall_vision node published nothing. Nothing to check.")
        return 1
    print("\nworst disagreement seen: %.3f m" % worst_seen)
    print("(a sonar that check_all reports FROZEN is not evidence of anything;")
    print(" ignore its column until it is reseated)")
    if worst_seen > 0.15:
        print("Too large to drive on. Re-run --auto, or --fit with the car")
        print("moved between captures, and check hfov_deg with --solve-hfov.")
    elif worst_seen > 0.05:
        print("Usable, but re-check after any change to the camera mount.")
    else:
        print("Good agreement.")
    return 0


def cmd_fit(node, args):
    import time
    samples = []
    print("\nSolving camera height and pitch.")
    print("Point the car squarely at a wall. Capture a sample, move the car")
    print("(or the wall) to a different distance, capture again. Use five or")
    print("six spread between about 0.3 m and 1.5 m -- the spread is what")
    print("separates height from pitch.\n")
    while True:
        try:
            raw = input("[enter] capture, 'd' done, 'q' quit: ").strip().lower()
        except EOFError:
            break
        if raw == "q":
            return 1
        if raw == "d":
            break
        t0 = time.time()
        while time.time() - t0 < 0.6:
            rclpy.spin_once(node, timeout_sec=0.1)
        row, _res = node.boundary_row()
        truth = node.lidar_front(args.angle_offset, args.angle_sign)
        if row is None:
            print("  no floor/wall edge in the middle of the frame -- is the")
            print("  wall in view and the mat lit?")
            continue
        if truth is None:
            print("  no lidar return straight ahead; type the tape measurement")
            try:
                truth = float(input("  distance to the wall, metres: "))
            except (ValueError, EOFError):
                continue
        samples.append((row, truth))
        print("  sample %d: row %.1f  <-  %.3f m" % (len(samples), row, truth))

    if len(samples) < 3:
        print("\nNeed at least three distances; got %d." % len(samples))
        return 1
    with node.lock:
        geom = node.vision.geom
    got = fit_height_pitch(samples, geom.fy, geom.cy, args.cam_x)
    if got is None:
        print("\nNo mounting fits those samples. Check that the boundary row")
        print("really is the wall base and not a shadow.")
        return 1
    h, pitch, rms = got
    print("\n  camera height %.3f m, pitch %.1f deg down, residual %.1f mm"
          % (h, pitch, 1000.0 * rms))
    if rms > 0.03:
        print("  Residual over 30 mm: the samples disagree with any single")
        print("  mounting. Re-take them, keeping the car square to the wall.")
    print("\nPaste into config/params.yaml under wall_vision:\n")
    print("    cam_height_m: %.3f" % h)
    print("    cam_pitch_deg: %.1f" % pitch)
    return 0


def cmd_hfov(node, args):
    import time
    if args.left is None or args.right is None:
        print("--solve-hfov needs --left and --right, the tape-measured")
        print("distances from the car's centreline to each wall.")
        return 2
    print("\nSolving the field of view from the measured side distances.")
    t0 = time.time()
    while time.time() - t0 < 1.5:
        rclpy.spin_once(node, timeout_sec=0.1)
    with node.lock:
        frame = None if node.frame is None else node.frame.copy()
        vision = node.vision
    if frame is None:
        print("No camera frames.")
        return 1
    res = vision.process(frame)
    if res.left_m is None or res.right_m is None:
        print("The camera did not fit both walls (L=%s R=%s). Park nearer the"
              % (res.left_m, res.right_m))
        print("middle of a straight, where both wall bases are in view.")
        return 1
    # Lateral distance scales as 1/fx, so the correction is a plain ratio.
    ratio = 0.5 * (res.left_m / args.left + res.right_m / args.right)
    fx = focal_from_fov(vision.geom.width, args.hfov) * ratio
    hfov = 2.0 * math.degrees(math.atan((vision.geom.width * 0.5) / fx))
    print("  camera says L=%.3f R=%.3f; you measured L=%.3f R=%.3f"
          % (res.left_m, res.right_m, args.left, args.right))
    print("  scale %.3f  ->  hfov %.1f deg (was %.1f)"
          % (ratio, hfov, args.hfov))
    if abs(ratio - 1.0) > 0.35:
        print("  That is a big correction. Check the height and pitch with")
        print("  --fit first: an error there also skews the side distances.")
    print("\nPaste into config/params.yaml under wall_vision:\n")
    print("    hfov_deg: %.1f" % hfov)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--auto", action="store_true",
                    help="solve height and pitch from one frame + the lidar")
    ap.add_argument("--solve-hfov", dest="solve_hfov",
                    action="store_true",
                    help="solve the field of view from measured side distances")
    ap.add_argument("--left", type=float, default=None)
    ap.add_argument("--right", type=float, default=None)
    ap.add_argument("--cam-height", dest="cam_height", type=float, default=0.12)
    ap.add_argument("--cam-pitch", dest="cam_pitch", type=float, default=15.0)
    ap.add_argument("--cam-x", dest="cam_x", type=float, default=0.05)
    ap.add_argument("--hfov-deg", dest="hfov", type=float, default=60.0)
    ap.add_argument("--angle-offset", dest="angle_offset", type=float,
                    default=170.0)
    ap.add_argument("--angle-sign", dest="angle_sign", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--max-range", dest="max_range", type=float, default=2.2,
                    help="wall_vision's max_range_m; readings pinned there are "
                         "treated as saturated rather than wrong")
    ap.add_argument("--inliers", type=float, default=0.5,
                    help="fraction of columns the --auto fit trusts (0-1)")
    args = ap.parse_args()

    rclpy.init()
    node = Collector(args.hfov, args.cam_height, args.cam_pitch, args.cam_x)
    import time
    t0 = time.time()
    while node.frame is None and time.time() - t0 < 6.0:
        rclpy.spin_once(node, timeout_sec=0.2)
    if node.frame is None:
        print("No frames on /image_raw after 6 s -- is the stack running?")
        node.destroy_node()
        rclpy.shutdown()
        return 1
    try:
        if args.auto:
            rc = cmd_auto(node, args)
        elif args.fit:
            rc = cmd_fit(node, args)
        elif args.solve_hfov:
            rc = cmd_hfov(node, args)
        else:
            rc = cmd_check(node, args)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
