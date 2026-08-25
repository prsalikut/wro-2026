#!/usr/bin/env python3
"""Record every sensor and every command while a human drives the car.

The point is to learn what a good lap looks like in sensor terms -- how close
the walls get, what the ultrasonics read approaching a corner, where the corner
lines appear, how far the heading swings through a turn -- and then tune the
autonomous driver to those numbers.

What is legitimate to take from this, and what is not: rule 9.9 forbids entering
data into the program by adjusting the vehicle, and the field layout is
randomised after check time, so a recorded ROUTE is both illegal in spirit and
useless in practice. General quantities are a different matter, and 13.18
explicitly grants calibration time: metres per percent of duty, degrees of yaw
per second of turn, the front distance at which a corner is really upon you, how
wide the corridors are, which sensors tell the truth. Those are what --analyse
extracts.

    python3 record_drive.py --out /tmp/drive.jsonl        # record until Ctrl-C
    python3 record_drive.py --analyse /tmp/drive.jsonl    # turn it into numbers
"""

import argparse
import json
import math
import sys
import time

import numpy as np


def record(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu, LaserScan, Range
    from std_msgs.msg import Float32, String

    class Rec(Node):
        def __init__(self):
            super().__init__("record_drive")
            self.sonar = {}
            self.yaw = None
            self.gyro = None
            self.gyro_xyz = None
            self.accel = None
            self.quat = None
            self.scan = None
            self.lane = None
            self.steer = 0.0
            self.drive = 0.0
            for s in ("front", "right", "rear", "left"):
                self.create_subscription(
                    Range, "sonar/{}".format(s),
                    lambda m, k=s: self.sonar.__setitem__(k, float(m.range)), 10)
            self.create_subscription(Float32, "imu/yaw",
                                     lambda m: setattr(self, "yaw", float(m.data)), 10)
            self.create_subscription(Imu, "imu/data", self._imu,
                                     qos_profile_sensor_data)
            self.create_subscription(LaserScan, "scan",
                                     lambda m: setattr(self, "scan", m),
                                     qos_profile_sensor_data)
            self.create_subscription(String, "vision/lane", self._lane, 10)
            self.create_subscription(Float32, "steering_cmd",
                                     lambda m: setattr(self, "steer", float(m.data)), 10)
            self.create_subscription(Float32, "drive_cmd",
                                     lambda m: setattr(self, "drive", float(m.data)), 10)

        def _imu(self, m):
            self.gyro = float(m.angular_velocity.z)
            self.gyro_xyz = (round(float(m.angular_velocity.x), 4),
                             round(float(m.angular_velocity.y), 4),
                             round(float(m.angular_velocity.z), 4))
            self.accel = (round(float(m.linear_acceleration.x), 3),
                          round(float(m.linear_acceleration.y), 3),
                          round(float(m.linear_acceleration.z), 3))
            q = m.orientation
            self.quat = (round(q.x, 4), round(q.y, 4), round(q.z, 4),
                         round(q.w, 4))

        def _lane(self, m):
            try:
                self.lane = json.loads(m.data)
            except ValueError:
                pass

        def scan_bins(self, offset, sign):
            """The whole scan, one bin per degree in the CAR frame.

            Sector medians were enough to drive on but not to analyse: a corner
            looks completely different from a straight in the full profile, and
            that shape is what the driver's thresholds should be derived from.
            """
            m = self.scan
            if m is None:
                return None
            out = [None] * 360
            for i, r in enumerate(m.ranges):
                if not math.isfinite(r) or r <= m.range_min:
                    continue
                a = math.degrees(m.angle_min + i * m.angle_increment)
                phi = (((a - offset) / sign) + 180.0) % 360.0 - 180.0
                b = int(round(phi)) % 360
                if out[b] is None or r < out[b]:
                    out[b] = round(r, 3)
            return out

        def lidar(self, offset, sign, mount, half=12.0):
            m = self.scan
            if m is None:
                return None
            vals = []
            for i, r in enumerate(m.ranges):
                if not math.isfinite(r) or r <= m.range_min:
                    continue
                a = math.degrees(m.angle_min + i * m.angle_increment)
                phi = (((a - offset) / sign) + 180.0) % 360.0 - 180.0
                if abs(((phi - mount) + 180.0) % 360.0 - 180.0) <= half:
                    vals.append(r)
            return float(np.median(vals)) if len(vals) >= 3 else None

    rclpy.init()
    n = Rec()
    t0 = time.time()
    while not n.sonar and time.time() - t0 < 10:
        rclpy.spin_once(n, timeout_sec=0.2)
    if not n.sonar:
        print("no sonar data -- is the stack up?")
        return 1

    fh = open(args.out, "w", buffering=1)
    print("\nRecording to %s at %.0f Hz. Drive the car; Ctrl-C to stop.\n"
          % (args.out, args.hz))
    print("%7s %7s %7s %7s %7s %8s %7s %7s"
          % ("t", "front", "left", "right", "yaw", "steer", "drive", "lines"))
    t0 = time.time()
    n_rows = 0
    last_print = 0.0
    period = 1.0 / args.hz
    try:
        while True:
            # Drain the callback queue for a whole period rather than taking
            # one message per cycle. rclpy.spin_once handles a SINGLE callback,
            # and with the IMU at 100 Hz, four sonars at 20 Hz each and the
            # camera on top, one-per-cycle meant the lidar almost never got
            # serviced -- it was absent from every recorded row while
            # publishing perfectly well at 11 Hz.
            slice_end = time.time() + period
            while time.time() < slice_end:
                rclpy.spin_once(n, timeout_sec=0.002)
            now = time.time() - t0
            lane = n.lane or {}
            row = {
                "t": round(now, 3),
                "sonar": {k: round(v, 3) for k, v in n.sonar.items()},
                "yaw": None if n.yaw is None else round(n.yaw, 2),
                "gyro_z": None if n.gyro is None else round(n.gyro, 4),
                "steer": round(n.steer, 2),
                "drive": round(n.drive, 1),
                "gyro_xyz": n.gyro_xyz,
                "accel": n.accel,
                "quat": n.quat,
                "lidar": {m: n.lidar(args.angle_offset, args.angle_sign, d)
                          for m, d in (("front", 0.0), ("left", 90.0),
                                       ("right", -90.0), ("rear", 180.0))},
                "scan": n.scan_bins(args.angle_offset, args.angle_sign)
                        if args.full_scan else None,
                "cam": {k: lane.get(k) for k in
                        ("front", "left", "right", "front_wall", "front_free",
                         "heading", "lane_width", "open_left", "open_right",
                         "open_bearing", "floor_frac", "points", "ok",
                         "reason")},
                "lines": {k: {"v": v.get("v_frac"), "d": v.get("distance_m"),
                              "bearing": v.get("bearing_deg"),
                              "area": v.get("area_frac")}
                          for k, v in (lane.get("lines") or {}).items()},
            }
            fh.write(json.dumps(row) + "\n")
            n_rows += 1
            missing = []
            if len(n.sonar) < 4:
                missing.append("sonar(%d/4)" % len(n.sonar))
            if n.scan is None:
                missing.append("lidar")
            if n.yaw is None:
                missing.append("imu")
            if n.lane is None:
                missing.append("camera")
            row["missing"] = missing
            if now - last_print >= 1.0:
                last_print = now
                f = lambda v: "   --  " if v is None else "%7.3f" % v  # noqa
                print("%7.1f %s %s %s %s %8.1f %7.1f %7s%s"
                      % (now, f(n.sonar.get("front")), f(n.sonar.get("left")),
                         f(n.sonar.get("right")),
                         f(n.yaw), n.steer, n.drive,
                         ",".join(sorted(row["lines"])) or "-",
                         ("   MISSING: " + ",".join(missing)) if missing else ""))

    except KeyboardInterrupt:
        pass
    finally:
        fh.close()
        n.destroy_node()
        rclpy.shutdown()
    print("\n%d rows over %.0f s -> %s" % (n_rows, time.time() - t0, args.out))
    return 0


def analyse(path):
    rows = [json.loads(l) for l in open(path)]
    if len(rows) < 50:
        print("only %d rows" % len(rows))
        return 1
    dur = rows[-1]["t"] - rows[0]["t"]
    print("\n%d rows over %.0f s of driving\n" % (len(rows), dur))

    def col(get):
        return [v for v in (get(r) for r in rows) if v is not None]

    # --- speed: how fast does the car actually go per percent of duty? ------
    # Front range falling while driving straight is the cleanest speed signal
    # available without encoders.
    pairs = []
    for a, b in zip(rows, rows[1:]):
        dt = b["t"] - a["t"]
        if not (0.02 < dt < 0.5):
            continue
        if abs(a["steer"]) > 5.0 or a["drive"] < 5.0:
            continue
        fa, fb = a["sonar"].get("front"), b["sonar"].get("front")
        if fa is None or fb is None:
            continue
        v = (fa - fb) / dt
        if 0.02 < v < 2.5:
            pairs.append((a["drive"], v))
    if len(pairs) > 20:
        d = np.array([p[0] for p in pairs])
        v = np.array([p[1] for p in pairs])
        k = float(np.sum(d * v) / np.sum(d * d))
        print("speed while driving straight: %.3f m/s per 100%% duty"
              % (k * 100))
        print("  -> speed_per_pct  %.4f   (n=%d, median duty %.0f%%)"
              % (k, len(pairs), float(np.median(d))))
    else:
        print("not enough straight-line driving to measure speed (n=%d)"
              % len(pairs))

    # --- turn rate: degrees of yaw per second at a given steering ----------
    turns = []
    for a, b in zip(rows, rows[1:]):
        dt = b["t"] - a["t"]
        if not (0.02 < dt < 0.5) or a["yaw"] is None or b["yaw"] is None:
            continue
        if abs(a["steer"]) < 15.0 or abs(a["drive"]) < 60.0:
            continue
        dpsi = ((b["yaw"] - a["yaw"]) + 180.0) % 360.0 - 180.0
        rate = abs(dpsi) / dt
        if rate < 1.0:            # commanded, but the car was not turning
            continue
        turns.append((abs(a["steer"]), rate, a["drive"]))
    if len(turns) > 20:
        rr = np.array([t[1] for t in turns])
        st = float(np.median([t[0] for t in turns]))
        print("\nturn rate while actually moving, at %.0f deg steering:" % st)
        print("  %.0f deg/s median, %.0f at the 75th pct, %.0f at the 90th (n=%d)"
              % (np.median(rr), np.percentile(rr, 75), np.percentile(rr, 90),
                 len(turns)))
        print("  -> a 90 degree corner takes %.1f s at the median, %.1f s at "
              "the 75th" % (90.0 / max(1, np.median(rr)),
                            90.0 / max(1, np.percentile(rr, 75))))

    # --- how close did a human let the walls get? -------------------------
    print("\nclearances a human driver actually used:")
    for k in ("front", "left", "right"):
        v = col(lambda r, k=k: r["sonar"].get(k))
        if not v:
            continue
        a = np.array(v)
        print("  %-6s min %.2f, 5th pct %.2f, median %.2f m"
              % (k, a.min(), np.percentile(a, 5), np.median(a)))
    fr = np.array(col(lambda r: r["sonar"].get("front")))
    if fr.size:
        print("  -> brake_m near %.2f, turn_at_m near %.2f"
              % (max(0.25, np.percentile(fr, 2)), np.percentile(fr, 10)))

    # --- corridor width ----------------------------------------------------
    w = [r["sonar"]["left"] + r["sonar"]["right"] for r in rows
         if r["sonar"].get("left") is not None
         and r["sonar"].get("right") is not None]
    if w:
        a = np.array(w)
        print("\ncorridor width: median %.2f m (%.2f-%.2f)"
              % (np.median(a), np.percentile(a, 5), np.percentile(a, 95)))

    # --- what did the corner lines do? ------------------------------------
    seen = {}
    for r in rows:
        for k, v in (r.get("lines") or {}).items():
            vf = v.get("v_frac", v.get("v")) if isinstance(v, dict) else v
            seen.setdefault(k, []).append((r["t"], vf, r["sonar"].get("front")))
    print("\ncorner lines seen: %s" % (", ".join(sorted(seen)) or "none"))
    for k, v in sorted(seen.items()):
        highs = [x for x in v if x[1] and x[1] > 0.6]
        print("  %-7s %d frames, %d with the line low in the frame" % (k, len(v), len(highs)))
        fronts = [x[2] for x in highs if x[2] is not None]
        if fronts:
            print("           front range when the line was underfoot: "
                  "%.2f m median" % float(np.median(fronts)))

    # --- sensor honesty ----------------------------------------------------
    print("\nsensor behaviour:")
    for k in ("front", "left", "right", "rear"):
        v = col(lambda r, k=k: r["sonar"].get(k))
        if not v:
            print("  %-6s no data" % k)
            continue
        print("  %-6s %.2f-%.2f m, %d distinct%s"
              % (k, min(v), max(v), len(set(round(x, 3) for x in v)),
                 "   <-- FROZEN" if len(set(round(x, 3) for x in v)) <= 3 else ""))
    both = [(r["sonar"].get("front"), r["lidar"].get("front")) for r in rows]
    both = [(a, b) for a, b in both if a is not None and b is not None]
    if both:
        print("  sonar vs lidar front: %.3f m median disagreement (n=%d)"
              % (float(np.median([abs(a - b) for a, b in both])), len(both)))
    camf = [(r["cam"].get("front"), r["sonar"].get("front")) for r in rows]
    camf = [(a, b) for a, b in camf if a is not None and b is not None]
    if camf:
        ratio = float(np.median([a / b for a, b in camf if b > 0.1]))
        print("  camera front / sonar front: %.2f (1.00 would be agreement)"
              % ratio)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/drive.jsonl")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--no-full-scan", dest="full_scan", action="store_false",
                    help="omit the 360-bin lidar scan (smaller files)")
    ap.add_argument("--analyse", default=None)
    ap.add_argument("--angle-offset", dest="angle_offset", type=float, default=176.0)
    ap.add_argument("--angle-sign", dest="angle_sign", type=float, default=1.0)
    args = ap.parse_args()
    if args.analyse:
        return analyse(args.analyse)
    return record(args)


if __name__ == "__main__":
    sys.exit(main())
