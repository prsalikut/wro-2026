#!/usr/bin/env python3
"""Record every sensor at once, then solve the lidar's mounting from the IMU.

The problem: `angle_offset_deg` and `angle_sign` say where the lidar's zero sits
relative to the car, they have never been verified, and a stationary snapshot
cannot settle them -- matching one frame against the camera left a 0.77 m
residual and several candidate angles that all looked equally bad.

The fix is to move the car and use the IMU as the reference, because rotation
turns one ambiguous snapshot into hundreds of constraints. A wall does not move.
So if the car's heading changes by X degrees, then in the CAR's frame that wall
must appear to swing by exactly -X. Only the correct (offset, sign) makes that
true across a whole rotation, and a sign error shows up immediately as the
bearing moving the wrong way.

Nothing here commands the motor or the servo: you turn the car by hand.

    # 1. record while you rotate the car through at least half a turn,
    #    slowly, and slide it around a bit too
    python3 collect_cal.py --collect --seconds 60 --out /tmp/cal.jsonl

    # 2. solve
    python3 collect_cal.py --analyse --out /tmp/cal.jsonl
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

BINS = 360


def _collect(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu, LaserScan, Range
    from std_msgs.msg import Float32, String

    class Rec(Node):
        def __init__(self):
            super().__init__("collect_cal")
            self.scan = None
            self.yaw = None
            self.lane = None
            self.gyro_z = None
            self.sonar = {}
            self.create_subscription(LaserScan, "scan", self._scan,
                                     qos_profile_sensor_data)
            self.create_subscription(Float32, "imu/yaw",
                                     lambda m: setattr(self, "yaw", float(m.data)), 10)
            self.create_subscription(Imu, "imu/data", self._imu,
                                     qos_profile_sensor_data)
            self.create_subscription(String, "vision/lane", self._lane, 10)
            for s in ("front", "right", "rear", "left"):
                self.create_subscription(
                    Range, "sonar/{}".format(s),
                    lambda m, k=s: self.sonar.__setitem__(k, float(m.range)), 10)

        def _scan(self, m):
            self.scan = m

        def _imu(self, m):
            self.gyro_z = float(m.angular_velocity.z)

        def _lane(self, m):
            try:
                self.lane = json.loads(m.data)
            except ValueError:
                pass

    rclpy.init()
    n = Rec()
    t0 = time.time()
    while (n.scan is None or n.yaw is None) and time.time() - t0 < 8:
        rclpy.spin_once(n, timeout_sec=0.2)
    if n.scan is None:
        print("No /scan.")
        return 1
    if n.yaw is None:
        print("No /imu/yaw -- the IMU is the reference here, so this will not")
        print("work without it.")
        return 1

    print("\nRecording for %.0f s to %s." % (args.seconds, args.out))
    print("TURN THE CAR BY HAND, slowly, through at least half a turn, and")
    print("slide it around a little as well. The more the heading changes, the")
    print("better the fit. Nothing will drive itself.\n")

    fh = open(args.out, "w")
    n_rec = 0
    yaws = []
    t_start = time.time()
    try:
        while time.time() - t_start < args.seconds:
            rclpy.spin_once(n, timeout_sec=0.05)
            if n.scan is None or n.yaw is None:
                continue
            m = n.scan
            binned = [None] * BINS
            for i, r in enumerate(m.ranges):
                if not math.isfinite(r) or r <= m.range_min:
                    continue
                a = math.degrees(m.angle_min + i * m.angle_increment)
                b = int(round(a)) % BINS
                if binned[b] is None or r < binned[b]:
                    binned[b] = round(r, 3)
            lane = n.lane or {}
            fh.write(json.dumps({
                "t": round(time.time() - t_start, 3),
                "yaw": round(n.yaw, 2),
                "gyro_z": None if n.gyro_z is None else round(n.gyro_z, 4),
                "scan": binned,
                "sonar": {k: round(v, 3) for k, v in n.sonar.items()},
                "cam": {k: lane.get(k) for k in
                        ("front", "left", "right", "heading", "ok")},
                "cam_profile": [[round(b), round(x, 3)]
                                for b, x, _y in (lane.get("profile") or [])],
            }) + "\n")
            n_rec += 1
            yaws.append(n.yaw)
            if n_rec % 40 == 0:
                span = _yaw_span(yaws)
                print("  %4d samples, heading covered %.0f deg" % (n_rec, span))
            time.sleep(max(0.0, 1.0 / args.hz - 0.05))
    except KeyboardInterrupt:
        print("\nstopped early")
    finally:
        fh.close()
        n.destroy_node()
        rclpy.shutdown()

    span = _yaw_span(yaws)
    print("\n%d samples, heading covered %.0f deg -> %s"
          % (n_rec, span, args.out))
    if span < 60:
        print("That is not much rotation. Re-record turning the car further;")
        print("below about 60 degrees the offset and the sign stay ambiguous.")
    return 0


def _yaw_span(yaws):
    if len(yaws) < 2:
        return 0.0
    un = _unwrap(yaws)
    return max(un) - min(un)


def _unwrap(yaws):
    out = [yaws[0]]
    for y in yaws[1:]:
        d = ((y - out[-1]) + 180.0) % 360.0 - 180.0
        out.append(out[-1] + d)
    return out


def _analyse(args):
    rows = [json.loads(l) for l in open(args.out)]
    if len(rows) < 20:
        print("Only %d samples; record more." % len(rows))
        return 1
    yaw_un = _unwrap([r["yaw"] for r in rows])
    span = max(yaw_un) - min(yaw_un)
    print("\n%d samples, %.0f deg of heading covered" % (len(rows), span))

    # --- 1. SIGN, from how the scan turns when the car turns ---------------
    # Between two samples a fraction of a second apart the car has barely
    # moved but its heading has changed, so a fixed object's scan bin shifts
    # by -dpsi*sign. Comparing consecutive scans is therefore insensitive to
    # the car being slid around, which is what defeated the naive
    # "fixed objects sit at a constant world bearing" test: that assumption
    # only holds for pure rotation, and this recording was not.
    print("\n1. Angle sign, from scan-to-scan rotation vs the IMU")
    votes = []
    for (r0, y0), (r1, y1) in zip(zip(rows, yaw_un), zip(rows[1:], yaw_un[1:])):
        dpsi = y1 - y0
        if abs(dpsi) < 1.0 or abs(dpsi) > 25.0:
            continue
        a = np.array([np.nan if v is None else v for v in r0["scan"]])
        b = np.array([np.nan if v is None else v for v in r1["scan"]])
        best_s, best_e = None, None
        for shift in range(-30, 31):
            bb = np.roll(b, shift)
            m = ~np.isnan(a) & ~np.isnan(bb)
            if m.sum() < 60:
                continue
            e = float(np.mean(np.abs(a[m] - bb[m])))
            if best_e is None or e < best_e:
                best_s, best_e = shift, e
        if best_s is None or best_s in (-30, 30):
            continue
        # scan shifts by -dpsi*sign, so sign = -shift/dpsi
        votes.append(-best_s / dpsi)
    if len(votes) >= 20:
        med = float(np.median(votes))
        sign = 1.0 if med > 0 else -1.0
        print("   %d usable rotations, measured sign %+.2f -> angle_sign = %+.0f"
              % (len(votes), med, sign))
        if abs(abs(med) - 1.0) > 0.35:
            print("   (magnitude is off 1.0; the scan may not be in the units")
            print("    assumed, or the IMU yaw is scaled)")
    else:
        sign = float(args.sign_hint)
        print("   only %d usable rotations; assuming sign %+.0f"
              % (len(votes), sign))

    # --- 2. OFFSET AND SIGN TOGETHER, from the ultrasonics ------------------
    # The four HC-SR04 point forward, right, back and left by construction, so
    # they are four independent angular references. Search offset and sign
    # JOINTLY: the scan-to-scan sign test above is easily defeated by a slow
    # rotation (a degree per sample is a single bin) and by the car being slid,
    # so it must not be allowed to fix the sign before this runs.
    print("\n2. Angle offset AND sign, from the sonars over all samples")
    MOUNT = {"front": 0.0, "left": 90.0, "rear": 180.0, "right": -90.0}
    live = {}
    for k in MOUNT:
        vals = [r["sonar"].get(k) for r in rows if r["sonar"].get(k) is not None]
        if len(vals) > 50 and len(set(vals)) > 5:
            live[k] = MOUNT[k]
    print("   using sonars: %s" % (", ".join(sorted(live)) or "none"))
    scored = []
    if live:
        for cand_sign in (1.0, -1.0):
            for off in np.arange(-180.0, 180.0, args.step):
                errs = []
                for r in rows[::3]:
                    scan = r["scan"]
                    for k, mount in live.items():
                        sv = r["sonar"].get(k)
                        if sv is None or sv > args.max_range:
                            continue
                        got = []
                        for b in range(BINS):
                            rng = scan[b]
                            if rng is None or rng > args.max_range:
                                continue
                            phi = (((b - off) / cand_sign) + 180.0) % 360.0 - 180.0
                            if abs(((phi - mount) + 180.0) % 360.0 - 180.0) <= 12.0:
                                got.append(rng)
                        if len(got) >= 3:
                            errs.append(abs(float(np.median(got)) - sv))
                if len(errs) > 80:
                    scored.append((float(np.median(errs)), float(off),
                                   cand_sign, len(errs)))
    if scored:
        scored.sort()
        print("   %9s %6s %12s %8s" % ("offset", "sign", "median_err", "pairs"))
        for e, off, sg, n_ in scored[:8]:
            print("   %9.1f %6.0f %12.3f %8d" % (off, sg, e, n_))
        best_err, best_off, sign, _ = scored[0]
        cur = [x for x in scored
               if abs(x[1] - 170.0) < args.step and x[2] == 1.0]
        print("\n   Best: angle_offset_deg = %.1f, angle_sign = %.0f "
              "(median error %.3f m)" % (best_off, sign, best_err))
        if cur:
            print("   Configured 170.0 / sign +1 scores %.3f m" % cur[0][0])
            d = abs(((best_off - 170.0) + 180) % 360 - 180)
            if best_err < cur[0][0] - 0.03:
                print("   -> configured value is %.0f deg out%s" % (
                    d, " AND the sign is flipped" if sign < 0 else ""))
            else:
                print("   -> configured value is as good as anything; keep it")
        # How sharply is this determined? A flat landscape means the recording
        # did not constrain it, whatever the winner happens to be.
        best_by_off = {}
        for e, off, sg, _n in scored:
            best_by_off[off] = min(e, best_by_off.get(off, 9e9))
        spread = sorted(best_by_off.values())
        print("   best %.3f m, median over all offsets %.3f m"
              % (spread[0], spread[len(spread) // 2]))
        if spread[0] > 0.8 * spread[len(spread) // 2]:
            print("   WARNING: the landscape is nearly flat -- this recording")
            print("   does not pin the mounting. Re-record with the car turned")
            print("   in place, near walls, without sliding it.")
        else:
            print("\n   paste into params.yaml under open_round:")
            print("       angle_offset_deg: %.1f" % best_off)
            print("       angle_sign: %.0f" % sign)
    else:
        print("   Not enough live sonar data to pin the offset.")

    # --- do the sonars respond to the world at all? -------------------------
    print("\nUltrasonics -- variation over the recording:")
    for k in ("front", "right", "rear", "left"):
        vals = [r["sonar"].get(k) for r in rows if r["sonar"].get(k) is not None]
        if not vals:
            print("  %-6s no data" % k)
            continue
        uniq = len(set(vals))
        print("  %-6s %5.2f-%5.2f m, %d distinct values in %d samples%s"
              % (k, min(vals), max(vals), uniq, len(vals),
                 "   <-- FROZEN, ignores the world" if uniq <= 2 else ""))

    # --- camera sanity ------------------------------------------------------
    ok = sum(1 for r in rows if (r["cam"] or {}).get("ok"))
    print("\nCamera reported ok in %d of %d samples" % (ok, len(rows)))
    heads = [r["cam"].get("heading") for r in rows
             if r["cam"].get("heading") is not None]
    if len(heads) > 10:
        print("  camera heading ranged %.0f to %.0f deg" % (min(heads), max(heads)))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--out", default="/tmp/cal.jsonl")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--hz", type=float, default=8.0)
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--bin-step", dest="bin_step", type=int, default=3)
    ap.add_argument("--max-range", dest="max_range", type=float, default=3.0)
    ap.add_argument("--sign-hint", dest="sign_hint", type=float, default=1.0)
    args = ap.parse_args()
    if args.collect:
        return _collect(args)
    if args.analyse:
        return _analyse(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
