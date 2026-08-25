#!/usr/bin/env python3
"""Fit the open-round controller's gains to human-driven rounds.

What this takes from the recordings is a CONTROL LAW -- how much steering for a
given lane error, at what front distance a corner begins, how far a corner
swings, what duty holds a straight -- all as functions of what the sensors say
at that instant. What it deliberately does not take is a route: the field
layout is randomised after check time, so a remembered path is both against the
spirit of rule 9.9 and useless on the day. Rule 13.18 explicitly allows
calibrating to the field, and a fitted gain is calibration in the same sense a
tape-measured wheelbase is.

    python3 fit_policy.py /tmp/drives/*.jsonl
"""

import json
import math
import sys

import numpy as np


def load(paths):
    runs = []
    for p in paths:
        rows = [json.loads(l) for l in open(p)]
        runs.append((p.split("/")[-1], rows))
    return runs


def segments(rows, pred, min_s=0.4):
    """Contiguous stretches where `pred` holds for at least min_s."""
    out, i = [], 0
    while i < len(rows) - 1:
        if pred(rows[i]):
            j = i
            while j < len(rows) - 1 and pred(rows[j]):
                j += 1
            if rows[j - 1]["t"] - rows[i]["t"] >= min_s:
                out.append(rows[i:j])
            i = j
        else:
            i += 1
    return out


def main():
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return 2
    runs = load(paths)
    rows = [r for _n, rs in runs for r in rs]
    print("\nfitting to %d samples from %d rounds\n" % (len(rows), len(runs)))

    MOVING = 60.0

    # ---- 1. lane keeping: steering vs the left/right error ----------------
    # Only where the driver was tracking a corridor, not cornering: both walls
    # in sight, modest steering, moving.
    xs, ys = [], []
    for r in rows:
        l, rt = r["sonar"].get("left"), r["sonar"].get("right")
        if l is None or rt is None or l > 1.2 or rt > 1.2:
            continue
        if abs(r["drive"]) < MOVING or abs(r["steer"]) > 18:
            continue
        xs.append(rt - l)
        ys.append(r["steer"])
    if len(xs) > 200:
        x = np.array(xs)
        y = np.array(ys)
        A = np.vstack([x, np.ones_like(x)]).T
        (kp, bias), *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - (kp * x + bias)
        print("LANE KEEPING  (n=%d)" % len(x))
        print("  steer = %.1f * (right - left) %+.2f deg" % (kp, bias))
        print("  residual %.1f deg rms -- how tightly the human held the law"
              % float(np.sqrt(np.mean(resid ** 2))))
        print("  => centre_kp %.0f    trim_deg %+.2f" % (abs(kp) * 2.0, bias))
    else:
        kp, bias = 45.0, 0.0
        print("LANE KEEPING  too few clean samples; keeping defaults")

    # ---- 2. corners: when did the driver commit, and how far? -------------
    turns = []
    for _n, rs in runs:
        for seg in segments(rs, lambda r: abs(r["steer"]) >= 18
                            and abs(r["drive"]) >= MOVING, 0.35):
            pre = None
            idx = rs.index(seg[0])
            for k in range(max(0, idx - 12), idx):
                if rs[k]["sonar"].get("front") is not None:
                    pre = rs[k]
            yaws = [r["yaw"] for r in seg if r["yaw"] is not None]
            if pre is None or len(yaws) < 3:
                continue
            sw = 0.0
            for a, b in zip(yaws, yaws[1:]):
                sw += ((b - a) + 180) % 360 - 180
            lines = pre.get("lines") or {}
            turns.append({
                "front": pre["sonar"].get("front"),
                "lidar_front": (pre.get("lidar") or {}).get("front"),
                "swept": abs(sw),
                "dur": seg[-1]["t"] - seg[0]["t"],
                "duty": float(np.median([abs(r["drive"]) for r in seg])),
                "on_line": bool(lines),
                "line_v": max([(v.get("v") if isinstance(v, dict) else v) or 0
                               for v in lines.values()] or [0]),
            })
    real = [t for t in turns if t["swept"] > 45]
    print("\nCORNERS  %d committed turns, %d swung past 45 deg" % (len(turns), len(real)))
    if real:
        f = np.array([t["front"] for t in real if t["front"] is not None])
        sw = np.array([t["swept"] for t in real])
        du = np.array([t["dur"] for t in real])
        dy = np.array([t["duty"] for t in real])
        onl = sum(1 for t in real if t["on_line"])
        print("  front range at commit : %.2f m median (%.2f-%.2f)"
              % (np.median(f), np.percentile(f, 10), np.percentile(f, 90)))
        print("  yaw swung per corner  : %.0f deg median (%.0f-%.0f)"
              % (np.median(sw), np.percentile(sw, 10), np.percentile(sw, 90)))
        print("  duration              : %.1f s median" % np.median(du))
        print("  duty through the turn : %.0f%% median" % np.median(dy))
        print("  a corner line was in view at commit: %d of %d" % (onl, len(real)))
        print("  => turn_at_m %.2f   turn_angle_deg %.0f   max_turn_s %.1f"
              % (np.median(f), min(88.0, np.median(sw)), 2.5 * np.median(du)))

    # ---- 3. straights: how fast, and how close to the walls ---------------
    st = [r for r in rows if abs(r["steer"]) < 10 and abs(r["drive"]) >= MOVING]
    if st:
        d = np.array([abs(r["drive"]) for r in st])
        fr = np.array([r["sonar"]["front"] for r in st
                       if r["sonar"].get("front") is not None])
        print("\nSTRAIGHTS  (n=%d)" % len(st))
        print("  duty held             : %.0f%% median" % np.median(d))
        print("  front clearance       : %.2f m median, %.2f at the 5th pct"
              % (np.median(fr), np.percentile(fr, 5)))
        print("  => sustain_pct %.0f" % np.median(d))

    # ---- 4. how close did the human ever get? ----------------------------
    for k in ("left", "right"):
        a = np.array([r["sonar"][k] for r in rows if r["sonar"].get(k) is not None])
        print("  %-5s wall closest approach %.2f m (1st pct %.2f)"
              % (k, a.min(), np.percentile(a, 1)))

    # ---- 5. consistency: how repeatable were the rounds? ------------------
    print("\nCONSISTENCY between rounds")
    for name, rs in runs:
        f = np.array([r["sonar"]["front"] for r in rs
                      if r["sonar"].get("front") is not None])
        w = [r["sonar"]["left"] + r["sonar"]["right"] for r in rs
             if r["sonar"].get("left") is not None
             and r["sonar"].get("right") is not None
             and r["sonar"]["left"] < 1.2 and r["sonar"]["right"] < 1.2]
        secs = rs[-1]["t"] - rs[0]["t"]
        print("  %-22s %5.0f s  front med %.2f  corridor med %.2f"
              % (name, secs, np.median(f), np.median(w) if w else float("nan")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
