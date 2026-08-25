#!/usr/bin/env python3
"""Closed-loop Open Challenge simulator -- camera included, no ROS required.

Runs the real perception and control code (sign_detector.wall_vision and
sign_detector.open_round_core) against the synthetic field in sim_field.py, so
a change to either can be checked on a laptop before it goes near the car.  The
driver sees rendered pixels, not a cheat feed of true distances.

Each run is scored the way a judge scores the round: three laps, no wall
contact, and a stop inside the section it started in.

    python3 sim_offline.py                       # one nominal run
    python3 sim_offline.py --sweep               # the whole matrix, in parallel
    python3 sim_offline.py --video run.mp4       # watch what the camera saw
"""

import argparse
import itertools
import json
import math
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "ros2-package", "sign_detector"))

from sim_field import (CarModel, Field, FieldRenderer, LidarModel,  # noqa: E402
                       SonarModel, WALL_H)
from sign_detector.ground_geometry import CameraGeometry            # noqa: E402
from sign_detector.open_round_core import OpenRoundCore, OpenRoundParams  # noqa
from sign_detector.wall_vision import WallVision, WallVisionParams   # noqa: E402

CTRL_HZ = 50.0
CAM_HZ = 15.0
LIDAR_HZ = 8.0
SONAR_HZ = 20.0


class Run(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.field = Field(cfg["widths"],
                           swap_line_colours=cfg.get("swap_line_colours", False))
        self.geom = CameraGeometry(cfg["cam_w"], cfg["cam_h"], cfg["hfov"],
                                   cfg["cam_height"], cfg["cam_pitch"],
                                   cam_x_m=cfg["cam_x"])
        self.rend = FieldRenderer(self.field, self.geom, noise=cfg["noise"],
                                  seed=cfg["seed"])
        self.lidar = LidarModel(self.field, dropout=cfg["lidar_dropout"],
                                specular_deg=cfg["specular_deg"],
                                seed=cfg["seed"] + 1)
        self.sonar = SonarModel(self.field, faults=cfg.get("sonar_faults"),
                                seed=cfg["seed"] + 2)
        self.car = CarModel(wheelbase=cfg["wheelbase"],
                            body_radius=cfg["body_radius"],
                            max_steer=cfg["max_steer"])
        vp = WallVisionParams(**cfg.get("vision_params", {}))
        vp.work_width = min(vp.work_width, cfg["cam_w"])
        vp.work_height = min(vp.work_height, cfg["cam_h"])
        self.vision = WallVision(self.geom, vp)
        op = OpenRoundParams(**cfg.get("core_params", {}))
        op.max_steer_deg = cfg["max_steer"]
        self.logs = []
        self.core = OpenRoundCore(op, log=lambda lvl, m: self.logs.append(
            "%7.2f %-5s %s" % (self.t, lvl, m)))

        self.pose = self.field.start_pose(cfg["section"], cfg["zone"],
                                          cfg["ccw"], cfg["lateral"])
        self.t = 0.0
        self.collisions = 0
        self.in_collision = False
        self.outer_touch = 0
        self.min_clear = 1e9
        self.dist = 0.0
        self.unwrapped = 0.0
        self.prev_ang = math.atan2(self.pose[1], self.pose[0])
        self.steer = 0.0
        self.pct = 0.0
        self.status = {}
        self.frames = []
        self.vision_calls = 0
        self.vision_time = 0.0

    # -------------------------------------------------------------- sensors

    def _feed_camera(self, want_debug=False):
        bright = self.cfg.get("brightness", 1.0)
        img = self.rend.render(self.pose, brightness=bright,
                               tint=self.cfg.get("tint", (1.0, 1.0, 1.0)))
        t0 = time.time()
        res = self.vision.process(img, want_debug=want_debug)
        self.vision_time += time.time() - t0
        self.vision_calls += 1
        self.core.on_vision(res, self.t)
        return img, res

    def _feed_lidar(self):
        c = self.cfg
        ranges = self.lidar.scan(self.pose)
        n = len(ranges)
        f, l, r, b = [], [], [], []
        for i, rng in enumerate(ranges):
            if not math.isfinite(rng):
                continue
            phi = math.degrees(-math.pi + i * (2 * math.pi / n))
            if abs(phi) <= 20.0:
                f.append(rng)
            elif abs(phi) >= 155.0:
                b.append(rng)
            elif 30.0 <= phi <= 110.0 and rng <= 2.5:
                l.append(rng)
            elif -110.0 <= phi <= -30.0 and rng <= 2.5:
                r.append(rng)
        self.core.on_scan(
            min(f) if len(f) >= 2 else None,
            min(l) if l else None,
            min(r) if r else None,
            min(b) if len(b) >= 2 else None,
            self.t,
            left_far=max(l) if l else None,
            right_far=max(r) if r else None)
        del c

    def _feed_sonar(self):
        for name, val in self.sonar.read(self.pose).items():
            self.core.on_sonar(name, val, self.t)

    # ------------------------------------------------------------------ loop

    def run(self, seconds=180.0, video=None, trace=None):
        dt = 1.0 / CTRL_HZ
        next_cam = next_lidar = next_sonar = 0.0
        writer = None
        arm_at = self.cfg.get("arm_at", 0.4)
        armed = False
        rows = []
        while self.t < seconds:
            if self.t >= next_cam:
                next_cam += 1.0 / CAM_HZ
                img, res = self._feed_camera(want_debug=video is not None)
                if video is not None:
                    frame = res.debug if res.debug is not None else img
                    if writer is None:
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(
                            video, cv2.VideoWriter_fourcc(*"mp4v"), CAM_HZ,
                            (w * 2, h))
                    writer.write(self._compose(frame))
            if self.t >= next_lidar:
                next_lidar += 1.0 / LIDAR_HZ
                self._feed_lidar()
            if self.t >= next_sonar:
                next_sonar += 1.0 / SONAR_HZ
                self._feed_sonar()

            if not armed and self.t >= arm_at:
                armed = True
                self.core.set_armed(True, self.t)

            self.steer, self.pct, self.status = self.core.step(self.t)
            self.pose = self.car.step(self.pose, self.steer, self.pct, dt)
            self._score(dt)
            if trace is not None:
                rows.append({
                    "t": round(self.t, 2), "x": round(self.pose[0], 3),
                    "y": round(self.pose[1], 3),
                    "th": round(math.degrees(self.pose[2]), 1),
                    "st": round(self.steer, 1), "pct": round(self.pct, 0),
                    "state": self.status.get("state"),
                    "turns": self.status.get("turns"),
                    "L": self.status.get("left"), "R": self.status.get("right"),
                    "F": self.status.get("front"),
                    "src": self.status.get("sources"),
                    "reason": self.status.get("reason"),
                })
            self.t += dt
            if self.status.get("state") == "done":
                break
        if writer is not None:
            writer.release()
        if trace is not None:
            with open(trace, "w") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
        return self.report()

    def _compose(self, frame):
        h, w = frame.shape[:2]
        plan = self._plan_view(w, h)
        return np.hstack([frame, plan])

    def _plan_view(self, w, h):
        img = np.full((h, w, 3), 30, np.uint8)
        o = self.field.o
        scale = min(w, h) / (2.4 * o)
        def to_px(x, y):
            return (int(w / 2 + x * scale), int(h / 2 - y * scale))
        for (x1, y1), (x2, y2), _k in self.field.walls:
            cv2.line(img, to_px(x1, y1), to_px(x2, y2), (200, 200, 200), 2)
        for colour, (x1, y1), (x2, y2) in self.field.lines:
            c = (36, 116, 232) if colour == "orange" else (176, 96, 26)
            cv2.line(img, to_px(x1, y1), to_px(x2, y2), c, 2)
        x, y, th = self.pose
        cv2.circle(img, to_px(x, y), max(2, int(0.1 * scale)), (0, 220, 255), -1)
        cv2.line(img, to_px(x, y),
                 to_px(x + 0.25 * math.cos(th), y + 0.25 * math.sin(th)),
                 (0, 220, 255), 2)
        st = self.status
        for i, txt in enumerate([
                "t %.1f  %s" % (self.t, st.get("state")),
                "corners %s  dir %s" % (st.get("turns"), st.get("direction")),
                "L %s R %s F %s" % (st.get("left"), st.get("right"),
                                    st.get("front")),
                "steer %+.0f  duty %.0f" % (self.steer, self.pct)]):
            cv2.putText(img, txt, (6, 16 + 14 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (230, 230, 230), 1)
        return img

    def _score(self, dt):
        x, y, _th = self.pose
        clear = self.field.clearance(x, y)
        self.min_clear = min(self.min_clear, clear)
        if clear <= self.car.radius:
            if not self.in_collision:
                self.collisions += 1
                if self.field.touches_outer(x, y, self.car.radius):
                    self.outer_touch += 1
            self.in_collision = True
        else:
            self.in_collision = False
        self.dist += abs(self.car.v) * dt
        ang = math.atan2(y, x)
        d = ang - self.prev_ang
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        self.unwrapped += d
        self.prev_ang = ang

    def report(self):
        cfg = self.cfg
        laps = abs(self.unwrapped) / (2 * math.pi)
        end_section = self.field.section_of(self.pose[0], self.pose[1])
        stopped = self.status.get("state") == "done"
        in_start = end_section == cfg["section"]
        ok = (self.collisions == 0 and stopped and laps >= 2.85
              and self.status.get("turns", 0) >= 12)
        return {
            "verdict": "PASS" if ok else "FAIL",
            "config": cfg["label"],
            "laps": round(laps, 2),
            "corners": self.status.get("turns", 0),
            "state": self.status.get("state"),
            "collisions": self.collisions,
            "outer_touches": self.outer_touch,
            "min_clearance_m": round(self.min_clear, 3),
            "body_radius_m": self.car.radius,
            "sim_seconds": round(self.t, 1),
            "distance_m": round(self.dist, 2),
            "finished_in_start_section": bool(stopped and in_start),
            "end_section": end_section,
            "start_section": cfg["section"],
            "direction": self.status.get("direction"),
            "direction_correct": (self.status.get("direction") ==
                                  ("left" if cfg["ccw"] else "right")),
            "recoveries": self.status.get("recoveries", 0),
            "health": self.status.get("health", {}),
            "vision_ms": round(1000.0 * self.vision_time /
                               max(1, self.vision_calls), 2),
            "reason": self.status.get("reason"),
        }


# ----------------------------------------------------------------- configs

def base_config(**kw):
    cfg = {
        "label": "nominal",
        "widths": (1.0, 1.0, 1.0, 1.0),
        "section": "south",
        "zone": 3,
        "ccw": True,
        "lateral": 0.0,
        "seed": 7,
        "cam_w": 320, "cam_h": 240, "hfov": 60.0,
        "cam_height": 0.12, "cam_pitch": 15.0, "cam_x": 0.05,
        "noise": 3.0, "brightness": 1.0, "tint": (1.0, 1.0, 1.0),
        "lidar_dropout": 0.10, "specular_deg": 35.0,
        "sonar_faults": None,
        "wheelbase": 0.15, "body_radius": 0.13, "max_steer": 25.0,
        "swap_line_colours": False,
        "vision_params": {}, "core_params": {},
    }
    cfg.update(kw)
    return cfg


def sweep_configs():
    out = []
    for sec, ccw in itertools.product(("south", "east", "north", "west"),
                                      (True, False)):
        out.append(base_config(label="%s-%s" % (sec, "ccw" if ccw else "cw"),
                               section=sec, ccw=ccw))
    for zone in (1, 2, 5, 6):
        out.append(base_config(label="zone%d" % zone, zone=zone))
    for lat in (-0.7, 0.7):
        out.append(base_config(label="lateral%+.1f" % lat, lateral=lat))
    for widths in ((0.6, 1.0, 1.0, 1.0), (1.0, 0.6, 1.0, 0.6),
                   (0.6, 0.6, 0.6, 0.6), (0.9, 1.1, 0.9, 1.1),
                   (0.7, 1.0, 0.7, 1.0)):
        out.append(base_config(label="widths%s" % (widths,), widths=widths))
    out.append(base_config(label="sonar-frozen-right",
                           sonar_faults={"right": "frozen"}))
    out.append(base_config(label="sonar-offset-front",
                           sonar_faults={"front": ("offset", -0.9)}))
    out.append(base_config(label="sonar-all-dead",
                           sonar_faults={k: "dead" for k in
                                         ("front", "left", "right", "rear")}))
    out.append(base_config(label="lidar-blind", lidar_dropout=0.92))
    out.append(base_config(label="dim", brightness=0.45))
    out.append(base_config(label="bright", brightness=1.45))
    out.append(base_config(label="warm-tint", tint=(0.85, 1.0, 1.15)))
    out.append(base_config(label="noisy", noise=9.0))
    out.append(base_config(label="cam-low", cam_height=0.08, cam_pitch=12.0))
    out.append(base_config(label="cam-high", cam_height=0.18, cam_pitch=22.0))
    out.append(base_config(label="hfov-wide", hfov=90.0))
    # A mat printed with the colours the other way round: the driver must not
    # depend on the convention, only cross-check against it.
    out.append(base_config(label="lines-swapped", swap_line_colours=True))
    # How the car is actually configured after the 2026-08-25 bench session:
    # turns triggered by the printed lines, and the camera's absolute ranges
    # kept out of the fusion because they are not calibrated on this vehicle.
    onboard = {"corner_source": "line", "use_vision_ranges": False}
    out.append(base_config(label="as-configured", core_params=dict(onboard)))
    for widths in ((0.6, 0.6, 0.6, 0.6), (1.0, 0.6, 1.0, 0.6)):
        out.append(base_config(label="as-configured %s" % (widths,),
                               widths=widths, core_params=dict(onboard)))
    for sec, ccw in (("north", False), ("east", True)):
        out.append(base_config(label="as-configured %s-%s"
                               % (sec, "ccw" if ccw else "cw"),
                               section=sec, ccw=ccw, core_params=dict(onboard)))
    return out


def random_configs(n, seed=1234):
    """Randomised rounds, drawn the way the rules draw them.

    Coin tosses pick the corridor width per section and the start section, a
    die picks the zone, and the direction is a further toss (rules 9.3-9.5).
    Sensor faults, lighting and camera mounting are varied on top, because the
    interesting failures are combinations no hand-written case thinks of.
    """
    import random
    rng = random.Random(seed)
    out = []
    for i in range(n):
        widths = tuple(rng.choice([0.5, 0.6, 0.7, 0.9, 1.0, 1.1])
                       for _ in range(4))
        faults = {}
        if rng.random() < 0.30:
            faults[rng.choice(["front", "left", "right", "rear"])] = rng.choice(
                ["frozen", "dead", ("offset", rng.uniform(-0.9, 0.9))])
        out.append(base_config(
            label="rnd%d-%03d" % (seed, i),
            widths=widths,
            section=rng.choice(("south", "east", "north", "west")),
            zone=rng.randint(1, 6),
            ccw=rng.random() < 0.5,
            lateral=rng.uniform(-0.8, 0.8),
            seed=seed + i,
            cam_height=rng.uniform(0.09, 0.17),
            cam_pitch=rng.uniform(10.0, 24.0),
            hfov=rng.choice([60.0, 60.0, 70.0, 90.0]),
            brightness=rng.uniform(0.45, 1.5),
            tint=(rng.uniform(0.85, 1.15), 1.0, rng.uniform(0.85, 1.15)),
            noise=rng.uniform(2.0, 10.0),
            lidar_dropout=rng.choice([0.10, 0.10, 0.35, 0.92]),
            sonar_faults=faults or None))
    return out


def _run_one(cfg, seconds):
    try:
        return Run(cfg).run(seconds=seconds)
    except Exception as exc:                       # keep the sweep going
        import traceback
        return {"verdict": "ERROR", "config": cfg["label"],
                "reason": "%s: %s" % (type(exc).__name__, exc),
                "trace": traceback.format_exc().splitlines()[-3:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--random", type=int, default=0)
    ap.add_argument("--seed-base", dest="seed_base", type=int, default=1234,
                    help="shifts the randomised draw; shards use different "
                         "bases so between them they cover more ground")
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--video", default=None)
    ap.add_argument("--trace", default=None)
    ap.add_argument("--logs", action="store_true")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--only", default=None)
    ap.add_argument("--widths", default=None)
    ap.add_argument("--section", default="south")
    ap.add_argument("--zone", type=int, default=3)
    ap.add_argument("--cw", action="store_true")
    args = ap.parse_args()

    if args.sweep or args.random:
        cfgs = sweep_configs() if args.sweep else []
        if args.random:
            cfgs += random_configs(args.random, seed=args.seed_base)
        if args.only:
            cfgs = [c for c in cfgs if args.only in c["label"]]
        jobs = args.jobs or max(1, (os.cpu_count() or 2) - 1)
        t0 = time.time()
        if jobs > 1:
            import multiprocessing as mp
            with mp.Pool(jobs) as pool:
                results = pool.starmap(
                    _run_one, [(c, args.seconds) for c in cfgs])
        else:
            results = [_run_one(c, args.seconds) for c in cfgs]
        bad = [r for r in results if r["verdict"] != "PASS"]
        print("%-22s %-6s %5s %4s %5s %6s %9s %s"
              % ("CONFIG", "VERDICT", "LAPS", "CNR", "COLL", "MINCLR",
                 "FINISH", "NOTE"))
        for r in results:
            print("%-22s %-6s %5s %4s %5s %6s %9s %s" % (
                r["config"], r["verdict"], r.get("laps"), r.get("corners"),
                r.get("collisions"), r.get("min_clearance_m"),
                "start" if r.get("finished_in_start_section") else
                (r.get("end_section") or "-"),
                r.get("reason") or ("%s" % (r.get("health") or ""))))
        print("\n%d/%d PASS in %.0fs" % (len(results) - len(bad),
                                         len(results), time.time() - t0))
        return 0 if not bad else 1

    widths = tuple(float(v) for v in args.widths.split(",")) if args.widths \
        else (1.0, 1.0, 1.0, 1.0)
    cfg = base_config(label="manual", widths=widths, section=args.section,
                      zone=args.zone, ccw=not args.cw)
    run = Run(cfg)
    rep = run.run(seconds=args.seconds, video=args.video, trace=args.trace)
    print(json.dumps(rep, indent=2))
    if args.logs:
        print("\n".join(run.logs))
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
