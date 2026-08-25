#!/usr/bin/env python3
"""Survey a lap one short step at a time, to measure what the real track does.

This is NOT how the round gets driven. It is how the numbers that the round is
driven ON get measured. Every cycle is: measure while stopped -> decide -> one
short pulse -> stop. The car is stationary whenever a decision is made and never
carries speed into one, so it cannot run into a wall, and a survey that ends
with somebody picking the car up teaches nothing.

What comes out of it, via --analyse, is a params.yaml block: how far a pulse
actually travels, how wide each corridor really is, what front distance a corner
line appears at, how fast the car turns per pulse, and which sensors are telling
the truth. The continuous driver is then tuned from those, instead of from a
simulator's assumptions.

Every cycle is: measure while stopped -> decide -> one short pulse -> stop.
The car is stationary whenever a decision is made and never carries speed into
one, so the failure that wrecked the continuous driver -- braking distance
longer than the distance it could see -- cannot happen here. It is slow. It is
meant to be.

Sensor roles, deliberately narrow:
  ultrasonics  primary distance, front/left/right/rear
  IMU          heading: how far a turn has actually come, and drift while straight
  camera       which way the round runs (orange line first = clockwise), and
               when a corner line is underfoot
  lidar        logged for comparison, and used for the front only when it
               agrees with the front sonar

Every step is written to a JSONL log, so a run that goes wrong can be read back
afterwards instead of re-run.

    python3 inch_round.py --dry              # decide and log, never drive
    python3 inch_round.py --run              # survey, inching
    python3 inch_round.py --run --turns 4    # one lap only
    python3 inch_round.py --analyse LOG      # turn the survey into parameters
"""

import argparse
import json
import math
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, Range
from std_msgs.msg import Float32, String


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


class InchRound(Node):

    def __init__(self, a):
        super().__init__("inch_round")
        self.a = a
        self.sonar = {}
        self.sonar_hist = {k: [] for k in ("front", "right", "rear", "left")}
        self.yaw = None
        self.scan = None
        self.lines = {}
        self.lane_ok = None
        self.pub_steer = self.create_publisher(Float32, "steering_cmd", 10)
        self.pub_drive = self.create_publisher(Float32, "drive_cmd", 10)
        self.pub_status = self.create_publisher(String, "inch_status", 10)
        for s in ("front", "right", "rear", "left"):
            self.create_subscription(Range, "sonar/{}".format(s),
                                     lambda m, k=s: self._son(k, m), 10)
        self.create_subscription(Float32, "imu/yaw",
                                 lambda m: setattr(self, "yaw", float(m.data)), 10)
        self.create_subscription(LaserScan, "scan",
                                 lambda m: setattr(self, "scan", m),
                                 qos_profile_sensor_data)
        self.create_subscription(String, "vision/lane", self._lane, 10)

    def _son(self, k, m):
        v = float(m.range)
        self.sonar[k] = v
        h = self.sonar_hist[k]
        h.append(v)
        del h[:-12]

    def _lane(self, m):
        try:
            d = json.loads(m.data)
        except ValueError:
            return
        self.lane_ok = d.get("ok")
        self.lines = d.get("lines") or {}

    # ------------------------------------------------------------- measuring

    def settle(self, secs):
        t0 = time.time()
        while time.time() - t0 < secs:
            rclpy.spin_once(self, timeout_sec=0.02)

    def measure(self):
        """Median of each sonar over a short window, with the car stopped."""
        for k in self.sonar_hist:
            self.sonar_hist[k] = []
        self.settle(self.a.settle)
        out = {}
        for k, h in self.sonar_hist.items():
            good = [v for v in h if self.a.son_min <= v <= self.a.son_max]
            out[k] = float(np.median(good)) if len(good) >= 3 else None
        out["yaw"] = self.yaw
        out["lidar_front"] = self.lidar_front()
        out["lines"] = {k: v.get("v_frac") for k, v in self.lines.items()}
        return out

    def lidar_front(self, half=10.0):
        m = self.scan
        if m is None:
            return None
        vals = []
        for i, r in enumerate(m.ranges):
            if not math.isfinite(r) or r <= m.range_min:
                continue
            ang = math.degrees(m.angle_min + i * m.angle_increment)
            phi = wrap((ang - self.a.angle_offset) / self.a.angle_sign)
            if abs(phi) <= half:
                vals.append(r)
        return float(np.median(vals)) if len(vals) >= 3 else None

    def front_of(self, m):
        """Front distance: the sonar, unless the lidar agrees and reads shorter.

        The ultrasonic is primary because it is the one that sees these walls.
        The lidar only gets a say when the two already agree, which keeps a
        stray lidar return from stopping the car and a lidar dropout from
        hiding a wall.
        """
        s, l = m.get("front"), m.get("lidar_front")
        if s is None:
            return l
        if l is not None and abs(l - s) <= self.a.agree:
            return min(s, l)
        return s

    # ------------------------------------------------------------- actuating

    def hold(self, steer, duty, secs):
        t0 = time.time()
        while time.time() - t0 < secs:
            self.pub_steer.publish(Float32(data=float(steer)))
            self.pub_drive.publish(Float32(data=float(duty)))
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.01)

    def stop(self, steer=0.0):
        for _ in range(8):
            self.pub_steer.publish(Float32(data=float(steer)))
            self.pub_drive.publish(Float32(data=0.0))
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.01)

    def pulse(self, steer, duty, secs, why):
        """One short push, then a full stop. The only way this car moves."""
        if self.a.dry:
            return
        self.hold(steer, duty, secs)
        self.stop(steer)
        del why


def choose_direction(n, m, log):
    """Clockwise or counter-clockwise, from the mat and then the walls.

    Measured off the official artwork: driving CLOCKWISE the car crosses the
    ORANGE line first at every corner, counter-clockwise the blue one. That is
    a property of the printed mat rather than the rules, so if no line has been
    seen yet the side walls decide instead -- the outer wall is the near one,
    and the car turns away from it.
    """
    for colour, turn, why in (("orange", +1.0, "orange line first = clockwise"),
                              ("blue", -1.0, "blue line first = anticlockwise")):
        if colour in (m.get("lines") or {}):
            log("direction: %s -> turn %s" % (why, "RIGHT" if turn > 0 else "LEFT"))
            return turn
    l, r = m.get("left"), m.get("right")
    if l is not None and r is not None and abs(l - r) > 0.08:
        turn = +1.0 if r > l else -1.0
        log("direction: more room on the %s (%.2f vs %.2f) -> turn %s"
            % ("right" if r > l else "left", r, l,
               "RIGHT" if turn > 0 else "LEFT"))
        return turn
    return 0.0


def analyse(path, args):
    """Read a survey log and print the numbers the real driver needs."""
    rows = [json.loads(l) for l in open(path)]
    steps = [r for r in rows if "step" in r]
    if len(steps) < 20:
        print("Only %d surveyed steps -- not enough to tune from." % len(steps))
        return 1
    print("\n%d surveyed steps, %d corners\n"
          % (len(steps), max(r.get("turns", 0) for r in steps)))

    # --- how far does one pulse actually travel? ---------------------------
    moves = []
    for a, b in zip(steps, steps[1:]):
        if a.get("decision") != "inch":
            continue
        fa, fb = a.get("front"), b.get("front")
        if fa is None or fb is None:
            continue
        d = fa - fb
        if 0.002 < d < 0.25:
            moves.append(d)
    if moves:
        per = float(np.median(moves))
        print("travel per %.2f s pulse at %.0f%%: %.3f m median (%.3f-%.3f, "
              "n=%d)" % (args.pulse, args.duty, per, min(moves), max(moves),
                         len(moves)))
        print("  -> speed_per_pct  %.4f   (%.2f m/s while the pulse is on)"
              % (per / args.pulse / args.duty, per / args.pulse))
    else:
        print("no clean forward moves found")

    # --- how wide is the corridor, per straight? ---------------------------
    widths = [(r["left"] + r["right"]) for r in steps
              if r.get("left") is not None and r.get("right") is not None
              and r.get("state") == "drive"]
    if widths:
        w = np.array(widths)
        print("\ncorridor width: %.2f m median, %.2f-%.2f (n=%d)"
              % (np.median(w), np.percentile(w, 5), np.percentile(w, 95), len(w)))
        print("  -> single_target_m %.2f, tight_lane_m %.2f"
              % (np.median(w) / 2.0, np.median(w) * 1.25))

    # --- what front distance does a corner actually happen at? -------------
    at = [r.get("front") for r in steps
          if r.get("decision") == "start turn" and r.get("front") is not None]
    if at:
        print("\ncorners started at front = %s m"
              % ", ".join("%.2f" % v for v in at))
        print("  -> turn_at_m %.2f" % (float(np.median(at))))

    # --- how fast does it turn per pulse? ----------------------------------
    rates = []
    for a, b in zip(steps, steps[1:]):
        if a.get("state") != "turn" or b.get("state") != "turn":
            continue
        d = b.get("swept", 0) - a.get("swept", 0)
        if 0.5 < d < 45:
            rates.append(d)
    if rates:
        print("\nturn per %.2f s pulse: %.1f deg median (n=%d)"
              % (args.turn_pulse, float(np.median(rates)), len(rates)))
        print("  -> about %d pulses for a 90 degree corner"
              % max(1, round(90.0 / float(np.median(rates)))))

    # --- which sensors told the truth? -------------------------------------
    print("\nsensor behaviour over the survey:")
    for k in ("front", "left", "right", "rear"):
        v = [r[k] for r in steps if r.get(k) is not None]
        if not v:
            print("  %-6s no data" % k)
            continue
        print("  %-6s %.2f-%.2f m, %d distinct in %d readings%s"
              % (k, min(v), max(v), len(set(round(x, 3) for x in v)), len(v),
                 "   <-- FROZEN" if len(set(round(x, 3) for x in v)) <= 3 else ""))
    both = [(r["front"], r["lidar_front"]) for r in steps
            if r.get("front") is not None and r.get("lidar_front") is not None]
    if both:
        diff = [abs(a - b) for a, b in both]
        print("  sonar vs lidar front: %.3f m median disagreement (n=%d)"
              % (float(np.median(diff)), len(diff)))

    # --- headings of the straights -----------------------------------------
    yaws = [r["yaw"] for r in steps
            if r.get("yaw") is not None and r.get("state") == "drive"]
    if len(yaws) > 20:
        print("\nheading while driving straights spanned %.0f to %.0f deg"
              % (min(yaws), max(yaws)))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="actually drive")
    ap.add_argument("--dry", action="store_true", help="decide and log only")
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--duty", type=float, default=72.0)
    ap.add_argument("--pulse", type=float, default=0.16)
    ap.add_argument("--turn-pulse", dest="turn_pulse", type=float, default=0.20)
    ap.add_argument("--settle", type=float, default=0.35)
    ap.add_argument("--turn-at", dest="turn_at", type=float, default=0.60)
    ap.add_argument("--hard-min", dest="hard_min", type=float, default=0.30)
    ap.add_argument("--turn-angle", dest="turn_angle", type=float, default=85.0)
    ap.add_argument("--max-steer", dest="max_steer", type=float, default=25.0)
    ap.add_argument("--kp", type=float, default=45.0)
    ap.add_argument("--centre-max", dest="centre_max", type=float, default=14.0)
    ap.add_argument("--trim", type=float, default=2.67)
    ap.add_argument("--agree", type=float, default=0.25)
    ap.add_argument("--son-min", dest="son_min", type=float, default=0.03)
    ap.add_argument("--son-max", dest="son_max", type=float, default=3.5)
    ap.add_argument("--angle-offset", dest="angle_offset", type=float, default=176.0)
    ap.add_argument("--angle-sign", dest="angle_sign", type=float, default=1.0)
    ap.add_argument("--log", default="/tmp/inch_round.jsonl")
    ap.add_argument("--analyse", default=None,
                    help="read a survey log and print tuned parameters")
    args = ap.parse_args()
    if args.analyse:
        return analyse(args.analyse, args)
    if not args.run:
        args.dry = True

    rclpy.init()
    n = InchRound(args)
    t0 = time.time()
    while (not n.sonar or n.yaw is None) and time.time() - t0 < 12:
        rclpy.spin_once(n, timeout_sec=0.2)
    if not n.sonar:
        print("no sonar data")
        return 1
    if n.yaw is None:
        print("no /imu/yaw -- turns need a heading reference")
        return 1

    fh = open(args.log, "w", buffering=1)
    def log(msg):
        print(msg)
        fh.write(json.dumps({"t": round(time.time() - t0, 2), "msg": msg}) + "\n")

    log("inch round: %s, %d corners, %.0f%% duty in %.2f s pulses"
        % ("DRY (no motor)" if args.dry else "DRIVING", args.turns,
           args.duty, args.pulse))
    log("stops between every step; it cannot carry speed into a wall")

    turn_dir = 0.0
    turns = 0
    trigger_colour = None
    step = 0
    state = "drive"
    swept = 0.0
    prev_yaw = None
    since_corner = 0

    try:
        while step < args.steps and turns < args.turns:
            step += 1
            m = n.measure()
            f = n.front_of(m)
            l, r = m.get("left"), m.get("right")
            rec = {"step": step, "state": state, "front": f, "left": l,
                   "right": r, "rear": m.get("rear"), "yaw": m.get("yaw"),
                   "lidar_front": m.get("lidar_front"), "lines": m["lines"],
                   "turns": turns, "swept": round(swept, 1)}

            if turn_dir == 0.0:
                turn_dir = choose_direction(n, m, log)

            if state == "drive":
                lines = m.get("lines") or {}
                if trigger_colour is None and lines:
                    trigger_colour = sorted(lines, key=lambda k: -lines[k])[0]
                    log("corner line colour locked: %s" % trigger_colour)
                on_line = (trigger_colour in lines
                           and lines.get(trigger_colour, 0) >= 0.70)
                too_close = f is not None and f <= args.turn_at
                if (on_line or too_close) and since_corner >= 6:
                    if turn_dir == 0.0:
                        turn_dir = -1.0
                        log("no direction evidence; assuming LEFT")
                    state = "turn"
                    swept = 0.0
                    prev_yaw = m.get("yaw")
                    turns += 1
                    since_corner = 0
                    log("corner %d: %s (front=%s, %s)"
                        % (turns, "RIGHT" if turn_dir > 0 else "LEFT",
                           None if f is None else round(f, 2),
                           "on the line" if on_line else "wall ahead"))
                    rec["decision"] = "start turn"
                    fh.write(json.dumps(rec) + "\n")
                    continue

                if f is not None and f <= args.hard_min:
                    log("  %.2f m ahead: too close to inch, backing up" % f)
                    n.pulse(-turn_dir * args.max_steer, -args.duty,
                            args.pulse, "back off")
                    rec["decision"] = "back off"
                    fh.write(json.dumps(rec) + "\n")
                    continue

                steer = args.trim
                if l is not None and r is not None:
                    err = r - l
                    steer += max(-args.centre_max,
                                 min(args.centre_max, args.kp * err * 0.5))
                elif l is not None:
                    steer += max(-args.centre_max,
                                 min(args.centre_max, args.kp * (0.45 - l)))
                elif r is not None:
                    steer += max(-args.centre_max,
                                 min(args.centre_max, args.kp * (r - 0.45)))
                steer = max(-args.max_steer, min(args.max_steer, steer))
                rec["decision"] = "inch"
                rec["steer"] = round(steer, 1)
                n.pulse(steer, args.duty, args.pulse, "inch")
                since_corner += 1
                fh.write(json.dumps(rec) + "\n")
                print("  %3d  F=%-6s L=%-6s R=%-6s steer%+5.1f  turns=%d"
                      % (step, _f(f), _f(l), _f(r), steer, turns))

            else:                                    # turning
                yaw = m.get("yaw")
                if yaw is not None and prev_yaw is not None:
                    # The BNO055 reports a compass heading, which grows
                    # clockwise; a right turn therefore increases it.
                    swept += turn_dir * wrap(yaw - prev_yaw)
                prev_yaw = yaw
                rec["swept"] = round(swept, 1)
                if swept >= args.turn_angle:
                    log("corner %d complete: swept %.0f deg" % (turns, swept))
                    state = "drive"
                    n.stop()
                    rec["decision"] = "turn done"
                    fh.write(json.dumps(rec) + "\n")
                    continue
                if f is not None and f <= args.hard_min:
                    rec["decision"] = "turn: back up"
                    n.pulse(-turn_dir * args.max_steer, -args.duty,
                            args.turn_pulse, "back up mid-corner")
                else:
                    rec["decision"] = "turn: forward"
                    n.pulse(turn_dir * args.max_steer, args.duty,
                            args.turn_pulse, "turn")
                fh.write(json.dumps(rec) + "\n")
                print("  %3d  TURN %s swept=%5.1f  F=%-6s"
                      % (step, "R" if turn_dir > 0 else "L", swept, _f(f)))

            n.pub_status.publish(String(data=json.dumps(rec)))
    except KeyboardInterrupt:
        log("interrupted")
    finally:
        n.stop()
        n.stop()
        log("stopped after %d steps, %d corners" % (step, turns))
        fh.close()
        n.destroy_node()
        rclpy.shutdown()
    print("\nlog: %s" % args.log)
    return 0


def _f(v):
    return "  --  " if v is None else "%6.3f" % v


if __name__ == "__main__":
    sys.exit(main())
