#!/usr/bin/env python3
"""Closed-loop simulator for the open-round driver: square track, kinematic bicycle
model, specular lidar. Publishes sim_scan, consumes sim_steer/sim_drive, prints JSON."""

import json
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, String

OUTER = 3.0
INNER = 1.0


def _segments():
    o = OUTER / 2.0
    i = INNER / 2.0
    return [
        ((-o, -o), (o, -o)), ((o, -o), (o, o)),
        ((o, o), (-o, o)), ((-o, o), (-o, -o)),
        ((-i, -i), (i, -i)), ((i, -i), (i, i)),
        ((i, i), (-i, i)), ((-i, i), (-i, -i)),
    ]


SEGS = _segments()

# SUPERSEDED by tools/sim_field.py, which models the real mat.
# WARNING: the colours below are the wrong way round, and the lines are
# modelled as chords across the lane rather than the 30-degree radial
# spokes the official artwork actually prints.  On the real mat a
# CLOCKWISE car crosses ORANGE first at every corner.  Kept only so the
# old range-only regression still runs; do not copy this table.
LINES = [
    ("orange", (0.5, -1.5), (0.5, -0.5)), ("blue", (0.5, -0.5), (1.5, -0.5)),
    ("orange", (0.5, 0.5), (1.5, 0.5)), ("blue", (0.5, 0.5), (0.5, 1.5)),
    ("orange", (-0.5, 0.5), (-0.5, 1.5)), ("blue", (-1.5, 0.5), (-0.5, 0.5)),
    ("orange", (-1.5, -0.5), (-0.5, -0.5)), ("blue", (-0.5, -1.5), (-0.5, -0.5)),
]


def _seg_cross(ax, ay, bx, by, cx_, cy_, dx_, dy_):
    d1 = (bx - ax) * (cy_ - ay) - (by - ay) * (cx_ - ax)
    d2 = (bx - ax) * (dy_ - ay) - (by - ay) * (dx_ - ax)
    d3 = (dx_ - cx_) * (ay - cy_) - (dy_ - cy_) * (ax - cx_)
    d4 = (dx_ - cx_) * (by - cy_) - (dy_ - cy_) * (bx - cx_)
    return d1 * d2 < 0 and d3 * d4 < 0


def _ray_seg(px, py, dx, dy, seg):
    """Return (distance along ray, |cos incidence|) or None on miss."""
    (x1, y1), (x2, y2) = seg
    sx, sy = x2 - x1, y2 - y1
    den = dx * sy - dy * sx
    if abs(den) < 1e-12:
        return None
    t = ((x1 - px) * sy - (y1 - py) * sx) / den
    u = ((x1 - px) * dy - (y1 - py) * dx) / den
    if t <= 0.0 or not (0.0 <= u <= 1.0):
        return None
    n = math.hypot(sx, sy)
    if n < 1e-12:
        return None
    nx, ny = -sy / n, sx / n
    return t, abs(dx * nx + dy * ny)


def _dist_to_walls(px, py):
    best = 1e9
    for (x1, y1), (x2, y2) in SEGS:
        sx, sy = x2 - x1, y2 - y1
        l2 = sx * sx + sy * sy
        u = 0.0 if l2 < 1e-12 else max(0.0, min(1.0, ((px - x1) * sx + (py - y1) * sy) / l2))
        best = min(best, math.hypot(px - (x1 + u * sx), py - (y1 + u * sy)))
    return best


class Sim(Node):

    def __init__(self, secs, specular_deg, seed_pose):
        super().__init__("open_round_sim")
        self.declare_parameters("", [
            ("n_beams", 270),
            ("scan_hz", 11.7),
            ("range_max", 8.0),
            ("range_min", 0.10),
            ("specular_deg", specular_deg),
            ("dropout", 0.10),  # black-surface random miss rate
            ("wheelbase_m", 0.15),
            ("body_radius_m", 0.13),
            ("stiction_pct", 30.0),
            ("v_at_45_ms", 0.50),
            ("tau_s", 0.15),
            ("angle_offset_deg", 170.0),  # must match node under test
            ("trace_path", ""),
            ("lines_enable", True),
            ("line_probe_m", 0.45),
            ("line_refire_s", 3.0),
        ])
        self.n = int(self._p("n_beams"))
        self.rmax = float(self._p("range_max"))
        self.rmin = float(self._p("range_min"))
        self.cos_min = math.cos(math.radians(float(self._p("specular_deg"))))
        self.dropout = float(self._p("dropout"))
        self.wheelbase = float(self._p("wheelbase_m"))
        self.rad = float(self._p("body_radius_m"))
        self.stiction = float(self._p("stiction_pct"))
        self.kv = float(self._p("v_at_45_ms")) / 45.0
        self.tau = float(self._p("tau_s"))
        self.angle_off = float(self._p("angle_offset_deg"))

        self.x, self.y, self.th = seed_pose
        self.v = 0.0
        self.steer = 0.0
        self.duty = 0.0

        self.collisions = 0
        self.in_collision = False
        self.min_clear = 1e9
        self.dist = 0.0
        self.unwrapped = 0.0
        self.prev_ang = math.atan2(self.y, self.x)
        self.secs = secs
        self.rng = 12345

        self.lines_on = bool(self._p("lines_enable"))
        self.probe = float(self._p("line_probe_m"))
        self.refire = float(self._p("line_refire_s"))
        self.line_fired = [-1e9] * len(LINES)
        self.pub_line = self.create_publisher(String, "line_event", 10)

        self.status = None
        self.trace = None
        trace_path = str(self._p("trace_path"))
        if trace_path:
            self.trace = open(trace_path, "w", buffering=1)
            self.trace.write(json.dumps({
                "k": "h", "outer": OUTER, "inner": INNER, "rad": self.rad,
                "angle_off": self.angle_off, "n": self.n,
                "lines": [[c, a[0], a[1], b[0], b[1]] for c, a, b in LINES],
            }) + "\n")
            self.create_subscription(String, "open_status", self._on_status, 10)
        self._trace_step = 0

        self.pub_scan = self.create_publisher(LaserScan, "sim_scan", 10)
        self.create_subscription(Float32, "sim_steer", self._on_steer, 10)
        self.create_subscription(Float32, "sim_drive", self._on_drive, 10)
        self.dt = 0.02
        self.t = 0.0
        self.create_timer(self.dt, self.step)
        self.create_timer(1.0 / float(self._p("scan_hz")), self.emit_scan)

    def _p(self, name):
        return self.get_parameter(name).value

    def _rand(self):
        self.rng = (1103515245 * self.rng + 12345) & 0x7FFFFFFF  # glibc LCG
        return self.rng / float(0x7FFFFFFF)

    def _on_steer(self, msg):
        self.steer = max(-25.0, min(25.0, float(msg.data)))

    def _on_drive(self, msg):
        self.duty = float(msg.data)

    def _on_status(self, msg):
        try:
            self.status = json.loads(msg.data)
        except ValueError:
            pass

    def _check_lines(self):
        if not self.lines_on:
            return
        ex = self.x + self.probe * math.cos(self.th)
        ey = self.y + self.probe * math.sin(self.th)
        for i, (color, a, b) in enumerate(LINES):
            if self.t - self.line_fired[i] < self.refire:
                continue
            if _seg_cross(self.x, self.y, ex, ey, a[0], a[1], b[0], b[1]):
                self.line_fired[i] = self.t
                self.pub_line.publish(String(data=color))
                if self.trace:
                    self.trace.write(json.dumps({
                        "k": "l", "t": round(self.t, 2), "c": color, "i": i,
                    }) + "\n")

    def step(self):
        self._advance()
        self._check_lines()
        self._check_walls()
        self._update_progress()
        if self.trace:
            self._trace_step += 1
            if self._trace_step % 2 == 0:
                st = self.status or {}
                self.trace.write(json.dumps({
                    "k": "s", "t": round(self.t, 2),
                    "x": round(self.x, 3), "y": round(self.y, 3),
                    "th": round(self.th, 4), "v": round(self.v, 3),
                    "st": round(self.steer, 1), "du": round(self.duty, 0),
                    "state": st.get("state"), "f": st.get("front"),
                }) + "\n")
        if self.t >= self.secs:
            self.report()
            rclpy.shutdown()

    def _advance(self):
        target = 0.0 if abs(self.duty) < self.stiction else self.kv * self.duty
        self.v += (target - self.v) * (self.dt / self.tau)
        if abs(self.v) < 1e-4:
            self.v = 0.0
        # positive steer = right = clockwise
        self.th -= (self.v / self.wheelbase) * math.tan(math.radians(self.steer)) * self.dt
        self.x += self.v * math.cos(self.th) * self.dt
        self.y += self.v * math.sin(self.th) * self.dt
        self.dist += abs(self.v) * self.dt
        self.t += self.dt

    def _check_walls(self):
        clear = _dist_to_walls(self.x, self.y)
        self.min_clear = min(self.min_clear, clear)
        if clear <= self.rad:
            if not self.in_collision:
                self.collisions += 1
                self.get_logger().error(
                    "COLLISION #{} at t={:.1f}s xy=({:.2f},{:.2f}) clear={:.3f}"
                    .format(self.collisions, self.t, self.x, self.y, clear))
            self.in_collision = True
            if self.trace:
                self.trace.write(json.dumps({
                    "k": "c", "t": round(self.t, 2),
                    "x": round(self.x, 3), "y": round(self.y, 3),
                }) + "\n")
        else:
            self.in_collision = False

    def _update_progress(self):
        ang = math.atan2(self.y, self.x)
        d = ang - self.prev_ang
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        self.unwrapped += d
        self.prev_ang = ang

    def _cast(self, dx, dy):
        best, best_cos = None, 0.0
        for seg in SEGS:
            hit = _ray_seg(self.x, self.y, dx, dy, seg)
            if hit is None:
                continue
            t, c = hit
            if best is None or t < best:
                best, best_cos = t, c
        return best, best_cos

    def emit_scan(self):
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "laser_frame"
        msg.angle_min = -math.pi
        msg.angle_max = math.pi
        msg.angle_increment = (2 * math.pi) / self.n
        msg.range_min = self.rmin
        msg.range_max = self.rmax
        msg.scan_time = 0.085
        out = []
        for i in range(self.n):
            alpha = msg.angle_min + i * msg.angle_increment
            phi = math.degrees(alpha) - self.angle_off  # scan angle -> robot-frame bearing
            world = self.th + math.radians(phi)
            best, best_cos = self._cast(math.cos(world), math.sin(world))
            if (best is None or best > self.rmax
                    or best_cos < self.cos_min  # too grazing, no return
                    or self._rand() < self.dropout):
                out.append(float("inf"))
            else:
                out.append(max(self.rmin, best + (self._rand() - 0.5) * 0.01))
        msg.ranges = out
        self.pub_scan.publish(msg)
        if self.trace:
            self.trace.write(json.dumps({
                "k": "z", "t": round(self.t, 2),
                "r": [None if not math.isfinite(r) else round(r, 2) for r in out],
            }) + "\n")

    def report(self):
        laps = abs(self.unwrapped) / (2 * math.pi)
        ok = self.collisions == 0 and self.min_clear > self.rad
        print(json.dumps({
            "verdict": "PASS" if ok else "FAIL",
            "collisions": self.collisions,
            "min_clearance_m": round(self.min_clear, 3),
            "body_radius_m": self.rad,
            "laps": round(laps, 2),
            "distance_m": round(self.dist, 2),
            "sim_seconds": round(self.t, 1),
            "end_xy": [round(self.x, 2), round(self.y, 2)],
        }, indent=2))
        if self.trace:
            self.trace.close()


def main():
    rclpy.init()
    boot = rclpy.create_node("sim_boot")
    boot.declare_parameter("sim_seconds", 60.0)
    boot.declare_parameter("specular_deg", 35.0)
    secs = float(boot.get_parameter("sim_seconds").value)
    spec = float(boot.get_parameter("specular_deg").value)
    boot.destroy_node()
    pose = (0.0, -(OUTER / 2.0 + INNER / 2.0) / 2.0, 0.0)  # mid-lane, lower straight, +x
    node = Sim(secs, spec, pose)
    try:
        rclpy.spin(node)
    except Exception:
        pass


if __name__ == "__main__":
    main()
