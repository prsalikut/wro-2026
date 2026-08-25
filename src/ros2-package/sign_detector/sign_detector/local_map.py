"""A short-memory map of the walls around the car (no ROS, no numpy needed).

The problem this solves: the forward sensors point where the car is AIMED, not
where its curving path is GOING. Turning into a corner swings them onto a wall
that was never in front of the car until it was -- recorded on this vehicle as
a front range sitting near one metre and then reading six centimetres in a
single sample. No amount of reacting faster helps, because there is nothing to
react to until it is too late.

But the car saw that wall a second earlier, off to the side, and then forgot it.
So: every ultrasonic reading is dropped into a small map in world coordinates,
using the IMU for heading and the drive odometry for distance. The car can then
ask "what is inside the arc I am about to drive through", including things
nothing is currently pointing at.

The map is deliberately short-lived. Dead reckoning drifts, so a point is only
trusted for a few seconds and a few metres, which is long enough to remember the
wall you are turning into and short enough that drift never accumulates.
"""

import math

__all__ = ["LocalMap"]


class LocalMap(object):

    def __init__(self, keep_s=6.0, keep_m=2.5, max_points=900, merge_m=0.04):
        self.keep_s = float(keep_s)
        self.keep_m = float(keep_m)
        self.max_points = int(max_points)
        self.merge_m = float(merge_m)
        self.pts = []                      # (x, y, t) in the world frame
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0                      # radians, + = counter-clockwise

    # ------------------------------------------------------------------ pose

    def set_heading(self, yaw_deg):
        """Absolute heading from the IMU, in the car's convention."""
        self.th = math.radians(yaw_deg)

    def advance(self, distance_m):
        """Move the car forward along its current heading."""
        self.x += distance_m * math.cos(self.th)
        self.y += distance_m * math.sin(self.th)

    # ------------------------------------------------------------------- add

    def add(self, now, mounts, readings, max_range=2.0):
        """Record one sweep of ultrasonic returns.

        `mounts` maps a sensor name to (forward_m, left_m, bearing_deg) on the
        car; `readings` maps the same names to a range or None.
        """
        for name, rng in readings.items():
            if rng is None or rng <= 0.02 or rng > max_range:
                continue
            mount = mounts.get(name)
            if mount is None:
                continue
            fx, fy, bearing = mount
            # sensor position in the world
            sx = self.x + fx * math.cos(self.th) - fy * math.sin(self.th)
            sy = self.y + fx * math.sin(self.th) + fy * math.cos(self.th)
            a = self.th + math.radians(bearing)
            self._put(sx + rng * math.cos(a), sy + rng * math.sin(a), now)
        self._prune(now)

    def add_polar(self, now, bearings_ranges, max_range=2.5):
        """Record a sweep given as (bearing_deg, range_m) in the car frame."""
        for bearing, rng in bearings_ranges:
            if rng is None or rng <= 0.05 or rng > max_range:
                continue
            a = self.th + math.radians(bearing)
            self._put(self.x + rng * math.cos(a),
                      self.y + rng * math.sin(a), now)
        self._prune(now)

    def _put(self, x, y, t):
        # Merge into a nearby point rather than piling up duplicates: the same
        # wall gets hit many times a second.
        m2 = self.merge_m * self.merge_m
        for i, (px, py, _pt) in enumerate(self.pts):
            if (px - x) ** 2 + (py - y) ** 2 <= m2:
                self.pts[i] = (px, py, t)
                return
        self.pts.append((x, y, t))

    def _prune(self, now):
        keep = []
        for (px, py, pt) in self.pts:
            if now - pt > self.keep_s:
                continue
            if math.hypot(px - self.x, py - self.y) > self.keep_m:
                continue
            keep.append((px, py, pt))
        if len(keep) > self.max_points:
            keep.sort(key=lambda p: p[2])
            keep = keep[-self.max_points:]
        self.pts = keep

    # ----------------------------------------------------------------- query

    def nearest(self, lo_deg, hi_deg, max_range=2.0):
        """Closest remembered obstacle in a bearing window, car frame.

        Returns (range, bearing_deg) or (None, None).
        """
        best = None
        for (px, py, _t) in self.pts:
            dx, dy = px - self.x, py - self.y
            r = math.hypot(dx, dy)
            if r < 0.03 or r > max_range:
                continue
            b = math.degrees(math.atan2(dy, dx) - self.th)
            b = (b + 180.0) % 360.0 - 180.0
            if lo_deg <= b <= hi_deg and (best is None or r < best[0]):
                best = (r, b)
        return best if best else (None, None)

    def clearance_along_arc(self, turn_dir, radius_m, sweep_deg=95.0,
                            half_width_m=0.13, step_deg=6.0):
        """Smallest gap between the car's body and anything remembered, if it
        drives an arc of `radius_m` turning `turn_dir` (+1 right, -1 left).

        This is the question the forward sensors cannot answer: not "is
        anything ahead" but "is anything inside the curve I am about to take".
        """
        if radius_m <= 0.05 or not self.pts:
            return None
        # Centre of the arc, in the car frame: directly to the side.
        side = -1.0 if turn_dir > 0 else 1.0        # +y is left
        cx_c, cy_c = 0.0, side * radius_m
        worst = None
        n = max(1, int(sweep_deg / step_deg))
        # Start a little way into the arc. The car is already where the arc
        # begins, and on a narrow corridor a wall is already alongside it
        # there -- including that point made every arc look equally blocked,
        # whichever way it turned.
        skip = 0.18
        for k in range(n + 1):
            a = math.radians(k * sweep_deg / n) * (-1.0 if turn_dir > 0 else 1.0)
            # point on the arc, car frame
            px = cx_c + (0.0 - cx_c) * math.cos(a) - (0.0 - cy_c) * math.sin(a)
            py = cy_c + (0.0 - cx_c) * math.sin(a) + (0.0 - cy_c) * math.cos(a)
            if math.hypot(px, py) < skip:
                continue
            # to world
            wx = self.x + px * math.cos(self.th) - py * math.sin(self.th)
            wy = self.y + px * math.sin(self.th) + py * math.cos(self.th)
            for (qx, qy, _t) in self.pts:
                d = math.hypot(qx - wx, qy - wy) - half_width_m
                if worst is None or d < worst:
                    worst = d
        return worst

    def summary(self):
        return {"points": len(self.pts),
                "x": round(self.x, 2), "y": round(self.y, 2),
                "heading": round(math.degrees(self.th), 1)}
