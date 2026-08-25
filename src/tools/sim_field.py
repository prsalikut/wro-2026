#!/usr/bin/env python3
"""A synthetic WRO Future Engineers field, rendered to a camera image.

The point of this file is that the open-round driver is now camera-first, and a
camera-first driver cannot be trusted on the strength of a range-only simulator.
So the field is modelled in 3D -- mat, ten-centimetre black walls, the orange and
blue corner lines -- and projected through the same pinhole model the real node
uses (sign_detector.ground_geometry), which means the vision code under test sees
pixels, finds the wall base, and reports a distance, exactly as it will on the
mat.

It also models the two sensors that have actually failed on this car:

  * the lidar, with a specular cutoff, so glossy walls at grazing incidence
    return nothing (that is the documented X2 behaviour on this track);
  * the ultrasonics, with optional fault injection -- frozen output and a
    constant offset -- because DEPLOY.md records the right sonar sitting at
    exactly 0.43 m for 213 consecutive samples while the wall was 2.94 m away.

Geometry follows the 2026 rules: a 3 m outer square, and a corridor that is
1.0 m or 0.6 m per section, chosen independently, which makes the inner block a
rectangle rather than a square.

Standalone: `python3 sim_field.py --preview out.png` renders one frame.
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "ros2-package",
    "sign_detector"))
from sign_detector.ground_geometry import CameraGeometry  # noqa: E402

OUTER = 3.0                 # inner face to inner face of the outer walls
WALL_H = 0.10
NARROW = 0.6
WIDE = 1.0

# Sections, counter-clockwise, with the compass edge each one hugs.
STRAIGHTS = ("south", "east", "north", "west")
CORNERS_CCW = ("SE", "NE", "NW", "SW")


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


class Field(object):
    """Mat geometry for one round configuration.

    `widths` is the corridor width for the south, east, north and west
    straights, in that order; the rules pick each independently.
    """

    def __init__(self, widths=(WIDE, WIDE, WIDE, WIDE), outer=OUTER,
                 swap_line_colours=False):
        self.outer = float(outer)
        o = self.outer * 0.5
        self.o = o
        w_s, w_e, w_n, w_w = [float(w) for w in widths]
        self.widths = (w_s, w_e, w_n, w_w)
        # Inner block: each face sits `width` in from the outer wall it faces.
        self.y0 = -o + w_s            # inner block south face
        self.x1 = o - w_e             # east face
        self.y1 = o - w_n             # north face
        self.x0 = -o + w_w            # west face
        if self.x0 >= self.x1 or self.y0 >= self.y1:
            raise ValueError("corridors leave no inner block")

        self.swap_line_colours = bool(swap_line_colours)
        self.walls = self._walls()
        self.lines = self._lines()

    # ------------------------------------------------------------ geometry

    def _walls(self):
        o = self.o
        outer = [((-o, -o), (o, -o)), ((o, -o), (o, o)),
                 ((o, o), (-o, o)), ((-o, o), (-o, -o))]
        x0, x1, y0, y1 = self.x0, self.x1, self.y0, self.y1
        inner = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                 ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
        return [(a, b, "outer") for a, b in outer] + \
               [(a, b, "inner") for a, b in inner]

    def _lines(self):
        """The eight corner lines, as (colour, (x1,y1), (x2,y2)).

        These are NOT chords across the lane.  On the real mat each corner
        carries two 20 mm bands that run from the corner of the *nominal*
        1000 x 1000 mm inner square out to the outer wall, meeting it 420 mm
        from the field corner -- so they are radial spokes about 30 and 60
        degrees off the wall, printed at fixed positions that do not move when
        the inner block changes size.

        Colour convention, measured from the official artwork: a CLOCKWISE car
        crosses ORANGE first at every corner, a counter-clockwise one crosses
        BLUE first.  It is a property of the printed mat, not of the rulebook,
        so the driver only ever uses it as a cross-check.
        """
        o, a = self.o, LINE_APEX
        w = o - LINE_WALL_OFFSET
        out = []
        for sx in (1.0, -1.0):
            for sy in (1.0, -1.0):
                horiz = (sx * sy * a, sy * o)       # meets the wall y = sy*o
                vert = (sx * o, sy * a * sy)        # meets the wall x = sx*o
                horiz = (sx * w, sy * o)
                vert = (sx * o, sy * w)
                # Orange reaches the horizontal wall at NE and SW, the vertical
                # wall at SE and NW.
                if sx * sy > 0:
                    pairs = (("orange", horiz), ("blue", vert))
                else:
                    pairs = (("orange", vert), ("blue", horiz))
                apex = (sx * a, sy * a)
                for colour, end in pairs:
                    seg = self._clip_outside_block(apex, end)
                    if seg is not None:
                        name = colour
                        if self.swap_line_colours:
                            name = "blue" if colour == "orange" else "orange"
                        out.append((name, seg[0], seg[1]))
        return out

    def _clip_outside_block(self, apex, end):
        """Trim the part of a line that the inner block is standing on.

        With a narrow corridor the block is bigger than the nominal square, so
        the apex end of each band is hidden under it and only the outer part is
        painted on visible mat.
        """
        def inside(pt):
            return (self.x0 - 1e-6 <= pt[0] <= self.x1 + 1e-6
                    and self.y0 - 1e-6 <= pt[1] <= self.y1 + 1e-6)

        if not inside(apex):
            return apex, end
        if inside(end):
            return None
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            pt = (apex[0] + (end[0] - apex[0]) * mid,
                  apex[1] + (end[1] - apex[1]) * mid)
            if inside(pt):
                lo = mid
            else:
                hi = mid
        start = (apex[0] + (end[0] - apex[0]) * hi,
                 apex[1] + (end[1] - apex[1]) * hi)
        return start, end

    def markings(self):
        """Printed grey and yellow-green marks that are not lines and not walls.

        The yellow-green ring runs 50 mm inside every wall -- a bright chromatic
        stripe right at the wall base, which is exactly where the vision looks
        for the floor/wall join -- and the grey section grid crosses the lane
        everywhere.  Both belong in any honest test of the segmentation.
        """
        o = self.o
        out = []
        r = o - RING_INSET
        for a, b in (((-r, -r), (r, -r)), ((r, -r), (r, r)),
                     ((r, r), (-r, r)), ((-r, r), (-r, -r))):
            out.append((RING_BGR, RING_W, a, b))
        # Section and zone boundaries, drawn solid rather than dashed: more grey
        # pixels than the real mat has, so passing this is the harder result.
        for v in (-1.1, -0.9, -0.5, 0.0, 0.5, 0.9, 1.1):
            out.append((GRID_BGR, GRID_W, (v, -o), (v, o)))
            out.append((GRID_BGR, GRID_W, (-o, v), (o, v)))
        return out

    def clearance(self, x, y):
        """Distance from (x, y) to the nearest wall, metres."""
        best = 1e9
        for (x1, y1), (x2, y2), _kind in self.walls:
            sx, sy = x2 - x1, y2 - y1
            l2 = sx * sx + sy * sy
            u = 0.0 if l2 < 1e-12 else _clamp(
                ((x - x1) * sx + (y - y1) * sy) / l2, 0.0, 1.0)
            best = min(best, math.hypot(x - (x1 + u * sx), y - (y1 + u * sy)))
        return best

    def touches_outer(self, x, y, radius):
        o = self.o
        return (abs(x) > o - radius) or (abs(y) > o - radius)

    def section_of(self, x, y):
        """Which of the eight sections a point is in."""
        east = x > self.x1
        west = x < self.x0
        north = y > self.y1
        south = y < self.y0
        if north and east:
            return "NE"
        if north and west:
            return "NW"
        if south and east:
            return "SE"
        if south and west:
            return "SW"
        if north:
            return "north"
        if south:
            return "south"
        if east:
            return "east"
        if west:
            return "west"
        return "inside"          # on top of the block: impossible for the car

    def lane_centre(self, section):
        """Mid-corridor point of a straight section."""
        o = self.o
        if section == "south":
            return (0.5 * (self.x0 + self.x1), 0.5 * (-o + self.y0))
        if section == "north":
            return (0.5 * (self.x0 + self.x1), 0.5 * (self.y1 + o))
        if section == "east":
            return (0.5 * (self.x1 + o), 0.5 * (self.y0 + self.y1))
        if section == "west":
            return (0.5 * (-o + self.x0), 0.5 * (self.y0 + self.y1))
        raise ValueError(section)

    def start_pose(self, section, zone=3, ccw=True, lateral=0.0):
        """A legal start pose: inside a straight section, facing the direction
        of travel.  `zone` 1..6 slides the car along the section as the dice
        roll does; `lateral` (-1..1) offsets it across the corridor."""
        o = self.o
        cx, cy = self.lane_centre(section)
        along = (zone - 3.5) / 3.0            # -0.83 .. +0.83 of half-length
        if section in ("south", "north"):
            half_len = 0.5 * (self.x1 - self.x0)
            half_w = 0.5 * ((self.y0 + o) if section == "south"
                            else (o - self.y1))
            x = cx + along * half_len * 0.8
            y = cy + lateral * half_w * 0.5
            head = 0.0 if (section == "south") == ccw else math.pi
        else:
            half_len = 0.5 * (self.y1 - self.y0)
            half_w = 0.5 * ((o - self.x1) if section == "east"
                            else (self.x0 + o))
            y = cy + along * half_len * 0.8
            x = cx + lateral * half_w * 0.5
            up = (section == "east") == ccw
            head = math.pi / 2 if up else -math.pi / 2
        return (x, y, head)


# --------------------------------------------------------------------- render

# Measured from the official 2026 playfield artwork (see the research notes in
# README, "Where the numbers come from"): the mat is WHITE, the corner lines are
# PANTONE 151 C / 2728 C, and the printed grey grid and yellow-green rings are
# the two markings most likely to be mistaken for something they are not.
MAT_BGR = (255, 255, 255)
WALL_BGR = (28, 28, 30)
WALL_TOP_BGR = (48, 48, 50)
LINE_BGR = {"orange": (29, 128, 240), "blue": (157, 77, 15)}   # BGR
GRID_BGR = (192, 190, 188)          # PANTONE Cool Gray 5 C
RING_BGR = (0, 220, 222)            # CMYK 20/0/100/0, yellow-green
BORDER_BGR = (71, 17, 27)           # the dark navy surround, seen over the wall
PANEL_BGR = (74, 40, 26)            # centre logo panel, seen over a low block
ROOM_BGR = (196, 200, 205)

LINE_W = 0.020                      # rules 13.9
LINE_WALL_OFFSET = 0.420            # centreline hits the wall this far from the
                                    # field corner (Fig. 11: 420 / 580 mm)
LINE_APEX = 0.500                   # ... running from the NOMINAL inner corner
GRID_W = 0.004
RING_W = 0.003
RING_INSET = 0.050
BORDER_W = 0.100


class FieldRenderer(object):
    """Projects the field into a camera image with a painter's algorithm.

    Polygons are drawn far to near, which resolves occlusion correctly for a
    field made only of the floor and vertical walls.
    """

    def __init__(self, field, geometry, noise=3.0, vignette=0.22,
                 line_width_m=0.02, seed=7):
        self.f = field
        self.geom = geometry
        self.noise = float(noise)
        self.vignette = float(vignette)
        self.line_w = float(line_width_m)
        self.rng = np.random.RandomState(seed)
        self._gain_key = None
        self._gain_cache = None
        self._noise_pool = None

    # camera-space transform -------------------------------------------------

    def _to_cam(self, pts, pose):
        """World (x, y, z) -> camera space (x_right, y_down, z_forward)."""
        px, py, th = pose
        t = math.radians(self.geom.pitch_deg)
        st, ct = math.sin(t), math.cos(t)
        cth, sth = math.cos(th), math.sin(th)
        # The camera sits `cam_x` ahead of the car origin, at height h.
        ox = px + self.geom.cam_x * cth
        oy = py + self.geom.cam_x * sth
        out = []
        for (X, Y, Z) in pts:
            dx, dy, dz = X - ox, Y - oy, Z - self.geom.h
            xr = dx * cth + dy * sth
            yr = -dx * sth + dy * cth
            out.append((-yr, -xr * st - dz * ct, xr * ct - dz * st))
        return out

    @staticmethod
    def _clip_near(poly, znear=0.03):
        """Sutherland-Hodgman against the near plane, in camera space."""
        if not poly:
            return []
        out = []
        n = len(poly)
        for i in range(n):
            cur = poly[i]
            nxt = poly[(i + 1) % n]
            cin, nin = cur[2] > znear, nxt[2] > znear
            if cin:
                out.append(cur)
            if cin != nin:
                t = (znear - cur[2]) / (nxt[2] - cur[2])
                out.append((cur[0] + t * (nxt[0] - cur[0]),
                            cur[1] + t * (nxt[1] - cur[1]), znear))
        return out

    def _project(self, poly_cam):
        g = self.geom
        return [(g.cx + g.fx * x / z, g.cy + g.fy * y / z)
                for (x, y, z) in poly_cam]

    def _visible(self, poly, pose, margin=0.45):
        """Cheap reject for a quad that cannot land on screen.

        Half the field is behind the car and most of the rest is outside a 60
        degree lens, so testing before projecting removes the great majority of
        the work.
        """
        px, py, th = pose
        cth, sth = math.cos(th), math.sin(th)
        half = math.radians(self.geom.hfov_deg) * 0.5 + margin
        ahead = False
        for (X, Y, _Z) in poly:
            dx, dy = X - px, Y - py
            xr = dx * cth + dy * sth
            if xr <= 0.02:
                continue
            yr = -dx * sth + dy * cth
            if abs(math.atan2(yr, xr)) <= half:
                return True
            ahead = True
        # Every corner is forward but off to one side: it can still cross the
        # frame if the quad straddles it, so keep anything that spans the axis.
        if ahead:
            sides = set()
            for (X, Y, _Z) in poly:
                dx, dy = X - px, Y - py
                sides.add(math.copysign(1.0, -dx * sth + dy * cth))
            return len(sides) > 1
        return False

    def _fill(self, img, world_poly, pose, colour):
        cam = self._to_cam(world_poly, pose)
        cam = self._clip_near(cam)
        if len(cam) < 3:
            return
        pix = self._project(cam)
        pts = np.array([[int(round(u)), int(round(v))] for u, v in pix],
                       dtype=np.int32)
        # Guard against the enormous coordinates a near-horizon vertex produces.
        np.clip(pts, -20000, 20000, out=pts)
        cv2.fillConvexPoly(img, pts, colour, lineType=cv2.LINE_AA)

    # ------------------------------------------------------------------ draw

    @staticmethod
    def _split(x1, y1, x2, y2, step=0.25):
        """Chop a wall into short pieces.

        A painter's algorithm sorts whole polygons, so one long wall running
        from very near to very far cannot be ordered correctly against another.
        Splitting removes the artefact.
        """
        n = max(1, int(math.ceil(math.hypot(x2 - x1, y2 - y1) / step)))
        out = []
        for i in range(n):
            a, b = i / float(n), (i + 1) / float(n)
            out.append((x1 + (x2 - x1) * a, y1 + (y2 - y1) * a,
                        x1 + (x2 - x1) * b, y1 + (y2 - y1) * b))
        return out

    def _quads(self, pose):
        """Every drawable quad, far to near."""
        px, py, _th = pose
        quads = []

        def add(poly, colour):
            cx = sum(q[0] for q in poly) / len(poly)
            cy = sum(q[1] for q in poly) / len(poly)
            quads.append((math.hypot(cx - px, cy - py), poly, colour))

        def step_for(x1, y1, x2, y2):
            """Fine subdivision near the car, coarse far away.

            Splitting exists to keep the painter's algorithm ordering correct,
            and ordering errors only show where things are close.
            """
            d = min(math.hypot(x1 - px, y1 - py), math.hypot(x2 - px, y2 - py))
            return 0.25 if d < 1.0 else (0.6 if d < 2.0 else 1.5)

        painted = [(LINE_BGR[c], self.line_w, a, b) for c, a, b in self.f.lines]
        painted += self.f.markings()
        for colour, width, (x1, y1), (x2, y2) in painted:
            dx, dy = x2 - x1, y2 - y1
            n = math.hypot(dx, dy)
            if n < 1e-9:
                continue
            nx, ny = -dy / n * width * 0.5, dx / n * width * 0.5
            for (ax, ay, bx, by) in self._split(
                    x1, y1, x2, y2, step=step_for(x1, y1, x2, y2)):
                add([(ax + nx, ay + ny, 0.0), (bx + nx, by + ny, 0.0),
                     (bx - nx, by - ny, 0.0), (ax - nx, ay - ny, 0.0)],
                    colour)

        for (x1, y1), (x2, y2), _kind in self.f.walls:
            for (ax, ay, bx, by) in self._split(
                    x1, y1, x2, y2, step=step_for(x1, y1, x2, y2)):
                add([(ax, ay, 0.0), (bx, by, 0.0),
                     (bx, by, WALL_H), (ax, ay, WALL_H)], WALL_BGR)

        x0, x1b, y0, y1b = self.f.x0, self.f.x1, self.f.y0, self.f.y1
        steps = 8
        for i in range(steps):
            ya = y0 + (y1b - y0) * i / float(steps)
            yb = y0 + (y1b - y0) * (i + 1) / float(steps)
            add([(x0, ya, WALL_H), (x1b, ya, WALL_H),
                 (x1b, yb, WALL_H), (x0, yb, WALL_H)], WALL_TOP_BGR)

        quads.sort(key=lambda q: q[0], reverse=True)
        return quads

    def render(self, pose, brightness=1.0, tint=(1.0, 1.0, 1.0)):
        g = self.geom
        img = np.empty((g.height, g.width, 3), np.uint8)
        img[:] = ROOM_BGR

        o = self.f.o
        # The dark surround first: a camera above the 100 mm wall sees it.
        b = o + BORDER_W
        self._fill(img, [(-b, -b, 0.0), (b, -b, 0.0), (b, b, 0.0), (-b, b, 0.0)],
                   pose, BORDER_BGR)
        # Then the mat: it is flat, so nothing on the field can hide behind it.
        self._fill(img, [(-o, -o, 0.0), (o, -o, 0.0), (o, o, 0.0), (-o, o, 0.0)],
                   pose, MAT_BGR)
        # The centre logo panel, which shows through a low inner block.
        self._fill(img, [(-0.4, -0.4, 0.0), (0.4, -0.4, 0.0),
                         (0.4, 0.4, 0.0), (-0.4, 0.4, 0.0)], pose, PANEL_BGR)
        for _d, poly, colour in self._quads(pose):
            if self._visible(poly, pose):
                self._fill(img, poly, pose, colour)
        return self._post(img, brightness, tint)

    def _gain(self, shape, brightness, tint):
        """Combined vignette x brightness x tint, computed once per setting.

        This used to be three full-frame float multiplies per frame; the field
        does not change, so neither does the answer.
        """
        key = (shape, round(brightness, 4), tuple(round(c, 4) for c in tint))
        if self._gain_key == key:
            return self._gain_cache
        h, w = shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.sqrt(((xx - w / 2.0) / (w / 2.0)) ** 2 +
                    ((yy - h / 2.0) / (h / 2.0)) ** 2)
        vig = 1.0 - self.vignette * np.clip(r, 0, 1.6) ** 2
        gain = (vig * float(brightness))[:, :, None] * \
            np.array(tint, np.float32)[None, None, :]
        self._gain_key, self._gain_cache = key, gain.astype(np.float32)
        return self._gain_cache

    def _noise(self, shape):
        """A frame of sensor noise, drawn from a pre-rolled pool.

        Generating a quarter of a million gaussians per frame cost more than
        everything the simulator was built to test; a pool of a few frames'
        worth, offset differently each time, is indistinguishable downstream.
        """
        if self.noise <= 0.0:
            return None
        n = int(np.prod(shape))
        if self._noise_pool is None or self._noise_pool.size < n * 2:
            self._noise_pool = self.rng.normal(
                0.0, 1.0, n * 4).astype(np.float32)
        off = int(self.rng.randint(0, self._noise_pool.size - n))
        return self._noise_pool[off:off + n].reshape(shape) * self.noise

    def _post(self, img, brightness, tint):
        out = img.astype(np.float32)
        out *= self._gain(img.shape, brightness, tint)
        noise = self._noise(img.shape)
        if noise is not None:
            out += noise
        return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------- range models

def _ray_segment(px, py, dx, dy, seg):
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
    nx, ny = -sy / n, sx / n
    return t, abs(dx * nx + dy * ny)


class LidarModel(object):
    """YDLIDAR X2 with the specular dropout this track actually shows."""

    def __init__(self, field, n_beams=270, specular_deg=35.0, dropout=0.10,
                 range_max=8.0, range_min=0.10, noise=0.01, seed=11):
        self.f = field
        self.n = int(n_beams)
        self.cos_min = math.cos(math.radians(specular_deg))
        self.dropout = float(dropout)
        self.range_max = float(range_max)
        self.range_min = float(range_min)
        self.noise = float(noise)
        self.rng = np.random.RandomState(seed)
        self.segs = [(a, b) for a, b, _k in field.walls]

    def scan(self, pose):
        """Returns a list of `n` ranges on bearings -180..180 (index 0 = -180),
        in the CAR frame: bearing = -pi + i*2pi/n, + = left."""
        px, py, th = pose
        out = []
        for i in range(self.n):
            phi = -math.pi + i * (2 * math.pi / self.n)
            world = th + phi
            dx, dy = math.cos(world), math.sin(world)
            best, best_cos = None, 0.0
            for seg in self.segs:
                hit = _ray_segment(px, py, dx, dy, seg)
                if hit is None:
                    continue
                t, c = hit
                if best is None or t < best:
                    best, best_cos = t, c
            if (best is None or best > self.range_max
                    or best_cos < self.cos_min
                    or self.rng.rand() < self.dropout):
                out.append(float("inf"))
            else:
                out.append(max(self.range_min,
                               best + self.rng.normal(0.0, self.noise)))
        return out


SONAR_MOUNTS = {           # (forward offset m, lateral offset m, bearing deg)
    "front": (0.10, 0.0, 0.0),
    "right": (0.02, -0.07, -90.0),
    "rear": (-0.08, 0.0, 180.0),
    "left": (0.02, 0.07, 90.0),
}


class SonarModel(object):
    """HC-SR04 cone, with the failure modes seen on this car.

    `faults` maps a channel to 'frozen', 'dead', or ('offset', metres).
    """

    def __init__(self, field, cone_deg=15.0, rays=5, max_m=4.0, min_m=0.02,
                 noise=0.004, faults=None, seed=13):
        self.f = field
        self.cone = math.radians(cone_deg)
        self.rays = int(rays)
        self.max_m = float(max_m)
        self.min_m = float(min_m)
        self.noise = float(noise)
        self.faults = dict(faults or {})
        self.rng = np.random.RandomState(seed)
        self.segs = [(a, b) for a, b, _k in field.walls]
        self._frozen = {}

    def read(self, pose):
        px, py, th = pose
        out = {}
        for name, (fx, fy, bearing) in SONAR_MOUNTS.items():
            fault = self.faults.get(name)
            if fault == "dead":
                out[name] = None
                continue
            ox = px + fx * math.cos(th) - fy * math.sin(th)
            oy = py + fx * math.sin(th) + fy * math.cos(th)
            base = th + math.radians(bearing)
            best = None
            for k in range(self.rays):
                frac = (k / float(max(1, self.rays - 1))) - 0.5
                ang = base + frac * self.cone
                dx, dy = math.cos(ang), math.sin(ang)
                for seg in self.segs:
                    hit = _ray_segment(ox, oy, dx, dy, seg)
                    if hit is None:
                        continue
                    t, _c = hit
                    if best is None or t < best:
                        best = t
            if best is None or best > self.max_m:
                out[name] = None
                continue
            val = max(self.min_m, best + self.rng.normal(0.0, self.noise))
            if isinstance(fault, tuple) and fault[0] == "offset":
                val = max(self.min_m, val + float(fault[1]))
            if fault == "frozen":
                val = self._frozen.setdefault(name, round(val, 2))
            out[name] = round(val, 2)      # the firmware reports whole cm
        return out


class CarModel(object):
    """Kinematic bicycle with the measured stiction of this drivetrain."""

    def __init__(self, wheelbase=0.15, body_radius=0.13, stiction_pct=30.0,
                 v_at_45=0.50, tau=0.15, max_steer=25.0):
        self.L = float(wheelbase)
        self.radius = float(body_radius)
        self.stiction = float(stiction_pct)
        self.kv = float(v_at_45) / 45.0
        self.tau = float(tau)
        self.max_steer = float(max_steer)
        self.v = 0.0

    def step(self, pose, steer_deg, duty_pct, dt):
        x, y, th = pose
        steer = _clamp(float(steer_deg), -self.max_steer, self.max_steer)
        target = 0.0 if abs(duty_pct) < self.stiction else self.kv * duty_pct
        self.v += (target - self.v) * (dt / self.tau)
        if abs(self.v) < 1e-4:
            self.v = 0.0
        th -= (self.v / self.L) * math.tan(math.radians(steer)) * dt
        x += self.v * math.cos(th) * dt
        y += self.v * math.sin(th) * dt
        return (x, y, th)


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", default="field_preview.png")
    ap.add_argument("--widths", default="1.0,1.0,1.0,1.0")
    ap.add_argument("--section", default="south")
    ap.add_argument("--zone", type=int, default=3)
    ap.add_argument("--cw", action="store_true")
    ap.add_argument("--height", type=float, default=0.12)
    ap.add_argument("--pitch", type=float, default=15.0)
    args = ap.parse_args()

    widths = tuple(float(v) for v in args.widths.split(","))
    field = Field(widths)
    geom = CameraGeometry(640, 480, 60.0, args.height, args.pitch, cam_x_m=0.05)
    rend = FieldRenderer(field, geom)
    pose = field.start_pose(args.section, args.zone, ccw=not args.cw)
    img = rend.render(pose)
    cv2.imwrite(args.preview, img)
    print("pose %.2f %.2f %.1fdeg -> %s"
          % (pose[0], pose[1], math.degrees(pose[2]), args.preview))
    lid = LidarModel(field)
    son = SonarModel(field)
    good = [r for r in lid.scan(pose) if math.isfinite(r)]
    print("lidar returns %d/%d   sonar %s"
          % (len(good), lid.n, son.read(pose)))


if __name__ == "__main__":
    _main()
