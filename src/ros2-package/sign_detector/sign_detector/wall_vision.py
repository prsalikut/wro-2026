"""Camera-only lane perception for the open round (no ROS; OpenCV + numpy).

The lidar cannot see the side walls -- they are glossy black and at grazing
incidence they return nothing or nonsense -- and the ultrasonics have already
been caught freezing on a constant value.  A camera does not share either
failure: the wall is a large black region sitting on a light mat, which is about
the easiest thing there is to segment, and its *base line* is exactly where the
wall meets the floor, so it converts straight into a distance.

The pipeline is:

  1. learn what "floor" looks like from a patch immediately in front of the car,
     which is floor whenever the car is not already crashed;
  2. mark every pixel that is not floor;
  3. per image column, walk up from the bottom to the first sustained run of
     not-floor -- that pixel is where the floor ends;
  4. project those pixels onto the ground plane (ground_geometry) to get a
     metric free-space profile;
  5. fit straight lines to the left, right and front groups of that profile,
     which yields perpendicular wall distances AND the car's yaw relative to the
     corridor -- a heading reference that needs no IMU.

Nothing is hard-coded to the official mat's shade of grey.  The floor model is
learned every frame and low-pass filtered, so the same code works on the WRO mat,
on a wooden practice floor, and under whatever the venue lighting turns out to
be.  The coloured corner lines are found separately by hue, and are reported with
a distance rather than as a bare "seen" flag, so the driver can time a corner
instead of reacting to one.
"""

import math

import cv2
import numpy as np

from .ground_geometry import CameraGeometry

__all__ = ["WallVisionParams", "WallVisionResult", "WallVision",
           "fit_line_trimmed"]

# OpenCV hue is 0..179.  Bands are deliberately generous; saturation and the
# "differs from the floor" test do the real work of rejecting the mat.
HUE_BANDS = {
    "orange": ((0, 22), (168, 179)),     # wraps through red
    "blue": ((90, 132), None),
    "magenta": ((140, 168), None),       # parking-lot markers: never a corner
}


def fit_line_trimmed(u, w, trims=2, keep=2.5, min_pts=6):
    """Least squares w = a*u + b, re-fitted with gross outliers dropped.

    Returns (a, b, rms, n_used) or None.  A handful of stray boundary pixels --
    a cable on the mat, a glare edge -- otherwise drag the fit badly, and the
    walls are genuinely straight, so trimming is safe here.
    """
    u = np.asarray(u, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if u.size < min_pts:
        return None
    keep_mask = np.ones(u.size, dtype=bool)
    a = b = 0.0
    for _ in range(trims + 1):
        uu, ww = u[keep_mask], w[keep_mask]
        if uu.size < min_pts:
            return None
        n = uu.size
        su, sw = uu.sum(), ww.sum()
        suu = float(np.dot(uu, uu))
        suw = float(np.dot(uu, ww))
        den = n * suu - su * su
        if abs(den) < 1e-12:
            return None
        a = (n * suw - su * sw) / den
        b = (sw - a * su) / n
        resid = np.abs(w - (a * u + b))
        med = float(np.median(resid[keep_mask]))
        mad = float(np.median(np.abs(resid[keep_mask] - med))) + 1e-6
        new_mask = resid <= (med + keep * 1.4826 * mad + 0.01)
        if new_mask.sum() < min_pts or np.array_equal(new_mask, keep_mask):
            keep_mask = new_mask if new_mask.sum() >= min_pts else keep_mask
            break
        keep_mask = new_mask
    uu, ww = u[keep_mask], w[keep_mask]
    rms = float(np.sqrt(np.mean((ww - (a * uu + b)) ** 2))) if uu.size else 9.9
    return a, b, rms, int(uu.size)


class WallVisionParams(object):
    """Tunables.  Every one is exposed as a ROS parameter by the node."""

    def __init__(self, **kw):
        # --- work buffer -------------------------------------------------
        self.work_width = 320
        self.work_height = 240

        # --- floor model -------------------------------------------------
        self.sample_top = 0.86        # patch used to learn the floor, as a
        self.sample_bottom = 0.99     # fraction of image height ...
        self.sample_left = 0.30       # ... and width
        self.sample_right = 0.70
        self.model_alpha = 0.15       # EMA rate for the learned floor colour
        self.model_max_mad = 26.0     # patch this noisy is not plain floor
        self.min_floor_l = 60.0       # darker than this is not floor at all
        self.illum_fit = True         # fit a smooth brightness surface first
        self.illum_min_frac = 0.10    # too little floor to fit: use a flat model
        self.illum_sample = 6         # pixel stride for the fit
        self.dark_ratio = 0.62        # L below this fraction of floor L = wall
        self.dark_abs = 34.0          # ... but never a smaller step than this
        self.bright_ratio = 1.80      # blown-out highlight is still floor
        self.chroma_min = 20.0        # LAB (a,b) distance that counts as colour

        # --- free-space scan --------------------------------------------
        self.scan_top = 0.05          # hard cap; the horizon normally wins
        self.scan_bottom = 0.995
        self.run_px = 4               # sustained not-floor rows = a real edge
        self.col_step = 4             # profile column spacing, work pixels
        self.horizon_margin_px = 5.0

        # --- geometry / fitting ------------------------------------------
        self.min_range_m = 0.16
        self.max_range_m = 2.2        # beyond this one pixel row is > 5 cm
        self.side_min_y_m = 0.045     # |y| below this is "ahead", not a side
        self.side_fit_min_x = 0.20
        self.side_fit_max_x = 1.90
        self.front_corridor_m = 0.16  # half-width of the "straight ahead" strip
        self.open_bearing_lo = 12.0   # wedge used to measure open space aside
        self.open_bearing_hi = 30.0
        self.front_fit_max_x = 1.90
        self.front_fit_half_y = 0.50
        self.front_max_slope = 0.85   # dx/dy; steeper is a side wall, not front
        self.eval_x_m = 0.15          # where side distances are reported
        self.max_fit_rms_m = 0.055
        # A wall nearer than the car's own half-width is inside the car:
        # it is a bad fit, not a close wall, and reporting it once in a
        # while is enough to make the driver swerve at nothing.
        self.side_min_m = 0.12        # a fit outside these is nonsense
        self.side_max_m = 1.60
        self.min_side_pts = 5
        self.min_side_span_m = 0.18   # x-extent a side fit must cover
        self.side_beyond_tol_m = 0.10  # a ray this far past a "wall" disproves it
        self.side_beyond_frac = 0.22
        self.side_support_margin_m = 0.18  # a wall must be seen where it would
                                           # have to be visible from
        self.front_clear_margin_m = 0.14  # keep front-wall pixels out
                                          # of the side-wall fits
        self.min_front_pts = 7

        # --- coloured corner lines ---------------------------------------
        self.line_sat_min = 70
        self.line_val_min = 45
        self.line_min_area_px = 90
        self.line_max_range_m = 1.60

        # --- health -------------------------------------------------------
        self.min_floor_frac = 0.06    # less floor than this: do not trust a frame
        self.block_magenta = True     # parking-lot markers are objects, not paint
        self.paint_is_floor = True    # a dark but saturated pixel is a line

        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError("unknown WallVisionParams field %r" % (k,))
            setattr(self, k, type(getattr(self, k))(v))


class WallVisionResult(object):
    __slots__ = ("ok", "reason", "left_m", "right_m", "front_m", "front_wall_m",
                 "heading_deg", "lane_width_m", "profile", "lines",
                 "floor_frac", "floor_lab", "left_fit", "right_fit",
                 "front_fit", "debug", "n_points", "front_free_m",
                 "open_left_m", "open_right_m", "open_bearing_deg")

    def __init__(self):
        self.ok = False
        self.reason = "init"
        self.left_m = None
        self.right_m = None
        self.front_m = None          # nearest obstacle straight ahead
        self.front_free_m = None     # how far the corridor is clear ahead
        self.front_wall_m = None     # perpendicular distance to a fitted wall
        self.heading_deg = None      # + = car nose left of the corridor
        self.lane_width_m = None
        self.profile = []            # [(bearing_deg, x_m, y_m), ...]
        self.lines = {}              # colour -> dict(distance_m, ...)
        self.floor_frac = 0.0
        self.floor_lab = None
        self.left_fit = None
        self.right_fit = None
        self.front_fit = None
        self.n_points = 0
        self.open_left_m = None      # how far the view runs open to each side:
        self.open_right_m = None     # at a corner one of them is the way out
        self.open_bearing_deg = None  # and which bearing runs furthest
        self.debug = None

    def as_dict(self):
        def r(v, n=3):
            return None if v is None else round(float(v), n)
        return {
            "ok": self.ok, "reason": self.reason,
            "left": r(self.left_m), "right": r(self.right_m),
            "front": r(self.front_m), "front_free": r(self.front_free_m),
            "front_wall": r(self.front_wall_m),
            "open_left": r(self.open_left_m),
            "open_right": r(self.open_right_m),
            "open_bearing": r(self.open_bearing_deg, 1),
            "heading": r(self.heading_deg, 1),
            "lane_width": r(self.lane_width_m),
            "floor_frac": r(self.floor_frac, 3),
            "points": self.n_points,
            "lines": {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                          for kk, vv in v.items()}
                      for k, v in self.lines.items()},
            "line_seen": sorted(self.lines),
        }


class WallVision(object):

    def __init__(self, geometry, params=None):
        """`geometry` is a CameraGeometry for the FULL-resolution camera."""
        self.p = params or WallVisionParams()
        self.full_geom = geometry
        self.geom = geometry.scaled(self.p.work_width, self.p.work_height)
        self._floor = None           # np.array([L, a, b]) float
        self._floor_mad = None
        self._frames = 0
        self._basis = None
        self._illum_surf = None

    # ---------------------------------------------------------------- floor

    def _update_floor(self, lab):
        h, w = lab.shape[:2]
        r0, r1 = int(h * self.p.sample_top), int(h * self.p.sample_bottom)
        c0, c1 = int(w * self.p.sample_left), int(w * self.p.sample_right)
        patch = lab[max(0, r0):max(1, r1), max(0, c0):max(1, c1)]
        if patch.size == 0:
            return False
        flat = patch.reshape(-1, 3).astype(np.float32)
        med = np.median(flat, axis=0)
        mad = float(np.median(np.abs(flat[:, 0] - med[0])))
        # A patch this mottled is not plain floor -- the car is nose-on to a
        # wall, or something is lying in front of it.  Keep the old model.
        # A car parked nose-first against a wall sees black in the sample
        # patch too.  Learning THAT as "floor" would make the wall the
        # reference and the whole frame look clear, so a patch this dark is
        # refused outright and the node reports that it cannot see the mat.
        fresh = (mad <= self.p.model_max_mad
                 and float(med[0]) >= self.p.min_floor_l)
        if self._floor is None:
            if not fresh and self._frames > 0:
                return False
            self._floor = med.astype(np.float64)
            self._floor_mad = mad
            return True
        if fresh:
            a = self.p.model_alpha
            self._floor = (1.0 - a) * self._floor + a * med.astype(np.float64)
            self._floor_mad = (1.0 - a) * (self._floor_mad or mad) + a * mad
        return fresh

    def _illumination(self, L, seed_floor):
        """Smooth brightness surface over the pixels that look like floor.

        Every lens vignettes.  On this camera the image corners come out about
        40% darker than the centre, which is enough for plain mat to fall under
        any fixed "this is a black wall" threshold -- and the corners project to
        ground points about 20 cm in front of the car, so the car brakes for its
        own lens.  Venue lighting does the same thing more slowly across the
        mat.  Fitting a quadratic to the floor's own brightness and comparing
        each pixel against the local value removes both.
        """
        h, w = L.shape[:2]
        if self._basis is None or self._basis[0].shape != L.shape:
            yy, xx = np.mgrid[0:h, 0:w]
            u = (xx / float(w - 1) * 2.0 - 1.0).astype(np.float32)
            v = (yy / float(h - 1) * 2.0 - 1.0).astype(np.float32)
            self._basis = (np.ones_like(u), u, v, u * u, v * v, u * v)
        basis = self._basis
        k = max(1, int(self.p.illum_sample))
        m = seed_floor[::k, ::k]
        if m.mean() < self.p.illum_min_frac:
            return None
        cols = [b[::k, ::k][m].astype(np.float64) for b in basis]
        A = np.stack(cols, axis=1)
        y = L[::k, ::k][m].astype(np.float64)
        if A.shape[0] < 60:
            return None
        try:
            coef, _res, _rank, _sv = np.linalg.lstsq(A, y, rcond=None)
        except np.linalg.LinAlgError:
            return None
        surf = np.zeros(L.shape, np.float32)
        for c, b in zip(coef, basis):
            surf += float(c) * b
        return np.clip(surf, 8.0, 255.0)

    def _masks(self, lab, hsv):
        """(floor, blocked): what looks like mat, and what stops the car.

        These are deliberately NOT complements.  The orange and blue lines are
        painted ON the mat -- a car drives straight over them -- so a "not the
        mat colour" test would read every corner line as a wall and stop the
        free-space scan a metre early.  What actually blocks this car is a
        matt-black wall, so the blocking test is darkness relative to the local
        floor brightness, plus the magenta parking-lot elements, which are real
        objects rather than paint.
        """
        L = lab[:, :, 0].astype(np.int16)
        A = lab[:, :, 1].astype(np.int16)
        B = lab[:, :, 2].astype(np.int16)
        L0, A0, B0 = self._floor
        chroma = np.abs(A - int(round(A0))) + np.abs(B - int(round(B0)))

        def cuts(ref):
            dark = ref * self.p.dark_ratio
            step = np.where(ref > self.p.dark_abs * 1.5, ref - self.p.dark_abs,
                            dark)
            return np.maximum(np.minimum(dark, step), 3.0), \
                ref * self.p.bright_ratio + 25.0

        flat = np.full(L.shape, float(L0), np.float32)
        dark_cut, bright_cut = cuts(flat)
        seed = (L > dark_cut) & (L < bright_cut) & (chroma < self.p.chroma_min * 1.5)

        ref = self._illumination(L, seed) if self.p.illum_fit else None
        self._illum_surf = ref
        if ref is not None:
            dark_cut, bright_cut = cuts(ref)

        dark = L <= dark_cut
        coloured = chroma >= self.p.chroma_min
        floor = (~dark) & (L < bright_cut) & (chroma < self.p.chroma_min * 1.5)

        # The blue corner line is dark enough to pass a brightness test for a
        # wall -- it sits around L 107 against a mat near 184 -- so darkness
        # alone would have the car brake for a stripe of paint half a metre
        # ahead of it.  A matt black wall has almost no chroma; paint has
        # plenty, and that is the difference that separates them.
        blocked = dark & (~coloured if self.p.paint_is_floor else
                          np.ones_like(dark))
        if self.p.block_magenta:
            H = hsv[:, :, 0]
            S = hsv[:, :, 1]
            lo, hi = HUE_BANDS["magenta"][0]
            blocked = blocked | ((H >= lo) & (H <= hi)
                                 & (S >= self.p.line_sat_min)
                                 & (chroma >= self.p.chroma_min))
        return floor, blocked

    # -------------------------------------------------------------- profile

    def _boundary_rows(self, blocked, v_top, v_bottom):
        """Lowest row per column at which `run_px` blocked rows start.

        Scanning up from the bottom finds the *near* edge of the first
        obstruction, which is what a range sensor would report; scanning down
        from the top would find the far side of the wall instead.
        """
        p = self.p
        run = max(1, int(p.run_px))
        band = blocked[v_top:v_bottom, :].astype(np.uint8)
        if band.shape[0] < run:
            return None
        # A column is "solid at row r" when rows r..r+run-1 are all blocked.
        # Anchor at the BOTTOM of the kernel, so a column reports the last row
        # of the dark run rather than its first: that pixel is the actual
        # floor/wall join, and taking the first instead reads the wall a whole
        # run short -- 10 cm of range error at 1.5 m.
        acc = cv2.boxFilter(band, -1, (1, run), anchor=(0, run - 1),
                            normalize=False, borderType=cv2.BORDER_REPLICATE)
        solid = acc >= run
        flipped = solid[::-1, :]
        any_solid = flipped.any(axis=0)
        idx = flipped.argmax(axis=0)
        rows = (solid.shape[0] - 1 - idx) + v_top
        return rows, any_solid

    def _profile(self, blocked):
        p, geom = self.p, self.geom
        h, w = blocked.shape[:2]
        v_top = max(int(h * p.scan_top),
                    int(math.ceil(geom.horizon_row() + p.horizon_margin_px)))
        v_top = max(0, min(v_top, h - 2))
        v_bottom = min(h, int(h * p.scan_bottom))
        got = self._boundary_rows(blocked, v_top, v_bottom)
        if got is None:
            return [], v_top
        rows, any_solid = got

        pts = []
        for c in range(0, w, max(1, int(p.col_step))):
            u = c + 0.5
            if any_solid[c]:
                # +0.5: the join sits between the last dark row and the first
                # floor row, not on either of them.
                v = float(rows[c]) + 0.5
                g = geom.to_ground(u, v, max_range=1e9)
                if g is None:
                    continue
                x, y = g
                if x < p.min_range_m:
                    continue
                if x > p.max_range_m:
                    # Seen, but too far to measure honestly: report it as
                    # free space out to the range we are willing to claim.
                    s_ = p.max_range_m / x
                    pts.append((u, v, p.max_range_m, y * s_, False))
                    continue
                pts.append((u, v, x, y, True))
            else:
                # Nothing blocks this column inside the scan band: free to the
                # furthest range the camera is allowed to claim.
                g = geom.to_ground(u, v_top + 1.0, max_range=1e9)
                if g is None:
                    continue
                x = min(p.max_range_m, g[0])
                y = g[1] * (x / g[0]) if g[0] > 1e-6 else 0.0
                pts.append((u, v_top + 1.0, x, y, False))
        return pts, v_top

    # ----------------------------------------------------------- colour lines

    def _colour_lines(self, hsv, lab, v_top):
        p = self.p
        H = hsv[:, :, 0]
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]
        A = lab[:, :, 1].astype(np.int16)
        B = lab[:, :, 2].astype(np.int16)
        L0, A0, B0 = self._floor
        chroma = np.abs(A - int(round(A0))) + np.abs(B - int(round(B0)))
        base = ((S >= p.line_sat_min) & (V >= p.line_val_min)
                & (chroma >= p.chroma_min))
        base[:v_top, :] = False

        out = {}
        for colour, (band, wrap) in HUE_BANDS.items():
            m = (H >= band[0]) & (H <= band[1])
            if wrap is not None:
                m |= (H >= wrap[0]) & (H <= wrap[1])
            m &= base
            mask = m.astype(np.uint8)
            if int(mask.sum()) < p.line_min_area_px:
                continue
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
            best, best_area = None, 0
            for i in range(1, n):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area > best_area:
                    best, best_area = i, area
            if best is None or best_area < p.line_min_area_px:
                continue
            ys, xs = np.where(labels == best)
            v_near = float(ys.max())
            u_at = float(np.median(xs[ys >= v_near - 2]))
            g = self.geom.to_ground(u_at, v_near, max_range=p.line_max_range_m)
            if g is None:
                continue
            x, y = g
            out[colour] = {
                # Fraction of the way down the frame that the nearest pixel of
                # the line sits. Unlike distance_m this needs no camera height
                # or pitch, so a corner can be timed off it even when the
                # mounting is not calibrated -- "the line has reached the
                # bottom of the view" means "I am on it", whatever the optics.
                "v_frac": float(v_near) / float(self.p.work_height),
                "distance_m": float(x),
                "lateral_m": float(y),
                "bearing_deg": math.degrees(math.atan2(y, x)),
                "area_px": int(best_area),
                "area_frac": float(best_area) / float(mask.size),
                "u": u_at, "v": v_near,
            }
        return out

    # -------------------------------------------------------------- pipeline

    def process(self, bgr, want_debug=False):
        p = self.p
        res = WallVisionResult()
        if bgr is None or bgr.size == 0:
            res.reason = "empty frame"
            return res
        self._frames += 1

        small = cv2.resize(bgr, (p.work_width, p.work_height),
                           interpolation=cv2.INTER_AREA)
        blur = cv2.GaussianBlur(small, (3, 3), 0)
        lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        self._update_floor(lab)
        if self._floor is None:
            res.reason = ("nothing that looks like a floor in view"
                          if self._frames > 3 else "no floor model yet")
            res.floor_frac = 0.0
            return res
        res.floor_lab = [float(v) for v in self._floor]

        floor, blocked = self._masks(lab, hsv)
        kernel = np.ones((3, 3), np.uint8)
        blocked = cv2.morphologyEx(blocked.astype(np.uint8), cv2.MORPH_OPEN,
                                   kernel).astype(bool)
        res.floor_frac = float(floor.mean())

        pts, v_top = self._profile(blocked)
        res.n_points = len(pts)
        res.profile = [(math.degrees(math.atan2(y, x)), x, y)
                       for (_u, _v, x, y, _hit) in pts]

        res.lines = self._colour_lines(hsv, lab, v_top)

        if want_debug:
            res.debug = self._draw_debug(small, blocked, pts, res)

        if res.floor_frac < p.min_floor_frac:
            res.reason = "floor only %.0f%% of frame" % (100.0 * res.floor_frac)
            return res
        if len(pts) < 8:
            res.reason = "profile too sparse"
            return res

        self._measure(pts, res)
        if want_debug and res.debug is not None:
            self._draw_fits(res.debug, res)
        return res

    def _measure(self, pts, res):
        p = self.p
        hits = [(x, y) for (_u, _v, x, y, hit) in pts if hit]
        allx = np.array([x for (_u, _v, x, y, _h) in pts], dtype=np.float64)
        ally = np.array([y for (_u, _v, x, y, _h) in pts], dtype=np.float64)

        # --- straight ahead ------------------------------------------------
        ahead = [x for (x, y) in hits if abs(y) <= p.front_corridor_m]
        if ahead:
            res.front_m = float(np.percentile(np.array(ahead), 20.0))
        corridor = allx[np.abs(ally) <= p.front_corridor_m]
        if corridor.size:
            res.front_free_m = float(np.percentile(corridor, 20.0))

        # --- how far the view runs open to each side -------------------------
        # At a corner the way out is simply the side the camera can see
        # furthest along, and it is the most direct evidence there is of which
        # way this round runs: the outer wall is always the near one.
        lo, hi = p.open_bearing_lo, p.open_bearing_hi
        for name, sign in (("open_left_m", 1.0), ("open_right_m", -1.0)):
            wedge = [x for (_u, _v, x, y, _h) in pts
                     if lo <= sign * math.degrees(math.atan2(y, x)) <= hi]
            if len(wedge) >= 4:
                setattr(res, name,
                        float(np.percentile(np.array(wedge), 80.0)))

        # The bearing the view runs furthest along, weighted so distant rays
        # dominate.  Coming out of a corner this is the new corridor, and
        # driving until it sits at zero is a direct optical measurement of
        # "square to the corridor" -- which is what the corner is trying to
        # achieve, and what dead reckoning can only guess at.
        if len(pts) >= 8:
            bear = np.array([math.degrees(math.atan2(y, x))
                             for (_u, _v, x, y, _h) in pts])
            rng = np.array([x for (_u, _v, x, _y, _h) in pts])
            w = np.clip(rng - p.min_range_m, 0.0, None) ** 3
            tot = float(w.sum())
            if tot > 1e-9:
                res.open_bearing_deg = float((bear * w).sum() / tot)

        # --- front wall as a fitted line ------------------------------------
        fx, fy = [], []
        for (x, y) in hits:
            if x <= p.front_fit_max_x and abs(y) <= p.front_fit_half_y:
                fx.append(y)
                fy.append(x)
        fit = fit_line_trimmed(fx, fy, min_pts=p.min_front_pts)
        if fit is not None:
            a, b, rms, n = fit
            spread = (max(fx) - min(fx)) if fx else 0.0
            if (abs(a) <= p.front_max_slope and rms <= p.max_fit_rms_m
                    and spread >= 0.22):
                res.front_wall_m = float(b * math.cos(math.atan(a)))
                res.front_fit = (a, b, rms, n)

        # --- side walls -----------------------------------------------------
        # A wall straight ahead spans the full width of the frame, so its pixels
        # land in BOTH side groups and drag the fits into nonsense.  Cut the
        # profile off short of whatever is in front before splitting it.
        if res.front_fit is not None:
            fa, fb = res.front_fit[0], res.front_fit[1]

            def on_front(x, y):
                return abs(x - (fa * y + fb)) < p.front_clear_margin_m
            cut = p.side_fit_max_x
        else:
            fa = fb = None

            def on_front(x, y):
                return False
            cut = p.max_range_m
            for v in (res.front_m, res.front_free_m):
                if v is not None:
                    cut = min(cut, v)
            cut -= p.front_clear_margin_m

        lx, ly, rx, ry = [], [], [], []
        for (x, y) in hits:
            if not (p.side_fit_min_x <= x <= min(p.side_fit_max_x, cut)):
                continue
            if on_front(x, y):
                continue
            if y >= p.side_min_y_m:
                lx.append(x)
                ly.append(y)
            elif y <= -p.side_min_y_m:
                rx.append(x)
                ry.append(y)

        lfit = fit_line_trimmed(lx, ly, min_pts=p.min_side_pts)
        rfit = fit_line_trimmed(rx, ry, min_pts=p.min_side_pts)
        # A front wall shows up in both side groups as a steep line; drop fits
        # that are really the front wall seen edge-on.
        def see_past(fit, sign):
            """How many rays on this side reach past the fitted wall.

            A wall alongside the car blocks everything behind it, so a line
            that rays fly straight through is not a wall there.  This is what
            separates a genuine corridor wall from the inner block's FAR face
            seen ahead in a corner: the block's face is real, but extrapolating
            it back alongside the car invents a wall 15 cm away when the true
            clearance is two and a half metres.
            """
            a, b = fit[0], fit[1]
            n_side = beyond = 0
            for (_u, _v, x, y, _hit) in pts:
                if not (p.side_fit_min_x <= x <= p.side_fit_max_x):
                    continue
                if sign * y < p.side_min_y_m:
                    continue
                n_side += 1
                if sign * (y - (a * x + b)) > p.side_beyond_tol_m:
                    beyond += 1
            return n_side, beyond

        def usable(fit, xs, sign):
            if fit is None or not xs:
                return None
            a, b, rms, n = fit
            if rms > p.max_fit_rms_m or abs(a) > 1.0:
                return None
            if (max(xs) - min(xs)) < p.min_side_span_m:
                return None
            n_side, beyond = see_past(fit, sign)
            if beyond >= 3 and beyond > p.side_beyond_frac * max(1, n_side):
                return None
            # A wall `d` metres to the side enters a half-angle `hw` field of
            # view at d/tan(hw) ahead.  If the nearest pixel supporting the fit
            # is much further away than that, the line is being extrapolated
            # into space the camera can see is empty -- which is what happens
            # in a corner, where the inner block's FAR face fits beautifully
            # and then projects back into a wall alongside the car that is not
            # there.
            d = abs(a * p.eval_x_m + b)
            need = d / math.tan(math.radians(self.geom.hfov_deg * 0.5))
            if min(xs) > need + p.side_support_margin_m:
                return None
            return fit

        lfit = usable(lfit, lx, 1.0)
        rfit = usable(rfit, rx, -1.0)
        res.left_fit, res.right_fit = lfit, rfit

        slopes = [f[0] for f in (lfit, rfit) if f is not None]
        if slopes:
            a_avg = float(np.mean(slopes))
            res.heading_deg = -math.degrees(math.atan(a_avg))
        psi = math.radians(res.heading_deg or 0.0)
        cpsi = math.cos(psi)

        ex = p.eval_x_m

        def sane(v):
            return v if (v is not None and p.side_min_m < v < p.side_max_m) \
                else None

        if lfit is not None:
            res.left_m = sane(float((lfit[0] * ex + lfit[1]) * cpsi))
        if rfit is not None:
            res.right_m = sane(float(-(rfit[0] * ex + rfit[1]) * cpsi))
        if res.left_m is not None and res.right_m is not None:
            res.lane_width_m = res.left_m + res.right_m

        res.ok = (res.left_m is not None or res.right_m is not None
                  or res.front_m is not None)
        res.reason = "ok" if res.ok else "no wall fitted"

    # ---------------------------------------------------------------- debug

    def _draw_debug(self, small, blocked, pts, res):
        img = small.copy()
        overlay = img.copy()
        overlay[blocked] = (0, 0, 160)
        img = cv2.addWeighted(overlay, 0.28, img, 0.72, 0)
        for (u, v, x, y, hit) in pts:
            col = (0, 255, 255) if hit else (90, 200, 90)
            cv2.circle(img, (int(u), int(v)), 1, col, -1)
        hz = int(self.geom.horizon_row())
        if 0 <= hz < img.shape[0]:
            cv2.line(img, (0, hz), (img.shape[1], hz), (255, 120, 0), 1)
        for colour, info in res.lines.items():
            c = {"orange": (0, 140, 255), "blue": (255, 130, 0),
                 "magenta": (200, 0, 200)}.get(colour, (255, 255, 255))
            cv2.circle(img, (int(info["u"]), int(info["v"])), 5, c, 2)
            cv2.putText(img, "%s %.2fm" % (colour[:3], info["distance_m"]),
                        (int(info["u"]) - 20, int(info["v"]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)
        return img

    def _draw_fits(self, img, res):
        for fit, col in ((res.left_fit, (0, 255, 0)), (res.right_fit, (0, 255, 0))):
            if fit is None:
                continue
            a, b = fit[0], fit[1]
            p0 = self.geom.to_pixel(self.p.side_fit_min_x,
                                    a * self.p.side_fit_min_x + b)
            p1 = self.geom.to_pixel(self.p.side_fit_max_x,
                                    a * self.p.side_fit_max_x + b)
            if p0 and p1:
                cv2.line(img, (int(p0[0]), int(p0[1])),
                         (int(p1[0]), int(p1[1])), col, 1)
        if res.front_fit is not None:
            a, b = res.front_fit[0], res.front_fit[1]
            p0 = self.geom.to_pixel(a * -0.45 + b, -0.45)
            p1 = self.geom.to_pixel(a * 0.45 + b, 0.45)
            if p0 and p1:
                cv2.line(img, (int(p0[0]), int(p0[1])),
                         (int(p1[0]), int(p1[1])), (0, 0, 255), 1)
        txt = "L%s R%s F%s hd%s" % (
            "-" if res.left_m is None else "%.2f" % res.left_m,
            "-" if res.right_m is None else "%.2f" % res.right_m,
            "-" if res.front_m is None else "%.2f" % res.front_m,
            "-" if res.heading_deg is None else "%+.0f" % res.heading_deg)
        cv2.putText(img, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1)
        return img
