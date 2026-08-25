"""Pinhole camera <-> ground plane mapping (no ROS, no OpenCV).

Everything the open round needs from the camera is a distance on the mat, so the
one piece of maths that matters is turning a pixel that sits on the floor into a
metric point in the car's own frame.

Frames
------
Car frame: +x forward, +y left, +z up, origin on the ground under the rear axle.
Camera frame: the usual optical convention, +z along the lens axis, +x right,
+y down.  The camera sits at height `h` above the mat, `cam_x` ahead of the car
origin, pitched DOWN by `pitch` degrees, no roll and no yaw.

With that mounting the camera axes expressed in the car frame are

    x_cam = ( 0,        -1,  0        )      # image right  = car right
    y_cam = (-sin(t),    0, -cos(t)   )      # image down
    z_cam = ( cos(t),    0, -sin(t)   )      # lens axis

so a pixel (u, v) with normalised offsets a = (u-cx)/fx, b = (v-cy)/fy casts the
ray  d = (cos t - b sin t, -a, -sin t - b cos t)  from the camera centre.  It
meets z = 0 at  s = h / (sin t + b cos t), which is positive only below the
horizon row  v_h = cy - fy*tan(t).  Substituting gives the closed forms in
`pixel_to_ground`, and the algebra inverts exactly (see the self-test).

Nothing here is specific to the WRO field, so it is unit-testable on a laptop.
"""

import math

__all__ = [
    "CameraGeometry",
    "focal_from_fov",
    "pixel_to_ground",
    "ground_to_pixel",
    "horizon_row",
]


def focal_from_fov(pixels, fov_deg):
    """Focal length in pixels for a sensor `pixels` wide covering `fov_deg`."""
    half = math.radians(float(fov_deg)) * 0.5
    if half <= 0.0 or half >= math.pi / 2:
        raise ValueError("fov must be in (0, 180) degrees, got %r" % (fov_deg,))
    return (float(pixels) * 0.5) / math.tan(half)


def horizon_row(cy, fy, pitch_deg):
    """Image row where the ground plane recedes to infinity.

    Rows at or above this never touch the mat, however bright they look.
    """
    return cy - fy * math.tan(math.radians(pitch_deg))


def pixel_to_ground(u, v, h, pitch_deg, fx, fy, cx, cy, cam_x=0.0,
                    max_range=None):
    """Pixel -> (forward_m, left_m) on the mat, or None if the ray misses it.

    `max_range` (metres, optional) rejects points so close to the horizon that
    a one-pixel error would move them further than the answer is worth; the
    caller normally passes the range it is willing to trust.
    """
    t = math.radians(pitch_deg)
    st, ct = math.sin(t), math.cos(t)
    b = (float(v) - cy) / fy
    den = st + b * ct
    if den <= 1e-9:                      # at or above the horizon
        return None
    a = (float(u) - cx) / fx
    x = h * (ct - b * st) / den + cam_x
    y = -h * a / den
    if x <= 0.0:
        return None
    if max_range is not None and x > max_range:
        return None
    return x, y


def ground_to_pixel(x, y, h, pitch_deg, fx, fy, cx, cy, cam_x=0.0):
    """(forward_m, left_m) -> pixel, or None when the point is behind the lens."""
    t = math.radians(pitch_deg)
    st, ct = math.sin(t), math.cos(t)
    xc = float(x) - cam_x
    z = xc * ct + h * st                 # depth along the lens axis
    if z <= 1e-9:
        return None
    u = cx - fx * float(y) / z
    v = cy + fy * (h * ct - xc * st) / z
    return u, v


class CameraGeometry(object):
    """Bound version of the free functions above, plus a few conveniences.

    Held by the vision node so the mounting numbers are declared once.
    """

    def __init__(self, width, height, hfov_deg, height_m, pitch_deg,
                 cam_x_m=0.0, max_range_m=3.5):
        self.width = int(width)
        self.height = int(height)
        self.hfov_deg = float(hfov_deg)
        self.h = float(height_m)
        self.pitch_deg = float(pitch_deg)
        self.cam_x = float(cam_x_m)
        self.max_range = float(max_range_m)
        self.fx = focal_from_fov(self.width, self.hfov_deg)
        self.fy = self.fx                       # square pixels
        self.cx = self.width * 0.5
        self.cy = self.height * 0.5

    @property
    def vfov_deg(self):
        return math.degrees(2.0 * math.atan((self.height * 0.5) / self.fy))

    def scaled(self, width, height):
        """Same optics, different image size (for a downscaled work buffer)."""
        return CameraGeometry(width, height, self.hfov_deg, self.h,
                              self.pitch_deg, self.cam_x, self.max_range)

    def horizon_row(self):
        return horizon_row(self.cy, self.fy, self.pitch_deg)

    def to_ground(self, u, v, max_range=None):
        return pixel_to_ground(
            u, v, self.h, self.pitch_deg, self.fx, self.fy, self.cx, self.cy,
            self.cam_x, self.max_range if max_range is None else max_range)

    def to_pixel(self, x, y):
        return ground_to_pixel(
            x, y, self.h, self.pitch_deg, self.fx, self.fy, self.cx, self.cy,
            self.cam_x)

    def row_for_range(self, x):
        """Image row at which the mat is `x` metres ahead, straight in front."""
        px = self.to_pixel(x, 0.0)
        return None if px is None else px[1]

    def bearing_deg(self, u):
        """Bearing of an image column, degrees, + = left. Independent of pitch."""
        return math.degrees(math.atan((self.cx - float(u)) / self.fx))

    def range_resolution(self, x):
        """Metres of forward range covered by one pixel row at range `x`.

        Grows as x^2; the number that decides how far the camera may be
        believed.
        """
        v = self.row_for_range(x)
        if v is None:
            return float("inf")
        near = self.to_ground(self.cx, v + 1.0, max_range=1e9)
        far = self.to_ground(self.cx, v - 1.0, max_range=1e9)
        if near is None or far is None:
            return float("inf")
        return abs(far[0] - near[0]) * 0.5

    def usable_range(self, tolerance_m=0.05, hi=6.0):
        """Furthest range whose per-pixel-row resolution stays under tolerance."""
        lo, best = 0.05, 0.05
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self.range_resolution(mid) <= tolerance_m:
                best, lo = mid, mid
            else:
                hi = mid
        return best


def _self_test():
    cam = CameraGeometry(640, 480, 60.0, 0.12, 15.0, cam_x_m=0.05)
    worst = 0.0
    for x in [0.2, 0.35, 0.5, 0.8, 1.2, 2.0, 3.0]:
        for y in [-0.6, -0.25, 0.0, 0.25, 0.6]:
            px = cam.to_pixel(x, y)
            assert px is not None, (x, y)
            back = cam.to_ground(px[0], px[1], max_range=1e9)
            assert back is not None, (x, y, px)
            worst = max(worst, abs(back[0] - x), abs(back[1] - y))
    assert worst < 1e-9, worst
    assert cam.to_ground(cam.cx, cam.horizon_row() - 1.0) is None
    assert cam.to_ground(cam.cx, cam.horizon_row() + 0.5,
                         max_range=1e9) is not None

    print("round-trip error: %.2e m" % worst)
    print("hfov %.1f  vfov %.1f  horizon row %.1f  usable to %.2f m (5 cm/row)"
          % (cam.hfov_deg, cam.vfov_deg, cam.horizon_row(),
             cam.usable_range(0.05)))
    print("%6s %8s %10s" % ("row", "range_m", "m_per_row"))
    for v in range(int(cam.horizon_row()) + 2, cam.height + 1, 24):
        g = cam.to_ground(cam.cx, v, max_range=1e9)
        if g is None:
            continue
        print("%6d %8.3f %10.4f" % (v, g[0], cam.range_resolution(g[0])))


if __name__ == "__main__":
    _self_test()
