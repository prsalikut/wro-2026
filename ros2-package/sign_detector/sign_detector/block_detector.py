"""Red/green block detector (pure OpenCV, no ROS). Returns Detection objects.

Profiles:
  "lab"      - RECOMMENDED. Lighting-robust CIELAB a*/b* thresholds. Hue shifts a lot
               with lighting (green measured hue ~36 warm vs ~71 bright) but a* stays
               put: green a*<=124, red a*>=135, background ~127. Validated on the 142
               warm training photos AND a bright white-wall live frame.
  "practice" - legacy HSV tuned to the warm-lit cardboard blocks.
  "official" - HSV for the official WRO pillar colors; recalibrate at the venue.
"""
import cv2
import numpy as np
from dataclasses import dataclass
from math import atan, tan, radians, degrees


@dataclass
class Detection:
    color: str
    cx: int
    cy: int
    w: int
    h: int
    area: int
    bearing_deg: float
    distance_m: float


PROFILES = {
    "lab": dict(
        GREEN_A_MAX=115, GREEN_B_MIN=138,
        RED_A_MIN=154, RED_B_MIN=130,
        L_MIN=12,
    ),
    "practice": dict(
        RED=[((0, 120, 50), (16, 255, 255)), ((166, 120, 50), (179, 255, 255))],
        GREEN=[((28, 70, 45), (46, 255, 255))],
    ),
    "official": dict(
        RED=[((0, 100, 70), (10, 255, 255)), ((168, 100, 70), (179, 255, 255))],
        GREEN=[((40, 80, 70), (85, 255, 255))],
    ),
}


class BlockDetector:
    def __init__(self, profile="practice", hfov_deg=60.0,
                 sign_height_m=0.10, min_area_frac=0.004):
        self.hfov = hfov_deg
        self.sign_h = sign_height_m
        self.min_area_frac = min_area_frac
        self.k_open = np.ones((5, 5), np.uint8)
        self.k_close = np.ones((11, 11), np.uint8)
        self.set_profile(profile)

    def set_profile(self, profile):
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile!r} (use {list(PROFILES)})")
        self.profile = profile
        p = PROFILES[profile]
        if profile == "lab":
            self.lab = p
            self.ranges = None
        else:
            self.lab = None
            self.ranges = {
                "red":   [(np.array(lo), np.array(hi)) for lo, hi in p["RED"]],
                "green": [(np.array(lo), np.array(hi)) for lo, hi in p["GREEN"]],
            }

    def _clean(self, m):
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.k_open)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.k_close)
        return m

    def _mask(self, conv, color):
        if self.lab is not None:
            L, a, b = conv[:, :, 0], conv[:, :, 1], conv[:, :, 2]
            p = self.lab
            if color == "green":
                mk = (a <= p["GREEN_A_MAX"]) & (b >= p["GREEN_B_MIN"]) & (L >= p["L_MIN"])
            else:
                mk = (a >= p["RED_A_MIN"]) & (b >= p["RED_B_MIN"]) & (L >= p["L_MIN"])
            return self._clean(mk.astype(np.uint8) * 255)
        m = np.zeros(conv.shape[:2], np.uint8)
        for lo, hi in self.ranges[color]:
            m |= cv2.inRange(conv, lo, hi)
        return self._clean(m)

    def detect(self, bgr):
        H, W = bgr.shape[:2]
        fx = (W / 2.0) / tan(radians(self.hfov) / 2.0)
        area_min = self.min_area_frac * H * W
        blur = cv2.GaussianBlur(bgr, (5, 5), 0)
        conv = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB if self.lab is not None
                            else cv2.COLOR_BGR2HSV)
        out = []
        for color in ("red", "green"):
            cnts, _ = cv2.findContours(self._mask(conv, color),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area < area_min:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                if not (0.3 < w / float(h) < 3.5):
                    continue
                if area / float(w * h) < 0.55:
                    continue
                cx, cy = x + w // 2, y + h // 2
                bearing = degrees(atan((cx - W / 2.0) / fx))
                dist = (self.sign_h * fx) / h
                out.append(Detection(color, cx, cy, w, h, int(area),
                                     round(bearing, 1), round(dist, 2)))
        out.sort(key=lambda d: d.area, reverse=True)
        return out
