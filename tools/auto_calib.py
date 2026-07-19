"""Auto-calibrate camera<->LiDAR angular alignment + effective block height.

Scene requirement: several boxes may be visible, but EXACTLY ONE object within
~0.5 m of the camera (the calibration target = the detector's largest detection,
ideally placed off-centre so the rotation direction is unambiguous).

Rig geometry: lidar at (x=-0.195, y=0) in the camera frame (19.5 cm behind).
Prints recommended angle_sign / angle_offset_deg / sign_height_m and exits.
Run inside the sign_detector container: python3 auto_calib.py
"""
import math, time, collections
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
from sign_detector.block_detector import BlockDetector

LIDAR_DX = -0.195
MONO_GATE_M = 0.65
CLUSTER_BAND = (0.15, 0.80)
CLUSTER_W_DEG = (2.0, 45.0)
N_TARGET = 70
TIMEOUT_S = 75.0
MAX_GAP = 2
JUMP_M = 0.12
PRIOR_OFF = math.pi
PRIOR_TOL = math.radians(28.0)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def circ_stats(offsets):
    """Circular centre + spread (rad) via mean direction then median/std of deviations."""
    if not offsets:
        return None, None
    m = math.atan2(np.mean([math.sin(o) for o in offsets]),
                   np.mean([math.cos(o) for o in offsets]))
    dev = [wrap(o - m) for o in offsets]
    return wrap(m + float(np.median(dev))), float(np.std(dev))


def clusters_of(scan):
    """All clusters in the full scan: list of (median_range, centre_angle_rad, width_rad)."""
    out, cur, last_j, last_r = [], [], None, None
    for j, r in enumerate(scan.ranges):
        ok = math.isfinite(r) and scan.range_min < r < 8.0
        if ok and cur and (j - last_j - 1) <= MAX_GAP and abs(r - last_r) <= JUMP_M:
            cur.append((j, r))
        elif ok:
            if len(cur) >= 2:
                out.append(cur)
            cur = [(j, r)]
        if ok:
            last_j, last_r = j, r
    if len(cur) >= 2:
        out.append(cur)
    res = []
    for cl in out:
        med = float(np.median([r for _, r in cl]))
        a0 = scan.angle_min + cl[0][0] * scan.angle_increment
        a1 = scan.angle_min + cl[-1][0] * scan.angle_increment
        res.append((med, wrap(0.5 * (a0 + a1)), abs(a1 - a0)))
    return res


class Calib(Node):
    def __init__(self):
        super().__init__("auto_calib")
        self.br = CvBridge()
        self.det = BlockDetector(profile="lab", hfov_deg=60.0)
        self.scan = None
        self.mono_hist = collections.deque(maxlen=5)
        self.samples = []
        self.skipped = 0
        self.n_img = 0
        self.create_subscription(Image, "/image_raw", self.on_img, qos_profile_sensor_data)
        self.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data)

    def on_scan(self, m):
        self.scan = m

    def on_img(self, msg):
        self.n_img += 1
        if self.scan is None:
            return
        dets = self.det.detect(self.br.imgmsg_to_cv2(msg, "bgr8"))
        if not dets:
            return
        d = dets[0]
        self.mono_hist.append(d.distance_m)
        d_mono = float(np.median(self.mono_hist))
        if d_mono > MONO_GATE_M or len(self.mono_hist) < 3:
            return
        phi = -math.radians(d.bearing_deg)
        px, py = d_mono * math.cos(phi), d_mono * math.sin(phi)
        vx, vy = px - LIDAR_DX, py
        phi_l = math.atan2(vy, vx)
        r_exp = math.hypot(vx, vy)
        allc = [c for c in clusters_of(self.scan)
                if math.radians(CLUSTER_W_DEG[0]) <= c[2] <= math.radians(CLUSTER_W_DEG[1])]
        cands = [c for c in allc if abs(c[0] - r_exp) <= 0.22
                 and min(abs(wrap(wrap(c[1] - s * phi_l) - PRIOR_OFF))
                         for s in (1.0, -1.0)) <= PRIOR_TOL]
        if len(cands) != 1:
            self.skipped += 1
            if self.skipped % 60 == 1:
                near = sorted([(round(c[0], 2), round(math.degrees(c[1])), round(math.degrees(c[2])))
                               for c in allc if c[0] < 1.0])
                print(f"  [diag] d_mono={d_mono:.2f} r_exp={r_exp:.2f} "
                      f"cands={len(cands)}; clusters<1m (r,ang,w): {near}", flush=True)
            return
        r_cl, a_cl, _ = cands[0]
        self.samples.append((phi_l, a_cl, r_cl, d_mono, d.bearing_deg))
        if len(self.samples) % 10 == 0:
            print(f"  collected {len(self.samples)}/{N_TARGET} "
                  f"(skipped {self.skipped})", flush=True)


def main():
    rclpy.init()
    n = Calib()
    t0 = time.time()
    while len(n.samples) < N_TARGET and time.time() - t0 < TIMEOUT_S:
        rclpy.spin_once(n, timeout_sec=0.2)
        if n.n_img == 0 and time.time() - t0 > 20:
            print("ERROR: no images arriving on /image_raw"); break

    S = n.samples
    print(f"\n==== CALIBRATION RESULT ====")
    print(f"samples_used: {len(S)}   skipped: {n.skipped}")
    if len(S) < 8:
        print("ERROR: too few valid samples - is exactly ONE box within 50 cm,")
        print("detected by the camera, and the lidar spinning?")
        rclpy.shutdown(); return

    res = {}
    for sign in (+1.0, -1.0):
        offs = [wrap(a - sign * p) for p, a, _, _, _ in S]
        med, spread = circ_stats(offs)
        res[sign] = (med, spread)
    best = min(res, key=lambda s: res[s][1])
    alt = -best
    off_med, off_spread = res[best]

    d_true, heights = [], []
    for phi_l, a_cl, r_cl, d_mono, _ in S:
        phil = wrap((a_cl - off_med) / best)
        x = LIDAR_DX + r_cl * math.cos(phil)
        y = r_cl * math.sin(phil)
        dt = math.hypot(x, y)
        d_true.append(dt)
        heights.append(0.10 * dt / d_mono)

    print(f"angle_sign: {best:+.0f}")
    print(f"angle_offset_deg: {math.degrees(off_med):.1f}")
    print(f"offset_spread_deg: {math.degrees(off_spread):.1f}   "
          f"(alt hypothesis spread: {math.degrees(res[alt][1]):.1f})")
    if res[alt][1] < 1.2 * off_spread:
        print("WARNING: sign ambiguous (target too close to centre) - re-run with the")
        print("block clearly to one side; using the lower-spread hypothesis anyway.")
    print(f"lidar cluster: range_med={np.median([s[2] for s in S]):.2f} m  "
          f"angle_med={math.degrees(np.median([s[1] for s in S])):.1f} deg (scan frame)")
    print(f"camera: bearing_med={np.median([s[4] for s in S]):.1f} deg  "
          f"d_mono_med={np.median([s[3] for s in S]):.2f} m")
    print(f"d_true_med (from lidar, camera frame): {np.median(d_true):.2f} m")
    print(f"sign_height_m recommended: {np.median(heights):.3f}")
    print("============================")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
