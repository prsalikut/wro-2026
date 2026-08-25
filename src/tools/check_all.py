"""Pre-flight check: subscribe to every sensor topic at once and report what is
live, what is silent, and what is publishing nonsense.

Run it on the Pi after wiring or reflashing, before trusting a round:

    docker exec signstack bash -lc \
      'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && \
       python3 /ros2_ws/src/sign_detector/tools/check_all.py'

It is read-only: it never commands the motor or the servo, so it is safe to run
with the car on the bench or on the mat. Exit status is 0 only if every required
component passes, so it can gate a launch script."""

import argparse
import collections
import json
import math
import statistics as st
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu, LaserScan, Range
from std_msgs.msg import Float32, String

SONARS = ("front", "right", "rear", "left")
VISION = ("left", "right", "front")

# name -> (topic, type, qos, required)
CHECKS = [
    ("camera",   "/image_raw",     Image,     qos_profile_sensor_data, True),
    ("lidar",    "/scan",          LaserScan, qos_profile_sensor_data, True),
    ("bridge",   "bridge_status",  String,    10,                      True),
    ("imu",      "imu/data",       Imu,       qos_profile_sensor_data, False),
    ("imu_yaw",  "imu/yaw",        Float32,   10,                      False),
    ("buttons",  "start_status",   String,    10,                      True),
    # The open round drives on the camera first, so a silent wall_vision is a
    # failure, not a nicety.
    ("vision",   "vision/lane",    String,    10,                      True),
]
for _v in VISION:
    CHECKS.append(("vis_" + _v, "vision/" + _v, Range, 10, False))
for _s in SONARS:
    CHECKS.append(("sonar_" + _s, "sonar/" + _s, Range, 10, True))


class Check(Node):

    def __init__(self, duration):
        super().__init__("check_all")
        self.duration = duration
        self.count = collections.Counter()
        self.last = {}
        self.vals = collections.defaultdict(list)
        self.text = {}
        for name, topic, typ, qos, _req in CHECKS:
            self.create_subscription(
                typ, topic, lambda m, n=name: self.on(n, m), qos)

    def on(self, name, msg):
        self.count[name] += 1
        self.last[name] = time.time()
        if isinstance(msg, Range):
            self.vals[name].append(float(msg.range))
        elif isinstance(msg, Float32):
            self.vals[name].append(float(msg.data))
        elif isinstance(msg, LaserScan):
            good = [r for r in msg.ranges
                    if not math.isinf(r) and not math.isnan(r) and r > 0.0]
            self.vals[name].append(len(good) / float(max(1, len(msg.ranges))))
        elif isinstance(msg, Imu):
            q = msg.orientation
            self.vals[name].append(math.sqrt(
                q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w))
        elif isinstance(msg, String):
            self.text[name] = msg.data


def _vision_note(raw):
    """Summarise the camera's own view of the lane."""
    if not raw:
        return "no data"
    try:
        d = json.loads(raw)
    except ValueError:
        return "unparsable"
    if not d.get("ok"):
        return "NOT OK: %s (floor %.0f%%)" % (
            d.get("reason"), 100.0 * (d.get("floor_frac") or 0.0))
    walls = [k for k in ("left", "right") if d.get(k) is not None]
    if not walls:
        return "no walls fitted -- front %s, floor %.0f%%" % (
            d.get("front"), 100.0 * (d.get("floor_frac") or 0.0))
    return "L=%s R=%s F=%s heading=%s lines=%s" % (
        d.get("left"), d.get("right"), d.get("front"), d.get("heading"),
        sorted(d.get("lines") or {}) or "none")


def rate(n, secs):
    return n / secs if secs > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()

    rclpy.init()
    node = Check(args.seconds)
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        rclpy.spin_once(node, timeout_sec=0.1)
    secs = time.time() - t0

    print("\n==== COMPONENT CHECK (%.0fs) ====\n" % secs)
    print("%-11s %8s %9s   %s" % ("COMPONENT", "MSGS", "Hz", "NOTE"))
    print("-" * 62)

    failures, warnings = [], []
    for name, topic, _typ, _qos, required in CHECKS:
        n = node.count[name]
        hz = rate(n, secs)
        note = ""
        bad = False

        if n == 0:
            note = "SILENT (%s)" % topic
            bad = required
            if not required:
                note += " - optional"
        elif name.startswith("sonar_"):
            v = node.vals[name]
            note = "%.2f-%.2f m" % (min(v), max(v))
            if st.pstdev(v) < 1e-6:
                note += "  FROZEN, same value every frame"
                bad = True
            elif hz < 5.0:
                note += "  slow"
                warnings.append(name)
        elif name == "lidar":
            frac = st.mean(node.vals[name])
            note = "%.0f%% of beams return" % (100.0 * frac)
            if frac < 0.15:
                note += "  LOW"
                warnings.append(name)
        elif name == "imu":
            norm = st.mean(node.vals[name])
            note = ("fused quaternion ok" if abs(norm - 1.0) < 0.05
                    else "quaternion norm %.2f - not NDOF?" % norm)
        elif name == "imu_yaw":
            v = node.vals[name]
            note = "yaw %.1f..%.1f deg" % (min(v), max(v))
        elif name == "vision":
            note = _vision_note(node.text.get("vision"))
            if note.startswith("no walls"):
                warnings.append(name)
        elif name.startswith("vis_"):
            v = node.vals[name]
            note = "%.2f-%.2f m" % (min(v), max(v))
        elif name in node.text:
            note = node.text[name]

        if bad:
            failures.append(name)
        print("%-11s %8d %8.1f   %s%s" % (
            name, n, hz, "FAIL  " if bad else "", note))

    print()
    if node.count["vision"] == 0:
        print("wall_vision is not publishing. The open round can still run on")
        print("  lidar and sonar alone, but that is the configuration that has")
        print("  already failed on this car -- check the wall_vision node.")
    else:
        print("Camera lane view:", _vision_note(node.text.get("vision")))
        print("  Cross-check it against the ranges with:")
        print("    python3 tools/vision_calib.py --check")
    if node.count["imu"] == 0:
        print("IMU absent: heading hold is off, open_round falls back to")
        print("  derivative damping from the ranges. Not fatal.")
    live = [s for s in SONARS if node.count["sonar_" + s]]
    if live and len(live) < len(SONARS):
        print("Only %d/4 sonars reporting: %s" % (len(live), ", ".join(live)))
        print("  Check the shared trigger on Nano D12 and the A0-A3 echoes.")
    if len(live) == len(SONARS):
        print("All 4 sonars live. Wave a hand 20 cm from ONE at a time and")
        print("  re-run: order must be front=A0 right=A1 rear=A2 left=A3.")
        print("  A swapped echo is silent and breaks lane centring.")

    print("\nRESULT:", "FAIL" if failures else ("CHECK" if warnings else "PASS"))
    if failures:
        print("required components down:", ", ".join(failures))

    node.destroy_node()
    rclpy.shutdown()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
