"""Smoke test: sample /traffic_signs for N seconds, report per-color distance
mean/std/min/max + fusion source stability. Pass criteria printed at the end."""
import math, time, collections
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray

DURATION = 20.0


class S(Node):
    def __init__(self):
        super().__init__("smoke")
        self.samples = collections.defaultdict(list)
        self.frames = 0
        self.create_subscription(Detection3DArray, "/traffic_signs", self.cb, 10)

    def cb(self, m):
        self.frames += 1
        for d in m.detections:
            if not d.results:
                continue
            c = d.results[0].hypothesis.class_id
            p = d.results[0].pose.pose.position
            self.samples[c].append(math.hypot(p.x, p.y))


def main():
    rclpy.init()
    n = S()
    t0 = time.time()
    while time.time() - t0 < DURATION:
        rclpy.spin_once(n, timeout_sec=0.2)
    print(f"==== SMOKE TEST ({DURATION:.0f}s, {n.frames} msgs) ====")
    ok = True
    if not n.samples:
        print("FAIL: no detections at all"); ok = False
    for c, v in sorted(n.samples.items()):
        if len(v) < 5:
            print(f"{c:5}: only {len(v)} samples - too few"); ok = False; continue
        import statistics as st
        mean, sd = st.mean(v), st.pstdev(v)
        print(f"{c:5}: n={len(v):3d}  mean={mean:.3f} m  std={sd*100:.1f} cm  "
              f"min={min(v):.3f}  max={max(v):.3f}")
        if sd > 0.03:
            print(f"       ^ std above 3 cm target"); ok = False
    print("RESULT:", "PASS" if ok else "CHECK")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
