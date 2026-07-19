"""TEST MODE: steer to FOLLOW the nearest block by its bearing (ignores colour).
Block to the LEFT -> wheels steer LEFT; block to the RIGHT -> wheels steer RIGHT;
centred/absent -> straight. Publishes std_msgs/Float32 on /steering_cmd (+ = right),
which steering_bridge forwards to the Arduino. This is a driving/steering sanity
test, NOT the WRO pass rule (that lives in sign_steering_node.py)."""
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from vision_msgs.msg import Detection3DArray


class FollowBlock(Node):
    def __init__(self):
        super().__init__("follow_block")
        self.declare_parameter("gain", 1.7)
        self.declare_parameter("max_deg", 25.0)
        self.declare_parameter("deadband_deg", 2.0)
        self.declare_parameter("engage_range_m", 1.6)
        self.declare_parameter("clear_center_s", 0.5)
        self.declare_parameter("smooth_alpha", 0.2)
        self.declare_parameter("min_step_deg", 1.0)
        g = self.get_parameter
        self.gain = g("gain").value
        self.max_deg = g("max_deg").value
        self.deadband = g("deadband_deg").value
        self.engage = g("engage_range_m").value
        self.clear_s = g("clear_center_s").value
        self.alpha = g("smooth_alpha").value
        self.min_step = g("min_step_deg").value
        self.cmd_ema = 0.0
        self.last_pub = 0.0

        self.pub = self.create_publisher(Float32, "/steering_cmd", 10)
        self.create_subscription(Detection3DArray, "/traffic_signs", self.on_dets, 10)
        self.last_seen = self.get_clock().now()
        self.centered = False
        self.create_timer(0.1, self.tick)
        self._log_t = self.get_clock().now()
        self.get_logger().info(
            f"follow_block: gain={self.gain} max={self.max_deg}deg "
            f"engage<{self.engage}m  (block left->steer left, right->right)")

    def on_dets(self, msg):
        best, best_d = None, 1e9
        for d in msg.detections:
            if not d.results:
                continue
            p = d.results[0].pose.pose.position
            dist = math.hypot(p.x, p.y)
            if dist <= self.engage and dist < best_d:
                best_d, best = dist, (d.results[0].hypothesis.class_id, p, dist)
        if best is None:
            return
        cid, p, dist = best
        bearing = math.degrees(math.atan2(-p.y, p.x))
        raw = 0.0 if abs(bearing) < self.deadband else bearing * self.gain
        raw = max(-self.max_deg, min(self.max_deg, raw))
        self.cmd_ema += self.alpha * (raw - self.cmd_ema)
        self.last_seen = self.get_clock().now()
        self.centered = False
        if abs(self.cmd_ema - self.last_pub) >= self.min_step:
            self.last_pub = round(self.cmd_ema, 1)
            self.pub.publish(Float32(data=float(self.last_pub)))
        now = self.get_clock().now()
        if (now - self._log_t).nanoseconds > 4e8:
            side = "RIGHT" if bearing > self.deadband else ("LEFT" if bearing < -self.deadband else "centre")
            self.get_logger().info(
                f"{cid} @ {dist:.2f}m  bearing={bearing:+5.1f} {side:5} "
                f"-> steer {self.last_pub:+5.1f}")
            self._log_t = now

    def tick(self):
        dt = (self.get_clock().now() - self.last_seen).nanoseconds / 1e9
        if not self.centered and dt > self.clear_s:
            self.cmd_ema = 0.0
            self.last_pub = 0.0
            self.pub.publish(Float32(data=0.0))
            self.centered = True
            self.get_logger().info("no block in range -> centre (0.0)")


def main():
    rclpy.init()
    n = FollowBlock()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
