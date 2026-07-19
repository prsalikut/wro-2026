"""
ROS 2 (Humble) node: sign_steering  -- DEMO of the WRO pass rule (servo only)

NOTE: this is a demonstration of the traffic-sign pass rule with the STEERING
SERVO ONLY - there is no drive motor here yet. It just proves the steering
direction is correct end-to-end.

WRO 2026 pass rule: a RED pillar means the vehicle keeps to the RIGHT side of the
lane -> steer RIGHT (POSITIVE degrees; the pillar passes on the car's LEFT).
A GREEN pillar means keep LEFT -> steer LEFT (NEGATIVE degrees).

Each frame it picks the NEAREST in-range red/green detection and publishes a
signed steering angle: FULL deflection once the pillar is inside full_range_m,
tapering linearly down to min_scale * full_deflection_deg at engage_range_m.
When no in-range sign has been seen for clear_center_s seconds it re-centers.

Drive: while vision is alive it also publishes a constant forward cruise duty on
/drive_cmd (drive_cruise_pct, slowed to slow_factor of that while actively
passing a pillar); when vision goes quiet both channels drop to 0.

Subscribes:  traffic_signs   vision_msgs/Detection3DArray
Publishes:   steering_cmd    std_msgs/Float32   (signed degrees, + = steer RIGHT)
             drive_cmd       std_msgs/Float32   (signed percent, + = forward)
Params:      engage_range_m, full_deflection_deg, full_range_m, min_scale,
             clear_center_s, drive_cruise_pct, slow_factor
"""
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from vision_msgs.msg import Detection3DArray


class SignSteering(Node):
    def __init__(self):
        super().__init__("sign_steering")

        self.declare_parameters("", [
            ("engage_range_m", 1.2),
            ("full_deflection_deg", 25.0),
            ("full_range_m", 0.6),
            ("min_scale", 0.4),
            ("clear_center_s", 0.7),
            ("drive_cruise_pct", 30.0),
            ("slow_factor", 0.7),
        ])
        g = self.get_parameter
        self.engage_range = float(g("engage_range_m").value)
        self.full_defl = float(g("full_deflection_deg").value)
        self.full_range = float(g("full_range_m").value)
        self.min_scale = float(g("min_scale").value)
        self.clear_center = float(g("clear_center_s").value)
        self.cruise = float(g("drive_cruise_pct").value)
        self.slow_factor = float(g("slow_factor").value)

        self.pub = self.create_publisher(Float32, "steering_cmd", 10)
        self.pub_drive = self.create_publisher(Float32, "drive_cmd", 10)
        self.create_subscription(Detection3DArray, "traffic_signs", self.on_signs, 10)
        self.create_timer(0.2, self._on_watch)

        self._last_seen = None
        self._last_frame = None
        self._last_log = 0.0
        self._last_key = None

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _publish(self, deg):
        msg = Float32()
        msg.data = float(deg)
        self.pub.publish(msg)

    def _publish_drive(self, pct):
        msg = Float32()
        msg.data = float(pct)
        self.pub_drive.publish(msg)

    def _log(self, key, text):
        """Log transitions immediately; steady state at most ~1 Hz."""
        now = self._now()
        if key != self._last_key or (now - self._last_log) >= 1.0:
            self.get_logger().info(text)
            self._last_log = now
            self._last_key = key

    def _on_watch(self):
        """on_signs stops firing entirely if the camera/detector dies; keep
        commanding center + motor stop then so the bridge never holds stale
        commands. (The bridge's own cmd_timeout_s failsafe backs THIS node.)"""
        now = self._now()
        if self._last_frame is None or (now - self._last_frame) >= self.clear_center:
            self._publish(0.0)
            self._publish_drive(0.0)
            self._log("center", "vision quiet -> center + stop")

    def on_signs(self, msg):
        now = self._now()
        self._last_frame = now

        best = None
        for det in msg.detections:
            if not det.results:
                continue
            hyp = det.results[0]
            cls = hyp.hypothesis.class_id.lower()
            if cls not in ("red", "green"):
                continue
            pos = hyp.pose.pose.position
            dist = math.hypot(pos.x, pos.y)
            if dist > self.engage_range:
                continue
            if best is None or dist < best[0]:
                best = (dist, cls, pos.x, pos.y)

        if best is None:
            self._publish_drive(self.cruise)
            if self._last_seen is None or (now - self._last_seen) >= self.clear_center:
                self._publish(0.0)
                self._log("center", "clear -> center (0.0 deg)")
            return

        dist, cls, _x, _y = best
        self._last_seen = now
        direction = 1.0 if cls == "red" else -1.0
        if dist <= self.full_range:
            scale = 1.0
        else:
            span = max(1e-6, self.engage_range - self.full_range)
            scale = (self.engage_range - dist) / span
        scale = max(self.min_scale, min(1.0, scale))
        deg = direction * self.full_defl * scale
        self._publish(deg)
        self._publish_drive(self.cruise * self.slow_factor)
        side = "keep right" if cls == "red" else "keep left"
        self._log(cls, "{} @ {:.2f}m -> {:+.1f} deg ({})".format(cls, dist, deg, side))


def main():
    rclpy.init()
    node = SignSteering()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
