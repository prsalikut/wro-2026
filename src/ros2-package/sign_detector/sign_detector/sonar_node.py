"""Republishes the Nano's four HC-SR04 ranges as sensor_msgs/Range.
Listens for streamed US lines; front/right/rear/left in metres."""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Range
from std_msgs.msg import String

NAMES = ("front", "right", "rear", "left")


class SonarNode(Node):

    def __init__(self):
        super().__init__("sonar")
        self.declare_parameters("", [
            ("enable_stream", True),
            ("min_range_m", 0.02),
            ("max_range_m", 4.0),
            ("fov_deg", 15.0),
            ("reply_timeout_s", 1.0),
        ])

        def g(k):
            return self.get_parameter(k).value

        self.min_m = float(g("min_range_m"))
        self.max_m = float(g("max_range_m"))
        self.fov = float(g("fov_deg")) * 3.14159265 / 180.0
        self.timeout = float(g("reply_timeout_s"))

        self.pubs = {
            n: self.create_publisher(Range, "sonar/{}".format(n), 10)
            for n in NAMES
        }
        self.pub_cmd = self.create_publisher(String, "arduino_cmd", 10)
        self.create_subscription(String, "arduino_reply", self.on_reply, 20)

        self.last_reply = self._now()
        self.warned = False
        self.enable = bool(g("enable_stream"))
        self.started = False
        # One USON at startup, not per-reading: every arduino_cmd opens a manual
        # window in the bridge, which suspends autonomy while it is open.
        self.create_timer(2.0, self.ensure_stream)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def ensure_stream(self):
        if not self.enable:
            return
        if (self._now() - self.last_reply) <= self.timeout:
            self.started = True
            self.warned = False
            return
        if self.started and not self.warned:
            self.get_logger().warning("sonar stream stopped")
            self.warned = True
        self.pub_cmd.publish(String(data="USON"))

    def on_reply(self, msg):
        text = msg.data.strip()
        if "US" not in text:
            return
        parts = text.split()
        try:
            idx = parts.index("US")
        except ValueError:
            return
        vals = parts[idx + 1:idx + 1 + len(NAMES)]
        if len(vals) < len(NAMES):
            return
        self.last_reply = self._now()
        self.warned = False
        stamp = self.get_clock().now().to_msg()
        for name, raw in zip(NAMES, vals):
            try:
                cm = int(raw)
            except ValueError:
                continue
            if cm < 0:
                continue
            m = cm / 100.0
            if m < self.min_m or m > self.max_m:
                continue
            r = Range()
            r.header.stamp = stamp
            r.header.frame_id = "sonar_{}".format(name)
            r.radiation_type = Range.ULTRASOUND
            r.field_of_view = self.fov
            r.min_range = self.min_m
            r.max_range = self.max_m
            r.range = m
            self.pubs[name].publish(r)


def main():
    rclpy.init()
    node = SonarNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
