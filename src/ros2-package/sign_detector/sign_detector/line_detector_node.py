"""Detects the mat's orange and blue corner lines in the lower camera band and
publishes a crossing event per line on line_event."""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String


class LineDetector(Node):

    def __init__(self):
        super().__init__("line_detector")
        self.declare_parameters("", [
            ("band_top", 0.55),
            ("band_bottom", 0.92),
            ("min_area_frac", 0.015),
            ("cooldown_s", 1.5),
            ("orange_h_lo", 2),
            ("orange_h_hi", 24),
            ("orange_sat_min", 60),
            ("orange_val_min", 60),
            ("blue_h_lo", 92),
            ("blue_h_hi", 135),
            ("blue_sat_min", 40),
            ("blue_val_min", 5),
            ("process_hz", 12.0),
        ])

        def g(k):
            return self.get_parameter(k).value

        self.band_top = float(g("band_top"))
        self.band_bottom = float(g("band_bottom"))
        self.min_frac = float(g("min_area_frac"))
        self.cooldown = float(g("cooldown_s"))
        self.ranges = {
            "orange": ((int(g("orange_h_lo")), int(g("orange_sat_min")),
                        int(g("orange_val_min"))),
                       (int(g("orange_h_hi")), 255, 255)),
            "blue": ((int(g("blue_h_lo")), int(g("blue_sat_min")),
                      int(g("blue_val_min"))),
                     (int(g("blue_h_hi")), 255, 255)),
        }
        self.min_period = 1.0 / max(1.0, float(g("process_hz")))

        self.bridge = CvBridge()
        self.pub = self.create_publisher(String, "line_event", 10)
        self.pub_frac = self.create_publisher(String, "line_debug", 10)
        self.create_subscription(
            Image, "image_raw", self.on_image, qos_profile_sensor_data)

        self.last_processed = 0.0
        self.last_fired = {"orange": -1e9, "blue": -1e9}
        self.visible = {"orange": False, "blue": False}
        self.get_logger().info("line_detector up")

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _mask_frac(self, hsv, color):
        lo, hi = self.ranges[color]
        mask = cv2.inRange(hsv, lo, hi)
        return float(cv2.countNonZero(mask)) / mask.size

    def on_image(self, msg):
        now = self._now()
        if now - self.last_processed < self.min_period:
            return
        self.last_processed = now

        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            self.get_logger().warning("cv_bridge failed: {}".format(exc))
            return

        h = bgr.shape[0]
        band = bgr[int(h * self.band_top):int(h * self.band_bottom)]
        band = cv2.resize(band, (160, max(24, band.shape[0] // 4)))
        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)

        fracs = {
            "orange": self._mask_frac(hsv, "orange"),
            "blue": self._mask_frac(hsv, "blue"),
        }
        self.pub_frac.publish(String(data=json_fracs(fracs)))

        for color, frac in fracs.items():
            seen = frac >= self.min_frac
            if (seen and not self.visible[color]
                    and now - self.last_fired[color] >= self.cooldown):
                self.last_fired[color] = now
                self.pub.publish(String(data=color))
                self.get_logger().info(
                    "{} line ({:.1f}% of band)".format(color, frac * 100))
            self.visible[color] = seen


def json_fracs(fracs):
    return '{{"orange": {:.4f}, "blue": {:.4f}}}'.format(
        fracs["orange"], fracs["blue"])


def main():
    rclpy.init()
    node = LineDetector()
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
