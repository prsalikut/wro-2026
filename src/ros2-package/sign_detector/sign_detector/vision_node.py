"""ROS 2 node: wall_vision -- camera-derived lane geometry for the open round.

Runs sign_detector.wall_vision over /image_raw and publishes what it finds in
two forms: a JSON summary on /vision/lane for the driver and the dashboard, and
plain sensor_msgs/Range on /vision/{left,right,front} so the camera's distances
can be fused with the lidar's and the ultrasonics' by exactly the same code.

Publishes
    /vision/lane          std_msgs/String   JSON: distances, heading, lines
    /vision/left|right    sensor_msgs/Range perpendicular wall distance
    /vision/front         sensor_msgs/Range nearest obstruction ahead
    /line_event           std_msgs/String   colour, once per corner line
    /vision/debug_image   sensor_msgs/Image annotated frame (optional)

The camera mounting parameters (height, pitch, hfov) are the only ones that
must be right for the distances to mean anything; tools/vision_calib.py
measures them against a tape measure.
"""

import json
import math

import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Range
from std_msgs.msg import String

from .ground_geometry import CameraGeometry
from .wall_vision import WallVision, WallVisionParams

SIDES = ("left", "right", "front")


class VisionNode(Node):

    def __init__(self):
        super().__init__("wall_vision")
        self.declare_parameters("", [
            # --- camera mounting, in metres and degrees --------------------
            ("cam_height_m", 0.12),
            ("cam_pitch_deg", 15.0),
            ("cam_x_m", 0.05),
            ("hfov_deg", 60.0),
            ("max_range_m", 2.2),

            ("image_topic", "image_raw"),
            ("process_hz", 15.0),
            ("publish_debug", True),
            ("debug_hz", 4.0),
            ("line_cooldown_s", 1.5),
            ("line_event_m", 0.55),

            # --- perception tuning (see wall_vision.WallVisionParams) -------
            ("work_width", 320),
            ("work_height", 240),
            ("dark_ratio", 0.62),
            ("dark_abs", 34.0),
            ("chroma_min", 20.0),
            ("run_px", 4),
            ("col_step", 4),
            ("min_range_m", 0.16),
            ("eval_x_m", 0.15),
            ("side_fit_min_x", 0.20),
            ("side_fit_max_x", 1.90),
            ("max_fit_rms_m", 0.055),
            ("front_corridor_m", 0.16),
            ("line_sat_min", 70),
            ("line_val_min", 45),
            ("line_min_area_px", 90),
            ("illum_fit", True),
            ("paint_is_floor", True),
            ("block_magenta", True),
        ])

        def g(k):
            return self.get_parameter(k).value

        self.hfov = float(g("hfov_deg"))
        self.cam_h = float(g("cam_height_m"))
        self.cam_pitch = float(g("cam_pitch_deg"))
        self.cam_x = float(g("cam_x_m"))
        self.max_range = float(g("max_range_m"))
        self.min_period = 1.0 / max(1.0, float(g("process_hz")))
        self.debug_period = 1.0 / max(0.1, float(g("debug_hz")))
        self.want_debug = bool(g("publish_debug"))
        self.line_cooldown = float(g("line_cooldown_s"))
        self.line_event_m = float(g("line_event_m"))

        self.vp = WallVisionParams(
            work_width=int(g("work_width")), work_height=int(g("work_height")),
            dark_ratio=float(g("dark_ratio")), dark_abs=float(g("dark_abs")),
            chroma_min=float(g("chroma_min")), run_px=int(g("run_px")),
            col_step=int(g("col_step")), min_range_m=float(g("min_range_m")),
            max_range_m=self.max_range, eval_x_m=float(g("eval_x_m")),
            side_fit_min_x=float(g("side_fit_min_x")),
            side_fit_max_x=float(g("side_fit_max_x")),
            max_fit_rms_m=float(g("max_fit_rms_m")),
            front_corridor_m=float(g("front_corridor_m")),
            line_sat_min=int(g("line_sat_min")),
            line_val_min=int(g("line_val_min")),
            line_min_area_px=int(g("line_min_area_px")),
            illum_fit=bool(g("illum_fit")),
            paint_is_floor=bool(g("paint_is_floor")),
            block_magenta=bool(g("block_magenta")))

        self.bridge = CvBridge()
        self.vision = None
        self.size = None

        self.pub_lane = self.create_publisher(String, "vision/lane", 10)
        self.pub_line = self.create_publisher(String, "line_event", 10)
        self.pub_range = {
            s: self.create_publisher(Range, "vision/{}".format(s), 10)
            for s in SIDES}
        self.pub_debug = self.create_publisher(Image, "vision/debug_image", 1)
        self.create_subscription(Image, str(g("image_topic")), self.on_image,
                                 qos_profile_sensor_data)

        self.last_processed = 0.0
        self.last_debug = 0.0
        self.last_fired = {}
        self.visible = {}
        self.frames = 0
        self.get_logger().info(
            "wall_vision up: camera {:.0f} deg hfov at {:.0f} mm, {:.0f} deg "
            "down; trusting ranges to {:.2f} m".format(
                self.hfov, self.cam_h * 1000.0, self.cam_pitch, self.max_range))

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _ensure(self, width, height):
        if self.vision is not None and self.size == (width, height):
            return
        geom = CameraGeometry(width, height, self.hfov, self.cam_h,
                              self.cam_pitch, self.cam_x, self.max_range)
        self.vp.work_width = min(self.vp.work_width, width)
        self.vp.work_height = min(self.vp.work_height, height)
        self.vision = WallVision(geom, self.vp)
        self.size = (width, height)
        self.get_logger().info(
            "camera {}x{}: vfov {:.0f} deg, horizon row {:.0f}, one pixel row "
            "is {:.0f} mm at 1 m; honest range {:.2f} m".format(
                width, height, geom.vfov_deg, geom.horizon_row(),
                1000.0 * geom.range_resolution(1.0), geom.usable_range(0.05)))

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

        self._ensure(bgr.shape[1], bgr.shape[0])
        debug = self.want_debug and (now - self.last_debug) >= self.debug_period
        res = self.vision.process(bgr, want_debug=debug)
        self.frames += 1

        data = res.as_dict()
        data["stamp"] = now
        self.pub_lane.publish(String(data=json.dumps(data)))

        stamp = self.get_clock().now().to_msg()
        for side in SIDES:
            value = getattr(res, "{}_m".format(side))
            if value is None:
                continue
            r = Range()
            r.header.stamp = stamp
            r.header.frame_id = "camera_{}".format(side)
            r.radiation_type = Range.INFRARED     # closest thing to "optical"
            r.field_of_view = math.radians(self.hfov)
            r.min_range = float(self.vp.min_range_m)
            r.max_range = float(self.max_range)
            r.range = float(value)
            self.pub_range[side].publish(r)

        self._emit_lines(res, now)

        if debug and res.debug is not None:
            self.last_debug = now
            try:
                out = self.bridge.cv2_to_imgmsg(res.debug, "bgr8")
                out.header.stamp = stamp
                self.pub_debug.publish(out)
            except Exception as exc:
                self.get_logger().warning("debug publish failed: {}".format(exc))

        if self.frames % 150 == 1:
            self.get_logger().info(
                "L={} R={} F={} heading={} lines={} floor={:.0f}%".format(
                    data["left"], data["right"], data["front"],
                    data["heading"], sorted(data["lines"]),
                    100.0 * (data["floor_frac"] or 0.0)))

    def _emit_lines(self, res, now):
        """One event per crossing, not one per frame."""
        for colour in ("orange", "blue"):
            info = res.lines.get(colour)
            near = (info is not None
                    and info.get("distance_m") is not None
                    and info["distance_m"] <= self.line_event_m)
            was = self.visible.get(colour, False)
            if near and not was and (now - self.last_fired.get(colour, -1e9)
                                     ) >= self.line_cooldown:
                self.last_fired[colour] = now
                self.pub_line.publish(String(data=colour))
                self.get_logger().info("{} line at {:.2f} m".format(
                    colour, info["distance_m"]))
            self.visible[colour] = near


def main():
    rclpy.init()
    node = VisionNode()
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
