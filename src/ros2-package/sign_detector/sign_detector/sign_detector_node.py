"""
ROS 2 (Humble) node: detect red/green traffic signs in the camera image, fuse each
detection's bearing with the LaserScan using the camera<->lidar extrinsics (the lidar
sits behind/above the camera), and publish 3D detections + RViz markers.

Fusion (per detection): smooth the noisy monocular distance, predict where the lidar
should see the block given the lidar offset, search the scan for the matching cluster,
and fall back to the (smoothed) monocular estimate when no cluster matches.

Subscribes:  <image_topic>  sensor_msgs/Image
             <scan_topic>   sensor_msgs/LaserScan
Publishes:   traffic_signs          vision_msgs/Detection3DArray   (color + x,y in target_frame)
             traffic_signs_markers  visualization_msgs/MarkerArray (RViz)
             sign_debug             sensor_msgs/Image              (annotated, optional)
"""
import math
from collections import deque
from statistics import median

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, LaserScan
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from .block_detector import BlockDetector


class SignDetectorNode(Node):
    ASSOC_TOL_DEG = 8.0
    MAX_GAP_BEAMS = 2
    CLUSTER_JUMP_M = 0.12
    MIN_CLUSTER_BEAMS = 2
    NEAR_RATIO = 1.8
    SCAN_STALE_S = 0.7

    def __init__(self):
        super().__init__("sign_detector")

        self.declare_parameters("", [
            ("detector", "hsv"),
            ("profile", "practice"),
            ("weights", ""),
            ("conf", 0.4),
            ("imgsz", 320),
            ("cnn_chroma_min", 28.0),
            ("cnn_conf_min", 0.60),
            ("cnn_input_size", 64),
            ("hfov_deg", 60.0),
            ("sign_height_m", 0.093),
            ("image_topic", "/image_raw"),
            ("scan_topic", "/scan"),
            ("target_frame", "base_link"),
            ("lidar_dx", -0.195),
            ("angle_sign", 1.0),
            ("angle_offset_deg", -93.7),
            ("search_window_deg", 10.0),
            ("match_tol_m", 0.25),
            ("max_range_m", 2.0),
            ("mono_smooth_n", 5),
            ("fused_smooth_n", 3),
            ("publish_debug", True),
        ])
        g = self.get_parameter
        self.hfov = g("hfov_deg").value
        self.frame = g("target_frame").value
        self.lidar_dx = g("lidar_dx").value
        self.angle_sign = g("angle_sign").value
        self.angle_off = math.radians(g("angle_offset_deg").value)
        self.win = math.radians(g("search_window_deg").value)
        self.match_tol = g("match_tol_m").value
        self.max_range = g("max_range_m").value
        self.mono_n = int(g("mono_smooth_n").value)
        self.fused_n = int(g("fused_smooth_n").value)
        self.publish_debug = g("publish_debug").value

        kind = g("detector").value.lower()
        if kind == "yolo":
            from .ml_block_detector import MLBlockDetector
            w = g("weights").value
            if not w:
                self.get_logger().fatal("detector:=yolo requires the 'weights' parameter")
                raise SystemExit(1)
            self.det = MLBlockDetector(w, hfov_deg=self.hfov,
                                       conf=g("conf").value, imgsz=g("imgsz").value)
            self.get_logger().info(f"YOLO detector: {w}")
        elif kind == "cnn":
            w = g("weights").value
            try:
                from .cnn_block_detector import CNNBlockDetector
                if not w:
                    raise ValueError("detector:=cnn requires 'weights' (path to .tflite/.h5)")
                self.det = CNNBlockDetector(
                    w, hfov_deg=self.hfov, sign_height_m=g("sign_height_m").value,
                    chroma_min=g("cnn_chroma_min").value,
                    conf_min=g("cnn_conf_min").value,
                    input_size=int(g("cnn_input_size").value))
                self.get_logger().info(
                    f"CNN detector: {w} (backend={self.det.clf.backend})")
            except Exception as e:
                self.get_logger().error(
                    f"CNN detector init failed ({e!r}); falling back to colour BlockDetector")
                self.det = BlockDetector(profile=g("profile").value, hfov_deg=self.hfov,
                                         sign_height_m=g("sign_height_m").value)
        else:
            self.det = BlockDetector(profile=g("profile").value, hfov_deg=self.hfov,
                                     sign_height_m=g("sign_height_m").value)
            self.get_logger().info(f"HSV detector (profile={g('profile').value})")

        self.bridge = CvBridge()
        self.scan = None
        self.scan_time = None
        self.tracks = {}
        self.create_subscription(LaserScan, g("scan_topic").value,
                                 self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Image, g("image_topic").value,
                                 self.on_image, qos_profile_sensor_data)
        self.pub_det = self.create_publisher(Detection3DArray, "traffic_signs", 10)
        self.pub_mk = self.create_publisher(MarkerArray, "traffic_signs_markers", 10)
        self.pub_dbg = self.create_publisher(Image, "sign_debug", 1) if self.publish_debug else None

    def on_scan(self, msg):
        self.scan = msg
        self.scan_time = self.get_clock().now().nanoseconds * 1e-9

    def _match_scan(self, alpha_exp, r_exp):
        """Search the scan within +/-win of alpha_exp and return the NEAREST cluster
           (>= MIN_CLUSTER_BEAMS beams) whose median range is plausible for this
           target: not farther than r_exp + match_tol, not nearer than
           r_exp / NEAR_RATIO.  The pillar is the nearest object on its own bearing
           - anything nearer on that exact bearing would occlude it - so
           nearest-cluster selection is robust to monocular scale error, while the
           near bound stops OTHER objects in the wedge (the second pillar, a side
           wall, a robot part) from hijacking the match.
           (Fixed 2026-07-11: the old best-|median - r_exp| rule let the WALL behind
           the pillar win whenever the mono estimate over-read, reporting 0.47 m
           for a block actually at 0.27 m.)  Returns (range_m, centre_angle) or None."""
        s = self.scan
        if s is None or not len(s.ranges):
            return None
        if (self.scan_time is None
                or self.get_clock().now().nanoseconds * 1e-9 - self.scan_time
                > self.SCAN_STALE_S):
            return None
        n = len(s.ranges)
        inc = s.angle_increment
        c = int(round((alpha_exp - s.angle_min) / inc))
        half = max(1, int(round(self.win / inc)))

        clusters, cur, last_j, last_r = [], [], None, None
        for j in range(c - half, c + half + 1):
            r = s.ranges[j % n]
            if not (math.isfinite(r) and s.range_min < r < self.max_range):
                continue
            if (cur and (j - last_j - 1) <= self.MAX_GAP_BEAMS
                    and abs(r - last_r) <= self.CLUSTER_JUMP_M):
                cur.append((j, r))
            else:
                if cur:
                    clusters.append(cur)
                cur = [(j, r)]
            last_j, last_r = j, r
        if cur:
            clusters.append(cur)

        best = None
        for cl in clusters:
            if len(cl) < self.MIN_CLUSTER_BEAMS:
                continue
            med = median([r for _, r in cl])
            if med > r_exp + self.match_tol:
                continue
            if med * self.NEAR_RATIO < r_exp:
                continue
            if best is None or med < best[1]:
                best = (cl, med)
        if best is None:
            return None
        cl, r_m = best
        alpha_m = s.angle_min + 0.5 * (cl[0][0] + cl[-1][0]) * inc
        return r_m, alpha_m

    def _track_mono(self, d):
        """Associate d with its colour track (or reset it) and return the median mono
           distance over the last mono_smooth_n frames."""
        t = self.tracks.get(d.color)
        if t is None or abs(d.bearing_deg - t["bearing"]) > self.ASSOC_TOL_DEG:
            t = {"mono": deque(maxlen=self.mono_n),
                 "fused": deque(maxlen=self.fused_n),
                 "bearing": d.bearing_deg}
            self.tracks[d.color] = t
        t["bearing"] = d.bearing_deg
        t["mono"].append(d.distance_m)
        return median(t["mono"])

    def _fuse(self, d, primary):
        """Fuse one detection with the scan. Returns (x, y, dist, src) in the camera frame.
           primary = largest detection of its colour (gets smoothing); others pass through."""
        phi = -math.radians(d.bearing_deg)
        d_cam = self._track_mono(d) if primary else d.distance_m

        px, py = d_cam * math.cos(phi), d_cam * math.sin(phi)
        vx, vy = px - self.lidar_dx, py
        r_exp = math.hypot(vx, vy)
        phi_l = math.atan2(vy, vx)
        alpha_exp = self.angle_sign * phi_l + self.angle_off

        match = self._match_scan(alpha_exp, r_exp)
        if match is not None:
            r_m, alpha_m = match
            phil = (alpha_m - self.angle_off) / self.angle_sign
            x, y = self.lidar_dx + r_m * math.cos(phil), r_m * math.sin(phil)
            src = "lidar"
        else:
            x, y = px, py
            src = "mono"

        dist = math.hypot(x, y)
        if primary:
            t = self.tracks[d.color]
            t["fused"].append(dist)
            dist_s = median(t["fused"])
            if dist > 1e-6:
                x *= dist_s / dist
                y *= dist_s / dist
            dist = dist_s
        return x, y, dist, src

    def on_image(self, msg):
        bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        dets = self.det.detect(bgr)

        arr = Detection3DArray()
        arr.header = msg.header
        arr.header.frame_id = self.frame

        mk = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        mk.markers.append(clear)

        seen = set()
        for i, d in enumerate(dets):
            primary = d.color not in seen
            seen.add(d.color)
            x, y, dist, src = self._fuse(d, primary)

            det = Detection3D()
            det.header = arr.header
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = d.color
            hyp.hypothesis.score = 1.0
            hyp.pose.pose.position.x = float(x)
            hyp.pose.pose.position.y = float(y)
            hyp.pose.pose.position.z = 0.05
            hyp.pose.pose.orientation.w = 1.0
            det.results.append(hyp)
            det.bbox.center.position.x = float(x)
            det.bbox.center.position.y = float(y)
            det.bbox.center.position.z = 0.05
            det.bbox.center.orientation.w = 1.0
            det.bbox.size.x, det.bbox.size.y, det.bbox.size.z = 0.05, 0.05, 0.10
            arr.detections.append(det)

            mk.markers.append(self._marker(i, d, x, y))
            if self.pub_dbg is not None:
                self._draw(bgr, d, dist, src)

        self.pub_det.publish(arr)
        self.pub_mk.publish(mk)
        if self.pub_dbg is not None:
            out = self.bridge.cv2_to_imgmsg(bgr, "bgr8")
            out.header = msg.header
            self.pub_dbg.publish(out)

    def _marker(self, i, d, x, y):
        m = Marker()
        m.header.frame_id = self.frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "signs"
        m.id = i
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = 0.05
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = 0.05, 0.05, 0.10
        m.color.a = 1.0
        m.color.r = 1.0 if d.color == "red" else 0.0
        m.color.g = 1.0 if d.color == "green" else 0.0
        m.color.b = 0.0
        m.lifetime = Duration(seconds=0.3).to_msg()
        return m

    def _draw(self, bgr, d, dist, src):
        col = (0, 0, 255) if d.color == "red" else (0, 200, 0)
        x1, y1 = d.cx - d.w // 2, d.cy - d.h // 2
        x2, y2 = d.cx + d.w // 2, d.cy + d.h // 2
        flag = "L" if src == "lidar" else "M"
        cv2.rectangle(bgr, (x1, y1), (x2, y2), col, 2)
        cv2.putText(bgr, f"{d.color} {d.bearing_deg:+.0f}deg {dist:.2f}m {flag}",
                    (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)


def main():
    rclpy.init()
    node = SignDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
