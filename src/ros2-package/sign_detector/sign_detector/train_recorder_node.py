"""Records human RC driving (raw scan, steering_cmd, drive_cmd, optional IMU yaw rate) to JSONL.

Controlled via train_cmd ("start [name]" / "stop"); publishes train_status JSON at 1 Hz.
"""
import json
import math
import os
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Float32, String

DEFAULT_DIR = "/ros2_ws/src/sign_detector/training_data"


class TrainRecorder(Node):
    def __init__(self):
        super().__init__("train_recorder")

        self.declare_parameters("", [
            ("out_dir", DEFAULT_DIR),
            ("rate_hz", 20.0),
            ("min_drive_pct", 3.0),
            ("scan_timeout_s", 0.5),
        ])
        g = self.get_parameter
        self.out_dir = str(g("out_dir").value)
        self.rate = float(g("rate_hz").value)
        self.min_drive = float(g("min_drive_pct").value)
        self.scan_timeout = float(g("scan_timeout_s").value)

        self.scan = None
        self.scan_t = None
        self.steer = 0.0
        self.drive = 0.0
        self.imu_yaw_rate = None
        self.imu_seen = False

        self.fh = None
        self.session = None
        self.rows = 0
        self.skipped = 0
        self.started_at = None

        self.create_subscription(
            LaserScan, "scan", self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Float32, "steering_cmd", self._on_steer, 10)
        self.create_subscription(Float32, "drive_cmd", self._on_drive, 10)
        self.create_subscription(Imu, "imu/data", self._on_imu,
                                 qos_profile_sensor_data)
        self.create_subscription(String, "train_cmd", self._on_cmd, 10)
        self.pub_status = self.create_publisher(String, "train_status", 10)

        self.create_timer(1.0 / max(1.0, self.rate), self._on_sample)
        self.create_timer(1.0, self._on_status)
        self.get_logger().info(
            "train_recorder idle; out_dir={} rate={:.0f}Hz".format(
                self.out_dir, self.rate))

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_scan(self, m):
        self.scan = m
        self.scan_t = self._now()

    def _on_steer(self, m):
        self.steer = float(m.data)

    def _on_drive(self, m):
        self.drive = float(m.data)

    def _on_imu(self, m):
        self.imu_seen = True
        self.imu_yaw_rate = float(m.angular_velocity.z)

    def _on_cmd(self, msg):
        parts = msg.data.strip().split()
        if not parts:
            return
        verb = parts[0].lower()
        if verb == "start":
            self.start(parts[1] if len(parts) > 1 else None)
        elif verb == "stop":
            self.stop()

    def start(self, name=None):
        if self.fh is not None:
            return False, "already recording"
        try:
            os.makedirs(self.out_dir, exist_ok=True)
        except Exception as exc:
            self.get_logger().error("mkdir failed: {}".format(exc))
            return False, str(exc)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        safe = "".join(c for c in (name or "run") if c.isalnum() or c in "-_")
        self.session = "{}-{}.jsonl".format(stamp, safe or "run")
        path = os.path.join(self.out_dir, self.session)
        try:
            self.fh = open(path, "w", buffering=1)
        except Exception as exc:
            self.get_logger().error("open failed: {}".format(exc))
            return False, str(exc)
        self.rows = 0
        self.skipped = 0
        self.started_at = self._now()
        self.fh.write(json.dumps(self._meta()) + "\n")
        self.get_logger().info("recording -> {}".format(path))
        return True, self.session

    def _meta(self):
        meta = {"type": "meta", "session": self.session}
        if self.scan is not None:
            meta.update({
                "angle_min": self.scan.angle_min,
                "angle_increment": self.scan.angle_increment,
                "range_min": self.scan.range_min,
                "range_max": self.scan.range_max,
                "n_beams": len(self.scan.ranges),
            })
        return meta

    def stop(self):
        if self.fh is None:
            return False, "not recording"
        try:
            self.fh.close()
        except Exception:
            pass
        done, n = self.session, self.rows
        self.fh = None
        self.session = None
        self.get_logger().info("stopped: {} rows -> {}".format(n, done))
        return True, "{} ({} rows)".format(done, n)

    def _on_sample(self):
        if self.fh is None:
            return
        now = self._now()
        if self.scan is None or self.scan_t is None:
            return
        if (now - self.scan_t) > self.scan_timeout:
            self.skipped += 1  # stale scan
            return
        if abs(self.drive) < self.min_drive:
            self.skipped += 1  # parked
            return

        ranges = [None if not math.isfinite(r) else round(float(r), 4)
                  for r in self.scan.ranges]
        row = {
            "t": round(now - self.started_at, 3),
            "ranges": ranges,
            "steer": round(self.steer, 2),
            "drive": round(self.drive, 2),
        }
        if self.imu_yaw_rate is not None:
            row["yaw_rate"] = round(self.imu_yaw_rate, 5)
        try:
            self.fh.write(json.dumps(row) + "\n")
            self.rows += 1
        except Exception as exc:
            self.get_logger().error("write failed: {}".format(exc))
            self.stop()

    def _on_status(self):
        self.pub_status.publish(String(data=json.dumps({
            "recording": self.fh is not None,
            "session": self.session,
            "rows": self.rows,
            "skipped": self.skipped,
            "secs": (None if self.started_at is None or self.fh is None
                     else round(self._now() - self.started_at, 1)),
            "imu": self.imu_seen,
            "lidar": self.scan is not None and self.scan_t is not None
                     and (self._now() - self.scan_t) <= self.scan_timeout,
            "out_dir": self.out_dir,
        })))


def main():
    rclpy.init()
    node = TrainRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
