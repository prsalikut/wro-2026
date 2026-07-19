"""
ROS 2 (Humble) node: steering_bridge

Bridges /steering_cmd (std_msgs/Float32, signed degrees, POSITIVE = steer RIGHT)
to the steering Arduino over serial (see arduino_link.ArduinoLink). It applies an
optional direction flip (steer_dir), an optional slew-rate limit, and clamps to
the mechanical limit (max_deg) before forwarding 'S <deg>'.

Also bridges /drive_cmd (std_msgs/Float32, signed percent, POSITIVE = forward)
to the BTS7960 drive motor via 'M <pct>', clamped to +/-drive_max_pct. A stale
/drive_cmd (cmd_timeout_s) stops the motor, and shutdown sends 'M 0' + center.

A control timer re-sends the current angle at least every 0.5 s so the firmware's
~1 s watchdog never auto-centers while the node is idle. Serial errors are logged
and trigger a reconnect attempt (via ArduinoLink) rather than crashing the node.

Perception-death failsafe: the keepalive deliberately feeds the firmware watchdog,
so the firmware cannot detect a dead PUBLISHER (camera unplugged, detector crash)
-- only a dead Pi. If no /steering_cmd arrives for cmd_timeout_s while holding a
non-zero angle, this node decays the command to center itself.

Subscribes:  steering_cmd   std_msgs/Float32   (signed degrees, + = right)
             drive_cmd      std_msgs/Float32   (signed percent, + = forward)
Params:      port, baud, steer_dir, max_deg, rate_limit_dps, cmd_timeout_s,
             drive_max_pct, center_on_shutdown
"""
import json
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String

from .arduino_link import ArduinoLink

TICK_S = 0.05
KEEPALIVE_S = 0.5
CHANGE_DEG = 0.5

MANUAL_WINDOW_S = 30.0
MANUAL_DRIVE_DEADMAN_S = 5.0
RAW_ALLOWED = ("PING", "GET", "C", "S", "U", "TRIM", "LIM", "M")


class SteeringBridge(Node):
    def __init__(self):
        super().__init__("steering_bridge")

        self.declare_parameters("", [
            ("port", "auto"),
            ("baud", 115200),
            ("steer_dir", 1.0),
            ("max_deg", 25.0),
            ("rate_limit_dps", 0.0),
            ("cmd_timeout_s", 1.0),
            ("drive_max_pct", 40.0),
            ("center_on_shutdown", True),
        ])
        g = self.get_parameter
        self.steer_dir = float(g("steer_dir").value)
        self.max_deg = float(g("max_deg").value)
        self.rate_limit = float(g("rate_limit_dps").value)
        self.cmd_timeout = float(g("cmd_timeout_s").value)
        self.drive_max = float(g("drive_max_pct").value)
        self.center_on_shutdown = bool(g("center_on_shutdown").value)

        self.link = ArduinoLink(port=g("port").value, baud=int(g("baud").value))
        self._link_ok = False
        try:
            self.link.connect()
            self._link_ok = True
            self.get_logger().info(
                "steering fw: {} on {}".format(self.link.pong or "connected", self.link.port))
        except Exception as exc:
            self.get_logger().error("steering connect failed: {}; will retry".format(exc))

        self.cmd_deg = 0.0
        self.out_deg = 0.0
        self.last_sent = None
        self.last_send_time = 0.0
        self.last_cmd_time = None
        self.drive_cmd = 0.0
        self.last_drive_sent = None
        self.last_drive_send_time = 0.0
        self.last_drive_cmd_time = None
        self._nan_warn_time = -1e9
        self._last_tick = self._now()

        self.manual_until = 0.0
        self._manual_active = False
        self._manual_m_live = False
        self._last_raw_time = 0.0
        self.last_ok_time = None

        self.create_subscription(Float32, "steering_cmd", self.on_cmd, 10)
        self.create_subscription(Float32, "drive_cmd", self.on_drive, 10)
        self.create_subscription(String, "arduino_cmd", self.on_raw, 10)
        self.pub_raw = self.create_publisher(String, "arduino_reply", 10)
        self.pub_status = self.create_publisher(String, "bridge_status", 10)
        self.create_timer(TICK_S, self._on_timer)
        self.create_timer(1.0, self._pub_bridge_status)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _shape(self, raw_deg):
        """Apply the direction flip and clamp to +/- max_deg."""
        d = self.steer_dir * raw_deg
        return max(-self.max_deg, min(self.max_deg, d))

    def on_cmd(self, msg):
        val = float(msg.data)
        if not math.isfinite(val):
            now = self._now()
            if now - self._nan_warn_time >= 5.0:
                self.get_logger().error("non-finite steering_cmd rejected")
                self._nan_warn_time = now
            return
        self.cmd_deg = self._shape(val)
        self.last_cmd_time = self._now()

    def on_drive(self, msg):
        val = float(msg.data)
        if not math.isfinite(val):
            now = self._now()
            if now - self._nan_warn_time >= 5.0:
                self.get_logger().error("non-finite drive_cmd rejected")
                self._nan_warn_time = now
            return
        self.drive_cmd = max(-self.drive_max, min(self.drive_max, val))
        self.last_drive_cmd_time = self._now()

    def on_raw(self, msg):
        """Whitelisted raw firmware commands from the test console. Opens a
        manual window during which autonomy forwarding is suspended and the
        firmware watchdog is fed with PINGs (see _on_timer)."""
        text = msg.data.strip()
        if not text:
            return
        parts = text.split()
        verb = parts[0].upper()
        now = self._now()
        if verb == "RESUME":
            self.manual_until = 0.0
            self._exit_manual()
            self._reply("OK RESUME (autonomy re-engaged)")
            return
        if verb not in RAW_ALLOWED:
            self._reply("ERR verb '{}' not allowed".format(verb))
            return
        if verb == "M" and len(parts) >= 2:
            try:
                v = float(parts[1])
                if not math.isfinite(v):
                    self._reply("ERR M bad number")
                    return
                v = max(-self.drive_max, min(self.drive_max, v))
                parts[1] = "{:.0f}".format(v)
                self._manual_m_live = v != 0.0
            except ValueError:
                pass
        self.manual_until = now + MANUAL_WINDOW_S
        self._last_raw_time = now
        if not self._manual_active:
            self._manual_active = True
            try:
                self.link.send("M 0")
            except Exception:
                pass
            self.get_logger().info("manual window OPEN (drive stopped, autonomy suspended)")
        cmd = " ".join(parts)
        try:
            reply = self.link.send(cmd)
            self.last_ok_time = self._now()
            self._link_ok = True
        except Exception as exc:
            reply = "ERR link: {}".format(exc)
        self._reply("{} -> {}".format(cmd, reply))

    def _reply(self, text):
        m = String()
        m.data = text
        self.pub_raw.publish(m)

    def _exit_manual(self):
        """Leave manual mode safely: motor stop, center, watchdog restored."""
        if not self._manual_active:
            return
        self._manual_active = False
        self._manual_m_live = False
        for cmd in ("M 0", "C", "WD 1000"):
            try:
                self.link.send(cmd)
            except Exception:
                break
        self.get_logger().info("manual window CLOSED (M 0 + C + WD 1000)")

    def _pub_bridge_status(self):
        now = self._now()
        m = String()
        m.data = json.dumps({
            "connected": self._link_ok,
            "port": str(self.link.port),
            "manual": now < self.manual_until,
            "manual_left": round(max(0.0, self.manual_until - now), 1),
            "last_ok_age": None if self.last_ok_time is None
                           else round(now - self.last_ok_time, 1),
            "steer_out": self.out_deg,
            "drive_out": self.drive_cmd,
            "drive_max": self.drive_max,
        })
        self.pub_status.publish(m)

    def _on_timer(self):
        now = self._now()
        dt = now - self._last_tick
        self._last_tick = now

        if now < self.manual_until:
            if (self._manual_m_live
                    and (now - self._last_raw_time) >= MANUAL_DRIVE_DEADMAN_S):
                try:
                    self.link.send("M 0")
                    self.get_logger().warning("manual drive deadman -> M 0")
                except Exception:
                    pass
                self._manual_m_live = False
            if (now - self.last_send_time) >= KEEPALIVE_S:
                try:
                    self.link.send("PING")
                    self.last_ok_time = now
                    self._link_ok = True
                except Exception as exc:
                    if self._link_ok:
                        self.get_logger().error(
                            "manual keepalive error: {}".format(exc))
                        self._link_ok = False
                self.last_send_time = now
            return
        if self._manual_active:
            self._exit_manual()

        if (self.cmd_timeout > 0.0 and self.cmd_deg != 0.0
                and self.last_cmd_time is not None
                and (now - self.last_cmd_time) >= self.cmd_timeout):
            self.cmd_deg = 0.0
            self.get_logger().warning(
                "steering_cmd stale for {:.1f}s -> centering".format(self.cmd_timeout))

        if self.rate_limit > 0.0 and self.out_deg != self.cmd_deg:
            step = self.rate_limit * dt
            delta = self.cmd_deg - self.out_deg
            if abs(delta) <= step:
                self.out_deg = self.cmd_deg
            else:
                self.out_deg += step if delta > 0 else -step
        else:
            self.out_deg = self.cmd_deg

        changed = self.last_sent is None or abs(self.out_deg - self.last_sent) > CHANGE_DEG
        stale = (now - self.last_send_time) >= KEEPALIVE_S
        if changed or stale:
            self._send_steer(self.out_deg)

        if (self.cmd_timeout > 0.0 and self.drive_cmd != 0.0
                and self.last_drive_cmd_time is not None
                and (now - self.last_drive_cmd_time) >= self.cmd_timeout):
            self.drive_cmd = 0.0
            self.get_logger().warning(
                "drive_cmd stale for {:.1f}s -> motor stop".format(self.cmd_timeout))
        d_changed = (self.last_drive_sent is None
                     or abs(self.drive_cmd - self.last_drive_sent) > 1.0)
        d_refresh = (now - self.last_drive_send_time) >= 2.0
        if d_changed or d_refresh:
            self._send_drive(self.drive_cmd)

    def _send_steer(self, deg):
        try:
            self.link.steer(deg)
            self.last_sent = deg
            self.last_send_time = self._now()
            self.last_ok_time = self.last_send_time
            if not self._link_ok:
                self.get_logger().info("steering link recovered")
                self._link_ok = True
        except Exception as exc:
            self.last_send_time = self._now()
            if self._link_ok:
                self.get_logger().error("steering serial error: {}; retrying".format(exc))
                self._link_ok = False

    def _send_drive(self, pct):
        try:
            self.link.drive(pct)
            self.last_drive_sent = pct
            self.last_drive_send_time = self._now()
            self.last_ok_time = self.last_drive_send_time
            if not self._link_ok:
                self.get_logger().info("steering link recovered")
                self._link_ok = True
        except Exception as exc:
            self.last_drive_send_time = self._now()
            if self._link_ok:
                self.get_logger().error("drive serial error: {}; retrying".format(exc))
                self._link_ok = False

    def destroy_node(self):
        if self.link is not None:
            try:
                self.link.drive(0)
            except Exception as exc:
                self.get_logger().warning("drive stop on shutdown failed: {}".format(exc))
        if self.center_on_shutdown and self.link is not None:
            try:
                self.link.center()
            except Exception as exc:
                self.get_logger().warning("center on shutdown failed: {}".format(exc))
        if self.link is not None:
            try:
                self.link.close()
            except Exception:
                pass
        super().destroy_node()


def main():
    rclpy.init()
    node = SteeringBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
