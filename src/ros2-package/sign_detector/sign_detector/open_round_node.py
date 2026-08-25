"""ROS 2 node: open_round -- the WRO Future Engineers Open Challenge driver.

All of the driving logic lives in sign_detector.open_round_core, which has no
ROS dependency and is exercised closed-loop against a rendered camera feed by
tools/sim_offline.py.  This file is the plumbing: it turns topics into core
inputs, sectorises the lidar scan, gates the round on the START button, and
publishes what the core decided.

Subscribes
    vision/lane    std_msgs/String    camera lane geometry (wall_vision)
    scan           sensor_msgs/LaserScan
    sonar/{front,right,rear,left}     sensor_msgs/Range
    imu/data, imu/yaw                 optional
    start_status   std_msgs/String    the START/MODE buttons
Publishes
    steering_cmd   std_msgs/Float32   signed degrees, + = right
    drive_cmd      std_msgs/Float32   signed percent, + = forward
    open_status    std_msgs/String    JSON state for the dashboard

Start gating (rules 9.11 / 9.14): the car boots into a waiting state and only
drives once START has been pressed.  start_button launches this node on the
press, so a "running" status is normally already latched when we come up; if no
start_button node is publishing at all the node arms itself after
`start_timeout_s` so a bench run still works.
"""

import json
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan, Range
from std_msgs.msg import Float32, String

from .open_round_core import OpenRoundCore, OpenRoundParams

# 50 Hz. At 20 Hz the car covered 2.5 cm between decisions, and a wall that
# appears in one sample -- which is exactly what happens when a turn swings the
# forward sensors onto something they could not previously see -- got a single
# tick of reaction. Faster ticking does not make the sensors see sooner, but it
# stops the controller adding its own delay on top.
TICK_S = 0.02
SONARS = ("front", "right", "rear", "left")

# Parameters forwarded straight to the control core.  Declared here so the whole
# thing is tunable from params.yaml without touching code.
CORE_PARAMS = (
    "vision_trust", "sonar_trust", "lidar_trust", "agree_tol_m",
    "sonar_freeze_n", "max_steer_deg", "trim_deg", "centre_kp", "centre_kd",
    "centre_max_deg", "heading_kp", "heading_max_deg", "single_target_m",
    "target_min_m", "target_max_m", "wall_min_m", "side_ignore_m",
    "turn_at_m", "turn_confirm_s", "turn_confirm_slow_s", "front_wall_tol_m",
    "min_corner_travel_m", "line_arm_m", "corner_lockout_s",
    "corner_bias_from_m", "corner_bias_m", "corner_bias_min_m",
    "turn_angle_deg", "turn_ramp_s", "min_turn_s", "max_turn_s",
    "turn_exit_front_m", "turn_exit_open_deg", "turn_dr_floor_deg",
    "turn_max_sweep_deg", "stop_after_turns", "recover_front_m",
    "recover_rear_min_m", "recover_max_s", "max_recoveries", "drive_pct",
    "kick_pct", "kick_s", "sustain_pct", "turn_sustain_pct", "min_move_pct",
    "slow_front_m", "slow_factor", "brake_m", "brake_turn_m", "resume_m",
    "backout_enable", "rear_min_m", "stuck_s", "max_run_s", "finish_enable",
    "finish_tol_m", "finish_max_s", "wheelbase_m", "speed_per_pct",
    "stiction_pct", "direction_margin_m", "pulse_enable", "pulse_period_s",
    "pulse_on_s", "tight_lane_m", "outer_min_m", "corner_min_side_m",
    "turn_inside_min_m", "turn_inside_stop_m", "kick_after_still_s", "map_enable", "map_keep_s", "map_keep_m",
    "map_arc_margin_m", "map_turn_radius_m", "map_use_lidar", "blind_floor_frac", "speed_cal_alpha",
    "corner_source", "corner_asym_m", "corner_asym_s", "line_trigger_v", "line_arm_v", "use_vision_ranges", "use_lidar_sides",
    "imu_sign_init", "imu_sign_learn", "tight_recoveries", "line_see_v", "pulse_straights", "line_grace_s", "wall_push_deg", "speed_tau_s", "min_kick_s",
    "kick_speed_mps", "restall_s", "turn_confirm_slow_s",
)


class OpenRoundNode(Node):

    def __init__(self):
        super().__init__("open_round")
        self.declare_parameters("", [
            # lidar geometry -- angle_offset_deg defines where "forward" is and
            # every steering decision depends on it being right.
            ("angle_offset_deg", 170.0),
            ("angle_sign", 1.0),
            ("min_valid_m", 0.15),
            ("front_half_deg", 20.0),
            ("front_quorum", 2),
            ("rear_half_deg", 25.0),
            ("side_lo_deg", 30.0),
            ("side_hi_deg", 110.0),
            ("side_max_m", 2.5),
            ("scan_timeout_s", 0.6),
            ("tick_hz", 50.0),
            # start gating
            ("require_start", True),
            ("start_timeout_s", 4.0),
            ("mode_name", "open"),
            ("use_vision", True),
            ("use_lidar", True),
            ("use_sonar", True),
        ])
        defaults = OpenRoundParams()
        self.declare_parameters(
            "", [(k, getattr(defaults, k)) for k in CORE_PARAMS])

        def g(k):
            return self.get_parameter(k).value

        self.angle_off = float(g("angle_offset_deg"))
        self.angle_sign = float(g("angle_sign"))
        self.min_valid = float(g("min_valid_m"))
        self.front_half = float(g("front_half_deg"))
        self.quorum = int(g("front_quorum"))
        self.rear_half = float(g("rear_half_deg"))
        self.side_lo = float(g("side_lo_deg"))
        self.side_hi = float(g("side_hi_deg"))
        self.side_max = float(g("side_max_m"))
        self.scan_timeout = float(g("scan_timeout_s"))
        self.require_start = bool(g("require_start"))
        self.start_timeout = float(g("start_timeout_s"))
        self.mode_name = str(g("mode_name"))
        self.use_vision = bool(g("use_vision"))
        self.use_lidar = bool(g("use_lidar"))
        self.use_sonar = bool(g("use_sonar"))

        params = OpenRoundParams(**{k: g(k) for k in CORE_PARAMS})
        self.core = OpenRoundCore(params, log=self._core_log)

        self.pub_steer = self.create_publisher(Float32, "steering_cmd", 10)
        self.pub_drive = self.create_publisher(Float32, "drive_cmd", 10)
        self.pub_status = self.create_publisher(String, "open_status", 10)

        if self.use_vision:
            self.create_subscription(String, "vision/lane", self.on_vision, 10)
        if self.use_lidar:
            self.create_subscription(LaserScan, "scan", self.on_scan,
                                     qos_profile_sensor_data)
        if self.use_sonar:
            for side in SONARS:
                self.create_subscription(
                    Range, "sonar/{}".format(side),
                    lambda m, s=side: self.on_sonar(s, m), 10)
        self.create_subscription(Imu, "imu/data", self.on_imu,
                                 qos_profile_sensor_data)
        self.create_subscription(Float32, "imu/yaw", self.on_yaw, 10)
        self.create_subscription(String, "start_status", self.on_start, 10)

        self.boot_t = self._now()
        self.last_scan = -1e9
        self.start_seen = False
        self.start_state = None
        self.start_mode = None
        self._armed_note = False
        self.create_timer(1.0 / max(5.0, float(g("tick_hz"))), self.on_tick)
        self.get_logger().info(
            "open_round up: {}vision {}lidar {}sonar; {}".format(
                "" if self.use_vision else "NO ",
                "" if self.use_lidar else "NO ",
                "" if self.use_sonar else "NO ",
                "waiting for START" if self.require_start else "self-arming"))

    # ------------------------------------------------------------------ util

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _core_log(self, level, msg):
        """Forward a core message at the right severity.

        Deliberately three separate statements: rclpy caches severity per CALL
        SITE, so routing every level through one getattr() line makes the
        second message at a different level raise "Logger severity cannot be
        changed between calls" and kill the node mid-round.
        """
        log = self.get_logger()
        if level == "error":
            log.error(msg)
        elif level == "warn":
            log.warning(msg)
        else:
            log.info(msg)

    # ---------------------------------------------------------------- inputs

    def on_vision(self, msg):
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        self.core.on_vision(data, self._now())

    def _phi(self, alpha_rad):
        """Raw scan angle -> car-frame bearing in degrees, 0 ahead, + left."""
        phi = (math.degrees(alpha_rad) - self.angle_off) / self.angle_sign
        return (phi + 180.0) % 360.0 - 180.0

    def on_scan(self, msg):
        lo = max(msg.range_min, self.min_valid)
        f, l, r, b = [], [], [], []
        for i, rng in enumerate(msg.ranges):
            if not math.isfinite(rng) or rng < lo:
                continue
            phi = self._phi(msg.angle_min + i * msg.angle_increment)
            if abs(phi) <= self.front_half:
                f.append(rng)
            elif abs(phi) >= (180.0 - self.rear_half):
                b.append(rng)
            elif self.side_lo <= phi <= self.side_hi and rng <= self.side_max:
                l.append(rng)
            elif -self.side_hi <= phi <= -self.side_lo and rng <= self.side_max:
                r.append(rng)
        now = self._now()
        self.last_scan = now
        # Hand the whole sweep to the map as well: sectors are enough to drive
        # on, but the map needs the shape.
        pts = []
        for i, rng in enumerate(msg.ranges):
            if not math.isfinite(rng) or rng < lo:
                continue
            pts.append((self._phi(msg.angle_min + i * msg.angle_increment), rng))
        self.core.on_scan_points(pts, now)
        self.core.on_scan(
            min(f) if len(f) >= self.quorum else None,
            min(l) if l else None,
            min(r) if r else None,
            min(b) if len(b) >= self.quorum else None,
            now,
            left_far=max(l) if l else None,
            right_far=max(r) if r else None)

    def on_sonar(self, side, msg):
        self.core.on_sonar(side, float(msg.range), self._now())

    def on_imu(self, msg):
        self.core.on_imu(rate=float(msg.angular_velocity.z), now=self._now())

    def on_yaw(self, msg):
        self.core.on_imu(yaw_deg=float(msg.data), now=self._now())

    def on_start(self, msg):
        self.start_seen = True
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        self.start_state = data.get("state")
        self.start_mode = data.get("mode")

    # ------------------------------------------------------------------ tick

    def _should_arm(self, now):
        if not self.require_start:
            return True
        if self.start_seen:
            if self.start_state == "no_gpio":
                return True          # no button fitted: nothing can ever press it
            if self.start_mode is not None and self.start_mode != self.mode_name:
                return False
            return self.start_state == "running"
        # Nothing is publishing start_status at all -- there is no start_button
        # node in this graph, so waiting for it would wait for ever.
        return (now - self.boot_t) >= self.start_timeout

    def on_tick(self):
        now = self._now()
        arm = self._should_arm(now)
        if arm and not self._armed_note:
            self._armed_note = True
            self.get_logger().info(
                "armed ({})".format("START pressed" if self.start_seen
                                    else "no start_button on this graph"))
        self.core.set_armed(arm, now)

        deg, pct, status = self.core.step(now)
        self.pub_steer.publish(Float32(data=float(deg)))
        self.pub_drive.publish(Float32(data=float(pct)))
        status["scan_age"] = round(now - self.last_scan, 2) \
            if self.last_scan > 0 else None
        status["start"] = {"seen": self.start_seen, "state": self.start_state,
                           "mode": self.start_mode}
        self.pub_status.publish(String(data=json.dumps(status)))


def main():
    rclpy.init()
    node = OpenRoundNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if rclpy.ok():
                node.pub_steer.publish(Float32(data=0.0))
                node.pub_drive.publish(Float32(data=0.0))
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
