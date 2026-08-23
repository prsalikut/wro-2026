"""State machine for the WRO Future Engineers open round: drive -> turn -> halt/backout.
Fuses ultrasonic range, lidar and IMU heading; each degrades independently."""
import json
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan, Range
from std_msgs.msg import Float32, String

TICK_S = 0.05


def _r(v):
    return round(v, 3) if v is not None else None


class OpenRound(Node):

    def __init__(self):
        super().__init__("open_round")
        self._declare_params()
        self._load_params()

        self.pub_steer = self.create_publisher(Float32, "steering_cmd", 10)
        self.pub_drive = self.create_publisher(Float32, "drive_cmd", 10)
        self.pub_status = self.create_publisher(String, "open_status", 10)
        self.create_subscription(
            LaserScan, "scan", self.on_scan, qos_profile_sensor_data)
        self.create_subscription(
            Imu, "imu/data", self.on_imu, qos_profile_sensor_data)
        self.create_subscription(String, "line_event", self.on_line, 10)
        self.create_subscription(Float32, "imu/yaw", self.on_yaw, 10)
        for side in ("front", "rear", "left", "right"):
            self.create_subscription(
                Range, "sonar/{}".format(side),
                lambda m, s=side: self.on_sonar(s, m), 10)

        self.front = None
        self.rear = None
        self.left = None
        self.right = None
        self.left_far = None
        self.right_far = None
        self._fhist = []
        self.last_scan = None
        self.front_lost_at = None

        self.state = "drive"
        self.halt_reason = None
        self.turn_dir = 0.0
        self.turn_started = 0.0
        self.below_since = None
        self.backout_started = 0.0
        self.boxed_until = 0.0
        self.turns = 0
        self.moving_since = None
        self.stuck_ref = None
        self.stuck_since = 0.0
        self.last_go = -1e9
        self.yaw_rate = None
        self.line_request = False
        self.last_line_corner = -1e9
        self.trigger_color = None
        self.yaw_deg = None
        self.yaw_time = -1e9
        self.turn_yaw0 = None
        self.hold_yaw = None
        self.sonar = {"front": None, "rear": None, "left": None, "right": None}
        self.sonar_t = {k: -1e9 for k in self.sonar}
        self.err_prev = None
        self.err_t = -1e9
        self.disagree = {"left": False, "right": False}
        self.t_start = self._now()
        self._last_log = 0.0
        self._last_key = None

        self.create_timer(TICK_S, self.on_tick)
        self.get_logger().info(
            "open_round up: brake={:.2f}m turn={:.2f}m clear={:.2f}m "
            "kick={:.0f}% for {:.1f}s then sustain {:.0f}%".format(
                self.brake, self.turn_at, self.clear,
                self.kick_pct, self.kick_s, self.sustain))

    def _declare_params(self):
        self.declare_parameters("", [
            ("angle_offset_deg", 170.0),    # lidar zero -> chassis forward
            ("angle_sign", 1.0),
            ("min_valid_m", 0.15),
            ("front_half_deg", 20.0),
            ("front_quorum", 2),
            ("front_median_n", 5),
            ("side_lo_deg", 30.0),
            ("side_hi_deg", 110.0),
            ("side_max_m", 2.5),

            ("brake_m", 0.70),
            ("brake_turn_m", 0.30),
            ("turn_m", 1.10),
            ("turn_confirm_s", 0.35),
            ("line_corner_cooldown_s", 3.0),
            ("turn_angle_deg", 87.0),
            ("heading_kp", 1.1),
            ("heading_max_deg", 16.0),
            ("imu_timeout_s", 0.5),
            ("sonar_timeout_s", 0.4),
            ("sonar_centre", True),
            ("sonar_front_guard", True),
            ("fuse_tol_m", 0.22),
            ("sonar_weight", 0.7),
            ("centre_kd", 9.0),
            ("stop_after_turns", 12),
            ("clear_m", 1.20),
            ("resume_m", 0.76),

            ("drive_pct", 45.0),            # 0 = steering-only dry run
            ("kick_safe_m", 1.30),          # full kick travels ~0.6 m
            ("kick_pct", 100.0),
            ("kick_s", 0.9),
            ("sustain_pct", 32.0),
            ("turn_sustain_pct", 38.0),
            ("restall_s", 1.2),
            ("reverse_enable", True),
            ("reverse_kick_pct", -100.0),
            ("reverse_kick_s", 1.0),
            ("reverse_sustain_pct", -40.0),
            ("reverse_max_s", 2.0),
            ("reverse_target_m", 0.76),     # must not exceed resume_m
            ("rear_half_deg", 25.0),
            ("rear_min_m", 0.40),
            ("boxed_cooldown_s", 5.0),

            ("max_steer_deg", 25.0),
            ("straight_trim_deg", 0.0),
            ("centre_kp_deg_per_m", 40.0),
            ("centre_max_deg", 14.0),
            ("single_target_m", 0.45),
            ("centre_valid_m", 0.85),       # beyond this, beams overshoot the wall
            ("centre_bias_m", 0.0),

            ("turn_ramp_s", 0.6),
            ("inside_min_m", 0.38),
            ("min_turn_s", 1.6),
            ("max_turn_s", 6.0),
            ("stuck_eps_m", 0.05),
            ("stuck_s", 2.5),
            ("max_run_s", 180.0),
            ("unknown_grace_s", 0.4),
            ("scan_timeout_s", 0.5),
        ])

    def _load_params(self):
        def g(k):
            return self.get_parameter(k).value

        self.angle_off = float(g("angle_offset_deg"))
        self.angle_sign = float(g("angle_sign"))
        self.min_valid = float(g("min_valid_m"))
        self.front_half = float(g("front_half_deg"))
        self.quorum = int(g("front_quorum"))
        self.med_n = max(1, int(g("front_median_n")))
        self.side_lo = float(g("side_lo_deg"))
        self.side_hi = float(g("side_hi_deg"))
        self.side_max = float(g("side_max_m"))
        self.brake = float(g("brake_m"))
        self.brake_turn = float(g("brake_turn_m"))
        self.turn_at = float(g("turn_m"))
        self.turn_confirm = float(g("turn_confirm_s"))
        self.line_cooldown = float(g("line_corner_cooldown_s"))
        self.turn_angle = float(g("turn_angle_deg"))
        self.head_kp = float(g("heading_kp"))
        self.head_max = float(g("heading_max_deg"))
        self.imu_timeout = float(g("imu_timeout_s"))
        self.sonar_timeout = float(g("sonar_timeout_s"))
        self.sonar_centre = bool(g("sonar_centre"))
        self.sonar_front_guard = bool(g("sonar_front_guard"))
        self.fuse_tol = float(g("fuse_tol_m"))
        self.sonar_w = float(g("sonar_weight"))
        self.c_kd = float(g("centre_kd"))
        self.stop_turns = int(g("stop_after_turns"))
        self.clear = float(g("clear_m"))
        self.resume = float(g("resume_m"))
        self.drive_pct = float(g("drive_pct"))
        self.kick_safe = float(g("kick_safe_m"))
        self.kick_pct = float(g("kick_pct"))
        self.kick_s = float(g("kick_s"))
        self.sustain = float(g("sustain_pct"))
        self.turn_sustain = float(g("turn_sustain_pct"))
        self.restall_s = float(g("restall_s"))
        self.rev_enable = bool(g("reverse_enable"))
        self.rev_kick = float(g("reverse_kick_pct"))
        self.rev_kick_s = float(g("reverse_kick_s"))
        self.rev_sustain = float(g("reverse_sustain_pct"))
        self.rev_max_s = float(g("reverse_max_s"))
        self.rev_target = float(g("reverse_target_m"))
        self.rear_half = float(g("rear_half_deg"))
        self.rear_min = float(g("rear_min_m"))
        self.boxed_cd = float(g("boxed_cooldown_s"))
        self.max_steer = float(g("max_steer_deg"))
        self.trim = float(g("straight_trim_deg"))
        self.c_kp = float(g("centre_kp_deg_per_m"))
        self.c_max = float(g("centre_max_deg"))
        self.c_target = float(g("single_target_m"))
        self.c_valid = float(g("centre_valid_m"))
        self.c_bias = float(g("centre_bias_m"))
        self.turn_ramp = float(g("turn_ramp_s"))
        self.inside_min = float(g("inside_min_m"))
        self.min_turn = float(g("min_turn_s"))
        self.max_turn = float(g("max_turn_s"))
        self.stuck_eps = float(g("stuck_eps_m"))
        self.stuck_s = float(g("stuck_s"))
        self.max_run = float(g("max_run_s"))
        self.unknown_grace = float(g("unknown_grace_s"))
        self.scan_timeout = float(g("scan_timeout_s"))

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_imu(self, m):
        self.yaw_rate = float(m.angular_velocity.z)

    def on_sonar(self, side, msg):
        self.sonar[side] = float(msg.range)
        self.sonar_t[side] = self._now()

    def sonar_get(self, side):
        if (self._now() - self.sonar_t[side]) > self.sonar_timeout:
            return None
        return self.sonar[side]

    def _fuse_side(self, side, lidar):
        """Combine sonar and lidar for one side.

        Ultrasound reads these walls under conditions where the lidar cannot,
        so it carries the larger weight, but the lidar still contributes when
        the two agree. A wide disagreement means one of them is wrong rather
        than both being noisy, and the sonar is the one that survives grazing
        incidence on a glossy surface."""
        sonar = self.sonar_get(side) if self.sonar_centre else None
        if lidar is not None and lidar > self.c_valid:
            lidar = None
        self.disagree[side] = False
        if sonar is not None and lidar is not None:
            if abs(sonar - lidar) <= self.fuse_tol:
                return self.sonar_w * sonar + (1.0 - self.sonar_w) * lidar
            self.disagree[side] = True
            return sonar
        if sonar is not None:
            return sonar
        return lidar

    def on_yaw(self, msg):
        self.yaw_deg = float(msg.data)
        self.yaw_time = self._now()

    def _imu_ok(self):
        return (self.yaw_deg is not None
                and (self._now() - self.yaw_time) <= self.imu_timeout)

    @staticmethod
    def _wrap(deg):
        return (deg + 180.0) % 360.0 - 180.0

    def _heading_hold(self):
        if not self._imu_ok() or self.hold_yaw is None:
            return 0.0
        err = self._wrap(self.hold_yaw - self.yaw_deg)
        return max(-self.head_max, min(self.head_max, -self.head_kp * err))

    def on_line(self, msg):
        if self.trigger_color is None:
            self.trigger_color = msg.data
            self.get_logger().info(
                "corner trigger color locked: {}".format(msg.data))
        if msg.data != self.trigger_color:
            return
        now = self._now()
        if (self.state == "drive"
                and now - self.last_line_corner >= self.line_cooldown):
            self.last_line_corner = now
            self.line_request = True
            self.get_logger().info("corner line: {}".format(msg.data))

    def _phi(self, alpha_rad):
        """Raw scan angle -> robot-frame bearing, deg. 0 = forward, + = left."""
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
        raw = min(f) if len(f) >= self.quorum else None
        self._fhist.append(raw)
        if len(self._fhist) > self.med_n:
            self._fhist.pop(0)
        vals = [v for v in self._fhist if v is not None]
        if len(vals) > len(self._fhist) // 2:
            vals.sort()
            self.front = vals[len(vals) // 2]
            self.front_lost_at = None
        else:
            self.front = None
            if self.front_lost_at is None:
                self.front_lost_at = now
        self.rear = min(b) if len(b) >= self.quorum else None
        self.left = min(l) if l else None
        self.right = min(r) if r else None
        self.left_far = max(l) if l else None
        self.right_far = max(r) if r else None
        self.last_scan = now

    def _front_is_clear(self, thresh=None):
        """No forward return counts as open only after unknown_grace_s."""
        thresh = self.clear if thresh is None else thresh
        if self.front is not None:
            return self.front > thresh
        if self.front_lost_at is None:
            return False
        return (self._now() - self.front_lost_at) >= self.unknown_grace

    def _go(self, now, sustain):
        """Kick to break stiction, then hold sustain; kick only with room ahead."""
        if self.drive_pct <= 0.0:
            return 0.0
        f = self.front
        room = (f is None) or (f > self.kick_safe)
        if self.moving_since is None or (now - self.last_go) > self.restall_s:
            if not room:
                self.last_go = now
                return sustain
            self.moving_since = now
        self.last_go = now
        if (now - self.moving_since) < self.kick_s:
            if not room:
                self.moving_since = now - self.kick_s
                self.get_logger().warning(
                    "kick aborted: front {} inside {:.2f}m".format(
                        _r(f), self.kick_safe))
                return sustain
            return self.kick_pct
        return sustain

    def _nudge(self):
        """Proportional lane centring on side ranges. + = steer RIGHT."""
        l = self._fuse_side("left", self.left)
        r = self._fuse_side("right", self.right)
        if l is not None and r is not None:
            err = r - l
        elif l is not None:
            err = self.c_target - l
        elif r is not None:
            err = r - self.c_target
        else:
            return 0.0
        err += self.c_bias * (-self.turn_dir)

        # Derivative of the centring error is a heading estimate the ranges can
        # supply on their own, so the car still damps its approach to a wall
        # when no IMU is fitted. With an IMU the two are complementary: this
        # reacts to lateral drift, heading hold holds the absolute bearing.
        now = self._now()
        dt = now - self.err_t
        rate = 0.0
        if self.err_prev is not None and 1e-3 < dt < 0.5:
            rate = (err - self.err_prev) / dt
        self.err_prev = err
        self.err_t = now

        steer = self.c_kp * err + self.c_kd * rate
        return max(-self.c_max, min(self.c_max, steer))

    def _halt(self, reason):
        if self.state != "halt":
            self.get_logger().error("HALT: {}".format(reason))
        self.state = "halt"
        self.halt_reason = reason
        self.moving_since = None
        return 0.0, 0.0, "halt"

    def _decide(self, now):
        f = self.front
        if self.sonar_front_guard:
            sf = self.sonar_get("front")
            # Nearest of the two wins: a sonar that sees a wall the lidar
            # missed must still be able to stop the car.
            if sf is not None:
                f = sf if f is None else min(f, sf)

        lim = self.brake_turn if self.state == "turn" else self.brake
        if (f is not None and f < lim
                and self.state not in ("halt", "backout")):
            return self._halt("front {:.2f}m < brake {:.2f}m".format(f, lim))

        if (now - self.t_start) > self.max_run:
            return self._halt("run exceeded {:.0f}s".format(self.max_run))

        if (self.state == "halt" and self.rev_enable and self.drive_pct > 0.0
                and now >= self.boxed_until):
            if f is not None and f < self.rev_target:
                self.state = "backout"
                self.backout_started = now
                self.get_logger().warning(
                    "reversing out: F={:.2f}m -> target {:.2f}m".format(
                        f, self.rev_target))

        if self.state == "backout":
            held = now - self.backout_started
            sr = self.sonar_get("rear")
            rear = self.rear if sr is None else (
                sr if self.rear is None else min(self.rear, sr))
            rear_blocked = rear is not None and rear < self.rear_min
            if rear_blocked:
                self.boxed_until = now + self.boxed_cd
                self.get_logger().warning(
                    "BOXED: front {} rear {:.2f}m -- holding {:.0f}s".format(
                        _r(f), rear, self.boxed_cd))
            if (rear_blocked or (f is None or f >= self.rev_target)
                    or held >= self.rev_max_s):
                self.get_logger().info(
                    "backout done after {:.1f}s (F={})".format(held, _r(f)))
                self.state = "halt"
                self.moving_since = None
                return 0.0, 0.0, "halt"
            pct = self.rev_kick if held < self.rev_kick_s else self.rev_sustain
            return 0.0, pct, "backout"

        if self.state == "halt":
            if self._front_is_clear(self.resume):
                self.get_logger().info("front clear -> resuming")
                self.state = "drive"
                self.halt_reason = None
                self.stuck_ref = None
                self.stuck_since = now
                self.moving_since = None
            else:
                return 0.0, 0.0, "halt"

        if self.state != "turn" and self.drive_pct > 0.0:
            sig = tuple(-1.0 if v is None else round(v / self.stuck_eps)
                        for v in (self.front, self.left, self.right))
            if self.stuck_ref != sig:
                self.stuck_ref = sig
                self.stuck_since = now
            elif (now - self.stuck_since) > self.stuck_s:
                return self._halt("no sector changed in {:.1f}s -- jammed"
                                  .format(now - self.stuck_since))
        else:
            self.stuck_ref = None
            self.stuck_since = now

        if self.state == "drive" and self.turns >= self.stop_turns:
            self.state = "done"
            self.get_logger().info(
                "course complete: {} corners".format(self.turns))
        if self.state == "done":
            return 0.0, 0.0, "done"

        if self.state == "drive":
            if f is None or f >= self.turn_at:
                self.below_since = None
            elif self.below_since is None:
                self.below_since = now

            confirmed = (self.below_since is not None
                         and (now - self.below_since) >= self.turn_confirm)
            if self.line_request:
                self.line_request = False
                confirmed = True
            if confirmed:
                if self.turn_dir == 0.0:
                    lv = self.left_far if self.left_far is not None else 0.0
                    rv = self.right_far if self.right_far is not None else 0.0
                    self.turn_dir = 1.0 if rv > lv else -1.0
                    self.get_logger().info(
                        "turn direction locked {} (far L={} R={})".format(
                            "RIGHT" if self.turn_dir > 0 else "LEFT",
                            _r(self.left_far), _r(self.right_far)))
                self.state = "turn"
                self.turn_started = now
                self.moving_since = None
                self.turn_yaw0 = self.yaw_deg if self._imu_ok() else None
                self.turns += 1
                self.get_logger().info(
                    "corner {} (F={})".format(self.turns, _r(f)))
                self.below_since = None
            else:
                if self.hold_yaw is None and self._imu_ok():
                    self.hold_yaw = self.yaw_deg
                return (self.trim + self._nudge() + self._heading_hold(),
                        self._go(now, self.sustain), "drive")

        if self.state == "turn":
            held = now - self.turn_started
            if held > self.max_turn:
                return self._halt("corner {} exceeded {:.0f}s".format(
                    self.turns, self.max_turn))
            swept = None
            if self.turn_yaw0 is not None and self._imu_ok():
                swept = -self.turn_dir * self._wrap(self.yaw_deg - self.turn_yaw0)
            done = (swept >= self.turn_angle if swept is not None
                    else self._front_is_clear())
            if held >= self.min_turn and done:
                if swept is not None:
                    self.get_logger().info(
                        "corner {} swept {:.0f} deg".format(self.turns, swept))
                self.hold_yaw = self.yaw_deg if self._imu_ok() else None
                self.state = "drive"
                self.moving_since = None
                self.stuck_ref = None
                return self.trim, self._go(now, self.sustain), "drive"
            frac = 1.0 if self.turn_ramp <= 0.0 else min(1.0, held / self.turn_ramp)
            inside = self.right if self.turn_dir > 0 else self.left
            if inside is not None and inside < self.inside_min:
                frac *= max(0.25, inside / self.inside_min)
            return (self.turn_dir * self.max_steer * frac,
                    self._go(now, self.turn_sustain), "turn")

        return self.trim, self._go(now, self.sustain), "drive"

    def on_tick(self):
        now = self._now()
        if self.last_scan is None or (now - self.last_scan) > self.scan_timeout:
            self.pub_steer.publish(Float32(data=0.0))
            self.pub_drive.publish(Float32(data=0.0))
            self._log("nolidar", "no /scan -> centre + stop")
            return

        deg, pct, state = self._decide(now)
        self.pub_steer.publish(Float32(data=float(deg)))
        self.pub_drive.publish(Float32(data=float(pct)))

        self.pub_status.publish(String(data=json.dumps({
            "state": state, "steer_deg": round(deg, 1), "drive_pct": pct,
            "front": _r(self.front), "left": _r(self.left),
            "right": _r(self.right), "rear": _r(self.rear), "turns": self.turns,
            "halt_reason": self.halt_reason,
            "turn_dir": ("right" if self.turn_dir > 0 else
                         "left" if self.turn_dir < 0 else None),
            "yaw_rate": _r(self.yaw_rate),
            "yaw": _r(self.yaw_deg), "hold_yaw": _r(self.hold_yaw),
            "sonar": {k: _r(self.sonar_get(k)) for k in self.sonar},
            "fused_l": _r(self._fuse_side("left", self.left)),
            "fused_r": _r(self._fuse_side("right", self.right)),
            "disagree": dict(self.disagree),
            "run_s": round(now - self.t_start, 1),
        })))
        self._log(state, "{} F={} L={} R={} turns={} -> {:+.1f}deg {:.0f}%".format(
            state, _r(self.front), _r(self.left), _r(self.right),
            self.turns, deg, pct))

    def _log(self, key, text):
        now = self._now()
        if key != self._last_key or (now - self._last_log) >= 1.0:
            self.get_logger().info(text)
            self._last_log = now
            self._last_key = key


def main():
    rclpy.init()
    node = OpenRound()
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
