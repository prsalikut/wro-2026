"""Open Challenge driver, as plain Python: no ROS, no OpenCV, no hardware.

Three laps of the WRO Future Engineers track, with the camera as the primary
sensor.  Everything that decides where the car goes lives here so it can be run
against a simulator on a laptop; open_round_node.py is only the ROS plumbing
that feeds it and forwards what it returns.

What the camera buys us, over the range-only driver it replaces:

  * a distance to the outer wall that survives grazing incidence and a latched
    ultrasonic, because it comes from where the wall visibly meets the mat;
  * a heading relative to the corridor, from the slope of that same wall base,
    which is a yaw reference the car otherwise only has when an IMU is fitted;
  * the orange and blue corner lines with a RANGE attached, so a corner is timed
    from a fixed mark painted on the mat rather than guessed from a wall
    distance that depends on how wide this round's corridors happen to be.

Geometry worth knowing before reading the corner code: this chassis steers 25
degrees on a 0.15 m wheelbase, so its tightest arc is 0.32 m.  A right-angle
corner between two 1.0 m corridors needs 0.42 m or less and is comfortable; the
rules also allow 0.6 m corridors, which need 0.19 m and are therefore
*impossible* in a single arc for this car.  So a corner that runs out of room
reverses and takes a second bite, which is a three-point turn and is legal
(rule 9.21 permits reversing within the section where it happened).
"""

import math

from .local_map import LocalMap
from .range_fusion import AXES, RangeFusion, SourceSpec

# Where the four ultrasonics sit on the car: (forward m, left m, bearing deg).
SONAR_MOUNTS = {
    "front": (0.10, 0.0, 0.0),
    "right": (0.02, -0.07, -90.0),
    "rear": (-0.08, 0.0, 180.0),
    "left": (0.02, 0.07, 90.0),
}

__all__ = ["OpenRoundParams", "OpenRoundCore"]

STATES = ("wait", "drive", "turn", "recover", "finish", "done", "halt",
          "backout")


def _r(v, n=3):
    return None if v is None else round(float(v), n)


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def _wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


class OpenRoundParams(object):

    def __init__(self, **kw):
        # --- fusion --------------------------------------------------------
        self.vision_trust = 1.3
        self.sonar_trust = 1.0
        self.lidar_trust = 0.7
        self.vision_timeout_s = 0.45
        self.sonar_timeout_s = 0.45
        self.lidar_timeout_s = 0.50
        self.agree_tol_m = 0.22
        self.sonar_freeze_n = 10
        self.vision_freeze_n = 0        # a fit never repeats bit-for-bit
        self.lidar_freeze_n = 0

        # --- lane keeping ---------------------------------------------------
        self.max_steer_deg = 25.0
        self.trim_deg = 0.0
        self.centre_kp = 42.0           # deg per metre of side error
        self.centre_kd = 8.0            # deg per (m/s) of error rate
        self.centre_max_deg = 15.0
        self.heading_kp = 0.60          # deg of steer per deg of yaw error
        self.heading_max_deg = 12.0
        self.vision_heading_timeout_s = 0.4
        self.single_target_m = 0.30     # hold-off when only one wall is seen
        self.target_min_m = 0.26
        self.target_max_m = 0.45
        self.lane_width_alpha = 0.1
        self.wall_min_m = 0.18          # never let any wall get closer
        self.wall_push_deg = 20.0
        self.outer_min_m = 0.22         # rule 9.18: never touch the OUTER wall
        self.outer_push_deg = 18.0
        self.side_ignore_m = 1.15       # past this the "side" is open space,
                                        # not a wall to centre against
        self.side_hit_alpha = 0.05      # how fast "which side has a wall"
        self.evidence_alpha = 0.06      # ... and the open-space evidence decay
        self.direction_margin_m = 0.12  # how much more room one side needs
                                        # before it counts as "the way out"
        self.outer_hit_min = 0.12       # evidence needed to name the outer wall
        self.outer_hit_gap = 0.07

        # --- corner ---------------------------------------------------------
        self.turn_at_m = 0.62           # front distance that starts a corner
        self.turn_confirm_s = 0.12
        self.turn_confirm_slow_s = 0.55  # ... when no transverse wall is seen
        self.front_wall_tol_m = 0.22     # fitted wall must match the range
        self.min_corner_travel_m = 0.75  # corners are never closer than this
        self.corner_min_side_m = 0.15    # hugging a wall? then a short front
                                         # range is that wall, not a corner
        self.line_arm_m = 0.60          # corner line this close = corner soon
        self.line_fallback_m = 0.16     # ... and this close with no front fix
        # The printed corner lines are the primary turn trigger: they are a
        # fixed mark on the mat, found by colour alone, so unlike a wall
        # distance they do not depend on the camera mounting being calibrated
        # or on how wide this round's corridors happen to be.
        self.corner_source = "sides"    # "sides" | "front"
        self.corner_asym_m = 0.30       # how far the two sides must diverge
        self.corner_asym_s = 0.10       # ... and for how long
        self.line_trigger_v = 0.72      # line this far down the frame = turn now
        self.line_arm_v = 0.45          # ... and this far down = corner ahead
        self.line_see_v = 0.12          # merely visible: enough to fix direction
        self.line_grace_s = 2.5         # no line for this long: trust the front
        self.line_lost_grace_s = 0.35
        self.use_vision_ranges = True   # feed camera distances into the fusion
        self.use_lidar_sides = False    # lidar for the front only; the sides
                                        # are the ultrasonics' job
        self.corner_lockout_s = 2.2
        self.corner_bias_from_m = 1.15  # start easing outward here
        self.corner_bias_m = 0.11       # metres to give up toward the outside
        self.corner_bias_min_m = 0.22   # but never closer than this to a wall
        self.turn_angle_deg = 84.0
        self.turn_ramp_s = 0.25
        self.turn_inside_min_m = 0.30   # widen the arc rather than clip the
                                        # inner corner on the way out
        self.turn_inside_stop_m = 0.20  # ... and below this, stop turning
        self.min_turn_s = 0.7
        self.max_turn_s = 7.0
        self.turn_exit_front_m = 0.80   # corridor is open again
        self.turn_exit_align_deg = 14.0  # camera says we are square to it
        self.turn_exit_open_deg = 11.0   # the open bearing is straight ahead
        self.turn_min_sweep_deg = 52.0
        self.turn_dr_floor_deg = 56.0   # dead reckoning is a floor, not a gate
        self.turn_exit_stable_n = 2     # optical exit must hold two frames
        self.turn_max_sweep_deg = 102.0
        self.stop_after_turns = 12

        # --- three-point corner ---------------------------------------------
        self.recover_front_m = 0.34     # too close mid-corner: back up
        self.recover_rear_min_m = 0.24
        self.recover_max_s = 1.6
        self.recover_pct = -55.0
        self.recover_kick_pct = -100.0
        self.recover_kick_s = 0.35
        self.max_recoveries = 4
        self.tight_recoveries = 10      # a 0.5 m corner is a parking manoeuvre

        # --- drive ----------------------------------------------------------
        self.drive_pct = 45.0           # 0 = steering-only dry run
        self.kick_pct = 100.0
        self.kick_s = 0.9
        self.kick_safe_m = 1.30
        self.sustain_pct = 32.0
        self.turn_sustain_pct = 38.0
        self.restall_s = 1.2
        self.slow_front_m = 0.90        # ease off when something is close
        self.slow_factor = 0.85
        self.min_move_pct = 34.0        # below this the drivetrain just stalls
        self.kick_speed_mps = 1.0       # how fast a full-duty kick travels
        self.kick_stop_margin_m = 0.35  # room to leave in front of a kick
        self.min_kick_s = 0.16
        self.stall_eps_m = 0.035
        self.kick_after_still_s = 0.5   # only kick a car that is really stopped
        self.wheelbase_m = 0.15         # for the dead-reckoned yaw
        self.speed_per_pct = 0.0111
        self.stiction_pct = 30.0
        # This drivetrain does not move below about 30% duty, so it has no slow
        # speed to drop into for a tight corner.  Pulsing the drive gives one:
        # short bursts above stiction separated by coasting, which roughly
        # halves the average speed and so doubles the number of decisions the
        # controller gets per metre.
        self.tight_lane_m = 0.78        # corridor narrower than this is "tight"
        self.pulse_enable = True
        self.pulse_period_s = 0.50
        self.pulse_on_s = 0.18
        self.pulse_straights = False    # pulse on the straights too, not just
                                        # in corners
        self.speed_tau_s = 0.15         # drivetrain lag; dr is wrong without it
        self.speed_cal_alpha = 0.05     # learn the real m/s per percent
        self.speed_cal_min = 0.45
        self.speed_cal_max = 2.2
        self.imu_sign_learn = True      # work out which way the IMU counts
        # The BNO055 in NDOF reports a COMPASS heading, which grows clockwise,
        # while everything here treats counter-clockwise as positive. Start
        # from -1 for that part rather than making every corner wait for the
        # learner to gather evidence.
        self.imu_sign_init = -1.0

        # --- safety ---------------------------------------------------------
        self.brake_m = 0.24
        self.brake_turn_m = 0.20
        self.resume_m = 0.42
        self.backout_enable = True
        self.backout_target_m = 0.42
        self.backout_max_s = 1.8
        self.rear_min_m = 0.26
        self.boxed_cooldown_s = 4.0
        self.stuck_eps_m = 0.04
        self.stuck_s = 3.0
        self.max_run_s = 172.0           # rounds are 3 minutes (rule 9.1)
        self.sensor_timeout_s = 0.8      # all three sources quiet = stop
        # A short-lived map of where walls were, built from the ultrasonics and
        # the IMU. It exists because the forward sensors cannot see the inside
        # of a turn until the car is already in it.
        self.map_enable = True
        self.map_keep_s = 6.0
        self.map_keep_m = 2.5
        self.map_arc_margin_m = 0.10     # refuse an arc tighter than this
        self.map_turn_radius_m = 0.9     # the arc the car actually drives
        self.map_use_lidar = True        # lidar feeds the MAP (not steering):
                                         # it is the only sensor that sees
                                         # ahead-and-to-the-side
        self.blind_floor_frac = 0.05     # less mat than this: view is blocked
        self.blind_front_m = 0.20

        # --- finish -----------------------------------------------------------
        self.finish_enable = True
        self.finish_tol_m = 0.12
        self.finish_min_s = 0.6
        self.finish_max_s = 9.0
        self.finish_hold_front_m = 0.55

        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError("unknown OpenRoundParams field %r" % (k,))
            setattr(self, k, type(getattr(self, k))(v))


class OpenRoundCore(object):

    def __init__(self, params=None, log=None):
        self.p = params or OpenRoundParams()
        self._log_fn = log or (lambda level, msg: None)
        p = self.p
        self.fusion = RangeFusion([
            SourceSpec("vision", p.vision_trust, p.vision_timeout_s,
                       0.05, 3.0, p.vision_freeze_n),
            SourceSpec("sonar", p.sonar_trust, p.sonar_timeout_s,
                       0.03, 3.5, p.sonar_freeze_n),
            SourceSpec("lidar", p.lidar_trust, p.lidar_timeout_s,
                       0.05, 6.0, p.lidar_freeze_n),
        ], agree_tol_m=p.agree_tol_m)

        self.map = LocalMap(keep_s=p.map_keep_s, keep_m=p.map_keep_m)
        self._map_odo = 0.0
        self.armed = False
        self.state = "wait"
        self.reason = None
        self.t0 = 0.0
        self.turns = 0
        self.turn_dir = 0.0             # +1 = right, -1 = left
        self.outer_side = None          # 'left' | 'right'
        self.trigger_colour = None
        self.front_at_start = None

        self._vision = {}
        self._vision_t = -1e9
        self._heading = None
        self._heading_t = -1e9
        self._open_bearing = None
        self._open_bearing_t = -1e9
        self._lines = {}
        self._lines_t = -1e9
        self._side_hits = {"left": 0.0, "right": 0.0}
        self._open = {"left": None, "right": None}
        self._near = {"left": None, "right": None}
        self._lane_w = None
        self._yaw = None
        self._yaw_t = -1e9
        self._yaw_rate = None
        self._imu_sign = float(self.p.imu_sign_init)
        self._imu_vote = 0.0
        self._imu_prev = None

        self._below_since = None
        self._asym_since = None
        self._corner_near = False
        self._line_corners = 0
        self._line_seen_t = -1e9
        self._turn_started = 0.0
        self._turn_yaw0 = None
        self._turn_dr0 = None
        self._turn_head0 = None
        self._last_corner_t = -1e9
        self._recover_started = 0.0
        self._turn_elapsed = 0.0
        self._recoveries = 0
        self._hold_yaw = None
        self._err_prev = None
        self._err_t = -1e9
        self._moving_since = None
        self._kick_until = -1e9
        self._pulse_t0 = 0.0
        self._last_go = -1e9
        self._dr_yaw = 0.0
        self._odo = 0.0
        self._last_corner_odo = -9.9
        self._v_est = 0.0
        self._speed_scale = 1.0
        self._cal_front = None
        self._cal_t = -1e9
        self._yaw_prev = None
        self._turn_swept = 0.0
        self._exit_votes = 0
        self._turn_dr0 = None
        self._last_step_t = None
        self._last_sweep_log = 0.0
        self._prev_tick_t = 0.0
        self._last_pct = 0.0
        self._progress_ref = None
        self._progress_t = 0.0
        self._last_motion_t = -1e9
        self._stuck_sig = None
        self._stuck_since = 0.0
        self._backout_started = 0.0
        self._boxed_until = 0.0
        self._finish_started = 0.0
        self._last_seen = -1e9
        self._notes = []

    # ------------------------------------------------------------------ log

    def _say(self, level, msg):
        self._notes.append(msg)
        del self._notes[:-12]
        self._log_fn(level, msg)

    # ---------------------------------------------------------------- inputs

    def set_armed(self, armed, now):
        armed = bool(armed)
        if armed == self.armed:
            return
        self.armed = armed
        if armed:
            self._begin(now)
        else:
            self.state = "wait"
            self.reason = "disarmed"
            self._say("info", "disarmed -> holding")

    def _begin(self, now):
        self.t0 = now
        self.turns = 0
        self.turn_dir = 0.0
        self.outer_side = None
        self.trigger_colour = None
        self.front_at_start = None
        self.state = "drive"
        self.reason = None
        self._recoveries = 0
        self._corner_near = False
        self._line_corners = 0
        self._last_corner_t = -1e9      # the first corner may come at once
        self._last_corner_odo = -9.9
        self._stuck_since = now
        self._moving_since = None
        self._side_hits = {"left": 0.0, "right": 0.0}
        self._say("info", "armed: open round starting")

    def on_vision(self, v, now):
        """`v` is WallVisionResult.as_dict() (or the object itself)."""
        if hasattr(v, "as_dict"):
            v = v.as_dict()
        p_blind_frac = self.p.blind_floor_frac
        self._vision = v or {}
        self._vision_t = now
        self._last_seen = now
        blind = (not self._vision.get("ok")
                 and (self._vision.get("floor_frac") or 1.0) < p_blind_frac)
        for axis, key in (("left", "left"), ("right", "right"),
                          ("front", "front")):
            if not self.p.use_vision_ranges:
                # The camera is still worth having for the corner lines and for
                # heading (a wall's SLOPE is scale free, so it survives a bad
                # height), but its absolute distances are only as good as the
                # mounting numbers. When those are not trusted, keep them out
                # of the fusion rather than let them outvote the lidar.
                continue
            value = self._vision.get(key)
            if axis == "front" and blind and value is None:
                # No floor anywhere in frame is not "no information": it is a
                # wall filling the view.  Reporting nothing here would let the
                # car drive on into whatever it is looking at.
                value = self.p.blind_front_m
            self.fusion.update("vision", axis, value, now)
        ob = self._vision.get("open_bearing")
        if ob is not None:
            self._open_bearing = float(ob)
            self._open_bearing_t = now
        hd = self._vision.get("heading")
        if hd is not None:
            self._heading = float(hd)
            self._heading_t = now
        for side in ("left", "right"):
            self._blend("_open", side, self._vision.get("open_" + side))
        lw = self._vision.get("lane_width")
        if lw is not None:
            self._note_lane_width(float(lw))
        lines = self._vision.get("lines") or {}
        if lines:
            self._lines = lines
            self._lines_t = now

    def on_scan_points(self, bearings_ranges, now):
        """Full lidar sweep, for the MAP only -- never for steering.

        A side ultrasonic maps the wall it is abeam of, which is behind the car
        by the time the car could turn into it. Nothing else on this vehicle
        sees ahead-and-to-the-side, so without the lidar the map cannot answer
        the only question worth asking: what is inside the arc I am about to
        drive. It stays out of the centring, where it reads these glossy walls
        long or not at all.
        """
        if not self.p.map_enable or not self.p.map_use_lidar:
            return
        self.map.add_polar(now, bearings_ranges, max_range=self.p.map_keep_m)

    def on_scan(self, front, left, right, rear, now, left_far=None,
                right_far=None):
        """Lidar in. By default it contributes to the FRONT only.

        Where the car sits between two walls is an ultrasonic measurement on
        this vehicle: the lidar reads these glossy side walls at grazing
        incidence and returns long or nothing, and letting those numbers into
        the centring is what had the car riding one wall. The lidar earns its
        place looking forward, where it is a narrow accurate beam and the
        ultrasonic cone is not.
        """
        self._last_seen = now
        axes = (("front", front), ("rear", rear))
        if self.p.use_lidar_sides:
            axes = axes + (("left", left), ("right", right))
        for axis, val in axes:
            self.fusion.update("lidar", axis, val, now)
        self._left_far = left_far
        self._right_far = right_far
        if self.p.use_lidar_sides and left is not None and right is not None:
            self._note_lane_width(left + right)

    def on_sonar(self, name, value, now):
        if name not in AXES:
            return
        self._last_seen = now
        self.fusion.update("sonar", name, value, now)

    def on_imu(self, yaw_deg=None, rate=None, now=0.0):
        if yaw_deg is not None:
            self._yaw = float(yaw_deg)
            self._yaw_t = now
        if rate is not None:
            self._yaw_rate = float(rate)

    def _blend(self, attr, key, value):
        if value is None:
            return
        d = getattr(self, attr)
        a = self.p.evidence_alpha
        cur = d[key]
        d[key] = float(value) if cur is None else (1 - a) * cur + a * float(value)

    def _note_lane_width(self, w):
        if not (0.35 <= w <= 1.6):
            return
        a = self.p.lane_width_alpha
        self._lane_w = w if self._lane_w is None else (1 - a) * self._lane_w + a * w

    # ------------------------------------------------------------- accessors

    def _vision_fresh(self, now):
        return (bool(self._vision)
                and (now - self._vision_t) <= self.p.vision_timeout_s)

    def _imu_ok(self, now):
        return self._yaw is not None and (now - self._yaw_t) <= 0.5

    def _heading_ok(self, now):
        return (self._heading is not None
                and (now - self._heading_t) <= self.p.vision_heading_timeout_s)

    def _line(self, colour, now):
        if (now - self._lines_t) > self.p.vision_timeout_s:
            return None
        info = self._lines.get(colour)
        return None if info is None else info.get("distance_m")

    def _nearest_line(self, now):
        """(colour, distance) of the closest corner line, ignoring magenta."""
        if (now - self._lines_t) > self.p.vision_timeout_s:
            return None, None
        best, bestd = None, None
        for colour in ("orange", "blue"):
            info = self._lines.get(colour)
            if info is None:
                continue
            d = info.get("distance_m")
            if d is None:
                continue
            if bestd is None or d < bestd:
                best, bestd = colour, d
        return best, bestd

    def _line_underfoot(self, now):
        """(colour, v_frac) of the line nearest the bottom of the frame.

        Purely an image measurement: no camera geometry involved.
        """
        if (now - self._lines_t) > self.p.vision_timeout_s:
            return None, None
        best, bestv = None, None
        for colour in ("orange", "blue"):
            info = self._lines.get(colour)
            if info is None:
                continue
            v = info.get("v_frac")
            if v is None:
                continue
            if bestv is None or v > bestv:
                best, bestv = colour, v
        return best, bestv

    def _side(self, which, now):
        """Side range, or None when what we can see is open space.

        Past the end of the inner block a side reading jumps to the far wall
        three metres away.  Centring on that would swing the car straight into
        the wall it can see, so a reading that cannot be a corridor wall is not
        a reading at all.
        """
        v = self.fusion.get(which, now).value
        if v is None or v > self.p.side_ignore_m:
            return None
        return v

    def _side_raw(self, which, now):
        """Side range with no "too far to be a wall" clamp.

        _side() throws away anything past side_ignore_m because centring on a
        wall three metres away steers into the near one. But that discarded
        long reading is exactly what says a corner is here, so corner detection
        reads the raw value.
        """
        return self.fusion.get(which, now).value

    def _target(self):
        """How far to hold off the one wall we can see."""
        if self._lane_w is None:
            return self.p.single_target_m
        return _clamp(self._lane_w * 0.5, self.p.target_min_m, self.p.target_max_m)

    def _outer(self):
        """Which side the outer wall is on, best guess.

        Once the direction is locked the answer is arithmetic -- the outer wall
        is opposite the way we turn.  Before that, it is the side the camera
        keeps finding a wall on, because the outer wall runs the whole 3 m side
        of the field while the inner block is only a metre long and drops out of
        a 60-degree field of view from mid-corridor.
        """
        if self.turn_dir > 0:
            return "left"
        if self.turn_dir < 0:
            return "right"
        l, r = self._side_hits["left"], self._side_hits["right"]
        if max(l, r) < self.p.outer_hit_min or abs(l - r) < self.p.outer_hit_gap:
            return None
        return "left" if l > r else "right"

    # ------------------------------------------------------------- steering

    def _corner_shift(self, now, front):
        """How far to sit toward the OUTSIDE of the corridor, in metres.

        The tightest arc this chassis can turn is 0.32 m, and a right-angle
        corner only leaves room for that if the car enters it from the outside
        of the corridor.  The shift ramps in over the approach instead of
        snapping, so the car is already placed when the corner arrives rather
        than swerving into it.
        """
        p = self.p
        if front is None or self._outer() is None:
            return 0.0
        span = max(0.05, p.corner_bias_from_m - p.turn_at_m)
        ramp = _clamp((p.corner_bias_from_m - front) / span, 0.0, 1.0)
        if self._corner_near:
            ramp = max(ramp, 0.75)
        shift = p.corner_bias_m * ramp
        half = (self._lane_w * 0.5) if self._lane_w else p.single_target_m
        return max(0.0, min(shift, half - p.corner_bias_min_m))

    def _centre_steer(self, now, shift=0.0):
        """Proportional lane keeping.  Positive result steers right."""
        p = self.p
        left = self._side("left", now)
        right = self._side("right", now)
        outer = self._outer()
        sign = 1.0 if outer == "right" else (-1.0 if outer == "left" else 0.0)
        target = self._target()

        if left is not None and right is not None:
            err = (right - left) + 2.0 * shift * sign
        elif left is not None:
            # sign is +1 when the outer wall is on the right, so a positive
            # shift means we want to sit FURTHER from the left (inner) wall.
            err = 2.0 * ((target + shift * sign) - left)
        elif right is not None:
            err = 2.0 * (right - (target + shift * sign))
        else:
            return 0.0, None

        dt = now - self._err_t
        rate = 0.0
        if self._err_prev is not None and 1e-3 < dt < 0.5:
            rate = (err - self._err_prev) / dt
        self._err_prev, self._err_t = err, now

        # With a camera heading available the derivative term is redundant and
        # noisy, so it only runs as the fallback it was written to be.
        kd = 0.0 if self._heading_ok(now) else p.centre_kd
        steer = 0.5 * p.centre_kp * err + kd * rate
        return _clamp(steer, -p.centre_max_deg, p.centre_max_deg), err

    def _heading_steer(self, now):
        p = self.p
        if self._heading_ok(now):
            # heading > 0 means the nose points left of the corridor: steer right
            return _clamp(p.heading_kp * self._heading, -p.heading_max_deg,
                          p.heading_max_deg)
        if self._imu_ok(now) and self._hold_yaw is not None:
            err = _wrap(self._hold_yaw - self._yaw)
            return _clamp(-p.heading_kp * err, -p.heading_max_deg,
                          p.heading_max_deg)
        return 0.0

    def _wall_push(self, now):
        """Last-ditch shove away from a wall that is too close."""
        p = self.p
        left = self._side("left", now)
        right = self._side("right", now)
        push = 0.0
        if left is not None and left < p.wall_min_m:
            push += p.wall_push_deg * (p.wall_min_m - left) / p.wall_min_m
        if right is not None and right < p.wall_min_m:
            push -= p.wall_push_deg * (p.wall_min_m - right) / p.wall_min_m
        return push

    # ---------------------------------------------------------------- drive

    def _tight(self):
        """Is this corridor too narrow to corner in one arc?

        At 25 degrees on a 0.15 m wheelbase the tightest arc is 0.32 m; a
        right-angle corner between 0.7 m corridors already needs 0.25 m and a
        0.6 m one needs 0.19 m.  Below `tight_lane_m` the corner is a
        multi-point manoeuvre, not a curve, and the car has to be driven
        accordingly.
        """
        return self._lane_w is not None and self._lane_w < self.p.tight_lane_m

    def _pulse(self, now, pct):
        """Duty-cycle the drive so the car can go slower than stiction allows."""
        p = self.p
        if not p.pulse_enable or pct == 0.0 or p.pulse_period_s <= 0.0:
            return pct
        phase = (now - self._pulse_t0) % p.pulse_period_s
        return pct if phase < p.pulse_on_s else 0.0

    def _outer_guard(self, now):
        """Hard shove away from the outer wall.

        Rule 9.18 singles it out: in the open rounds the car may not touch the
        outer boundary wall at all, while brushing the inner block is tolerated
        as long as it does not move.  So the outside gets its own margin,
        wider than the general one.
        """
        p = self.p
        outer = self._outer()
        if outer is None:
            return 0.0
        d = self._side(outer, now)
        if d is None or d >= p.outer_min_m:
            return 0.0
        away = -1.0 if outer == "right" else 1.0
        return away * p.outer_push_deg * (p.outer_min_m - d) / p.outer_min_m

    def _go(self, now, sustain, front):
        """Duty for this tick.

        The drivetrain does not move at all below about 30% and needs a burst of
        full duty to break stiction from rest, so two things must never happen:
        commanding a "gentle" duty that is really zero, and refusing to kick
        because the wall ahead is close -- which is exactly when the car is
        stopped and needs to move.  The kick is therefore shortened to fit the
        room instead of being skipped.
        """
        p = self.p
        if p.drive_pct <= 0.0:
            return 0.0
        sustain = max(float(sustain), p.min_move_pct)
        room = 9.9 if front is None else max(0.0, front - p.kick_stop_margin_m)

        # Kick only when the car is genuinely at rest. This used to trigger on
        # `_moving_since is None`, which every state change reset -- so
        # entering a corner fired a full-duty kick at a car already doing half
        # a metre a second, and it went into the turn at 100%. The kick is for
        # breaking stiction from standstill, so ask the world whether the car
        # is actually stopped.
        standing = (now - self._last_motion_t) >= p.kick_after_still_s
        if self._moving_since is None:
            self._moving_since = now
            self._pulse_t0 = now
            if standing:
                length = _clamp(room / max(0.1, p.kick_speed_mps),
                                p.min_kick_s, p.kick_s)
                self._kick_until = now + (length if room > 0.02 else 0.0)
            else:
                self._kick_until = 0.0
        self._last_go = now
        if now < self._kick_until:
            return p.kick_pct
        if front is not None and front < p.slow_front_m:
            return max(sustain * p.slow_factor, p.min_move_pct)
        return sustain

    def _update_map(self, now):
        """Fold the latest ultrasonic sweep into the local map."""
        if not self.p.map_enable:
            return
        if self._imu_ok(now):
            self.map.set_heading(self._imu_sign * self._yaw)
        else:
            self.map.set_heading(self._dr_yaw)
        step = self._odo - self._map_odo
        if step > 0.0:
            self._map_odo = self._odo
            self.map.advance(step if self._last_pct >= 0 else -step)
        readings = {}
        for name in ("front", "left", "right", "rear"):
            c = self.fusion.chan.get(("sonar", name))
            if (c is not None and c.healthy and c.value is not None
                    and (now - c.stamp) <= self.p.sonar_timeout_s):
                readings[name] = c.value
        if readings:
            self.map.add(now, SONAR_MOUNTS, readings)

    def _arc_is_clear(self, turn_dir):
        """Would an arc this way run the body into anything remembered?"""
        if not self.p.map_enable or turn_dir == 0.0:
            return True, None
        gap = self.map.clearance_along_arc(turn_dir, self.p.map_turn_radius_m)
        if gap is None:
            return True, None
        return gap >= self.p.map_arc_margin_m, gap

    def _note_sides(self, now):
        """Track which side keeps showing a corridor wall.

        This used to be fed only from the camera, which meant that turning the
        camera's ranges off also silently destroyed the evidence the direction
        vote runs on. Feeding it from the FUSED ranges keeps it working on
        whatever sensors are actually in play: the outer wall runs the whole
        three metres, so it is present on nearly every tick, while the inner
        block is a metre long and keeps dropping out.
        """
        a = self.p.side_hit_alpha
        for side in ("left", "right"):
            v = self._side(side, now)
            self._side_hits[side] = ((1.0 - a) * self._side_hits[side]
                                     + a * (1.0 if v is not None else 0.0))
            if v is not None:
                self._blend("_near", side, v)

    def _note_progress(self, now, pct):
        """Re-kick when the wheels are turning but the world is not moving."""
        p = self.p
        vals = tuple(self.fusion.get(a, now).value
                     for a in ("front", "left", "right"))
        if self._progress_ref is None:
            self._progress_ref, self._progress_t = vals, now
            return
        moved = any(a is not None and b is not None
                    and abs(a - b) > p.stall_eps_m
                    for a, b in zip(vals, self._progress_ref))
        if moved:
            self._last_motion_t = now
        if moved or abs(pct) < 1.0:
            self._progress_ref, self._progress_t = vals, now
            return
        if (now - self._progress_t) > p.restall_s:
            self._progress_ref, self._progress_t = vals, now
            self._moving_since = None            # forces a fresh kick
            self._say("warn", "no progress at %.0f%% -> re-kicking" % pct)

    def _speed_est(self, pct, dt=0.0):
        """Low-passed speed estimate.

        The motor does not reach the commanded speed instantly, and treating a
        -100% reverse pulse as an instant 1.1 m/s made the dead-reckoned corner
        angle read nearly double the truth.
        """
        target = 0.0 if abs(pct) < self.p.stiction_pct \
            else self.p.speed_per_pct * self._speed_scale * pct
        if dt > 0.0:
            a = _clamp(dt / max(1e-3, self.p.speed_tau_s), 0.0, 1.0)
            self._v_est += (target - self._v_est) * a
        return self._v_est

    def _calibrate_speed(self, now, deg, pct, front):
        """Learn metres per percent from how fast the wall ahead approaches.

        Dead reckoning is only as good as the speed model, and no bench number
        survives a different battery charge or carpet.  Driving straight at a
        wall makes the answer observable: the front range falls at exactly the
        car's speed, and every sensor on the car measures it.
        """
        p = self.p
        if (self.state != "drive" or abs(deg) > 6.0
                or abs(pct) < p.stiction_pct or front is None):
            self._cal_front, self._cal_t = front, now
            return
        dt = now - self._cal_t
        if self._cal_front is None or not (0.08 < dt < 0.6):
            self._cal_front, self._cal_t = front, now
            return
        v_obs = (self._cal_front - front) / dt
        self._cal_front, self._cal_t = front, now
        nominal = p.speed_per_pct * pct
        if v_obs <= 0.02 or nominal <= 0.02:
            return
        ratio = _clamp(v_obs / nominal, p.speed_cal_min, p.speed_cal_max)
        a = p.speed_cal_alpha
        self._speed_scale = _clamp((1 - a) * self._speed_scale + a * ratio,
                                   p.speed_cal_min, p.speed_cal_max)

    def _integrate_yaw(self, now, deg, pct):
        """Dead-reckoned heading, so a corner still knows how far it has come
        when no IMU is fitted.  Good for one corner, not for a lap."""
        if self._last_step_t is None:
            self._last_step_t = now
            return
        dt = _clamp(now - self._last_step_t, 0.0, 0.2)
        self._last_step_t = now
        v = self._speed_est(pct, dt)
        self._odo += abs(v) * dt
        d_dr = 0.0
        if dt > 0.0 and v != 0.0:
            # positive steer is right, which decreases a counter-clockwise yaw
            d_dr = -math.degrees(
                (v / max(0.02, self.p.wheelbase_m))
                * math.tan(math.radians(deg)) * dt)
            self._dr_yaw += d_dr
        self._learn_imu_sign(now, d_dr)
        yaw = self._yaw_estimate(now)
        if self._yaw_prev is not None and self.state in ("turn", "recover"):
            self._turn_swept += -self.turn_dir * _wrap(yaw - self._yaw_prev)
        self._yaw_prev = yaw

    # ------------------------------------------------------------------ tick

    def step(self, now):
        p = self.p
        if not self.armed or self.state in ("wait", "done"):
            if self.state == "done":
                return 0.0, 0.0, self._status(now, 0.0, 0.0)
            return 0.0, 0.0, self._status(now, 0.0, 0.0)

        if (now - self._last_seen) > p.sensor_timeout_s:
            self.reason = "no sensor data for %.1fs" % (now - self._last_seen)
            return 0.0, 0.0, self._status(now, 0.0, 0.0)

        self._prev_tick_t = getattr(self, "_tick_t", now)
        self._tick_t = now
        self.fusion.set_moving(abs(self._last_pct) > 1.0)
        deg, pct = self._decide(now)
        deg = _clamp(deg, -p.max_steer_deg, p.max_steer_deg)
        self._note_sides(now)
        self._update_map(now)
        self._calibrate_speed(now, deg, pct, self.fusion.get("front", now).value)
        self._integrate_yaw(now, deg, pct)
        self._note_progress(now, pct)
        self._last_pct = pct
        return deg, pct, self._status(now, deg, pct)

    def _front(self, now):
        """Consensus front range, and the nearest single healthy reading.

        Decisions use the consensus so one optimistic sensor cannot drive the
        car into a wall and one pessimistic sensor cannot stop the round; the
        brake uses the nearest, because a wall only one sensor can see is still
        a wall.
        """
        f = self.fusion.get("front", now)
        n = self.fusion.nearest("front", now)
        return f.value, n.value

    def _halt(self, reason, now):
        if self.state != "halt":
            self._say("error", "HALT: %s" % reason)
        self.state = "halt"
        self.reason = reason
        self._moving_since = None
        return 0.0, 0.0

    def _decide(self, now):
        p = self.p
        front, front_min = self._front(now)

        if self.front_at_start is None and front is not None:
            self.front_at_start = front
            self._say("info", "start front reference %.2f m" % front)

        if (now - self.t0) > p.max_run_s:
            self.state = "done"
            self.reason = "round time expired"
            self._say("warn", "3 minutes up -> stop")
            return 0.0, 0.0

        limit = p.brake_turn_m if self.state in ("turn", "recover") else p.brake_m
        if (front_min is not None and front_min < limit
                and self.state not in ("halt", "backout", "recover", "done")):
            return self._halt("front %.2f m under %.2f m" % (front_min, limit), now)

        if self.state == "halt":
            return self._do_halt(now, front, front_min)
        if self.state == "backout":
            return self._do_backout(now, front_min)
        if self.state == "recover":
            return self._do_recover(now, front, front_min)
        if self.state == "turn":
            return self._do_turn(now, front, front_min)
        if self.state == "finish":
            return self._do_finish(now, front)
        return self._do_drive(now, front, front_min)

    # ---------------------------------------------------------------- states

    def _do_drive(self, now, front, front_min):
        p = self.p

        if self.turns >= p.stop_after_turns:
            if p.finish_enable:
                self.state = "finish"
                self._finish_started = now
                self._say("info", "12 corners done -> running out to the "
                                  "start section")
            else:
                self.state = "done"
                self.reason = "three laps complete"
            return 0.0, 0.0

        if not self._check_stuck(now):
            return self._halt("nothing changed for %.1fs -- jammed" % p.stuck_s,
                              now)

        colour, line_d = self._nearest_line(now)
        lcolour, lv = self._line_underfoot(now)
        # The corner lines sit on the boundary between the straight and the
        # corner section, so seeing one at range means "the corner starts
        # here". Crossing one -- the line reaching the bottom of the frame --
        # is the event worth turning on.
        if lv is not None and lv >= self.p.line_see_v:
            self._line_seen_t = now
            if self.trigger_colour is None:
                self.trigger_colour = lcolour
                self._say("info", "corner line colour locked: %s (used to "
                                  "count corners, not to pick a direction)"
                          % lcolour)
        if lv is not None and lv >= self.p.line_arm_v:
            if lcolour == self.trigger_colour and not self._corner_near:
                self._line_corners += 1
                self._corner_near = True

        by_line_img = (lv is not None and lv >= p.line_trigger_v
                       and (self.trigger_colour is None
                            or lcolour == self.trigger_colour))

        wall = self._vision.get("front_wall") if self._vision_fresh(now) else None
        square = (wall is not None and front is not None
                  and abs(wall - front) <= p.front_wall_tol_m)
        corroborated = len(self.fusion.get("front", now).used) >= 2
        need = p.turn_confirm_s \
            if (square or corroborated or not self._vision_fresh(now)) \
            else p.turn_confirm_slow_s
        if front is None or front >= p.turn_at_m:
            self._below_since = None
        elif self._below_since is None:
            self._below_since = now
        by_front = (self._below_since is not None
                    and (now - self._below_since) >= need)

        # A corner announces itself as the two side ultrasonics diverging:
        # running down a corridor both read short and roughly equal (S and S),
        # and at the corner one opens up (S and L). That transition is the
        # trigger, and the long side is the way out. It needs no line, no
        # colour convention and no camera calibration -- just the two sensors
        # that are already looking down both walls.
        lraw = self._side_raw("left", now)
        rraw = self._side_raw("right", now)
        asym = False
        if lraw is not None and rraw is not None:
            asym = abs(lraw - rraw) >= p.corner_asym_m
        elif (lraw is None) != (rraw is None):
            asym = True          # one side sees nothing at all: wide open
        if asym:
            if self._asym_since is None:
                self._asym_since = now
        else:
            self._asym_since = None
        by_asym = (self._asym_since is not None
                   and (now - self._asym_since) >= p.corner_asym_s)

        src = p.corner_source
        if src == "front":
            trigger = by_front
        else:
            # Front range stays as a backstop for a corner the sides somehow
            # miss; the line is kept only to count corners.
            trigger = by_asym or by_front

        sides = [v for v in (self._side("left", now), self._side("right", now))
                 if v is not None]
        centred = (not sides) or min(sides) >= p.corner_min_side_m or square
        by_line = by_line_img

        far_enough = (self._odo - self._last_corner_odo) >= p.min_corner_travel_m
        if (centred and trigger
                and (now - self._last_corner_t) >= p.corner_lockout_s
                and far_enough):
            self._start_turn(now, front, colour, line_d, by_line)
            return self._do_turn(now, front, front_min)

        steer, _err = self._centre_steer(now, shift=self._corner_shift(now, front))
        steer += (self._heading_steer(now) + self._wall_push(now)
                  + self._outer_guard(now) + p.trim_deg)
        # Pulsing the straights is not about width, it is about reaction time.
        # This motor will not turn below about 60% duty, which is 1.0 m/s, and
        # at that speed the car needs roughly 0.4 m to stop once a sensor has
        # seen something -- further than it can usefully see. Pulsing halves
        # the average speed and so halves the stopping distance, which is the
        # difference between braking and hitting.
        pct = self._go(now, p.sustain_pct, front)
        if p.pulse_straights:
            pct = self._pulse(now, pct)
        return steer, pct

    def _start_turn(self, now, front, colour, line_d, by_line):
        p = self.p
        # Decide at EVERY threshold, not just the first. Crossing the line is
        # the moment the question can be answered: the corridor continues on
        # exactly one side and the ultrasonics are looking straight down both.
        # Deciding once and locking it means one bad reading at the first
        # corner is carried round the whole round.
        fresh = self._choose_direction(now)
        if fresh != 0.0:
            if self.turn_dir != 0.0 and fresh != self.turn_dir:
                self._say("warn", "corner %d: more room on the %s now -- "
                                  "turning that way instead"
                          % (self.turns + 1, "right" if fresh > 0 else "left"))
            self.turn_dir = fresh
        elif self.turn_dir == 0.0:
            self.turn_dir = -1.0
            self._say("warn", "no clear side at this corner: assuming LEFT")
        if self.trigger_colour is None and colour is not None and by_line:
            self.trigger_colour = colour
            self._say("info", "corner line colour locked: %s" % colour)
        self.state = "turn"
        self._turn_started = now
        self._last_corner_t = now
        self._last_corner_odo = self._odo
        self._turn_yaw0 = self._yaw if self._imu_ok(now) else None
        self._turn_dr0 = self._dr_yaw
        self._turn_swept = 0.0
        self._exit_votes = 0
        self._yaw_prev = self._yaw_estimate(now)
        self._turn_head0 = self._heading if self._heading_ok(now) else None
        self._recoveries = 0
        self._moving_since = None
        self.turns += 1
        self._say("info", "corner %d (%s) front=%s line=%s%s" % (
            self.turns, "RIGHT" if self.turn_dir > 0 else "LEFT",
            _r(front), _r(line_d), " by-line" if by_line else " by-front"))

    def _choose_direction(self, now):
        """Which way to turn: toward whichever side actually has room.

        No colour convention. An earlier version asserted that orange-first
        meant clockwise, from the mat artwork, and hard-coded the turn from it;
        on this track that produced a car that worked anticlockwise and drove
        into the wall clockwise. The mat may well follow that convention, but
        the car does not need to care -- at a corner the corridor continues on
        exactly one side, and the ultrasonics can see which.

        The line's job is to say WHEN the corner is; the ranges say WHICH WAY.
        """
        left = self._side_raw("left", now)
        right = self._side_raw("right", now)
        # A side that reads nothing at all is the most open of all.
        if (left is None) != (right is None):
            d = 1.0 if left is not None else -1.0
            self._say("info", "turning %s: nothing at all on that side"
                      % ("RIGHT" if d > 0 else "LEFT"))
            return d
        # Prefer the ultrasonics: they are what actually measure these walls.
        sl = self.fusion.chan.get(("sonar", "left"))
        sr = self.fusion.chan.get(("sonar", "right"))
        for c, name in ((sl, "left"), (sr, "right")):
            if c is not None and c.healthy and c.value is not None \
                    and (now - c.stamp) <= self.p.sonar_timeout_s:
                if name == "left":
                    left = c.value
                else:
                    right = c.value

        # Before trusting the ranges, ask the map whether the arc is drivable.
        # The side ultrasonic sees a gap; the map knows whether the wall just
        # past it is in the way.
        okr, gr = self._arc_is_clear(1.0)
        okl, gl = self._arc_is_clear(-1.0)
        if okr != okl and (gr is not None or gl is not None):
            d = 1.0 if okr else -1.0
            self._say("info", "turning %s: the map says the other way is "
                              "blocked (%.2f m vs %.2f m of clearance)"
                      % ("RIGHT" if d > 0 else "LEFT",
                         gr if gr is not None else 9.9,
                         gl if gl is not None else 9.9))
            return d

        if left is not None and right is not None:
            if abs(right - left) >= self.p.direction_margin_m:
                d = 1.0 if right > left else -1.0
                self._say("info", "turning %s: %.2f m of room that side "
                                  "against %.2f m the other"
                          % ("RIGHT" if d > 0 else "LEFT",
                             max(left, right), min(left, right)))
                return d
            self._say("info", "both sides similar (%.2f / %.2f); waiting for a "
                              "clearer view" % (left, right))
            return 0.0
        if left is None and right is not None:
            self._say("info", "turning LEFT: nothing on the left, %.2f m on "
                              "the right" % right)
            return -1.0
        if right is None and left is not None:
            self._say("info", "turning RIGHT: nothing on the right, %.2f m on "
                              "the left" % left)
            return 1.0
        lf = getattr(self, "_left_far", None)
        rf = getattr(self, "_right_far", None)
        if lf is not None and rf is not None and abs(lf - rf) > 0.15:
            d = 1.0 if rf > lf else -1.0
            self._say("info", "turning %s on the lidar's longer side"
                      % ("RIGHT" if d > 0 else "LEFT"))
            return d
        return 0.0

    def _do_turn(self, now, front, front_min):
        p = self.p
        held = now - self._turn_started
        if held > p.max_turn_s:
            return self._halt("corner %d took over %.0fs" % (self.turns,
                                                             p.max_turn_s), now)

        swept = self._swept(now)
        trusted_sweep = self._imu_ok(now)
        front_free = self._vision.get("front_free")
        front_ok = ((front is not None and front >= p.turn_exit_front_m)
                    or (front_free is not None
                        and front_free >= p.turn_exit_front_m))
        # The corner is finished when the way out is straight ahead, which the
        # camera measures directly.
        open_ok = (self._open_bearing is not None
                   and (now - self._open_bearing_t) <= p.vision_timeout_s
                   and abs(self._open_bearing) <= p.turn_exit_open_deg)
        # The camera's own heading is the best exit cue there is: it says the
        # car is square to the NEW corridor, which is the actual goal, rather
        # than that it has swept some nominal number of degrees.
        aligned = (self._heading_ok(now)
                   and abs(self._heading) <= p.turn_exit_align_deg
                   and self._side("left", now) is not None
                   or self._heading_ok(now)
                   and abs(self._heading) <= p.turn_exit_align_deg
                   and self._side("right", now) is not None)
        optical = (open_ok or aligned) and front_ok
        self._exit_votes = (self._exit_votes + 1) if optical else 0
        done = False
        if trusted_sweep:
            # With a working IMU the swept angle is a direct measurement of the
            # thing the corner is trying to achieve, so nothing else gets to end
            # the turn early. Letting an optical "looks clear ahead" cue do it
            # cut corners at 57 degrees and left the car aimed at the wall it
            # had just turned away from.
            done = swept >= p.turn_angle_deg
        elif self._exit_votes >= p.turn_exit_stable_n \
                and swept >= p.turn_dr_floor_deg:
            done = True
        elif front_ok and swept >= p.turn_angle_deg:
            done = True
        # Trace the sweep so an over-rotation can be diagnosed rather than
        # guessed at: if the numbers jump, the control tick is being starved.
        if now - self._last_sweep_log >= 0.4:
            self._last_sweep_log = now
            self._say("info", "  turn %d: swept %.0f deg, held %.1fs, "
                              "front %s, dt %.0f ms"
                      % (self.turns, swept or 0.0, held, _r(front),
                         1000.0 * (now - self._prev_tick_t)))
        if swept is not None and swept >= p.turn_max_sweep_deg:
            # An over-rotation guard that a minimum-duration gate can veto is
            # not a guard: it exits regardless of how long the corner has run.
            self._end_turn(now, swept)
            return self._do_drive(now, front, front_min)
        if held >= p.min_turn_s and done:
            self._end_turn(now, swept)
            return self._do_drive(now, front, front_min)

        # Out of room: back up and take a second bite.  A 0.6 m corridor cannot
        # be cornered in one arc by this chassis, so this is the normal path
        # there, not an error.
        budget = p.tight_recoveries if self._tight() else p.max_recoveries
        recover_at = p.recover_front_m
        if self._lane_w is not None:
            recover_at = _clamp(self._lane_w * 0.55, 0.26, p.recover_front_m)
        if (front_min is not None and front_min <= recover_at
                and self._recoveries < budget):
            rear = self.fusion.get("rear", now).value
            if rear is None or rear > p.recover_rear_min_m:
                self.state = "recover"
                self._turn_elapsed = held
                self._recover_started = now
                self._recoveries += 1
                self._say("info", "corner %d: %.2f m ahead, backing up (%d)"
                          % (self.turns, front_min, self._recoveries))
                return 0.0, 0.0

        frac = 1.0 if p.turn_ramp_s <= 0 else min(1.0, held / p.turn_ramp_s)
        # While turning, the forward sensors point where the car is AIMED, not
        # where its curving path is GOING -- so a wall on the inside of the turn
        # is invisible until the car has already swung into it. Every impact in
        # the last run looked like that: the front range sat near a metre and
        # then read 6 cm in one sample. The side ultrasonic on the inside of the
        # turn CAN see it, so it gets a veto.
        clear, gap = self._arc_is_clear(self.turn_dir)
        if not clear:
            self._say("warn", "corner %d: the map puts a wall %.2f m into this "
                              "arc -- straightening" % (self.turns, gap))
            return 0.0, 0.0
        inside = self._side("right" if self.turn_dir > 0 else "left", now)
        if inside is not None and inside < p.turn_inside_stop_m:
            self._say("warn", "corner %d: %.2f m on the inside -- straightening"
                      % (self.turns, inside))
            return 0.0, 0.0
        if inside is not None and inside < p.turn_inside_min_m:
            frac *= max(0.30, inside / p.turn_inside_min_m)
        steer = self.turn_dir * p.max_steer_deg * frac
        pct = self._go(now, p.turn_sustain_pct, front)
        if self._tight():
            pct = self._pulse(now, pct)
        return steer, pct

    def _swept(self, now):
        """Degrees turned so far this corner, or None if unknowable.

        Accumulated tick by tick rather than differenced against the angle the
        corner started at: a single difference has to be wrapped into
        +/-180, and an over-rotating corner then reads as a small NEGATIVE
        sweep and never satisfies its own exit test.
        """
        return self._turn_swept

    def _yaw_estimate(self, now):
        return (self._imu_sign * self._yaw) if self._imu_ok(now) \
            else self._dr_yaw

    def _learn_imu_sign(self, now, d_dr):
        """Decide whether the IMU counts yaw the same way we do.

        Some parts report a compass heading, which grows CLOCKWISE, and the
        rest of this file treats counter-clockwise as positive.  Get that
        backwards and the corner's own sweep runs negative, so the corner never
        satisfies its exit test and the car spins until the timeout.  Rather
        than make it a parameter nobody can check, compare the IMU against dead
        reckoning while turning and take the majority verdict.
        """
        if not self.p.imu_sign_learn or not self._imu_ok(now):
            self._imu_prev = None
            return
        if self._imu_prev is None:
            self._imu_prev = self._yaw
            return
        d_imu = _wrap(self._yaw - self._imu_prev)
        self._imu_prev = self._yaw
        if abs(d_imu) < 0.25 or abs(d_dr) < 0.25:
            return
        agree = 1.0 if (d_imu > 0) == (d_dr > 0) else -1.0
        self._imu_vote = _clamp(self._imu_vote + agree, -30.0, 30.0)
        want = 1.0 if self._imu_vote >= 0 else -1.0
        if want != self._imu_sign and abs(self._imu_vote) >= 12.0:
            self._imu_sign = want
            self._say("warn", "IMU yaw runs the opposite way: flipping sign")

    def _end_turn(self, now, swept):
        self.state = "drive"
        self._below_since = None
        self._corner_near = False
        self._moving_since = None
        self._stuck_sig = None
        self._err_prev = None
        self._hold_yaw = self._yaw if self._imu_ok(now) else None
        self._last_corner_t = now
        self._last_corner_odo = self._odo
        self._say("info", "corner %d out%s" % (
            self.turns, "" if swept is None else " (%.0f deg)" % swept))

    def _do_recover(self, now, front, front_min):
        p = self.p
        held = now - self._recover_started
        swept = self._swept(now)
        if swept is not None and swept >= p.turn_angle_deg:
            self._end_turn(now, swept)
            return self._do_drive(now, front, front_min)
        rear = self.fusion.get("rear", now).value
        blocked = rear is not None and rear < p.recover_rear_min_m
        clear = front_min is None or front_min > (p.recover_front_m + 0.16)
        if blocked or clear or held >= p.recover_max_s:
            self.state = "turn"
            self._turn_started = now - self._turn_elapsed
            self._say("info", "corner %d: resuming the turn (rear=%s)"
                      % (self.turns, _r(rear)))
            return 0.0, 0.0
        pct = p.recover_kick_pct if held < p.recover_kick_s else p.recover_pct
        if self._tight():
            pct = self._pulse(now, pct)
        # Counter-steer while reversing: it swings the nose further round the
        # corner, so the next forward bite starts from a better angle.
        return -self.turn_dir * p.max_steer_deg, pct

    def _do_halt(self, now, front, front_min):
        p = self.p
        if (p.backout_enable and p.drive_pct > 0.0 and now >= self._boxed_until
                and front_min is not None and front_min < p.backout_target_m):
            self.state = "backout"
            self._backout_started = now
            self._say("warn", "reversing out of %.2f m" % front_min)
            return 0.0, 0.0
        # Resume on the NEAREST healthy reading, not the consensus. The brake
        # already uses nearest -- using consensus here meant the car could stop
        # 0.14 m from a wall, average that against a lidar beam that missed it,
        # decide the way was clear, and drive straight back into it. That loop
        # accounted for nearly every halt in the last run.
        if front_min is not None and front_min > p.resume_m:
            self.state = "drive"
            self.reason = None
            # Without this the "front has been under turn_at for long enough"
            # timer is still running from before the halt, so the first tick
            # after resuming declares a corner that is not there.
            self._below_since = None
            self._stuck_sig = None
            self._stuck_since = now
            self._moving_since = None
            self._say("info", "front clear -> driving")
            return self._do_drive(now, front, front_min)
        return 0.0, 0.0

    def _do_backout(self, now, front):
        p = self.p
        front = self.fusion.nearest("front", now).value
        held = now - self._backout_started
        rear = self.fusion.get("rear", now).value
        blocked = rear is not None and rear < p.rear_min_m
        if blocked:
            self._boxed_until = now + p.boxed_cooldown_s
            self._say("warn", "boxed in: front %s rear %s" % (_r(front), _r(rear)))
        if (blocked or held >= p.backout_max_s
                or (front is not None and front >= p.backout_target_m)):
            self.state = "halt"
            self._moving_since = None
            return 0.0, 0.0
        pct = p.recover_kick_pct if held < p.recover_kick_s else p.recover_pct
        return 0.0, pct

    def _do_finish(self, now, front):
        """Roll out to where the round started, then stop (rule 9.24.2)."""
        p = self.p
        held = now - self._finish_started
        ref = self.front_at_start
        stop = False
        if ref is not None and front is not None and held >= p.finish_min_s:
            stop = front <= ref + p.finish_tol_m
        if front is not None and front <= p.finish_hold_front_m:
            stop = True                      # never start a thirteenth corner
        if held >= p.finish_max_s:
            stop = True
        if stop:
            self.state = "done"
            self.reason = "three laps complete"
            self._say("info", "stopped in the start section (front=%s, "
                              "reference %s)" % (_r(front), _r(ref)))
            return 0.0, 0.0
        steer, _e = self._centre_steer(now)
        steer += (self._heading_steer(now) + self._wall_push(now)
                  + self._outer_guard(now) + p.trim_deg)
        return steer, self._go(now, p.sustain_pct, front)

    def _check_stuck(self, now):
        p = self.p
        if p.drive_pct <= 0.0:
            return True
        vals = [self.fusion.get(a, now).value for a in ("front", "left", "right")]
        sig = tuple(-1 if v is None else int(round(v / p.stuck_eps_m))
                    for v in vals)
        if sig != self._stuck_sig:
            self._stuck_sig = sig
            self._stuck_since = now
            return True
        return (now - self._stuck_since) <= p.stuck_s

    # --------------------------------------------------------------- status

    def _status(self, now, deg, pct):
        f = self.fusion
        left = f.get("left", now)
        right = f.get("right", now)
        front = f.get("front", now)
        return {
            "state": self.state,
            "armed": self.armed,
            "steer_deg": round(deg, 1),
            "drive_pct": round(pct, 1),
            "turns": self.turns,
            "laps": round(self.turns / 4.0, 2),
            "direction": ("right" if self.turn_dir > 0 else
                          "left" if self.turn_dir < 0 else None),
            "outer_wall": self._outer(),
            "trigger_colour": self.trigger_colour,
            "line_v": (lambda t: t[1])(self._line_underfoot(now)),
            "front": _r(front.value),
            "left": _r(left.value),
            "right": _r(right.value),
            "rear": _r(f.get("rear", now).value),
            "front_src": front.used,
            "left_src": left.used,
            "right_src": right.used,
            "disagree": [a for a in AXES if f.get(a, now).disagree],
            "lane_width": _r(self._lane_w),
            "target": _r(self._target()),
            "heading": _r(self._heading, 1) if self._heading_ok(now) else None,
            "yaw": _r(self._yaw, 1) if self._imu_ok(now) else None,
            "lines": {c: _r(d) for c, d in
                      ((c, self._line(c, now)) for c in ("orange", "blue"))
                      if d is not None},
            "vision_ok": bool(self._vision.get("ok"))
                         and (now - self._vision_t) <= self.p.vision_timeout_s,
            "health": f.health(),
            "sources": f.snapshot(now),
            "recoveries": self._recoveries,
            "map": self.map.summary() if self.p.map_enable else None,
            "tight": self._tight(),
            "line_corners": self._line_corners,
            "odo_m": round(self._odo, 2),
            "speed_scale": round(self._speed_scale, 2),
            "reason": self.reason,
            "run_s": round(now - self.t0, 1) if self.armed else 0.0,
            "notes": list(self._notes[-4:]),
        }
