#!/usr/bin/env python3
"""Tests for the open-round stack that need no ROS and no hardware.

    python3 src/tests/test_open_round.py          # everything but the sim
    python3 src/tests/test_open_round.py --full   # plus closed-loop rounds

Plain unittest rather than pytest, so it runs inside the container as shipped.
"""

import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ros2-package", "sign_detector"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

import numpy as np  # noqa: E402

from sign_detector.ground_geometry import CameraGeometry  # noqa: E402
from sign_detector.open_round_core import (OpenRoundCore,  # noqa: E402
                                           OpenRoundParams)
from sign_detector.range_fusion import RangeFusion, SourceSpec  # noqa: E402
from sign_detector.wall_vision import WallVision, fit_line_trimmed  # noqa: E402


class TestGroundGeometry(unittest.TestCase):

    def setUp(self):
        self.cam = CameraGeometry(640, 480, 60.0, 0.12, 15.0, cam_x_m=0.05)

    def test_round_trip(self):
        for x in (0.25, 0.5, 1.0, 2.0):
            for y in (-0.5, 0.0, 0.35):
                u, v = self.cam.to_pixel(x, y)
                back = self.cam.to_ground(u, v, max_range=1e9)
                self.assertIsNotNone(back)
                self.assertAlmostEqual(back[0], x, places=9)
                self.assertAlmostEqual(back[1], y, places=9)

    def test_nothing_above_the_horizon_is_ground(self):
        self.assertIsNone(self.cam.to_ground(self.cam.cx,
                                             self.cam.horizon_row() - 0.5))

    def test_range_resolution_degrades_with_distance(self):
        near = self.cam.range_resolution(0.4)
        far = self.cam.range_resolution(1.6)
        self.assertLess(near, far)
        # The honest range is what the node advertises; if this moves, the
        # max_range_m default in params.yaml has to move with it.
        self.assertGreater(self.cam.usable_range(0.05), 1.5)

    def test_bearing_sign(self):
        self.assertGreater(self.cam.bearing_deg(10), 0.0)      # left of centre
        self.assertLess(self.cam.bearing_deg(630), 0.0)


class TestLineFit(unittest.TestCase):

    def test_outliers_do_not_drag_the_fit(self):
        xs = [0.3 + 0.05 * i for i in range(20)]
        ys = [0.4 + 0.0 * x for x in xs]
        ys[5] = 1.9                     # a stray boundary pixel
        ys[11] = 1.7
        got = fit_line_trimmed(xs, ys)
        self.assertIsNotNone(got)
        a, b, rms, _n = got
        self.assertLess(abs(a), 0.05)
        self.assertAlmostEqual(b, 0.4, delta=0.03)


class TestRangeFusion(unittest.TestCase):

    def _fusion(self):
        # Same shape as OpenRoundCore builds: only the ultrasonics latch, so
        # only they carry the frozen-value detector.
        return RangeFusion([SourceSpec("vision", 1.3, 0.45, 0.05, 3.0, 0),
                            SourceSpec("sonar", 1.0, 0.45, 0.03, 3.5, 10),
                            SourceSpec("lidar", 0.7, 0.5, 0.05, 6.0, 0)])

    def test_agreeing_sources_are_averaged(self):
        f = self._fusion()
        f.update("vision", "left", 0.50, 1.0)
        f.update("lidar", "left", 0.52, 1.0)
        r = f.get("left", 1.0)
        self.assertAlmostEqual(r.value, 0.507, delta=0.01)
        self.assertFalse(r.disagree)

    def test_a_latched_sonar_is_dropped(self):
        """DEPLOY.md records the right sonar stuck at 0.43 m for 213 samples."""
        f = self._fusion()
        t = 0.0
        for i in range(20):
            t += 0.05
            f.update("vision", "right", 0.50 + 0.002 * i, t)
            f.update("lidar", "right", 0.51 + 0.002 * i, t)
            f.update("sonar", "right", 0.43, t)
        self.assertIn("sonar/right", f.health())
        self.assertNotIn("sonar", f.get("right", t).used)

    def test_a_parked_car_does_not_look_latched(self):
        f = self._fusion()
        f.set_moving(False)
        t = 0.0
        for _ in range(40):
            t += 0.05
            f.update("sonar", "front", 0.80, t)
        self.assertEqual(f.health(), {})

    def test_a_biased_sensor_is_demoted_then_forgiven(self):
        f = self._fusion()
        t = 0.0
        for i in range(60):
            t += 0.05
            f.update("vision", "front", 1.20, t)
            f.update("lidar", "front", 1.22, t)
            f.update("sonar", "front", 0.30, t)          # 0.9 m short
        self.assertIn("sonar/front", f.health())
        for i in range(20):
            t += 0.05
            f.update("vision", "front", 1.20, t)
            f.update("lidar", "front", 1.22, t)
            # A real sonar reports whole centimetres and jitters between them;
            # a value that never moves at all is the latch case, tested above.
            f.update("sonar", "front", 1.21 + 0.01 * (i % 2), t)
        self.assertEqual(f.health(), {})

    def test_nearest_ignores_an_unhealthy_source(self):
        f = self._fusion()
        t = 0.0
        for i in range(60):
            t += 0.05
            f.update("vision", "front", 1.20, t)
            f.update("lidar", "front", 1.22, t)
            f.update("sonar", "front", 0.30, t)
        self.assertGreater(f.nearest("front", t).value, 1.0)


def _feed(core, t, front=1.5, left=0.5, right=0.5, rear=1.0, heading=0.0,
          lines=None, ok=True):
    core.on_vision({"ok": ok, "left": left, "right": right, "front": front,
                    "front_wall": front, "front_free": front,
                    "heading": heading, "open_bearing": 0.0,
                    "lane_width": None if (left is None or right is None)
                    else left + right,
                    "floor_frac": 0.8, "lines": lines or {}}, t)
    core.on_scan(front, left, right, rear, t, left_far=left, right_far=right)
    for name, v in (("front", front), ("left", left), ("right", right),
                    ("rear", rear)):
        core.on_sonar(name, v, t)


class TestOpenRoundCore(unittest.TestCase):

    def test_disarmed_car_does_not_move(self):
        core = OpenRoundCore()
        t = 0.0
        for _ in range(20):
            t += 0.05
            _feed(core, t)
            deg, pct, st = core.step(t)
            self.assertEqual(pct, 0.0)
            self.assertEqual(deg, 0.0)
            self.assertEqual(st["state"], "wait")

    def test_arming_starts_the_round(self):
        core = OpenRoundCore()
        core.set_armed(True, 0.0)
        t = 0.0
        for _ in range(10):
            t += 0.05
            _feed(core, t)
            _deg, pct, st = core.step(t)
        self.assertEqual(st["state"], "drive")
        self.assertGreater(pct, 0.0)

    def test_stop_after_disarm(self):
        core = OpenRoundCore()
        core.set_armed(True, 0.0)
        t = 0.0
        for _ in range(10):
            t += 0.05
            _feed(core, t)
            core.step(t)
        core.set_armed(False, t)
        t += 0.05
        _feed(core, t)
        _deg, pct, st = core.step(t)
        self.assertEqual(pct, 0.0)
        self.assertEqual(st["state"], "wait")

    def test_centring_steers_away_from_the_near_wall(self):
        core = OpenRoundCore()
        core.set_armed(True, 0.0)
        t = 0.0
        for _ in range(10):
            t += 0.05
            _feed(core, t, left=0.30, right=0.70)
            deg, _pct, _st = core.step(t)
        self.assertGreater(deg, 2.0, "closer on the left should steer right")
        core2 = OpenRoundCore()
        core2.set_armed(True, 0.0)
        t = 0.0
        for _ in range(10):
            t += 0.05
            _feed(core2, t, left=0.70, right=0.30)
            deg, _pct, _st = core2.step(t)
        self.assertLess(deg, -2.0)

    def test_open_space_on_one_side_is_not_a_wall(self):
        """Past the end of the inner block a side reading jumps to 3 m."""
        core = OpenRoundCore()
        core.set_armed(True, 0.0)
        t = 0.0
        for _ in range(10):
            t += 0.05
            core.on_vision({"ok": True, "left": None, "right": 0.20,
                            "front": 1.5, "front_wall": 1.5, "heading": 0.0,
                            "floor_frac": 0.8, "lines": {}}, t)
            core.on_scan(1.5, 2.6, 0.20, 1.0, t, left_far=2.6, right_far=0.2)
            deg, _pct, _st = core.step(t)
        self.assertLess(deg, -1.0, "should hold off the right wall, not chase "
                                   "the open side")

    def test_a_halt_does_not_produce_a_phantom_corner(self):
        core = OpenRoundCore(OpenRoundParams(drive_pct=45.0))
        core.set_armed(True, 0.0)
        t = 0.0
        for _ in range(30):        # close enough to arm the corner timer
            t += 0.05
            _feed(core, t, front=0.5)
            core.step(t)
        for _ in range(20):        # then a hard stop
            t += 0.05
            _feed(core, t, front=0.15)
            core.step(t)
        turns_at_halt = core.turns
        for _ in range(10):        # and the wall goes away
            t += 0.05
            _feed(core, t, front=1.6)
            _deg, _pct, st = core.step(t)
        self.assertEqual(core.turns, turns_at_halt,
                         "resuming from a halt must not count a corner")

    def test_round_timer_stops_the_car(self):
        core = OpenRoundCore(OpenRoundParams(max_run_s=2.0))
        core.set_armed(True, 0.0)
        t = 0.0
        for _ in range(60):
            t += 0.05
            _feed(core, t)
            _deg, pct, st = core.step(t)
        self.assertEqual(st["state"], "done")
        self.assertEqual(pct, 0.0)

    def test_direction_is_measured_at_the_corner_not_assumed(self):
        """Turn toward whichever side actually has room.

        There is deliberately no colour convention here. An earlier version
        asserted that an orange line first meant clockwise, taken from the mat
        artwork, and hard-coded the turn from it; on the real track that gave a
        car which worked one way round and drove into the wall the other. The
        line says WHEN a corner is; the ultrasonics say WHICH WAY.
        """
        for left, right, colour, want in ((0.25, 0.95, "orange", "right"),
                                          (0.95, 0.25, "orange", "left"),
                                          (0.25, 0.95, "blue", "right"),
                                          (0.95, 0.25, "blue", "left")):
            core = OpenRoundCore(OpenRoundParams(min_corner_travel_m=0.0,
                                                 corner_lockout_s=0.0))
            core.set_armed(True, 0.0)
            t = 0.0
            for i in range(60):
                t += 0.05
                front = 1.5 if i < 30 else 0.5
                # Jitter by a centimetre: a real HC-SR04 reports whole
                # centimetres and never repeats exactly, and feeding it a
                # bit-identical value trips the latch detector -- correctly,
                # but it leaves this test with no side readings at all.
                j = 0.01 * (i % 2)
                core.on_vision({"ok": True, "left": None, "right": None,
                                "front": front, "front_wall": front,
                                "front_free": front, "heading": 0.0,
                                "open_bearing": 0.0, "floor_frac": 0.8,
                                "lines": {colour: {"v_frac": 0.85,
                                                   "distance_m": 0.2}}}, t)
                core.on_scan(front, None, None, 1.0, t)
                for n, v in (("front", front + j), ("left", left + j),
                             ("right", right + j), ("rear", 1.0 + j)):
                    core.on_sonar(n, v, t)
                core.step(t)
            got = "right" if core.turn_dir > 0 else "left"
            with self.subTest(left=left, right=right, colour=colour):
                self.assertEqual(got, want)

    def test_lidar_does_not_reach_the_side_ranges(self):
        """The lidar reads these glossy walls long or not at all edge-on, and
        letting it into the centring is what made the car ride one wall."""
        core = OpenRoundCore()
        core.set_armed(True, 0.0)
        t = 0.0
        for i in range(10):
            t += 0.05
            j = 0.01 * (i % 2)
            core.on_scan(1.5, 0.9, 0.1, 1.0, t)     # nonsense sides
            core.on_sonar("left", 0.45 + j, t)
            core.on_sonar("right", 0.45 + j, t)
            core.on_sonar("front", 1.5 + j, t)
            core.step(t)
        self.assertNotIn("lidar", core.fusion.get("left", t).used)
        self.assertNotIn("lidar", core.fusion.get("right", t).used)
        self.assertIn("sonar", core.fusion.get("left", t).used)
        self.assertAlmostEqual(core.fusion.get("left", t).value, 0.45, delta=0.02)

    def test_the_camera_seeing_no_floor_at_all_counts_as_blocked(self):
        core = OpenRoundCore()
        core.set_armed(True, 0.0)
        t = 0.0
        for _ in range(20):
            t += 0.05
            core.on_vision({"ok": False, "reason": "floor only 1% of frame",
                            "left": None, "right": None, "front": None,
                            "floor_frac": 0.01, "lines": {}}, t)
            core.on_sonar("rear", 1.0, t)
            _deg, pct, st = core.step(t)
        self.assertLessEqual(pct, 0.0, "must not drive forward into whatever "
                                       "is filling the frame")
        self.assertIn(st["state"], ("halt", "backout"))

    def test_drive_never_commands_below_stiction(self):
        core = OpenRoundCore()
        core.set_armed(True, 0.0)
        t = 0.0
        seen = set()
        for _ in range(200):
            t += 0.05
            _feed(core, t, front=0.85)          # inside slow_front_m
            _deg, pct, _st = core.step(t)
            seen.add(round(pct))
        moving = [p for p in seen if p > 0]
        self.assertTrue(moving)
        self.assertTrue(all(p >= core.p.stiction_pct for p in moving),
                        "a duty below stiction is a stalled motor, not a slow "
                        "one: got %s" % sorted(seen))


class TestWallVision(unittest.TestCase):

    def _vision(self, h=0.12, pitch=15.0):
        geom = CameraGeometry(320, 240, 60.0, h, pitch, cam_x_m=0.05)
        return WallVision(geom), geom

    def _settle(self, wv, img, n=6):
        for _ in range(n):
            res = wv.process(img)
        return res

    def test_a_wall_ahead_is_measured(self):
        wv, geom = self._vision()
        img = np.full((240, 320, 3), 235, np.uint8)
        row = geom.to_pixel(1.20, 0.0)[1]
        img[:int(row)] = (30, 30, 30)
        res = self._settle(wv, img)
        self.assertTrue(res.ok)
        self.assertAlmostEqual(res.front_m, 1.20, delta=0.10)

    def test_painted_lines_are_not_obstacles(self):
        """The blue corner line is dark enough to fail a brightness test."""
        wv, geom = self._vision()
        img = np.full((240, 320, 3), 235, np.uint8)
        row = geom.to_pixel(1.20, 0.0)[1]
        img[:int(row)] = (30, 30, 30)
        stripe = int(geom.to_pixel(0.45, 0.0)[1])
        img[stripe - 3:stripe + 3, :] = (157, 77, 15)     # PANTONE 2728 C
        res = self._settle(wv, img)
        self.assertAlmostEqual(res.front_m, 1.20, delta=0.12)
        self.assertIn("blue", res.lines)
        self.assertAlmostEqual(res.lines["blue"]["distance_m"], 0.45,
                               delta=0.12)

    def test_vignetting_is_not_read_as_a_wall(self):
        wv, geom = self._vision()
        img = np.full((240, 320, 3), 235, np.float32)
        yy, xx = np.mgrid[0:240, 0:320]
        r = np.sqrt(((xx - 160) / 160.0) ** 2 + ((yy - 120) / 120.0) ** 2)
        img *= (1.0 - 0.30 * np.clip(r, 0, 1.6) ** 2)[:, :, None]
        img = img.astype(np.uint8)
        row = geom.to_pixel(1.40, 0.0)[1]
        img[:int(row)] = (30, 30, 30)
        res = self._settle(wv, img)
        self.assertGreater(res.front_m, 1.0,
                           "darkened image corners project to about 0.2 m "
                           "ahead and must not read as an obstruction")

    def test_no_floor_in_view_is_reported_rather_than_guessed(self):
        wv, _geom = self._vision()
        img = np.full((240, 320, 3), 20, np.uint8)
        res = self._settle(wv, img)
        self.assertFalse(res.ok)
        self.assertLess(res.floor_frac, 0.2)


class TestClosedLoop(unittest.TestCase):
    """Full rounds against the rendered field. Slow: --full to enable."""

    @unittest.skipUnless(os.environ.get("WRO_FULL"), "set WRO_FULL=1")
    def test_representative_rounds(self):
        from sim_offline import Run, base_config
        cases = [
            dict(label="ccw wide", section="south", ccw=True),
            dict(label="cw wide", section="north", ccw=False),
            dict(label="narrow", widths=(0.6, 0.6, 0.6, 0.6)),
            dict(label="mixed", widths=(1.0, 0.6, 1.0, 0.6)),
            dict(label="frozen sonar", sonar_faults={"right": "frozen"}),
            dict(label="blind lidar", lidar_dropout=0.92),
        ]
        for kw in cases:
            label = kw.pop("label")
            rep = Run(base_config(label=label, **kw)).run(seconds=175)
            with self.subTest(case=label):
                self.assertEqual(rep["collisions"], 0, rep)
                self.assertEqual(rep["outer_touches"], 0, rep)
                self.assertGreaterEqual(rep["laps"], 2.85, rep)
                self.assertTrue(rep["finished_in_start_section"], rep)


if __name__ == "__main__":
    if "--full" in sys.argv:
        sys.argv.remove("--full")
        os.environ["WRO_FULL"] = "1"
    unittest.main(verbosity=2)
