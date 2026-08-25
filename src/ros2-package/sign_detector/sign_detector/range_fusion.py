"""Combine camera, lidar and ultrasonic ranges, and notice when one is lying.

Three sensors see the same wall and all three fail differently:

  * the lidar reads long or not at all on glossy black at grazing incidence, so
    it is weak exactly where the side walls are;
  * an HC-SR04 can latch: DEPLOY.md records the right sonar reporting 0.43 m for
    213 consecutive samples while the wall was 2.94 m away, which drove the car
    into a wall it could not see;
  * the camera loses a wall when it swings outside a 60 degree field of view,
    and gives up entirely in the dark.

None of those is a fallback chain -- a chain trusts a broken sensor until it
goes silent, and a latched sonar never goes silent.  So each source is tracked
separately, with a health check that fires on *implausible behaviour* rather
than absence: a value that never changes, a value outside the physical range,
or a step no vehicle could have made.  What survives is combined by trust
weight, and any disagreement is reported rather than averaged away.
"""

import math

__all__ = ["SourceSpec", "FusedRange", "RangeFusion", "AXES"]

AXES = ("front", "left", "right", "rear")


class SourceSpec(object):

    def __init__(self, name, trust=1.0, timeout_s=0.5, min_m=0.02, max_m=4.0,
                 freeze_n=12, max_step_mps=6.0, outlier_tol_m=0.32,
                 outlier_s=1.5):
        self.name = name
        self.trust = float(trust)
        self.timeout_s = float(timeout_s)
        self.min_m = float(min_m)
        self.max_m = float(max_m)
        self.freeze_n = int(freeze_n)      # 0 disables the latch detector
        self.max_step_mps = float(max_step_mps)
        self.outlier_tol_m = float(outlier_tol_m)
        self.outlier_s = float(outlier_s)


class _Chan(object):
    __slots__ = ("value", "stamp", "same", "last_raw", "healthy", "fault",
                 "prev_value", "prev_stamp", "bad", "bias_since")

    def __init__(self):
        self.value = None
        self.stamp = -1e9
        self.same = 0
        self.last_raw = None
        self.prev_value = None
        self.prev_stamp = -1e9
        self.healthy = True
        self.fault = None
        self.bad = 0
        self.bias_since = None


class FusedRange(object):
    __slots__ = ("value", "used", "rejected", "disagree", "spread", "n")

    def __init__(self, value=None, used=(), rejected=(), disagree=False,
                 spread=0.0):
        self.value = value
        self.used = list(used)
        self.rejected = list(rejected)
        self.disagree = disagree
        self.spread = spread
        self.n = len(self.used)

    def __repr__(self):
        return "FusedRange(%s, used=%s%s)" % (
            "None" if self.value is None else "%.3f" % self.value,
            ",".join(self.used), " DISAGREE" if self.disagree else "")


class RangeFusion(object):

    def __init__(self, sources, agree_tol_m=0.20):
        self.specs = {s.name: s for s in sources}
        self.agree_tol = float(agree_tol_m)
        self.moving = True
        self.chan = {}
        for s in sources:
            for axis in AXES:
                self.chan[(s.name, axis)] = _Chan()

    def set_moving(self, moving):
        """A parked car legitimately reads the same number for ever, so the
        latch detector only counts while the wheels are turning."""
        moving = bool(moving)
        if moving != self.moving:
            self.moving = moving
            if not moving:
                for c in self.chan.values():
                    c.same = 0

    # ------------------------------------------------------------------ feed

    def update(self, source, axis, value, now):
        """Record one reading.  `value` None means the source has nothing."""
        key = (source, axis)
        c = self.chan.get(key)
        if c is None:
            return
        spec = self.specs[source]
        if value is None or not math.isfinite(value):
            c.last_raw = None
            c.same = 0
            return
        value = float(value)

        if c.fault is not None and c.fault.startswith("disagrees"):
            # Keep evaluating a demoted channel against the others so it can
            # come back when the disagreement stops.
            self._check_bias(source, axis, c, spec, value, now)
        if value < spec.min_m or value > spec.max_m:
            # One silly number is noise, not a broken sensor; only a run of
            # them means the channel should stop being believed.
            c.bad += 1
            c.last_raw = None
            if c.bad >= 6:
                c.fault = "out of range (%.2f)" % value
                c.healthy = False
            return
        c.bad = 0

        # A latched sensor keeps reporting, so absence is not the test: the same
        # number over and over while the car is moving is.
        if spec.freeze_n > 0 and self.moving:
            if c.last_raw is not None and abs(value - c.last_raw) < 1e-6:
                c.same += 1
            else:
                c.same = 0
            if c.same >= spec.freeze_n:
                c.fault = "frozen at %.2f m for %d samples" % (value, c.same)
                c.healthy = False
            elif c.fault is not None and c.fault.startswith("frozen"):
                c.fault = None
                c.healthy = True
        elif spec.freeze_n > 0:
            c.same = 0
        c.last_raw = value

        if (spec.max_step_mps > 0.0 and c.value is not None
                and (now - c.stamp) < spec.timeout_s):
            dt = max(1e-3, now - c.stamp)
            if abs(value - c.value) / dt > spec.max_step_mps:
                # One wild sample: drop it, but let a genuine change through if
                # it repeats, so a real step does not blind the axis for ever.
                if c.prev_value is not None and abs(value - c.prev_value) < 0.10:
                    pass
                else:
                    c.prev_value, c.prev_stamp = value, now
                    return

        c.prev_value, c.prev_stamp = c.value, c.stamp
        c.value = value
        c.stamp = now
        if c.fault is not None and c.fault.startswith("out of range"):
            c.fault = None
            c.healthy = True
            c.bad = 0
        self._check_bias(source, axis, c, spec, value, now)

    def _check_bias(self, source, axis, c, spec, value, now):
        """Demote a sensor that keeps disagreeing with everything else.

        The latch detector only catches a sensor that stops changing.  A sonar
        can also read a steady 0.9 m short -- DEPLOY.md records the front one
        doing exactly that -- and then the car believes it is permanently at a
        corner and never moves.  A reading that sits well outside what the
        other sensors agree on, for over a second, while they carry more trust
        between them, is the one that is wrong.
        """
        if spec.outlier_s <= 0.0:
            return
        others, weight = [], 0.0
        for name, ospec in self.specs.items():
            if name == source:
                continue
            oc = self.chan[(name, axis)]
            if (oc.value is None or not oc.healthy
                    or (now - oc.stamp) > ospec.timeout_s):
                continue
            others.append(oc.value)
            weight += ospec.trust
        if not others or weight <= spec.trust:
            c.bias_since = None
            return
        others.sort()
        med = (others[len(others) // 2] if len(others) % 2
               else 0.5 * (others[len(others) // 2 - 1]
                           + others[len(others) // 2]))
        if abs(value - med) > spec.outlier_tol_m:
            if c.bias_since is None:
                c.bias_since = now
            elif (now - c.bias_since) >= spec.outlier_s:
                c.fault = ("disagrees by %+.2f m with %d other sensor(s)"
                           % (value - med, len(others)))
                c.healthy = False
        else:
            c.bias_since = None
            if c.fault is not None and c.fault.startswith("disagrees"):
                c.fault = None
                c.healthy = True

    def clear_fault(self, source, axis=None):
        for ax in (AXES if axis is None else (axis,)):
            c = self.chan.get((source, ax))
            if c is not None:
                c.fault = None
                c.healthy = True
                c.same = 0

    # ------------------------------------------------------------------ read

    def _live(self, axis, now):
        out = []
        for name, spec in self.specs.items():
            c = self.chan[(name, axis)]
            if not c.healthy or c.value is None:
                continue
            if (now - c.stamp) > spec.timeout_s:
                continue
            out.append((name, c.value, spec.trust))
        return out

    def get(self, axis, now):
        """Trust-weighted consensus for one axis."""
        live = self._live(axis, now)
        if not live:
            return FusedRange()
        if len(live) == 1:
            name, v, _t = live[0]
            return FusedRange(v, used=[name])

        vals = sorted(v for _n, v, _t in live)
        mid = (vals[len(vals) // 2] if len(vals) % 2
               else 0.5 * (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]))
        keep = [(n, v, t) for (n, v, t) in live if abs(v - mid) <= self.agree_tol]
        dropped = [n for (n, v, _t) in live if abs(v - mid) > self.agree_tol]
        if not keep:                        # everything is an outlier: trust wins
            keep = [max(live, key=lambda e: e[2])]
            dropped = [n for (n, _v, _t) in live if n != keep[0][0]]
        wsum = sum(t for _n, _v, t in keep) or 1.0
        value = sum(v * t for _n, v, t in keep) / wsum
        return FusedRange(value, used=[n for n, _v, _t in keep],
                          rejected=dropped, disagree=bool(dropped),
                          spread=(max(vals) - min(vals)))

    def nearest(self, axis, now):
        """Closest healthy reading -- the number a brake should use."""
        live = self._live(axis, now)
        if not live:
            return FusedRange()
        name, v, _t = min(live, key=lambda e: e[1])
        return FusedRange(v, used=[name])

    def health(self):
        out = {}
        for (name, axis), c in self.chan.items():
            if c.fault:
                out["%s/%s" % (name, axis)] = c.fault
        return out

    def snapshot(self, now):
        out = {}
        for name in self.specs:
            row = {}
            for axis in AXES:
                c = self.chan[(name, axis)]
                fresh = (c.value is not None
                         and (now - c.stamp) <= self.specs[name].timeout_s)
                row[axis] = round(c.value, 3) if (fresh and c.healthy) else None
            out[name] = row
        return out
