"""Start and mode buttons on GPIO lines.

START toggles the round: first press launches the selected round, a later press
stops it. MODE picks which round START will launch, and is ignored while a round
is running, because switching the driver out from under a moving car is never
what the press meant.

Implements the WRO start procedure (rules 9.11/9.14): boot into a waiting state,
move on press."""

import subprocess
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

ROS_ENV = (
    "source /opt/ros/humble/setup.bash && "
    "source /ros2_ws/install/setup.bash && "
)


class GpioButton:
    """Active-low button with internal pull-up, via lgpio or libgpiod."""

    def __init__(self, chip, line):
        self.line = line
        self.backend = None
        try:
            import lgpio
            self._lg = lgpio
            self._h = lgpio.gpiochip_open(chip)
            lgpio.gpio_claim_input(self._h, line, lgpio.SET_PULL_UP)
            self.backend = "lgpio"
            return
        except Exception:
            pass
        try:
            import gpiod
            self._chip = gpiod.Chip(str(chip))
            self._line = self._chip.get_line(line)
            self._line.request(consumer="start_button",
                               type=gpiod.LINE_REQ_DIR_IN,
                               flags=gpiod.LINE_REQ_FLAG_BIAS_PULL_UP)
            self.backend = "gpiod"
        except Exception as exc:
            raise RuntimeError("no usable GPIO backend: {}".format(exc))

    def pressed(self):
        if self.backend == "lgpio":
            return self._lg.gpio_read(self._h, self.line) == 0
        return self._line.get_value() == 0

    def close(self):
        try:
            if self.backend == "lgpio":
                self._lg.gpiochip_close(self._h)
            else:
                self._line.release()
        except Exception:
            pass


class Edge:
    """Debounced press detector for one button.

    Fires on the press and then disarms until the button is seen released, so
    holding it down is one press, not a repeat every hold_off seconds. It also
    starts disarmed: a button held (or stuck) at boot must be released before it
    can launch a round.

    Each button carries its own timers. A shared one would let a press on either
    button swallow a press on the other."""

    def __init__(self, btn, debounce_s, hold_off_s):
        self.btn = btn
        self.debounce = debounce_s
        self.hold_off = hold_off_s
        self.last_level = False
        self.stable_since = 0.0
        self.last_edge = 0.0
        self.armed = False

    def fired(self, now):
        """True once per debounced press."""
        level = self.btn.pressed()
        if level != self.last_level:
            self.last_level = level
            self.stable_since = now
            return False
        if (now - self.stable_since) < self.debounce:
            return False
        if not level:
            self.armed = True
            return False
        if not self.armed:
            return False
        self.armed = False
        if (now - self.last_edge) < self.hold_off:
            return False        # bounced double-tap: swallow the whole press
        self.last_edge = now
        return True


class StartButton(Node):

    def __init__(self):
        super().__init__("start_button")
        self.declare_parameters("", [
            ("gpiochip", 4),
            ("line", 17),
            ("mode_line", 27),
            ("mode_button", True),
            ("poll_hz", 50.0),
            ("debounce_s", 0.05),
            ("hold_off_s", 1.0),
            ("run_command",
             "ros2 run sign_detector open_round --ros-args --params-file "
             "/ros2_ws/install/sign_detector/share/sign_detector/config/params.yaml"),
            ("obstacle_command",
             "ros2 run sign_detector sign_steering --ros-args --params-file "
             "/ros2_ws/install/sign_detector/share/sign_detector/config/params.yaml"),
        ])

        def g(k):
            return self.get_parameter(k).value

        debounce = float(g("debounce_s"))
        hold_off = float(g("hold_off_s"))

        self.modes = [("open", str(g("run_command"))),
                      ("obstacle", str(g("obstacle_command")))]
        self.mode_i = 0

        self.pub = self.create_publisher(String, "start_status", 10)
        self.proc = None

        chip = int(g("gpiochip"))
        self.buttons = []
        self.start_edge = self._claim(chip, int(g("line")), "start",
                                      debounce, hold_off)
        self.mode_edge = None
        if bool(g("mode_button")):
            self.mode_edge = self._claim(chip, int(g("mode_line")), "mode",
                                         debounce, hold_off)

        self.create_timer(1.0 / max(1.0, float(g("poll_hz"))), self.poll)
        self.create_timer(1.0, self.publish_status)

    def _claim(self, chip, line, what, debounce, hold_off):
        try:
            btn = GpioButton(chip, line)
        except Exception as exc:
            self.get_logger().error(
                "{} button unavailable: {}".format(what, exc))
            return None
        self.buttons.append(btn)
        self.get_logger().info(
            "{} button ready (gpiochip{} line {}, {})".format(
                what, chip, line, btn.backend))
        return Edge(btn, debounce, hold_off)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def mode(self):
        return self.modes[self.mode_i][0]

    def poll(self):
        now = self._now()
        for edge, action in ((self.start_edge, self.toggle),
                             (self.mode_edge, self.cycle_mode)):
            if edge is None:
                continue
            try:
                if edge.fired(now):
                    action()
            except Exception as exc:
                self.get_logger().error("gpio read failed: {}".format(exc))

    def cycle_mode(self):
        if self.running():
            self.get_logger().warning(
                "MODE pressed while running: ignored, stop the round first")
            return
        self.mode_i = (self.mode_i + 1) % len(self.modes)
        self.get_logger().info("MODE pressed: {}".format(self.mode()))
        self.publish_status()

    def toggle(self):
        if self.running():
            self.stop()
        else:
            self.start()

    def start(self):
        name, command = self.modes[self.mode_i]
        self.get_logger().info("START pressed: launching {} round".format(name))
        self.proc = subprocess.Popen(
            ["bash", "-lc", ROS_ENV + command],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.publish_status()

    def stop(self):
        self.get_logger().warning("STOP pressed: terminating round")
        try:
            self.proc.terminate()
            for _ in range(20):
                if self.proc.poll() is not None:
                    break
                time.sleep(0.05)
            if self.proc.poll() is None:
                self.proc.kill()
        except Exception:
            pass
        self.proc = None
        self.publish_status()

    def publish_status(self):
        state = "running" if self.running() else "waiting"
        if self.start_edge is None:
            state = "no_gpio"
        self.pub.publish(String(
            data='{"state": "%s", "mode": "%s"}' % (state, self.mode())))


def main():
    rclpy.init()
    node = StartButton()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node.running():
            node.stop()
        for btn in node.buttons:
            btn.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
