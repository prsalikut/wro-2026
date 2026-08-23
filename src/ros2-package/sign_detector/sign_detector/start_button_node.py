"""Start button on a GPIO pin: first press launches the round, a later press stops it.
Implements the WRO start procedure (rules 9.11/9.14): boot into a waiting state, move on press."""

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


class StartButton(Node):

    def __init__(self):
        super().__init__("start_button")
        self.declare_parameters("", [
            ("gpiochip", 4),
            ("line", 17),
            ("poll_hz", 50.0),
            ("debounce_s", 0.05),
            ("hold_off_s", 1.0),
            ("run_command",
             "ros2 run sign_detector open_round --ros-args --params-file "
             "/ros2_ws/install/sign_detector/share/sign_detector/config/params.yaml"),
        ])

        def g(k):
            return self.get_parameter(k).value

        self.debounce = float(g("debounce_s"))
        self.hold_off = float(g("hold_off_s"))
        self.run_command = str(g("run_command"))

        self.pub = self.create_publisher(String, "start_status", 10)
        self.proc = None
        self.last_edge = 0.0
        self.stable_since = 0.0
        self.last_level = False

        try:
            self.btn = GpioButton(int(g("gpiochip")), int(g("line")))
            self.get_logger().info(
                "waiting for start button (gpiochip{} line {}, {})".format(
                    g("gpiochip"), g("line"), self.btn.backend))
        except Exception as exc:
            self.btn = None
            self.get_logger().error("start button unavailable: {}".format(exc))

        self.create_timer(1.0 / max(1.0, float(g("poll_hz"))), self.poll)
        self.create_timer(1.0, self.publish_status)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def poll(self):
        if self.btn is None:
            return
        try:
            level = self.btn.pressed()
        except Exception as exc:
            self.get_logger().error("gpio read failed: {}".format(exc))
            return
        now = self._now()
        if level != self.last_level:
            self.last_level = level
            self.stable_since = now
            return
        if not level or (now - self.stable_since) < self.debounce:
            return
        if (now - self.last_edge) < self.hold_off:
            return
        self.last_edge = now
        self.toggle()

    def toggle(self):
        if self.running():
            self.stop()
        else:
            self.start()

    def start(self):
        self.get_logger().info("START pressed: launching round")
        self.proc = subprocess.Popen(
            ["bash", "-lc", ROS_ENV + self.run_command],
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
        if self.btn is None:
            state = "no_gpio"
        self.pub.publish(String(data='{"state": "%s"}' % state))


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
        if node.btn:
            node.btn.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
