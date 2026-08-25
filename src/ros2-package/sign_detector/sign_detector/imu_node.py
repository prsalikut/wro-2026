"""IMU node: BNO055 (0x28) or MPU6050 over I2C -> imu/data and imu/yaw (deg).
Hot-plug resilient: probes until a sensor appears, re-probes if reads fail."""
import fcntl
import math
import os
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, String

I2C_SLAVE = 0x0703

REG_PWR_MGMT_1 = 0x6B
REG_SMPLRT_DIV = 0x19
REG_CONFIG = 0x1A
REG_GYRO_CONFIG = 0x1B
REG_ACCEL_CONFIG = 0x1C
REG_ACCEL_XOUT_H = 0x3B
REG_WHO_AM_I = 0x75

GYRO_FS = {250: 0x00, 500: 0x08, 1000: 0x10, 2000: 0x18}
GYRO_LSB = {250: 131.0, 500: 65.5, 1000: 32.8, 2000: 16.4}
ACCEL_LSB = 16384.0  # +/-2 g
G = 9.80665

BNO055_ADDR = 0x28
PROBE_PERIOD_S = 2.0
MAX_READ_FAILS = 10


class MPU6050:
    def __init__(self, bus="/dev/i2c-1", addr=0x68, gyro_range=500, dlpf=3):
        self.fd = os.open(bus, os.O_RDWR)
        try:
            fcntl.ioctl(self.fd, I2C_SLAVE, addr)
            who = self._read(REG_WHO_AM_I, 1)[0]
            if who not in (0x68, 0x70, 0x71, 0x73, 0x98):
                raise RuntimeError("unexpected WHO_AM_I 0x%02x" % who)
            self.who = who
            self._write(REG_PWR_MGMT_1, 0x80)  # reset
            time.sleep(0.1)
            self._write(REG_PWR_MGMT_1, 0x01)  # wake, clock = gyro X PLL
            time.sleep(0.05)
            self._write(REG_CONFIG, dlpf & 0x07)
            self._write(REG_SMPLRT_DIV, 0x04)  # 1 kHz / 5 = 200 Hz
            self._write(REG_GYRO_CONFIG, GYRO_FS.get(gyro_range, 0x08))
            self._write(REG_ACCEL_CONFIG, 0x00)
            self.gyro_lsb = GYRO_LSB.get(gyro_range, 65.5)
            time.sleep(0.05)
            pwr = self._read(REG_PWR_MGMT_1, 1)[0]
            if pwr & 0x40:  # sleep bit stuck: chip not accepting writes
                raise RuntimeError("PWR_MGMT_1=0x%02x after wake" % pwr)
        except Exception:
            self.close()
            raise

    def _write(self, reg, val):
        os.write(self.fd, bytes([reg, val]))

    def _read(self, reg, n):
        os.write(self.fd, bytes([reg]))
        return os.read(self.fd, n)

    @staticmethod
    def _s16(hi, lo):
        v = (hi << 8) | lo
        return v - 65536 if v & 0x8000 else v

    def read(self):
        """-> (ax, ay, az) m/s^2, (gx, gy, gz) rad/s."""
        d = self._read(REG_ACCEL_XOUT_H, 14)
        ax = self._s16(d[0], d[1]) / ACCEL_LSB * G
        ay = self._s16(d[2], d[3]) / ACCEL_LSB * G
        az = self._s16(d[4], d[5]) / ACCEL_LSB * G
        gx = math.radians(self._s16(d[8], d[9]) / self.gyro_lsb)
        gy = math.radians(self._s16(d[10], d[11]) / self.gyro_lsb)
        gz = math.radians(self._s16(d[12], d[13]) / self.gyro_lsb)
        return (ax, ay, az), (gx, gy, gz)

    def close(self):
        try:
            os.close(self.fd)
        except Exception:
            pass


class BNO055:
    """BNO055 in NDOF mode: on-chip fused absolute heading."""

    CHIP_ID = 0x00
    PAGE_ID = 0x07
    OPR_MODE = 0x3D
    PWR_MODE = 0x3E
    SYS_TRIGGER = 0x3F
    UNIT_SEL = 0x3B
    EUL_HEADING_LSB = 0x1A
    GYR_DATA_X_LSB = 0x14
    ACC_DATA_X_LSB = 0x08
    CALIB_STAT = 0x35
    MODE_CONFIG = 0x00
    MODE_NDOF = 0x0C

    def __init__(self, bus="/dev/i2c-1", addr=0x28):
        self.fd = os.open(bus, os.O_RDWR)
        try:
            fcntl.ioctl(self.fd, I2C_SLAVE, addr)
            if not self._await_chip(2.0):
                raise RuntimeError("no BNO055 answering at 0x%02x" % addr)
            self._write(self.PAGE_ID, 0x00)
            self._write(self.OPR_MODE, self.MODE_CONFIG)
            time.sleep(0.03)
            self._write(self.SYS_TRIGGER, 0x20)  # reset
            # The chip is off the bus for roughly 650 ms while it reboots, and
            # a fixed sleep is a guess: too short and every transfer after it
            # fails, which is exactly how this sensor came to be written off as
            # dead. Wait for it to answer instead.
            time.sleep(0.65)
            fcntl.ioctl(self.fd, I2C_SLAVE, addr)
            if not self._await_chip(2.0):
                raise RuntimeError("BNO055 did not come back after reset")
            self._write(self.PWR_MODE, 0x00)
            time.sleep(0.02)
            self._write(self.SYS_TRIGGER, 0x00)
            self._write(self.UNIT_SEL, 0x00)  # deg, dps, m/s^2
            time.sleep(0.02)
            self._write(self.OPR_MODE, self.MODE_NDOF)
            time.sleep(0.03)
        except Exception:
            self.close()
            raise

    # The BNO055 stretches the I2C clock, and the Pi's controller does not
    # handle that well: the first transaction after an idle period fails with
    # EREMOTEIO and the very next one succeeds. Measured on this car -- a bare
    # probe failed, then CHIP_ID returned 0xa0 on the second attempt. Retrying
    # is the whole difference between "no IMU" and a working heading reference,
    # so every transfer goes through these.
    RETRIES = 8
    RETRY_S = 0.006

    def _await_chip(self, timeout_s):
        """Poll CHIP_ID until the device answers with 0xA0, or give up."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                if self._read(self.CHIP_ID, 1)[0] == 0xA0:
                    return True
            except OSError:
                pass
            time.sleep(0.05)
        return False

    def _write(self, reg, val):
        last = None
        for _ in range(self.RETRIES):
            try:
                os.write(self.fd, bytes([reg, val]))
                return
            except OSError as exc:
                last = exc
                time.sleep(self.RETRY_S)
        raise last

    def _read(self, reg, n):
        last = None
        for _ in range(self.RETRIES):
            try:
                os.write(self.fd, bytes([reg]))
                return os.read(self.fd, n)
            except OSError as exc:
                last = exc
                time.sleep(self.RETRY_S)
        raise last

    @staticmethod
    def _s16(lo, hi):
        v = (hi << 8) | lo
        return v - 65536 if v & 0x8000 else v

    def calib(self):
        c = self._read(self.CALIB_STAT, 1)[0]
        return dict(sys=(c >> 6) & 3, gyro=(c >> 4) & 3,
                    accel=(c >> 2) & 3, mag=c & 3)

    def heading_deg(self):
        d = self._read(self.EUL_HEADING_LSB, 2)
        return self._s16(d[0], d[1]) / 16.0

    def read(self):
        """-> (ax, ay, az) m/s^2, (gx, gy, gz) rad/s -- matches MPU6050."""
        a = self._read(self.ACC_DATA_X_LSB, 6)
        g = self._read(self.GYR_DATA_X_LSB, 6)
        acc = tuple(self._s16(a[i], a[i + 1]) / 100.0 for i in (0, 2, 4))
        gyr = tuple(math.radians(self._s16(g[i], g[i + 1]) / 16.0)
                    for i in (0, 2, 4))
        return acc, gyr

    def close(self):
        try:
            os.close(self.fd)
        except Exception:
            pass


class ImuNode(Node):
    def __init__(self):
        super().__init__("imu")
        self.declare_parameters("", [
            ("bus", "/dev/i2c-1"),
            ("address", 0x68),
            ("gyro_range_dps", 500),
            ("dlpf", 3),
            ("rate_hz", 100.0),
            ("frame_id", "imu_link"),
            ("calibrate_s", 2.0),
            ("deadband_dps", 0.6),
        ])
        self.bus = str(self._p("bus"))
        self.addr = int(self._p("address"))
        self.gyro_range = int(self._p("gyro_range_dps"))
        self.dlpf = int(self._p("dlpf"))
        self.frame = str(self._p("frame_id"))
        self.calib_s = float(self._p("calibrate_s"))
        self.deadband = math.radians(float(self._p("deadband_dps")))

        self.pub = self.create_publisher(Imu, "imu/data", 10)
        self.pub_yaw = self.create_publisher(Float32, "imu/yaw", 10)
        self.create_subscription(String, "imu/cmd", self._on_cmd, 10)

        self.dev = None
        self.kind = None
        self.fails = 0
        self.absent_logged = False
        self._reset_state()
        self._probe()
        self.next_probe = self._now() + PROBE_PERIOD_S
        self.create_timer(1.0 / float(self._p("rate_hz")), self.tick)

    def _p(self, name):
        return self.get_parameter(name).value

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _reset_state(self):
        self.bias = [0.0, 0.0, 0.0]
        self.n_bias = 0
        self.calibrating = True
        self.yaw = 0.0
        self.yaw_ref = 0.0
        self.t0 = self._now()
        self.last = self.t0

    def _probe(self):
        # Report BOTH failures. Reporting only the second one meant a BNO055
        # that was present but mis-initialising showed up as the MPU6050's
        # "nothing at 0x68", which sent the diagnosis in the wrong direction
        # for a long time.
        bno_err = mpu_err = None
        try:
            self.dev = BNO055(self.bus, BNO055_ADDR)
            self.kind = "bno055"
        except Exception as exc:
            bno_err = exc
            try:
                self.dev = MPU6050(self.bus, self.addr,
                                   self.gyro_range, self.dlpf)
                self.kind = "mpu6050"
            except Exception as exc2:
                mpu_err = exc2
                if not self.absent_logged:
                    self.get_logger().warning(
                        "no IMU; probing every {:.0f} s. BNO055 at 0x{:02x}: "
                        "{}: {}. MPU6050 at 0x{:02x}: {}: {}".format(
                            PROBE_PERIOD_S, BNO055_ADDR,
                            type(bno_err).__name__, bno_err,
                            self.addr, type(mpu_err).__name__, mpu_err))
                    self.absent_logged = True
                return
        self.absent_logged = False
        self.fails = 0
        self._reset_state()
        if self.kind == "bno055":
            self.calibrating = False
            self.get_logger().info(
                "BNO055 up on {} (NDOF, absolute heading)".format(self.bus))
        else:
            self.get_logger().info(
                "MPU6050 up on {} (WHO_AM_I=0x{:02x}), {} dps; yaw is"
                " integrated and drifts, zero it at each corner".format(
                    self.bus, self.dev.who, self.gyro_range))

    def _detach(self):
        self.dev.close()
        self.dev = None
        self.kind = None
        self.absent_logged = True
        self.get_logger().error(
            "{} consecutive i2c read failures; detached, re-probing".format(
                MAX_READ_FAILS))

    def _on_cmd(self, msg):
        if msg.data.strip().lower() == "zero":
            if self.kind == "bno055":
                try:
                    self.yaw_ref = self.dev.heading_deg()
                except Exception:
                    pass
            self.yaw = 0.0
            self.get_logger().info("yaw zeroed")

    def tick(self):
        now = self._now()
        if self.dev is None:
            if now >= self.next_probe:
                self.next_probe = now + PROBE_PERIOD_S
                self._probe()
            return
        try:
            acc, gyr = self.dev.read()
        except Exception:
            self.fails += 1
            if self.fails >= MAX_READ_FAILS:
                self._detach()
                self.next_probe = now + PROBE_PERIOD_S
            return
        self.fails = 0

        if self.kind == "bno055":
            try:
                self.yaw = math.radians(self.dev.heading_deg() - self.yaw_ref)
            except Exception:
                pass
        elif self.calibrating:
            # assumes car stationary; averages zero-rate offset
            self.bias = [b + v for b, v in zip(self.bias, gyr)]
            self.n_bias += 1
            if (now - self.t0) >= self.calib_s and self.n_bias > 10:
                self.bias = [b / self.n_bias for b in self.bias]
                self.calibrating = False
                self.get_logger().info(
                    "gyro bias {:.4f} {:.4f} {:.4f} rad/s ({} samples)".format(
                        self.bias[0], self.bias[1], self.bias[2], self.n_bias))
            self.last = now
            return

        gx, gy, gz = (v - b for v, b in zip(gyr, self.bias))
        dt = max(1e-4, min(0.1, now - self.last))
        self.last = now
        if abs(gz) > self.deadband:  # stationary noise must not integrate
            self.yaw += gz * dt

        m = Imu()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.frame
        m.angular_velocity.x = gx
        m.angular_velocity.y = gy
        m.angular_velocity.z = gz
        m.linear_acceleration.x = acc[0]
        m.linear_acceleration.y = acc[1]
        m.linear_acceleration.z = acc[2]
        m.orientation_covariance[0] = -1.0  # no orientation estimate
        self.pub.publish(m)
        self.pub_yaw.publish(Float32(data=float(math.degrees(self.yaw))))


def main():
    rclpy.init()
    node = ImuNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node.dev is not None:
            node.dev.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
