"""
Plain-Python serial link to the steering Arduino (NO ROS dependency, so it can be
reused from steer_test.py or a unit test). Speaks the steering-fw v1 line protocol
(115200 baud, newline-terminated ASCII):

    PING      -> PONG steering-fw v1
    S <deg>   signed degrees, POSITIVE = steer RIGHT   -> OK S <deg>
    C         center                                    -> OK ...
    U <us>    raw servo pulse (microseconds)            -> OK ...
    GET       -> STATE ...
    (boot banner on reset:  READY steering-fw v1)

The firmware watchdog auto-centers ~1 s after the last command, so a caller that
wants to hold an angle must re-send it periodically.

Port autodetect scans /dev/serial/by-id/* symlinks: it SKIPS the YDLIDAR X2
(Silicon Labs CP210x, which must never be opened), prefers the Arduino Uno, then
an FTDI/CH340 Nano, else the first remaining entry.
"""
import collections
import glob
import re
import threading
import time

_EXCLUDE = re.compile(r"cp210|silicon", re.IGNORECASE)
_PREFER = re.compile(r"arduino|2341", re.IGNORECASE)
# The Nano actually in this robot is an FTDI FT232R (VID 0403), enumerating as
# usb-FTDI_FT232R_USB_UART_<serial>-if00-port0. None of the CH340 patterns match
# it ("usb_serial" does not match "USB_UART"), so before FTDI was listed here it
# only ever worked by falling through to candidates[0] -- i.e. by luck, because
# the lidar was the sole other device. A second USB-serial device sorting before
# it would have silently opened the wrong port.
_SECOND = re.compile(r"ftdi|ft232|0403|ch340|1a86|usb_serial|wch", re.IGNORECASE)


def autodetect_port():
    """Return the best /dev/serial/by-id/* symlink for the steering controller.

    Excludes the CP210x lidar; prefers Arduino, then a CH340 clone, then the first
    remaining entry. Raises RuntimeError (listing what was found) if nothing fits.
    """
    found = sorted(glob.glob("/dev/serial/by-id/*"))
    candidates = [p for p in found if not _EXCLUDE.search(p)]
    for pattern in (_PREFER, _SECOND):
        for path in candidates:
            if pattern.search(path):
                return path
    if candidates:
        return candidates[0]
    raise RuntimeError(
        "ArduinoLink: could not autodetect a steering port.\n"
        "  /dev/serial/by-id/* = {}\n"
        "  (entries matching cp210/silicon are skipped - that is the YDLIDAR lidar,\n"
        "   which must never be opened). Pass port='/dev/ttyACM0' (or /dev/ttyUSB*)\n"
        "   explicitly, or check the Arduino's USB cable.".format(found or "(none)")
    )


class ArduinoLink:
    """Thread-safe request/reply serial link to the steering firmware."""

    def __init__(self, port="auto", baud=115200, timeout=1.0):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._ser = None
        self._lock = threading.Lock()
        self.banner = None
        self.pong = None
        # USON makes the Nano emit "US ..." lines unprompted. They share the port
        # with command/reply traffic, so they are parked here rather than being
        # mistaken for a reply (or discarded); poll_stream() hands them over.
        self._stream = collections.deque(maxlen=200)

    @staticmethod
    def _import_serial():
        try:
            import serial
            return serial
        except ImportError as exc:
            raise ImportError(
                "pyserial is required for ArduinoLink. Install it with "
                "'apt install python3-serial' (Debian/Pi) or 'pip install pyserial'."
            ) from exc

    def connect(self):
        """Open the port, wait out the Uno auto-reset, drain the READY banner, and
           verify the PING/PONG handshake. Retries once, then raises RuntimeError."""
        serial = self._import_serial()
        port = autodetect_port() if self.port == "auto" else self.port
        self.port = port

        last_err = None
        for _attempt in (1, 2):
            try:
                self._ser = serial.Serial(port, self.baud, timeout=self.timeout)
                time.sleep(2.0)
                self._drain_banner()
                reply = self._write_read("PING")
                if "PONG" not in reply.upper():
                    raise RuntimeError("no PONG from steering fw (got {!r})".format(reply))
                self.pong = reply
                return self
            except Exception as exc:
                last_err = exc
                self._safe_close()
                time.sleep(0.5)
        raise RuntimeError(
            "ArduinoLink: handshake with steering fw on {} failed after 2 tries: {}"
            .format(port, last_err)
        )

    def _reconnect(self):
        self._safe_close()
        return self.connect()

    def _drain_banner(self):
        """Read lines until the READY banner or a read timeout (blank line)."""
        self.banner = None
        deadline = time.time() + 2.5
        while time.time() < deadline:
            line = self._ser.readline().decode("ascii", "replace").strip()
            if not line:
                break
            if "READY" in line.upper():
                self.banner = line
                break

    @staticmethod
    def _is_stream(line):
        """True for an unprompted sonar broadcast (USON), not a command reply."""
        return line.startswith("US ")

    def _readline(self):
        return self._ser.readline().decode("ascii", "replace").strip()

    def _soak_pending(self):
        """Park already-buffered stream lines instead of dropping them.

        This replaces a reset_input_buffer() that discarded whatever had arrived
        since the last command -- with USON running that is the entire sonar
        feed, which is why streaming appeared to be stuck at a trickle. Stale
        non-stream lines are still dropped, since resyncing is the point."""
        try:
            while self._ser.in_waiting:
                line = self._readline()
                if not line:
                    break
                if self._is_stream(line):
                    self._stream.append(line)
        except Exception:
            pass

    def poll_stream(self):
        """Pop the sensor lines that arrived unprompted since the last call."""
        with self._lock:
            if self._ser is not None:
                self._soak_pending()
            out = list(self._stream)
            self._stream.clear()
            return out

    def _write_read(self, cmd):
        """Write one command line and return the stripped reply line. Raises on any
           serial error or if the link is not open."""
        if self._ser is None:
            raise RuntimeError("ArduinoLink not connected; call connect() first")
        cmd = cmd.strip()
        verb = cmd.split()[0].upper() if cmd else ""
        self._soak_pending()
        self._ser.write((cmd + "\n").encode("ascii"))
        self._ser.flush()
        # A one-shot US reply is byte-identical to a USON broadcast, so only the
        # US verb may claim one; for every other verb a US line is stream noise
        # to be stepped over until the real reply shows up.
        claims_us = verb == "US"
        deadline = time.time() + max(self.timeout, 0.2) * 4.0
        while True:
            line = self._readline()
            if not line:
                return ""
            if self._is_stream(line) and not claims_us:
                self._stream.append(line)
                if time.time() >= deadline:
                    return ""
                continue
            return line

    def send(self, cmd):
        """Thread-safe command/reply. On a serial error, drop the handle, try ONE
           reconnect and retry; if that also fails, raise RuntimeError."""
        with self._lock:
            try:
                return self._write_read(cmd)
            except Exception as first:
                self._safe_close()
                try:
                    self._reconnect()
                    return self._write_read(cmd)
                except Exception as second:
                    raise RuntimeError(
                        "ArduinoLink.send({!r}) failed and reconnect did not "
                        "recover: {}".format(cmd, second)
                    ) from first

    def steer(self, deg):
        """Steer to signed degrees (POSITIVE = RIGHT). Returns the firmware reply."""
        return self.send("S {:.1f}".format(float(deg)))

    def drive(self, pct):
        """Drive motor duty, signed percent (POSITIVE = forward, 0 = stop/coast)."""
        return self.send("M {:.0f}".format(float(pct)))

    def center(self):
        return self.send("C")

    def raw(self, us):
        """Send a raw servo pulse in microseconds."""
        return self.send("U {:d}".format(int(us)))

    def state(self):
        return self.send("GET")

    @property
    def connected(self):
        return self._ser is not None

    def _safe_close(self):
        try:
            if self._ser is not None:
                self._ser.close()
        except Exception:
            pass
        self._ser = None

    def close(self):
        with self._lock:
            self._safe_close()

    def __enter__(self):
        if self._ser is None:
            self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
