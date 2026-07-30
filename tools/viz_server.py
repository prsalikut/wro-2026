"""WRO robot test console + live dashboard (viz_server v3).

  http://<pi>:8080/          -> test console (modes, controls, health, stream)
  http://<pi>:8080/stream    -> raw MJPEG stream
  http://<pi>:8080/snapshot  -> single JPEG

Three operating modes (server-enforced safety caps, persisted across restarts):
  desktop   parts loose on the desk: per-component tests, motor <=40% / 2 s,
            autonomy BLOCKED.
  assembly  mounted + calibrating: everything in desktop plus calibration
            wizards (servo us-sweep, trim, motor start-duty, camera sliders,
            lighting margins, lidar<->camera alignment), motor <=60% / 3 s.
  final     on the car: pre-flight checklist, autonomy + cruise control,
            mission log, motor <=70%.

Control paths:
  - docker API over /var/run/docker.sock (stack container, node exec)
  - ROS topics from THIS process's node: /arduino_cmd -> steering_bridge manual
    channel (bridge suspends autonomy + feeds the firmware watchdog with PINGs
    while a manual window is open), /arduino_reply, /bridge_status.

Run in a container with --network host --ipc host and
  -v /var/run/docker.sock:/var/run/docker.sock -v /home/pi/pitest:/out
"""
import glob
import json
import math
import os
import shutil
import socket
import threading
import time
from collections import deque

import numpy as np, cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Float32, String
from vision_msgs.msg import Detection3DArray
from cv_bridge import CvBridge
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DOCKER_SOCK = "/var/run/docker.sock"
STACK = "signstack"
STATE_FILE = "/out/panel_state.json"
EVENT_FILE = "/out/panel_events.log"

CAL_SIGN = 1.0
CAL_OFF = math.radians(-93.7)
LIDAR_DX = -0.195

BG = (18, 18, 20)
FRAME = {"jpg": None, "q": 78}

MODES = ("desktop", "assembly", "final")
MODE_CAPS = {
    "desktop":  dict(duty=40.0, dur=2.0, autonomy=False),
    "assembly": dict(duty=60.0, dur=3.0, autonomy=True),
    "final":    dict(duty=70.0, dur=3.0, autonomy=True),
}
SERVO_US_MIN, SERVO_US_MAX = 850, 2150

LOCK = threading.RLock()
PANEL_EPOCH = int(time.time())
S = {
    "mode": "desktop",
    "camera_overrides": {},
    "motor_start_duty": None,
    "us_left": None, "us_right": None,
    "trim": 0.0,
}
LIVE = {
    "cam": None, "raw": None, "scan": None, "det": None,
    "steer": 0.0, "drive": 0.0, "drive_msg": None,
    "bridge": None, "bridge_time": None,
    "dets": [],
}
RATES = {"cam": deque(maxlen=200), "scan": deque(maxlen=100),
         "det": deque(maxlen=200), "raw": deque(maxlen=200)}
EVENTS = deque(maxlen=500)
EVENT_SEQ = [0]
REPLIES = deque(maxlen=50)
REPLY_SEQ = [0]
MISSION = deque(maxlen=100)

RC_STEER_LIMIT = 25.0
RC_DRIVE_CEIL = 70.0
RC = {"active": False, "max_pct": 55.0, "steer": 0.0, "drive": 0.0, "last": 0.0}


def now():
    return time.time()


def log_event(kind, msg):
    with LOCK:
        EVENT_SEQ[0] += 1
        e = {"i": EVENT_SEQ[0], "t": round(now(), 1), "kind": kind, "msg": msg}
        EVENTS.append(e)
    try:
        with open(EVENT_FILE, "a") as f:
            f.write("{}\t{}\t{}\n".format(time.strftime("%H:%M:%S"), kind, msg))
    except OSError:
        pass


def load_state():
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
        with LOCK:
            for k in S:
                if k in data:
                    S[k] = data[k]
            if S["mode"] not in MODES:
                S["mode"] = "desktop"
    except (OSError, ValueError):
        pass


def save_state():
    try:
        with LOCK:
            blob = json.dumps(S)
        with open(STATE_FILE, "w") as f:
            f.write(blob)
    except OSError:
        pass


def rate_of(key, window=3.0):
    t0 = now() - window
    with LOCK:
        n = sum(1 for t in RATES[key] if t >= t0)
    return n / window


def _sock_request(req_bytes, timeout):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(DOCKER_SOCK)
        s.sendall(req_bytes)
        data = b""
        while True:
            try:
                chunk = s.recv(65536)
            except socket.timeout:
                break
            if not chunk:
                break
            data += chunk
    finally:
        s.close()
    return data


def docker_api(method, path, body=None, timeout=15):
    payload = b"" if body is None else json.dumps(body).encode()
    req = ("{} {} HTTP/1.0\r\nHost: docker\r\nContent-Type: application/json\r\n"
           "Content-Length: {}\r\n\r\n").format(method, path, len(payload)).encode() + payload
    data = _sock_request(req, timeout)
    head, _, resp = data.partition(b"\r\n\r\n")
    try:
        status = int(head.split()[1])
    except (IndexError, ValueError):
        return 0, b""
    return status, resp


_STACK_CACHE = {"t": 0.0, "v": None, "autonomy": None, "auto_t": 0.0}


def stack_running():
    """NON-BLOCKING: returns the cached state maintained by stack_poller().
    Never call the docker socket from ROS callbacks/timers (it would stall
    the single-threaded rclpy executor for up to the socket timeout)."""
    return _STACK_CACHE["v"]


def _stack_probe():
    try:
        st, body = docker_api("GET", "/containers/{}/json".format(STACK), timeout=6)
        return bool(json.loads(body)["State"]["Running"]) if st == 200 else None
    except Exception:
        return None


def stack_start():
    st, _ = docker_api("POST", "/containers/{}/start".format(STACK))
    _STACK_CACHE["v"] = _stack_probe()
    return st in (204, 304)


def stack_stop():
    st, _ = docker_api("POST", "/containers/{}/stop?t=5".format(STACK), timeout=25)
    _STACK_CACHE["v"] = _stack_probe()
    return st in (204, 304)


def stack_poller():
    """Background: keeps the stack + autonomy liveness caches fresh so no ROS
    callback ever blocks on the docker socket. Also re-applies firmware trim
    when the bridge link comes back (Nano resets lose it)."""
    prev_link = None
    while True:
        _STACK_CACHE["v"] = _stack_probe()
        t = now()
        if _STACK_CACHE["v"] and t - _STACK_CACHE["auto_t"] > 5.0:
            try:
                _STACK_CACHE["autonomy"] = autonomy_running()
            except Exception:
                _STACK_CACHE["autonomy"] = None
            _STACK_CACHE["auto_t"] = t
            if (_STACK_CACHE["autonomy"]
                    and not MODE_CAPS[S["mode"]]["autonomy"]):
                autonomy_stop()
                _STACK_CACHE["autonomy"] = False
                log_event("mode", "{} mode: autonomy auto-killed after stack "
                                  "start".format(S["mode"]))
        elif not _STACK_CACHE["v"]:
            _STACK_CACHE["autonomy"] = False
        b = LIVE.get("bridge")
        link = bool(b and b.get("connected")) if b else None
        if link and prev_link is False and S.get("trim"):
            node = NODE.get("n")
            if node is not None:
                r = node.send_raw("TRIM {:.1f}".format(float(S["trim"])))
                node.send_raw("RESUME", wait=1.0)
                log_event("calib", "re-applied trim {} after link return -> {}".format(
                    S["trim"], r))
        prev_link = link
        time.sleep(2)


def exec_detached(cmd):
    st, body = docker_api("POST", "/containers/{}/exec".format(STACK),
                          {"Cmd": ["bash", "-c", cmd], "AttachStdout": False,
                           "AttachStderr": False, "Detach": True})
    if st != 201:
        return False
    eid = json.loads(body)["Id"]
    st, _ = docker_api("POST", "/exec/{}/start".format(eid), {"Detach": True})
    return st in (200, 204)


def exec_capture(cmd, timeout=25):
    """Run a command in the stack container and return (ok, combined_output)."""
    st, body = docker_api("POST", "/containers/{}/exec".format(STACK),
                          {"Cmd": ["bash", "-c", cmd], "AttachStdout": True,
                           "AttachStderr": True})
    if st != 201:
        return False, "exec create failed (stack running?)"
    eid = json.loads(body)["Id"]
    payload = json.dumps({"Detach": False, "Tty": False}).encode()
    req = ("POST /exec/{}/start HTTP/1.0\r\nHost: docker\r\n"
           "Content-Type: application/json\r\nContent-Length: {}\r\n\r\n"
           ).format(eid, len(payload)).encode() + payload
    data = _sock_request(req, timeout)
    _, _, stream = data.partition(b"\r\n\r\n")
    out = b""
    i = 0
    while i + 8 <= len(stream):
        ln = int.from_bytes(stream[i + 4:i + 8], "big")
        out += stream[i + 8:i + 8 + ln]
        i += 8 + ln
    try:
        st2, meta = docker_api("GET", "/exec/{}/json".format(eid))
        code = json.loads(meta).get("ExitCode") if st2 == 200 else None
    except Exception:
        code = None
    return (code == 0), out.decode("utf-8", "replace").strip()


ROS_PREFIX = ("source /opt/ros/humble/setup.bash && "
              "source /ros2_ws/install/setup.bash 2>/dev/null && ")

AUTONOMY_KILL = ("for d in /proc/[0-9]*; do if tr '\\0' ' ' < \"$d/cmdline\" 2>/dev/null"
                 " | grep -q sign_steering; then kill \"${d#/proc/}\"; fi; done")


def autonomy_running():
    ok, out = exec_capture(
        "for d in /proc/[0-9]*; do tr '\\0' ' ' < \"$d/cmdline\" 2>/dev/null"
        " | grep -q sign_steering && echo RUNNING && break; done; true", timeout=8)
    return ok and "RUNNING" in out


def autonomy_start(cruise=None, slow=None):
    exec_detached(AUTONOMY_KILL)
    time.sleep(0.6)
    overrides = ""
    if cruise is not None:
        overrides += " -p drive_cruise_pct:={}".format(float(cruise))
    if slow is not None:
        overrides += " -p slow_factor:={}".format(float(slow))
    cmd = (ROS_PREFIX + "nohup ros2 run sign_detector sign_steering --ros-args "
           "-r __node:=sign_steering --params-file "
           "/ros2_ws/install/sign_detector/share/sign_detector/config/params.yaml"
           + overrides + " >/tmp/sign_steering.log 2>&1 &")
    return exec_detached(cmd)


def autonomy_stop():
    return exec_detached(AUTONOMY_KILL)


def pi_health():
    temp = load = None
    undervolt = None
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            temp = round(int(f.read().strip()) / 1000.0, 1)
    except Exception:
        pass
    try:
        with open("/proc/loadavg") as f:
            load = float(f.read().split()[0])
    except Exception:
        pass
    try:
        for hw in glob.glob("/sys/class/hwmon/hwmon*/name"):
            with open(hw) as f:
                if "rpi_volt" in f.read():
                    with open(os.path.join(os.path.dirname(hw), "in0_lalarm")) as f2:
                        undervolt = f2.read().strip() == "1"
                    break
    except Exception:
        pass
    disk_free_gb = None
    try:
        disk_free_gb = round(shutil.disk_usage("/").free / 1e9, 1)
    except Exception:
        pass
    return temp, load, undervolt, disk_free_gb


NODE = {"n": None}


class VizNode(Node):
    def __init__(self):
        super().__init__("viz_server")
        self.br = CvBridge()
        self.img = None
        self.raw_img = None
        self.scan = None
        self.dets = []
        self.create_subscription(Image, "/sign_debug", self._img, 10)
        self.create_subscription(Image, "/image_raw", self._raw, qos_profile_sensor_data)
        self.create_subscription(LaserScan, "/scan", self._scan, qos_profile_sensor_data)
        self.create_subscription(Detection3DArray, "/traffic_signs", self._det, 10)
        self.create_subscription(Float32, "/steering_cmd", self._steer, 10)
        self.create_subscription(Float32, "/drive_cmd", self._drive, 10)
        self.create_subscription(String, "/arduino_reply", self._reply, 10)
        self.create_subscription(String, "/bridge_status", self._bstat, 10)
        self.pub_cmd = self.create_publisher(String, "/arduino_cmd", 10)
        self.pub_steer = self.create_publisher(Float32, "steering_cmd", 10)
        self.pub_drive = self.create_publisher(Float32, "drive_cmd", 10)
        self.create_timer(1 / 15.0, self.render)
        self.create_timer(0.05, self._rc_tick)
        self._pass_track = {}

    def rc_publish(self, steer_deg, drive_pct):
        ms = Float32(); ms.data = float(steer_deg); self.pub_steer.publish(ms)
        md = Float32(); md.data = float(drive_pct); self.pub_drive.publish(md)

    def _rc_tick(self):
        """While RC is armed, republish the latest teleop command at a steady
        20 Hz (the browser POST rate jitters). Zero the DRIVE if no browser
        command has arrived for >0.5 s -- steering holds, but the motor must
        not coast on a stale command. The bridge's 1 s failsafe backs this up."""
        if not RC["active"]:
            return
        stale = now() - RC["last"] > 0.5
        self.rc_publish(RC["steer"], 0.0 if stale else RC["drive"])

    def _img(self, m):
        self.img = self.br.imgmsg_to_cv2(m, "bgr8")
        LIVE["cam"] = now()
        with LOCK:
            RATES["cam"].append(LIVE["cam"])

    def _raw(self, m):
        self.raw_img = self.br.imgmsg_to_cv2(m, "bgr8")
        LIVE["raw"] = now()
        with LOCK:
            RATES["raw"].append(LIVE["raw"])

    def _scan(self, m):
        self.scan = m
        LIVE["scan"] = now()
        with LOCK:
            RATES["scan"].append(LIVE["scan"])

    def _det(self, m):
        self.dets = list(m.detections)
        LIVE["det"] = now()
        with LOCK:
            RATES["det"].append(LIVE["det"])
        dets = []
        for d in self.dets:
            if not d.results:
                continue
            p = d.results[0].pose.pose.position
            dets.append((d.results[0].hypothesis.class_id,
                         round(p.x, 2), round(p.y, 2),
                         round(math.hypot(p.x, p.y), 2)))
        LIVE["dets"] = dets
        self._mission_track(dets)

    def _steer(self, m):
        LIVE["steer"] = float(m.data)

    def _drive(self, m):
        LIVE["drive"] = float(m.data)
        LIVE["drive_msg"] = now()

    def _reply(self, m):
        with LOCK:
            REPLY_SEQ[0] += 1
            REPLIES.append((REPLY_SEQ[0], m.data, now()))

    def _bstat(self, m):
        try:
            LIVE["bridge"] = json.loads(m.data)
            LIVE["bridge_time"] = now()
        except ValueError:
            pass

    def _mission_track(self, dets):
        t = now()
        seen = {c: dist for c, _x, _y, dist in dets if dist < 1.2}
        for color, dist in seen.items():
            rec = self._pass_track.get(color)
            if rec is None:
                self._pass_track[color] = {"t0": t, "min": dist, "last": t}
                side = "RIGHT" if color == "red" else "LEFT"
                MISSION.append({"t": round(t, 1),
                                "msg": "ENGAGE {} @ {:.2f}m -> steer {}".format(
                                    color, dist, side)})
            else:
                rec["min"] = min(rec["min"], dist)
                rec["last"] = t
        for color in list(self._pass_track):
            rec = self._pass_track[color]
            if t - rec["last"] > 1.0:
                MISSION.append({"t": round(t, 1),
                                "msg": "CLEAR {} (closest {:.2f}m, {:.1f}s)".format(
                                    color, rec["min"], rec["last"] - rec["t0"])})
                del self._pass_track[color]

    RAW_LOCK = threading.Lock()

    def send_raw(self, cmd, wait=2.5):
        """Publish to the bridge manual channel and wait for its reply.
        Serialized: concurrent callers would otherwise steal each other's
        replies (the bridge tags no correlation id)."""
        with VizNode.RAW_LOCK:
            with LOCK:
                last = REPLY_SEQ[0]
            m = String()
            m.data = cmd
            self.pub_cmd.publish(m)
            t0 = now()
            while now() - t0 < wait:
                with LOCK:
                    for seq, text, _t in reversed(REPLIES):
                        if seq > last:
                            return text
                time.sleep(0.05)
            return None

    def lidar_nearest(self, half_deg=30.0, max_r=2.0):
        """Nearest cluster in the forward wedge, camera frame. For bench checks."""
        s = self.scan
        if s is None or LIVE["scan"] is None or now() - LIVE["scan"] > 2.0:
            return None
        pts = []
        a = s.angle_min
        for r in s.ranges:
            if math.isfinite(r) and s.range_min < r < max_r + abs(LIDAR_DX):
                phi = (a - CAL_OFF) / CAL_SIGN
                x = LIDAR_DX + r * math.cos(phi)
                y = r * math.sin(phi)
                if x > 0.03 and abs(math.degrees(math.atan2(y, x))) <= half_deg:
                    pts.append((math.hypot(x, y), math.degrees(math.atan2(y, x))))
            a += s.angle_increment
        if not pts:
            return None
        pts.sort()
        near = [p for p in pts if p[0] - pts[0][0] < 0.08]
        dist = sum(p[0] for p in near) / len(near)
        bear = sum(p[1] for p in near) / len(near)
        return {"dist": round(dist, 3), "bearing": round(bear, 1), "beams": len(near)}

    def camera_metrics(self):
        img = self.raw_img
        if img is None or LIVE["raw"] is None or now() - LIVE["raw"] > 2.0:
            return None
        H = img.shape[0]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fg = gray[H // 3:, :]
        b, g, r = [float(c.mean()) for c in cv2.split(img[H // 3:, :])]
        return {"fg_luma": int(np.median(fg)),
                "clip_pct": round(float((gray > 250).mean() * 100), 2),
                "cast_rg": round(r - g, 1), "cast_rb": round(r - b, 1)}

    def render(self):
        t = now()
        cam_fresh = LIVE["cam"] is not None and t - LIVE["cam"] < 3.0
        raw_fresh = LIVE["raw"] is not None and t - LIVE["raw"] < 3.0
        if cam_fresh:
            cam = self.img
        elif raw_fresh:
            cam = self.raw_img.copy()
            cv2.putText(cam, "RAW CAMERA (detector down)", (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 160, 255), 2)
        else:
            cam = self.img if self.img is not None else np.full((480, 640, 3), 30,
                                                                np.uint8)
        cam = cv2.resize(cam, (640, 480))
        if not cam_fresh and not raw_fresh:
            cam = (cam * 0.35).astype(np.uint8)
            msg = "STACK OFF" if stack_running() is False else "waiting for camera..."
            cv2.putText(cam, msg, (170, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (80, 160, 255), 2)
        cam_fresh = cam_fresh or raw_fresh
        bird = birdseye(self.scan if (LIVE["scan"] and t - LIVE["scan"] < 3.0) else None,
                        self.dets if cam_fresh else [])
        hdr = 44
        dash = np.full((480 + hdr, 640 + bird.shape[1], 3), BG, np.uint8)
        dash[hdr:hdr + 480, 0:640] = cam
        dash[hdr:hdr + bird.shape[0], 640:640 + bird.shape[1]] = bird
        nd = len(LIVE["dets"]) if cam_fresh else 0
        cv2.putText(dash, "WRO test console  |  mode: {}".format(S["mode"].upper()),
                    (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 238, 244), 1)
        drv = ("{:+.0f}%".format(LIVE["drive"])
               if (LIVE["drive_msg"] and t - LIVE["drive_msg"] < 1.5) else "OFF")
        cv2.putText(dash, "CAM {} LIDAR {} DET {} STEER {:+.0f} DRV {}".format(
                        "OK" if cam_fresh else "--",
                        "OK" if (LIVE["scan"] and t - LIVE["scan"] < 3.0) else "OFF",
                        nd, LIVE["steer"], drv),
                    (640 + 12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 200, 255), 1)
        ok, buf = cv2.imencode(".jpg", dash, [cv2.IMWRITE_JPEG_QUALITY, FRAME["q"]])
        if ok:
            FRAME["jpg"] = buf.tobytes()


def birdseye(scan, dets, size=480, maxr=2.5):
    im = np.full((size, size, 3), 22, np.uint8)
    cx, cy = size // 2, size // 2
    sc = (size // 2 - 26) / maxr
    for rr in (0.5, 1.0, 1.5, 2.0):
        cv2.circle(im, (cx, cy), int(rr * sc), (48, 54, 64), 1)
        cv2.putText(im, "{:.1f}m".format(rr), (cx + 3, cy - int(rr * sc) + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 86, 96), 1)
    cv2.line(im, (cx, cy), (cx, cy - int(maxr * sc)), (46, 66, 90), 1)
    cv2.putText(im, "FWD", (cx + 6, cy - int(maxr * sc) + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 110, 150), 1)
    npts = 0
    if scan is not None:
        a = scan.angle_min
        for r in scan.ranges:
            if math.isfinite(r) and scan.range_min < r < maxr + abs(LIDAR_DX):
                phi = (a - CAL_OFF) / CAL_SIGN
                x = LIDAR_DX + r * math.cos(phi)
                y = r * math.sin(phi)
                px = int(cx - y * sc); py = int(cy - x * sc)
                if 0 <= px < size and 0 <= py < size:
                    cv2.circle(im, (px, py), 1, (120, 132, 144), -1); npts += 1
            a += scan.angle_increment
    lpx, lpy = cx, cy - int(LIDAR_DX * sc)
    cv2.circle(im, (lpx, lpy), 3, (140, 140, 150), -1)
    cv2.circle(im, (cx, cy), 5, (0, 195, 255), -1)
    cv2.putText(im, "cam", (cx - 12, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 195, 255), 1)
    for d in dets:
        if not d.results:
            continue
        cid = d.results[0].hypothesis.class_id
        p = d.results[0].pose.pose.position
        px = int(cx - p.y * sc); py = int(cy - p.x * sc)
        col = (0, 0, 255) if cid == "red" else (0, 200, 0)
        cv2.circle(im, (px, py), 8, col, -1)
        cv2.circle(im, (px, py), 8, (240, 240, 240), 1)
        cv2.putText(im, "{} {:.2f}m".format(cid, p.x),
                    (min(px + 11, size - 96), py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1)
    cv2.putText(im, "LIDAR TOP-DOWN  {} pts".format(npts), (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 206, 216), 1)
    return im


BUSY = {"actuator": threading.Lock()}
ABORT = {"epoch": 0}


def _aborted(my_epoch):
    return ABORT["epoch"] != my_epoch


def act_servo(action, deg=None):
    node = NODE["n"]
    if not BUSY["actuator"].acquire(blocking=False):
        return {"ok": False, "err": "another actuator sequence is running"}
    epoch = ABORT["epoch"]

    def run():
        try:
            if action == "center":
                r = node.send_raw("C")
                log_event("servo", "center -> {}".format(r))
            elif action == "angle":
                d = max(-25.0, min(25.0, float(deg)))
                r = node.send_raw("S {:.1f}".format(d))
                log_event("servo", "S {:.1f} -> {}".format(d, r))
            elif action == "sweep":
                for cmd, hold in (("S 25", 1.2), ("C", 0.8), ("S -25", 1.2), ("C", 0.2)):
                    if _aborted(epoch):
                        log_event("servo", "sweep aborted")
                        return
                    r = node.send_raw(cmd)
                    log_event("servo", "{} -> {}".format(cmd, r))
                    time.sleep(hold)
        finally:
            BUSY["actuator"].release()

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True}


def act_motor(duty, dur):
    caps = MODE_CAPS[S["mode"]]
    try:
        duty = float(duty)
        dur = float(dur)
    except (TypeError, ValueError):
        return {"ok": False, "err": "bad duty/dur"}
    if not (math.isfinite(duty) and math.isfinite(dur)):
        return {"ok": False, "err": "bad duty/dur"}
    duty = max(-caps["duty"], min(caps["duty"], duty))
    dur = max(0.2, min(caps["dur"], dur))
    node = NODE["n"]
    if not BUSY["actuator"].acquire(blocking=False):
        return {"ok": False, "err": "another actuator sequence is running"}
    epoch = ABORT["epoch"]

    def run():
        try:
            if _aborted(epoch):
                return
            r = node.send_raw("M {:.0f}".format(duty))
            log_event("motor", "M {:.0f} for {:.1f}s -> {}".format(duty, dur, r))
            t_end = now() + dur
            while now() < t_end:
                if _aborted(epoch):
                    break
                time.sleep(0.05)
        finally:
            r2 = node.send_raw("M 0")
            if r2 is None or "OK" not in str(r2):
                r2b = node.send_raw("M 0")
                if r2b is None or "OK" not in str(r2b):
                    log_event("ESTOP", "motor stop unacknowledged -> stack stop fallback")
                    stack_stop()
                r2 = r2b
            log_event("motor", "M 0 -> {}".format(r2))
            BUSY["actuator"].release()

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True, "applied_duty": duty, "dur": dur}


def act_estop():
    """Instant preempt of everything, with verification and a guaranteed
    fallback: if the bridge/serial chain does not ACK the stop, stop the whole
    stack container -- the firmware watchdog then brakes+centers within ~1 s."""
    ABORT["epoch"] += 1
    RC["active"] = False
    node = NODE["n"]
    node.rc_publish(0.0, 0.0)
    r1 = node.send_raw("M 0", wait=1.5)
    r2 = node.send_raw("C", wait=1.5)
    autonomy_stop()
    acked = r1 is not None and "OK" in str(r1)
    fallback = None
    if not acked:
        fallback = "stack stopped (watchdog brake)" if stack_stop() else "STACK STOP FAILED"
    log_event("ESTOP", "M0={} C={} autonomy killed{}".format(
        r1, r2, " | FALLBACK: " + fallback if fallback else ""))
    return {"ok": acked or (fallback is not None and "FAILED" not in fallback),
            "acked": acked, "motor": r1, "servo": r2, "fallback": fallback}


def rc_engage():
    """Take manual control: stop the autonomous driver so it can't fight the
    teleop topics, close any open manual window (so the bridge forwards our
    steering_cmd/drive_cmd), and ARM the 20 Hz republish tick. The robot stays
    still (drive 0) until the operator holds the deadman AND pushes throttle."""
    autonomy_stop()
    node = NODE["n"]
    node.send_raw("RESUME", wait=1.0)
    with LOCK:
        RC["active"] = True
        RC["steer"] = 0.0
        RC["drive"] = 0.0
        RC["last"] = now()
    node.rc_publish(0.0, 0.0)
    log_event("rc", "RC ENGAGED (autonomy stopped, manual control armed)")
    return {"ok": True, "max_pct": RC["max_pct"]}


def rc_command(steer, drive, deadman, max_pct=None):
    """One teleop sample from the browser. Clamps hard; DRIVE is forced to 0
    unless the deadman is held. Does not arm RC (only rc_engage does, so an
    E-STOP stays latched); the 20 Hz _rc_tick republishes RC[] so the browser's
    jittery POST rate cannot open gaps in the command stream."""
    try:
        steer = float(steer); drive = float(drive)
    except (TypeError, ValueError):
        return {"ok": False, "err": "bad steer/drive"}
    if not (math.isfinite(steer) and math.isfinite(drive)):
        return {"ok": False, "err": "non-finite"}
    with LOCK:
        if max_pct is not None:
            try:
                RC["max_pct"] = max(0.0, min(RC_DRIVE_CEIL, float(max_pct)))
            except (TypeError, ValueError):
                pass
        cap = RC["max_pct"]
        RC["steer"] = max(-RC_STEER_LIMIT, min(RC_STEER_LIMIT, steer))
        RC["drive"] = max(-cap, min(cap, drive)) if deadman else 0.0
        RC["last"] = now()
        armed, out_s, out_d = RC["active"], RC["steer"], RC["drive"]
    return {"ok": True, "active": armed, "steer": out_s, "drive": out_d,
            "max_pct": cap, "armed": bool(deadman)}


def rc_release():
    """Operator let go / closed the tab: stop the motor now and disarm."""
    node = NODE["n"]
    with LOCK:
        RC["active"] = False
        RC["steer"] = 0.0
        RC["drive"] = 0.0
    node.rc_publish(0.0, 0.0)
    log_event("rc", "RC released (motor stop, disarmed)")
    return {"ok": True}


def preflight():
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    st = stack_running()
    add("stack container", st is True, "running" if st else "NOT RUNNING")
    cam_r = rate_of("cam")
    add("camera stream", cam_r > 10, "{:.1f} fps".format(cam_r))
    scan_r = rate_of("scan")
    add("lidar scan", scan_r > 5, "{:.1f} Hz".format(scan_r))
    det_r = rate_of("det")
    add("detector", det_r > 10, "{:.1f} Hz".format(det_r))
    b = LIVE["bridge"]
    fresh = b is not None and LIVE["bridge_time"] and now() - LIVE["bridge_time"] < 3
    add("bridge status", fresh and b.get("connected"),
        json.dumps(b) if fresh else "no /bridge_status")
    manual_open = bool(fresh and b.get("manual"))
    if manual_open:
        add("firmware state", False, "SKIPPED: manual window open (resume first)")
    elif fresh:
        fw = NODE["n"].send_raw("GET")
        fw_ok = (fw is not None and "STATE" in fw
                 and "lim=25.0/25.0" in fw and "wd=1000" in fw)
        add("firmware state", fw_ok, fw or "no reply")
        NODE["n"].send_raw("RESUME", wait=1.0)
    else:
        add("firmware state", False, "bridge not reporting")
    auto = _STACK_CACHE.get("autonomy")
    add("autonomy node", auto is True,
        "running" if auto else "NOT running (start it before GO)")
    sd = S.get("motor_start_duty")
    add("motor start-duty measured", sd is not None,
        "{}% - ensure cruise x slow_factor >= this".format(sd) if sd is not None
        else "not measured (assembly mode: start-duty finder)")
    temp, load, undervolt, disk = pi_health()
    add("cpu temp", temp is not None and temp < 75, "{} C".format(temp))
    add("cpu load", load is not None and load < 4.0, str(load))
    add("power (undervoltage)", undervolt is not True,
        "UNDERVOLT!" if undervolt else ("ok" if undervolt is False else "unknown"))
    add("disk free", disk is not None and disk > 1.0, "{} GB".format(disk))
    add("mode", S["mode"] == "final", S["mode"])
    ok_all = all(c["ok"] for c in checks)
    log_event("preflight", "PASS" if ok_all else "FAIL")
    return {"ok": ok_all, "checks": checks}


LIGHTING_SNIPPET = r'''
import json, time, math, sys
import numpy as np, cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
sys.path.insert(0, "/ros2_ws/src/sign_detector")
from sign_detector.block_detector import BlockDetector, PROFILES
imgs = []
rclpy.init(); n = Node("light"); br = CvBridge()
n.create_subscription(Image, "/image_raw", lambda m: imgs.append(br.imgmsg_to_cv2(m, "bgr8")), qos_profile_sensor_data)
t0 = time.time()
while len(imgs) < 3 and time.time() - t0 < 8: rclpy.spin_once(n, timeout_sec=0.3)
if not imgs: print(json.dumps({"err": "no frames"})); raise SystemExit
det = BlockDetector(profile="lab", hfov_deg=60.0, sign_height_m=0.13)
p = PROFILES["lab"]; out = {"thresholds": p, "colors": {}}
for img in imgs[-3:]:
    lab = cv2.cvtColor(cv2.GaussianBlur(img,(5,5),0), cv2.COLOR_BGR2LAB)
    for d in det.detect(img):
        x1, y1 = d.cx - d.w//4, d.cy - d.h//4
        roi = lab[max(0,y1):y1+max(1,d.h//2), max(0,x1):x1+max(1,d.w//2)]
        if roi.size == 0: continue
        a = float(np.median(roi[:,:,1])); b = float(np.median(roi[:,:,2])); L = float(np.median(roi[:,:,0]))
        c = out["colors"].setdefault(d.color, {"a": [], "b": [], "L": [], "n": 0})
        c["a"].append(a); c["b"].append(b); c["L"].append(L); c["n"] += 1
rep = {}
for color, c in out["colors"].items():
    a = float(np.median(c["a"])); b = float(np.median(c["b"])); L = float(np.median(c["L"]))
    if color == "red":
        m = {"a_margin": round(a - p["RED_A_MIN"],1), "b_margin": round(b - p["RED_B_MIN"],1)}
    else:
        m = {"a_margin": round(p["GREEN_A_MAX"] - a,1), "b_margin": round(b - p["GREEN_B_MIN"],1)}
    m.update({"a": round(a,1), "b": round(b,1), "L": round(L,1), "frames": c["n"]})
    rep[color] = m
print(json.dumps({"thresholds": {k: v for k, v in p.items()}, "measured": rep}))
rclpy.shutdown()
'''


def test_lighting():
    if stack_running() is not True:
        return {"ok": False, "err": "stack not running"}
    cmd = (ROS_PREFIX + "python3 - <<'PYEOF'\n" + LIGHTING_SNIPPET + "\nPYEOF")
    ok, out = exec_capture(cmd, timeout=30)
    try:
        data = json.loads(out.splitlines()[-1])
    except (ValueError, IndexError):
        return {"ok": False, "err": out[-400:]}
    verdict = {}
    for color, m in data.get("measured", {}).items():
        margins = [v for k, v in m.items() if k.endswith("_margin")]
        verdict[color] = ("HEALTHY" if all(v >= 8 for v in margins)
                          else "MARGINAL" if all(v >= 3 for v in margins) else "AT RISK")
    log_event("lighting", json.dumps(verdict) if verdict else "no blocks in view")
    return {"ok": True, "report": data, "verdict": verdict,
            "hint": "place red + green blocks 0.4-0.8m ahead for a full check"}


def test_align():
    """Camera-bearing vs lidar-cluster bearing for the current detection."""
    dets = LIVE["dets"]
    if not dets:
        return {"ok": False, "err": "no detection in view; place one block ahead"}
    near = NODE["n"].lidar_nearest(half_deg=25.0)
    if near is None:
        return {"ok": False, "err": "no lidar cluster in the forward wedge"}
    color, x, y, dist = dets[0]
    cam_bear = round(math.degrees(math.atan2(y, x)), 1)
    delta = round(near["bearing"] - cam_bear, 1)
    ddist = round(near["dist"] - dist, 3)
    if abs(delta) > 3:
        verdict = "CHECK angle_offset_deg (bearing off {} deg)".format(delta)
    elif abs(ddist) > 0.3:
        verdict = ("CHECK scan plane / mount height (range disagrees by {} m - "
                   "lidar may be seeing past the block)".format(ddist))
    else:
        verdict = "ALIGNED"
    log_event("align", "cam {}d/{}m vs lidar {}d/{}m -> {}".format(
        cam_bear, dist, near["bearing"], near["dist"], verdict))
    return {"ok": True, "camera": {"color": color, "bearing": cam_bear, "dist": dist},
            "lidar": near, "delta_deg": delta, "delta_dist_m": ddist,
            "verdict": verdict}


def calib_suggest():
    l, r = S.get("us_left"), S.get("us_right")
    if not l or not r:
        return {"ok": False, "err": "mark both limits first (U-sweep, then Mark)"}
    lo, hi = min(l, r), max(l, r)
    if not (lo < 1450 and hi > 1550):
        return {"ok": False,
                "err": "marks invalid: need one limit below 1450 and one above "
                       "1550 us (got {} / {}) - re-mark each side".format(lo, hi)}
    half = min(1500 - lo, hi - 1500)
    margin = int(half * 0.92)
    deg2us = round(margin / 25.0, 1)
    return {"ok": True, "us_left": l, "us_right": r,
            "usable_half_range_us": half,
            "suggest": {"DEG2US": deg2us, "US_MIN": 1500 - margin,
                        "US_MAX": 1500 + margin, "LIM": 25},
            "note": "edit arduino/steering_firmware.ino with these and reflash"}


def camera_set(name, value):
    allowed = {"brightness", "contrast", "saturation", "sharpness", "gain",
               "backlight_compensation", "hue"}
    if name not in allowed:
        return {"ok": False, "err": "control not allowed"}
    v = int(value)
    ok, out = exec_capture(
        ROS_PREFIX + "ros2 param set /camera {} {}".format(name, v), timeout=12)
    if ok and "successful" in out.lower():
        with LOCK:
            S["camera_overrides"][name] = v
        save_state()
        log_event("camera", "{} = {}".format(name, v))
        return {"ok": True}
    return {"ok": False, "err": out[-200:]}


def overrides_watchdog():
    """Re-apply saved camera overrides whenever the stack (re)starts. Fully
    guarded: a docker hiccup must never kill this daemon thread."""
    prev = None
    while True:
        try:
            st = stack_running()
            if st and prev is False:
                time.sleep(10)
                with LOCK:
                    items = dict(S["camera_overrides"])
                failed = []
                for k, v in items.items():
                    ok, out = exec_capture(
                        ROS_PREFIX + "ros2 param set /camera {} {}".format(k, v),
                        timeout=12)
                    if not (ok and "successful" in out.lower()):
                        failed.append(k)
                if items:
                    log_event("camera", "re-applied overrides: {}{}".format(
                        items, " FAILED: {}".format(failed) if failed else ""))
            prev = st
        except Exception as exc:
            log_event("panel", "overrides watchdog error: {}".format(exc))
        time.sleep(3)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        try:
            ln = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(ln).decode()) if ln else {}
        except (ValueError, OSError):
            return {}

    def do_POST(self):
        try:
            self._post()
        except Exception as exc:
            self._json({"ok": False, "err": str(exc)}, 500)

    def _post(self):
        p = self.path
        body = self._body()
        mode = S["mode"]
        caps = MODE_CAPS[mode]

        if p == "/api/estop":
            self._json(act_estop()); return
        if p == "/api/rc/engage":
            self._json(rc_engage()); return
        if p == "/api/rc/release":
            self._json(rc_release()); return
        if p == "/api/rc":
            self._json(rc_command(body.get("steer", 0.0), body.get("drive", 0.0),
                                  bool(body.get("deadman", False)),
                                  body.get("max_pct"))); return
        if p == "/api/mode":
            m = str(body.get("mode", ""))
            if m not in MODES:
                self._json({"ok": False, "err": "bad mode"}, 400); return
            prev = S["mode"]
            with LOCK:
                S["mode"] = m
            save_state()
            ABORT["epoch"] += 1
            if not MODE_CAPS[m]["autonomy"]:
                autonomy_stop()
                log_event("mode", "{} -> {} (autonomy auto-stopped)".format(prev, m))
            else:
                log_event("mode", "{} -> {}".format(prev, m))
            self._json({"ok": True, "mode": m}); return
        if p == "/api/stack/start":
            self._json({"ok": stack_start()}); log_event("stack", "start"); return
        if p == "/api/stack/stop":
            autonomy_stop()
            self._json({"ok": stack_stop()}); log_event("stack", "stop"); return
        if p == "/api/autonomy/start":
            if not caps["autonomy"]:
                self._json({"ok": False,
                            "err": "autonomy blocked in {} mode".format(mode)}, 403)
                return
            cruise = body.get("cruise")
            slow = body.get("slow")
            try:
                if cruise is not None:
                    cruise = float(cruise)
                    if not math.isfinite(cruise):
                        raise ValueError
                    cruise = 0.0 if cruise <= 0 else max(40.0, min(caps["duty"], cruise))
                if slow is not None:
                    slow = float(slow)
                    if not math.isfinite(slow):
                        raise ValueError
                    slow = max(0.5, min(1.0, slow))
            except (TypeError, ValueError):
                self._json({"ok": False, "err": "bad cruise/slow"}, 400); return
            RC["active"] = False
            NODE["n"].send_raw("RESUME", wait=1.0)
            ok = autonomy_start(cruise, slow)
            time.sleep(1.5)
            alive = autonomy_running()
            _STACK_CACHE["autonomy"] = alive
            _STACK_CACHE["auto_t"] = now()
            log_event("autonomy", "start cruise={} -> {}".format(
                cruise, "RUNNING" if alive else "NOT DETECTED (check logs)"))
            self._json({"ok": ok and alive, "running": alive,
                        "cruise": cruise}); return
        if p == "/api/autonomy/stop":
            self._json({"ok": autonomy_stop()}); log_event("autonomy", "stop"); return
        if p == "/api/raw":
            cmd = str(body.get("cmd", "")).strip()[:40]
            if not cmd:
                self._json({"ok": False, "err": "empty"}, 400); return
            parts = cmd.split()
            verb = parts[0].upper()
            if verb in ("WD", "RESUME"):
                self._json({"ok": False,
                            "err": "WD/RESUME not allowed here (WD is the failsafe; "
                                   "use the resume button)"}, 403); return
            if verb == "M":
                try:
                    v = float(parts[1]) if len(parts) > 1 else 0.0
                except ValueError:
                    v = None
                if v is None or not math.isfinite(v) or v != 0.0:
                    self._json({"ok": False,
                                "err": "raw M is limited to 'M 0' - use the motor "
                                       "buttons (mode-capped, auto-stop)"}, 403)
                    return
            if verb == "U":
                try:
                    u = int(float(parts[1]))
                except (ValueError, IndexError):
                    self._json({"ok": False, "err": "U needs a number"}, 400); return
                if not SERVO_US_MIN <= u <= SERVO_US_MAX:
                    self._json({"ok": False, "err": "U out of the verified envelope "
                                "[{}, {}]".format(SERVO_US_MIN, SERVO_US_MAX)}, 403)
                    return
            r = NODE["n"].send_raw(cmd)
            log_event("raw", "{} -> {}".format(cmd, r))
            self._json({"ok": r is not None, "reply": r}); return
        if p == "/api/resume":
            r = NODE["n"].send_raw("RESUME")
            self._json({"ok": True, "reply": r}); return
        if p == "/api/servo":
            self._json(act_servo(body.get("action", "center"), body.get("deg"))); return
        if p == "/api/motor":
            self._json(act_motor(body.get("duty", 0), body.get("dur", 1.0))); return
        if p == "/api/camera/param":
            self._json(camera_set(body.get("name", ""), body.get("value", 0))); return
        if p == "/api/calib/mark":
            side = body.get("side")
            us = int(body.get("us", 0))
            if side not in ("left", "right") or not SERVO_US_MIN <= us <= SERVO_US_MAX:
                self._json({"ok": False, "err": "bad mark"}, 400); return
            with LOCK:
                S["us_" + side] = us
            save_state()
            log_event("calib", "marked {} limit at {} us".format(side, us))
            self._json({"ok": True}); return
        if p == "/api/calib/suggest":
            self._json(calib_suggest()); return
        if p == "/api/calib/start_duty":
            with LOCK:
                S["motor_start_duty"] = float(body.get("duty", 0))
            save_state()
            log_event("calib", "motor start duty recorded: {}%".format(
                S["motor_start_duty"]))
            self._json({"ok": True}); return
        if p == "/api/calib/trim":
            try:
                delta = float(body.get("delta", 0))
            except (TypeError, ValueError):
                self._json({"ok": False, "err": "bad delta"}, 400); return
            with LOCK:
                S["trim"] = round(max(-4.0, min(4.0, S.get("trim", 0.0) + delta)), 1)
                val = S["trim"]
            save_state()
            r = NODE["n"].send_raw("TRIM {:.1f}".format(val))
            log_event("calib", "trim {:+.1f} -> {} (now {})".format(delta, r, val))
            self._json({"ok": r is not None, "trim": val, "reply": r}); return
        if p == "/api/calib/steer_check":
            passed = bool(body.get("passed"))
            log_event("calib", "steer-direction check: {}".format(
                "PASS (S+ = wheels RIGHT)" if passed
                else "FAIL - flip steer_dir in params.yaml or fix linkage!"))
            self._json({"ok": True}); return
        if p == "/api/test/lighting":
            self._json(test_lighting()); return
        if p == "/api/test/align":
            self._json(test_align()); return
        if p == "/api/preflight":
            self._json(preflight()); return
        if p == "/api/stream/quality":
            FRAME["q"] = max(40, min(95, int(body.get("q", 78))))
            self._json({"ok": True, "q": FRAME["q"]}); return
        self._json({"ok": False, "err": "unknown endpoint"}, 404)

    def do_GET(self):
        if self.path.startswith("/api/status"):
            t = now()
            temp, load, undervolt, disk = pi_health()
            b = LIVE["bridge"]
            b_fresh = b if (LIVE["bridge_time"] and t - LIVE["bridge_time"] < 3) else None
            self._json({
                "mode": S["mode"], "caps": MODE_CAPS[S["mode"]],
                "stack": stack_running(),
                "autonomy": _STACK_CACHE.get("autonomy"),
                "epoch": PANEL_EPOCH,
                "cam": {"age": _age(LIVE["cam"]), "fps": round(rate_of("cam"), 1)},
                "scan": {"age": _age(LIVE["scan"]), "hz": round(rate_of("scan"), 1)},
                "det": {"age": _age(LIVE["det"]), "hz": round(rate_of("det"), 1),
                        "list": LIVE["dets"]},
                "steer": LIVE["steer"], "drive": LIVE["drive"],
                "drive_alive": LIVE["drive_msg"] is not None
                               and t - LIVE["drive_msg"] < 1.5,
                "bridge": b_fresh,
                "camera_metrics": NODE["n"].camera_metrics(),
                "pi": {"temp_c": temp, "load": load, "undervolt": undervolt,
                       "disk_free_gb": disk},
                "state": {k: S[k] for k in
                          ("motor_start_duty", "us_left", "us_right", "trim",
                           "camera_overrides")},
                "mission": list(MISSION)[-12:],
            })
            return
        if self.path.startswith("/api/events"):
            try:
                after = int(self.path.split("after=")[1]) if "after=" in self.path else 0
            except ValueError:
                after = 0
            with LOCK:
                evs = [e for e in EVENTS if e["i"] > after]
            self._json({"events": evs[-80:]})
            return
        if self.path.startswith("/api/lidar_nearest"):
            self._json({"ok": True, "nearest": NODE["n"].lidar_nearest()})
            return
        if self.path.startswith("/api/logs"):
            what = "autonomy" if "autonomy" in self.path else "stack"
            if what == "autonomy":
                ok, out = exec_capture("tail -40 /tmp/sign_steering.log 2>/dev/null"
                                       " || echo '(no autonomy log)'", timeout=10)
            else:
                st, raw = docker_api("GET", "/containers/{}/logs?stdout=1&stderr=1"
                                     "&tail=40".format(STACK), timeout=10)
                out, i = b"", 0
                while i + 8 <= len(raw):
                    ln = int.from_bytes(raw[i + 4:i + 8], "big")
                    out += raw[i + 8:i + 8 + ln]
                    i += 8 + ln
                ok, out = st == 200, out.decode("utf-8", "replace")[-4000:]
            self._json({"ok": ok, "log": out})
            return
        if self.path.startswith("/snapshot"):
            j = FRAME["jpg"]
            if j is None:
                self.send_error(503); return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(j)))
            self.end_headers()
            self.wfile.write(j)
            return
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    j = FRAME["jpg"]
                    if j is not None:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(
                            "Content-Length: {}\r\n\r\n".format(len(j)).encode())
                        self.wfile.write(j)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.07)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        body = (RC_HTML if self.path.startswith("/rc") else PANEL_HTML).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _age(t):
    return None if t is None else round(now() - t, 1)


PANEL_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WRO test console</title>
<style>
 :root{--bg:#0e0e10;--card:#17191e;--line:#262a32;--txt:#e6e9ef;--dim:#8b93a1;
       --ok:#43d364;--bad:#ff6b6b;--warn:#ffb64d;--acc:#5aa2ff}
 body{margin:0;background:var(--bg);color:var(--txt);
      font-family:system-ui,sans-serif;font-size:14px}
 header{display:flex;gap:8px;align-items:center;padding:8px 14px;
        border-bottom:1px solid var(--line);flex-wrap:wrap;position:sticky;
        top:0;background:var(--bg);z-index:5}
 header b{font-size:15px;margin-right:6px}
 .tab{padding:6px 14px;border-radius:8px;border:1px solid var(--line);
      cursor:pointer;color:var(--dim)}
 .tab.active{color:var(--txt);border-color:var(--acc);background:#151b28}
 #estop{margin-left:auto;background:#3a1414;border:1px solid #7c2626;
        color:#ff9c9c;font-weight:700;padding:9px 18px;border-radius:8px;
        cursor:pointer}
 #estop:hover{background:#521b1b}
 .grid{display:grid;grid-template-columns:minmax(340px,1.2fr) minmax(300px,1fr);
       gap:12px;padding:12px;max-width:1400px;margin:0 auto}
 @media(max-width:900px){.grid{grid-template-columns:1fr}}
 .card{background:var(--card);border:1px solid var(--line);border-radius:10px;
       padding:12px}
 .card h3{margin:0 0 8px;font-size:13px;letter-spacing:.6px;color:var(--dim);
          text-transform:uppercase}
 img#view{width:100%;border-radius:8px;background:#000}
 .chips{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0}
 .chip{padding:3px 9px;border-radius:11px;font-size:12px;background:#1c2027;
       border:1px solid var(--line)}
 .ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}
 .dim{color:var(--dim)}
 button{padding:7px 13px;border-radius:7px;border:1px solid var(--line);
        background:#1c2027;color:var(--txt);cursor:pointer;font-size:13px}
 button:hover{background:#242a33}
 button:disabled{opacity:.4;cursor:wait}
 .row{display:flex;gap:7px;flex-wrap:wrap;align-items:center;margin:6px 0}
 input,select{background:#12141a;color:var(--txt);border:1px solid var(--line);
        border-radius:6px;padding:6px 8px;font-size:13px;width:90px}
 input[type=range]{width:150px}
 #log{height:170px;overflow-y:auto;font-family:ui-monospace,monospace;
      font-size:12px;background:#101216;border-radius:8px;padding:8px;
      border:1px solid var(--line)}
 #log div{padding:1px 0;border-bottom:1px solid #16181d}
 table{width:100%;border-collapse:collapse;font-size:13px}
 td{padding:4px 6px;border-bottom:1px solid var(--line)}
 .hide{display:none}
 .toast{position:fixed;bottom:16px;right:16px;background:#1c2333;
        border:1px solid var(--acc);padding:10px 16px;border-radius:8px;
        max-width:420px;z-index:9}
</style></head><body>
<header>
 <b>WRO console</b>
 <div class="tab" data-m="desktop" onclick="setMode('desktop')">🖥️ desktop</div>
 <div class="tab" data-m="assembly" onclick="setMode('assembly')">🔧 assembly</div>
 <div class="tab" data-m="final" onclick="setMode('final')">🏁 final</div>
 <a class="tab" href="/rc" target="_blank" style="text-decoration:none">🎮 RC drive</a>
 <span class="chip">stack <span id="c_stack" class="dim">?</span></span>
 <span class="chip">auto <span id="c_auto" class="dim">?</span></span>
 <span class="chip">cam <span id="c_cam" class="dim">?</span></span>
 <span class="chip">lidar <span id="c_lidar" class="dim">?</span></span>
 <span class="chip">link <span id="c_link" class="dim">?</span></span>
 <span class="chip">pi <span id="c_pi" class="dim">?</span></span>
 <span class="chip hide" id="lostchip" style="color:#ff6b6b">CONNECTION LOST</span>
 <button id="estop" onclick="estop()">■ E-STOP</button>
</header>
<div class="grid">
 <div>
  <div class="card"><img id="view" alt="live view">
   <div class="chips" id="detchips"></div>
   <div class="row dim" id="cmdline">steer — drive —</div>
  </div>
  <div class="card"><h3>event log</h3><div id="log"></div></div>
 </div>
 <div>
  <div class="card">
   <h3>stack & autonomy</h3>
   <div class="row">
    <button onclick="post('/api/stack/start')">▶ start stack</button>
    <button onclick="if(confirm('Stop camera+lidar+all nodes?'))post('/api/stack/stop')">■ stop stack</button>
   </div>
   <div class="row" id="autonomyrow">
    <button onclick="startAuto()">▶ autonomy</button>
    <button onclick="startAuto(0)">▶ DRY RUN (servo only)</button>
    <button onclick="post('/api/autonomy/stop')">■ autonomy off</button>
    cruise <input id="cruise" type="number" value="55" min="40" max="70">%
   </div>
   <div class="row">
    <button onclick="showLog('autonomy')">autonomy log</button>
    <button onclick="showLog('stack')">stack log</button>
   </div>
   <div class="row dim" id="calibvals"></div>
  </div>

  <div class="card" id="sec_parts">
   <h3>component tests</h3>
   <div class="row">
    <button onclick="raw('PING')">PING</button>
    <button onclick="raw('GET')">firmware state</button>
    <button onclick="post('/api/resume')">resume autonomy link</button>
   </div>
   <div class="row">servo:
    <button onclick="post('/api/servo',{action:'center'})">center</button>
    <button onclick="post('/api/servo',{action:'sweep'})">sweep L↔R</button>
    <button onclick="post('/api/servo',{action:'angle',deg:10})">+10°</button>
    <button onclick="post('/api/servo',{action:'angle',deg:-10})">−10°</button>
   </div>
   <div class="row">motor:
    duty <input id="duty" type="number" value="25" min="5" max="70">%
    for <input id="dur" type="number" value="1.5" min="0.2" max="3" step="0.1">s
    <button onclick="motor(1)">forward</button>
    <button onclick="motor(-1)">reverse</button>
   </div>
   <div class="row">
    <button onclick="get('/api/lidar_nearest')">lidar: nearest object</button>
    <span class="dim">place anything ~30 cm ahead</span>
   </div>
   <div class="row">raw cmd:
    <input id="rawcmd" style="width:160px" placeholder="e.g. U 1500">
    <button onclick="raw(document.getElementById('rawcmd').value)">send</button>
   </div>
  </div>

  <div class="card" id="sec_calib">
   <h3>calibration (assembly)</h3>
   <div class="row">servo pulse:
    <input id="usval" type="number" value="1500" min="850" max="2150" step="5">µs
    <button onclick="usSend(0)">go</button>
    <button onclick="usSend(-25)">−25</button>
    <button onclick="usSend(25)">+25</button>
   </div>
   <div class="row">
    <button onclick="mark('left')">mark LEFT limit</button>
    <button onclick="mark('right')">mark RIGHT limit</button>
    <button onclick="post('/api/calib/suggest',{},false,true)">suggest DEG2US/LIM</button>
   </div>
   <div class="row">trim:
    <button onclick="post('/api/calib/trim',{delta:-0.5})">−0.5°</button>
    <button onclick="post('/api/calib/trim',{delta:0.5})">+0.5°</button>
    <span id="trimval" class="dim">—</span>
   </div>
   <div class="row">steer direction check:
    <button onclick="steerCheck()">command RIGHT & verify</button>
   </div>
   <div class="row">motor start-duty finder:
    <span class="dim">pulse rising duties; record the first that MOVES</span></div>
   <div class="row" id="dutygrid"></div>
   <div class="row">camera:
    <select id="camctl"><option>brightness</option><option>contrast</option>
     <option>saturation</option><option>sharpness</option><option>gain</option>
     <option>backlight_compensation</option></select>
    <input id="camval" type="range" min="0" max="255" value="100"
           oninput="document.getElementById('camvaln').textContent=this.value">
    <span id="camvaln">100</span>
    <button onclick="camSet()">apply</button>
   </div>
   <div class="row">
    <button onclick="post('/api/test/lighting',{},false,true)">lighting margin check</button>
    <button onclick="post('/api/test/align',{},false,true)">lidar↔camera align</button>
   </div>
  </div>

  <div class="card" id="sec_final">
   <h3>final / mission</h3>
   <div class="row"><button onclick="post('/api/preflight',{},false,true)">🛫 run pre-flight</button></div>
   <div id="preflight"></div>
   <h3 style="margin-top:10px">mission log</h3>
   <div id="mission" class="dim" style="font-family:ui-monospace,monospace;font-size:12px"></div>
  </div>
 </div>
</div>
<script>
let MODE='desktop', lastEv=0, EPOCH=null, lostN=0;
const $=id=>document.getElementById(id);
function toast(t){const d=document.createElement('div');d.className='toast';
 d.textContent=t;document.body.appendChild(d);setTimeout(()=>d.remove(),4500);}
async function post(u,b={},conf=false,show=false){
 try{const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify(b)});const j=await r.json();
  if(!j.ok&&j.err)toast('✗ '+j.err);
  else if(show&&!j.checks)toast(JSON.stringify(j.verdict||j.suggest||j).slice(0,380));
  if(j.checks)renderPreflight(j);
  poll();return j;}catch(e){toast('✗ '+e);return{ok:false}}}
async function get(u){try{const r=await fetch(u);const j=await r.json();
 toast(JSON.stringify(j.nearest||j).slice(0,300));return j;}catch(e){toast('✗ '+e)}}
function renderPreflight(j){
 $('preflight').innerHTML='<table>'+j.checks.map(c=>
  '<tr><td class="'+(c.ok?'ok':'bad')+'">'+(c.ok?'✔':'✘')+'</td><td>'+c.name+
  '</td><td class="dim">'+String(c.detail).slice(0,90)+'</td></tr>').join('')+
  '</table><div class="'+(j.ok?'ok':'bad')+'" style="padding:6px 0;font-weight:700">'+
  (j.ok?'ALL CHECKS PASSED':'NOT READY')+'</div>';}
async function estop(){const j=await post('/api/estop');
 if(j.fallback)toast('⚠ E-STOP fallback: '+j.fallback);
 else if(j.acked)toast('E-STOP acknowledged');
 else toast('✗ E-STOP NOT CONFIRMED — cut battery power!');}
function setMode(m){post('/api/mode',{mode:m});}
async function raw(c){if(!c)return;const j=await post('/api/raw',{cmd:c});
 if(j.reply)toast(j.reply);}
function motor(dir){post('/api/motor',
 {duty:dir*parseFloat($('duty').value),dur:parseFloat($('dur').value)});}
function startAuto(cruise){const c=cruise!==undefined?cruise:parseFloat($('cruise').value);
 post('/api/autonomy/start',{cruise:c}).then(j=>{
  if(j.running)toast(c===0?'DRY RUN live (servo only)':'autonomy RUNNING at '+j.cruise+'%');
  else if(j.ok===false&&!j.err)toast('✗ autonomy did not come up — check autonomy log');});}
async function showLog(what){const r=await fetch('/api/logs?what='+what);
 const j=await r.json();alert((what+' log:\n\n')+(j.log||'(empty)'));}
async function steerCheck(){await post('/api/servo',{action:'angle',deg:15});
 setTimeout(async()=>{const ok=confirm('Did the wheels/linkage point RIGHT?\nOK = yes, Cancel = no');
  await post('/api/calib/steer_check',{passed:ok});
  await post('/api/servo',{action:'center'});
  toast(ok?'steer direction PASS':'FAIL: flip steer_dir in params.yaml!');},900);}
function usSend(delta){const el=$('usval');
 el.value=Math.max(850,Math.min(2150,parseInt(el.value)+delta));
 raw('U '+el.value);}
function mark(side){post('/api/calib/mark',{side:side,us:parseInt($('usval').value)});}
function camSet(){post('/api/camera/param',
 {name:$('camctl').value,value:parseInt($('camval').value)});}
// motor start-duty grid (armed: every pulse needs an explicit confirm)
(function(){const g=$('dutygrid');for(let d=20;d<=60;d+=5){
 const b=document.createElement('button');b.textContent=d+'%';
 b.onclick=()=>{if(!confirm('Pulse motor at '+d+'% for 1.2s?'))return;
  post('/api/motor',{duty:d,dur:1.2});
  setTimeout(()=>{if(confirm('Did the motor MOVE at '+d+'%?'))
   post('/api/calib/start_duty',{duty:d});},1600);};g.appendChild(b);}})();
// live image via snapshot polling (browser-safe, no infinite stream)
(function(){const im=$('view');let n=0;
 function tick(){im.src='/snapshot?t='+(n++);}
 im.onload=()=>setTimeout(tick,120);im.onerror=()=>setTimeout(tick,900);tick();})();
function chip(id,txt,cls){const e=$(id);e.textContent=txt;e.className=cls;}
async function poll(){
 try{
  const s=await(await fetch('/api/status')).json();
  lostN=0;$('lostchip').classList.add('hide');
  if(EPOCH===null)EPOCH=s.epoch;
  else if(EPOCH!==s.epoch){EPOCH=s.epoch;lastEv=0;toast('panel restarted — resynced');}
  MODE=s.mode;
  document.querySelectorAll('.tab').forEach(t=>
   t.classList.toggle('active',t.dataset.m===MODE));
  $('sec_calib').classList.toggle('hide',MODE==='desktop');
  $('sec_final').classList.toggle('hide',MODE!=='final');
  $('autonomyrow').classList.toggle('hide',!s.caps.autonomy);
  $('duty').max=s.caps.duty;$('dur').max=s.caps.dur;
  chip('c_stack',s.stack?'ON':'OFF',s.stack?'ok':'bad');
  chip('c_auto',s.autonomy===true?'RUN':(s.autonomy===false?'off':'?'),
   s.autonomy===true?'ok':'dim');
  chip('c_cam',s.cam.fps>10?s.cam.fps+'fps':'off',s.cam.fps>10?'ok':'bad');
  chip('c_lidar',s.scan.hz>5?s.scan.hz+'Hz':'off',s.scan.hz>5?'ok':'bad');
  const b=s.bridge;
  chip('c_link',b&&b.connected?
   (b.manual?'MANUAL '+Math.ceil(b.manual_left||0)+'s':'ok'):'down',
   b&&b.connected?(b.manual?'warn':'ok'):'bad');
  chip('c_pi',(s.pi.temp_c||'?')+'°C '+(s.pi.undervolt?'UNDERVOLT!':''),
   s.pi.undervolt?'bad':(s.pi.temp_c>72?'warn':'ok'));
  $('cmdline').textContent='steer '+s.steer.toFixed(0)+'°  drive '+
   s.drive.toFixed(0)+'%'+(s.camera_metrics?
   '  | luma '+s.camera_metrics.fg_luma+' clip '+s.camera_metrics.clip_pct+'%':'');
  const st=s.state||{};
  $('trimval').textContent=(st.trim!==undefined?st.trim:0).toFixed(1)+'°';
  $('calibvals').textContent='saved: start-duty '+
   (st.motor_start_duty!==null&&st.motor_start_duty!==undefined?
    st.motor_start_duty+'%':'—')+
   ' | limits '+(st.us_left||'—')+'/'+(st.us_right||'—')+'µs | trim '+
   (st.trim!==undefined?st.trim:0)+'°';
  $('detchips').innerHTML=s.det.list.map(d=>
   '<span class="chip" style="color:'+(d[0]==='red'?'#ff6b6b':'#43d364')+'">'+
   d[0]+' '+d[3]+'m</span>').join('')||'<span class="chip dim">no detections</span>';
  if(MODE==='final')$('mission').innerHTML=
   (s.mission||[]).map(m=>'<div>'+m.msg+'</div>').join('');
 }catch(e){if(++lostN>=2)$('lostchip').classList.remove('hide');}
 try{
  const ev=await(await fetch('/api/events?after='+lastEv)).json();
  for(const e of ev.events){lastEv=e.i;const d=document.createElement('div');
   d.textContent='['+e.kind+'] '+e.msg;$('log').prepend(d);}
 }catch(e){}
}
setInterval(poll,2000);poll();
</script></body></html>
"""


RC_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WRO RC drive</title>
<style>
 :root{--bg:#0e0e10;--card:#17191e;--line:#262a32;--txt:#e6e9ef;--dim:#8b93a1;
       --ok:#43d364;--bad:#ff6b6b;--warn:#ffb64d;--acc:#5aa2ff}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--txt);
      font-family:system-ui,sans-serif;font-size:14px;overflow-x:hidden}
 header{display:flex;gap:8px;align-items:center;padding:8px 14px;
        border-bottom:1px solid var(--line);flex-wrap:wrap}
 header b{font-size:16px}
 a.back{color:var(--dim);text-decoration:none;border:1px solid var(--line);
        padding:5px 10px;border-radius:7px}
 .pill{padding:4px 10px;border-radius:11px;font-size:12px;background:#1c2027;
       border:1px solid var(--line)}
 #engage{background:#123020;border:1px solid #2c6f47;color:#8ff0b6;
         font-weight:700;padding:8px 16px;border-radius:8px;cursor:pointer}
 #disarm{background:#1c2027;border:1px solid var(--line);color:var(--dim);
         padding:8px 14px;border-radius:8px;cursor:pointer}
 #estop{margin-left:auto;background:#3a1414;border:1px solid #7c2626;
        color:#ff9c9c;font-weight:800;padding:10px 22px;border-radius:8px;
        cursor:pointer;font-size:15px}
 #estop:hover{background:#521b1b}
 .wrap{display:grid;grid-template-columns:minmax(320px,1.4fr) minmax(280px,1fr);
       gap:12px;padding:12px;max-width:1250px;margin:0 auto}
 @media(max-width:820px){.wrap{grid-template-columns:1fr}}
 .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
 .card h3{margin:0 0 8px;font-size:12px;letter-spacing:.6px;color:var(--dim);
          text-transform:uppercase}
 #camwrap{position:relative}
 img#cam{width:100%;border-radius:8px;background:#000;display:block;aspect-ratio:4/3;object-fit:cover}
 #hud{position:absolute;left:0;right:0;bottom:0;top:0;pointer-events:none}
 .banner{position:absolute;top:10px;left:50%;transform:translateX(-50%);
         padding:6px 14px;border-radius:8px;font-weight:700;font-size:13px}
 .b_safe{background:rgba(30,34,42,.85);color:#9aa3b2;border:1px solid #333}
 .b_armed{background:rgba(18,60,32,.85);color:#8ff0b6;border:1px solid #2c6f47}
 .b_estop{background:rgba(80,20,20,.9);color:#ffb4b4;border:1px solid #7c2626}
 .gauges{display:flex;gap:14px;align-items:flex-end;margin-top:10px}
 .steerbar{position:relative;flex:1;height:26px;background:#101216;border:1px solid var(--line);
           border-radius:6px;overflow:hidden}
 .steerfill{position:absolute;top:0;bottom:0;background:#2b3d5c}
 .steermid{position:absolute;left:50%;top:0;bottom:0;width:2px;background:#4a5265}
 .thr{position:relative;width:54px;height:120px;background:#101216;border:1px solid var(--line);
      border-radius:6px;overflow:hidden}
 .thrfill{position:absolute;left:0;right:0}
 .thrmid{position:absolute;left:0;right:0;top:50%;height:2px;background:#4a5265}
 .lamp{display:inline-block;width:11px;height:11px;border-radius:50%;background:#3a3f49;
       vertical-align:middle;margin-right:5px}
 .lamp.on{background:var(--ok);box-shadow:0 0 8px var(--ok)}
 .num{font-family:ui-monospace,monospace;font-size:22px;font-weight:700}
 .row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:7px 0}
 label{color:var(--dim)}
 select,input[type=number]{background:#12141a;color:var(--txt);border:1px solid var(--line);
        border-radius:6px;padding:5px 7px;font-size:13px}
 input[type=range]{width:160px}
 .ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}.dim{color:var(--dim)}
 kbd{background:#101216;border:1px solid var(--line);border-bottom-width:2px;
     border-radius:4px;padding:1px 6px;font-family:ui-monospace,monospace;font-size:12px}
 #raw{font-family:ui-monospace,monospace;font-size:11px;color:var(--dim);
      background:#101216;border:1px solid var(--line);border-radius:6px;padding:7px;
      white-space:pre-wrap;max-height:120px;overflow:auto}
 details summary{cursor:pointer;color:var(--dim);margin:4px 0}
 .toast{position:fixed;bottom:16px;right:16px;background:#1c2333;border:1px solid var(--acc);
        padding:10px 16px;border-radius:8px;max-width:420px;z-index:9}
</style></head><body>
<header>
 <b>🎮 RC drive</b>
 <a class="back" href="/">← console</a>
 <span class="pill">pad <span id="gp" class="dim">scanning…</span></span>
 <span class="pill">link <span id="link" class="dim">?</span></span>
 <button id="gpscan" onclick="scanGamepads();setTimeout(()=>{$('gp').textContent=pad()?'connected':'none — press controller button';$('gp').className=pad()?'ok':'dim';},300);" style="background:#1c2030;border:1px solid var(--line);color:var(--dim);padding:5px 9px;border-radius:6px;cursor:pointer;font-size:12px">↻ scan</button>
 <button id="engage" onclick="engage()">▶ ENGAGE</button>
 <button id="disarm" onclick="disarm()">■ disarm</button>
 <button id="estop" onclick="estop()">■ E-STOP</button>
</header>
<div class="wrap">
 <div class="card" id="camwrap">
  <img id="cam" alt="FPV">
  <div id="hud">
   <div id="banner" class="banner b_safe">NOT ENGAGED — press ENGAGE</div>
  </div>
  <div class="gauges">
   <div style="flex:1">
    <div class="dim" style="font-size:12px">steering <span id="steern" class="num" style="font-size:15px">0°</span></div>
    <div class="steerbar"><div class="steermid"></div><div id="steerfill" class="steerfill"></div></div>
   </div>
   <div style="text-align:center">
    <div class="dim" style="font-size:12px">throttle</div>
    <div class="thr"><div class="thrmid"></div><div id="thrfill" class="thrfill"></div></div>
    <div id="driven" class="num" style="font-size:15px">0%</div>
   </div>
  </div>
  <div class="row"><span><span id="deadlamp" class="lamp"></span><span id="deadtxt" class="dim">deadman: release</span></span></div>
 </div>

 <div>
  <div class="card">
   <h3>how to drive</h3>
   <div class="dim" style="line-height:1.7">
    0. Click <b>"↻ scan"</b> then press any controller button to connect.<br>
    1. Press <b>ENGAGE</b> (enter key) — stops autonomy, takes direct control.<br>
    2. <b>Hold the deadman</b> (gamepad <b>RB</b> / keyboard <kbd>Shift</kbd>) — motor only runs while held.<br>
    3. Steer: <b>left stick</b> / <kbd>A</kbd><kbd>D</kbd> &nbsp; Throttle: <b>RT</b>/<b>LT</b> triggers / <kbd>W</kbd><kbd>S</kbd>.<br>
    <kbd>Esc</kbd> = disarm &nbsp; <kbd>Space</kbd> = E-STOP.<br>
    <span style="color:#ffb64d">Steam Deck:</span> run <code>flatpak override --user --device=all org.mozilla.firefox</code> once.
   </div>
  </div>
  <div class="card">
   <h3>controls</h3>
   <div class="row"><label>max speed</label>
    <input id="max" type="range" min="0" max="70" value="55"
     oninput="cfg.max=+this.value;document.getElementById('maxn').textContent=this.value+'%'">
    <span id="maxn">55%</span></div>
   <div class="row"><label>steer sens</label>
    <input id="sens" type="range" min="0.4" max="1" step="0.05" value="1"
     oninput="cfg.sens=+this.value;document.getElementById('sensn').textContent=this.value">
    <span id="sensn">1</span></div>
   <div class="row"><label>throttle</label>
    <select id="thrmode" onchange="cfg.thr=this.value">
     <option value="trig">triggers (RT/LT)</option>
     <option value="rstick">right stick Y</option></select>
    <label>steer axis</label>
    <select id="saxis" onchange="cfg.steerAxis=+this.value">
     <option value="0">left stick X</option><option value="2">right stick X</option></select>
   </div>
   <div class="row"><label>deadman btn</label>
    <select id="deadsel" onchange="cfg.dead=+this.value">
     <option value="5">RB (5)</option><option value="7">RT (7)</option>
     <option value="4">LB (4)</option><option value="0">A (0)</option></select>
    <label><input type="checkbox" id="inv" onchange="cfg.invert=this.checked"> invert steer</label>
   </div>
  </div>
  <div class="card">
   <details><summary>gamepad tester (find your buttons/axes)</summary>
    <div id="raw">connect a controller and press buttons…</div>
   </details>
  </div>
 </div>
</div>
<script>
const $=id=>document.getElementById(id);
let cfg={thr:'trig',steerAxis:0,dead:5,invert:false,max:55,sens:1.0,expo:0.22};
let armed=false, estopped=false, gpIdx=null, key={}, last={steer:0,drive:0,deadman:false};
let gpPollId=null, gpEver=false;
function toast(t){const d=document.createElement('div');d.className='toast';d.textContent=t;
 document.body.appendChild(d);setTimeout(()=>d.remove(),4000);}

// ---- gamepad (Steam Deck / Firefox proof) ----
addEventListener('gamepadconnected',e=>{gpIdx=e.gamepad.index;gpEver=true;setGp(e.gamepad.id);});
addEventListener('gamepaddisconnected',e=>{if(gpIdx===e.gamepad.index){gpIdx=null;gpEver=false;}setGp(null);});
function scanGamepads(){const gps=navigator.getGamepads?navigator.getGamepads():null;if(!gps)return;
 for(let i=0;i<gps.length;i++){if(gps[i]&&gps[i].connected){if(gpIdx!==i){gpIdx=i;gpEver=true;setGp(gps[i].id);}return;}}
 gpIdx=null;}
if(!gpPollId)gpPollId=setInterval(scanGamepads,1000);
function pad(){const gps=navigator.getGamepads?navigator.getGamepads():null;
 return(gpIdx!=null&&gps&&gps[gpIdx]&&gps[gpIdx].connected)?gps[gpIdx]:null;}
function setGp(id){$('gp').textContent=id?String(id).slice(0,26):'none';
 $('gp').className=id?'ok':'dim';}
function dz(x,d){return Math.abs(x)<(d||0.08)?0:x;}
function expo(x,e){return (1-e)*x+e*x*x*x;}

function readTriggers(p){var rt=0,lt=0;
 if(p.buttons){rt=(p.buttons[7]||{}).value||0;lt=(p.buttons[6]||{}).value||0;}
 if(rt===0&&lt===0&&p.axes&&p.axes.length>5){rt=p.axes[5]!==void 0?(p.axes[5]+1)/2:0;lt=p.axes[4]!==void 0?(p.axes[4]+1)/2:0;}
 return{rt:Math.max(0,Math.min(1,rt)),lt:Math.max(0,Math.min(1,lt))};}

function readControls(){
 let steer=0,thr=0,dead=false,src='keyboard';
 const p=pad();
 if(p){src='pad';
  steer=expo(dz(p.axes[cfg.steerAxis]||0),cfg.expo)*cfg.sens*(cfg.invert?-1:1);
  if(cfg.thr==='trig'){var t=readTriggers(p);thr=t.rt-t.lt;}
  else thr=-dz(p.axes[3]||0,0.12);
  dead=!!((p.buttons[cfg.dead]||{}).pressed);
 } else {
  steer=((key['arrowright']||key['d']?1:0)-(key['arrowleft']||key['a']?1:0))*cfg.sens*(cfg.invert?-1:1);
  thr=(key['arrowup']||key['w']?1:0)-(key['arrowdown']||key['s']?1:0);
  dead=!!(key['shift']||key['j']);
 }
 return {steer:Math.max(-25,Math.min(25,steer*25)),
         drive:Math.max(-cfg.max,Math.min(cfg.max,thr*cfg.max)),
         deadman:dead, src};
}

// ---- 20 Hz send ----
async function sendLoop(){
 if(!armed||estopped)return;
 const c=last;
 try{
  const r=await fetch('/api/rc',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({steer:c.steer,drive:c.deadman?c.drive:0,deadman:c.deadman,max_pct:cfg.max})});
  await r.json();
 }catch(e){/* transient */}
}
setInterval(sendLoop,50);

// ---- 60 Hz render ----
function frame(){
 last=readControls();
 const sPct=(last.steer/25)*50;              // -50..50 from centre
 const sf=$('steerfill');
 if(last.steer>=0){sf.style.left='50%';sf.style.width=sPct+'%';sf.style.background='#3a6ea5';}
 else{sf.style.left=(50+sPct)+'%';sf.style.width=(-sPct)+'%';sf.style.background='#a5713a';}
 $('steern').textContent=last.steer.toFixed(0)+'°';
 const dv=last.deadman?last.drive:0, dPct=(dv/70)*50;
 const tf=$('thrfill');
 if(dv>=0){tf.style.bottom='50%';tf.style.top='auto';tf.style.height=dPct+'%';tf.style.background='#43d364';}
 else{tf.style.top='50%';tf.style.bottom='auto';tf.style.height=(-dPct)+'%';tf.style.background='#ffb64d';}
 $('driven').textContent=dv.toFixed(0)+'%';
 const held=last.deadman;
 $('deadlamp').className='lamp'+(held?' on':'');
 $('deadtxt').textContent=held?'deadman: HELD (drive live)':'deadman: release';
 $('deadtxt').className=held?'ok':'dim';
 setBanner();
 drawRaw();
 requestAnimationFrame(frame);
}
function setBanner(){
 const b=$('banner');
 if(estopped){b.className='banner b_estop';b.textContent='E-STOPPED — press ENGAGE to re-arm';}
 else if(!armed){b.className='banner b_safe';b.textContent='NOT ENGAGED — press ENGAGE';}
 else if(last.deadman){b.className='banner b_armed';b.textContent='DRIVING — deadman held';}
 else{b.className='banner b_safe';b.textContent='ARMED — hold deadman to drive';}
}
function drawRaw(){const p=pad();if(!p){$('raw').textContent='no gamepad detected — press any button on your controller…';return;}
 const ax=Array.from(p.axes).map((a,i)=>i+':'+a.toFixed(2)).join(' ');
 const bt=[];for(let i=0;i<(p.buttons||[]).length;i++){const b=p.buttons[i];if(b.pressed||b.value>0.1)bt.push(i+(b.value>0.1?':'+b.value.toFixed(2):''));}
 var t=readTriggers(p);
 $('raw').textContent='axes '+ax+'\nbtns ['+bt.join(' ')+']\nRT='+t.rt.toFixed(2)+' LT='+t.lt.toFixed(2);}
requestAnimationFrame(frame);

// ---- engage / disarm / estop ----
async function engage(){try{const j=await(await fetch('/api/rc/engage',{method:'POST'})).json();
 armed=true;estopped=false;toast('RC engaged — autonomy stopped');}catch(e){toast('✗ engage failed');}updateBtns();}
async function disarm(){armed=false;updateBtns();try{await fetch('/api/rc/release',{method:'POST'});}catch(e){}}
async function estop(){estopped=true;armed=false;updateBtns();
 try{const j=await(await fetch('/api/estop',{method:'POST'})).json();
  toast(j.acked?'E-STOP acknowledged':(j.fallback||'E-STOP sent'));}catch(e){toast('✗ E-STOP — cut power!');}}
function updateBtns(){$('engage').style.display=armed?'none':'';
 $('disarm').style.display=armed?'':'none';}
updateBtns();

// ---- keyboard ----
addEventListener('keydown',e=>{var k=e.key.toLowerCase();
 if(k===' '){e.preventDefault();estop();return;}
 if(k==='enter'){e.preventDefault();if(!armed)engage();return;}
 if(k==='escape'){e.preventDefault();if(armed)disarm();return;}
 key[k]=true;
 if(['arrowup','arrowdown','arrowleft','arrowright'].includes(k))e.preventDefault();});
addEventListener('keyup',e=>{key[e.key.toLowerCase()]=false;});

// ---- safety: stop on tab hide / close ----
addEventListener('beforeunload',()=>{try{navigator.sendBeacon('/api/rc/release');}catch(e){}});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&armed){
 armed=false;updateBtns();try{navigator.sendBeacon('/api/rc/release');}catch(e){}}});
addEventListener('blur',()=>{key={};});   // drop stuck keys when focus leaves

// ---- FPV camera (snapshot polling; browser-safe) ----
(function(){const im=$('cam');let n=0;
 function tick(){im.src='/snapshot?t='+(n++);}
 im.onload=()=>setTimeout(tick,110);im.onerror=()=>setTimeout(tick,900);tick();})();

// ---- link/bridge health ----
async function health(){try{const s=await(await fetch('/api/status')).json();
 const b=s.bridge;const el=$('link');
 if(b&&b.connected){el.textContent=b.manual?'manual':'ok';el.className='ok';}
 else{el.textContent='down';el.className='bad';}
 if(s.stack===false){el.textContent='stack off';el.className='bad';}
 }catch(e){$('link').textContent='?';$('link').className='dim';}}
setInterval(health,1500);health();
</script></body></html>
"""


def main():
    load_state()
    rclpy.init()
    node = VizNode()
    NODE["n"] = node
    threading.Thread(target=stack_poller, daemon=True).start()
    threading.Thread(target=overrides_watchdog, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", 8080), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log_event("panel", "console up (mode={})".format(S["mode"]))
    print("test console on :8080 (mode={})".format(S["mode"]))
    rclpy.spin(node)


if __name__ == "__main__":
    main()
