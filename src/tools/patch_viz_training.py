#!/usr/bin/env python3
"""Add a Training (record) panel to the RC console in viz_server.py.

Run ON THE PI against /home/pi/sign_detector/tools/viz_server.py.

The Pi's viz_server.py is newer than the copy in this repo (2042 lines vs 1800)
and carries RC-console work that was never committed, so this PATCHES the live
file in place rather than shipping a replacement -- overwriting it would delete
that work. Every insertion is anchored on an existing line; if an anchor is
missing the script aborts without writing, so a drifted file fails loudly
instead of silently producing a broken console.

Idempotent: re-running is a no-op once the marker is present.
"""
import os
import re
import shutil
import sys

MARKER = "TRAIN_PANEL_INSTALLED"


def die(msg):
    print("ABORT: " + msg)
    sys.exit(1)


def anchor(src, needle, what):
    n = src.count(needle)
    if n != 1:
        die("expected exactly 1 anchor for {} ({!r}), found {}".format(
            what, needle[:60], n))
    return needle


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/pi/sign_detector/tools/viz_server.py"
    if not os.path.exists(path):
        die("no such file: " + path)
    src = open(path, encoding="utf-8").read()

    if MARKER in src:
        print("already patched; nothing to do")
        return

    # ---- 1. ROS plumbing: publish train_cmd, remember train_status ---------
    a = anchor(src, '        self.pub_drive = self.create_publisher(Float32, "drive_cmd", 10)',
               "publisher block")
    src = src.replace(a, a + "\n"
        '        # ' + MARKER + '\n'
        '        self.pub_train = self.create_publisher(String, "train_cmd", 10)\n'
        '        self.create_subscription(String, "train_status", self._train_stat, 10)\n'
        '        self.train = {"recording": False, "rows": 0, "session": None,\n'
        '                      "secs": None, "imu": False, "lidar": False}\n')

    a = anchor(src, "    def rc_publish(self, steer_deg, drive_pct):",
               "rc_publish method")
    src = src.replace(a,
        '    def _train_stat(self, m):\n'
        '        try:\n'
        '            self.train = json.loads(m.data)\n'
        '        except Exception:\n'
        '            pass\n\n'
        '    def train_send(self, verb, name=None):\n'
        '        """train_recorder owns the file; this only sends the command."""\n'
        '        txt = verb if not name else "{} {}".format(verb, name)\n'
        '        self.pub_train.publish(String(data=txt))\n\n' + a)

    # ---- 2. HTTP routes ---------------------------------------------------
    a = anchor(src, '        if p == "/api/rc/release":', "rc/release route")
    src = src.replace(a,
        '        if p == "/api/train/start":\n'
        '            node.train_send("start", (body or {}).get("name"))\n'
        '            log_event("train", "recording START")\n'
        '            return self._json({"ok": True})\n'
        '        if p == "/api/train/stop":\n'
        '            node.train_send("stop")\n'
        '            log_event("train", "recording STOP")\n'
        '            return self._json({"ok": True})\n'
        '        if p == "/api/train":\n'
        '            return self._json({"ok": True, "train": node.train})\n' + a)

    # ---- 3. RC page UI ----------------------------------------------------
    # Anchored on the closing </body> of the RC page template.
    m = list(re.finditer(r"</body>", src))
    if not m:
        die("no </body> found in RC_HTML")
    ins = m[0].start()
    ui = (
        '<div id="trainbox" style="position:fixed;left:8px;bottom:8px;z-index:99;'
        'background:rgba(12,14,18,.9);border:1px solid #2c6f47;border-radius:8px;'
        'padding:8px 10px;font:13px system-ui;color:#8ff0b6;min-width:190px">'
        '<b>Training</b> <span id="trst">idle</span><br>'
        '<button id="trgo" style="margin-top:6px;padding:8px 12px;font-size:15px">'
        '&#9679; REC</button> '
        '<button id="trstop" style="margin-top:6px;padding:8px 12px;font-size:15px">'
        '&#9632; STOP</button>'
        '</div>\n'
        '<script>\n'
        'async function trPoll(){try{const r=await(await fetch("/api/train")).json();\n'
        ' const t=r.train||{};\n'
        ' document.getElementById("trst").textContent = t.recording\n'
        '   ? ("REC "+(t.rows||0)+" rows "+(t.secs||0)+"s"+(t.imu?" +imu":""))\n'
        '   : ("idle"+(t.lidar?"":" (no lidar)"));\n'
        ' document.getElementById("trst").style.color = t.recording?"#ff6b6b":"#8ff0b6";\n'
        '}catch(e){}}\n'
        'setInterval(trPoll,1000); trPoll();\n'
        'document.getElementById("trgo").onclick=async()=>{\n'
        '  await fetch("/api/train/start",{method:"POST",\n'
        '    headers:{"Content-Type":"application/json"},\n'
        '    body:JSON.stringify({name:"lap"})}); trPoll();};\n'
        'document.getElementById("trstop").onclick=async()=>{\n'
        '  await fetch("/api/train/stop",{method:"POST"}); trPoll();};\n'
        '</script>\n')
    src = src[:ins] + ui + src[ins:]

    bak = path + ".bak-train"
    shutil.copyfile(path, bak)
    open(path, "w", encoding="utf-8").write(src)
    print("patched {}\nbackup  {}".format(path, bak))


if __name__ == "__main__":
    main()
