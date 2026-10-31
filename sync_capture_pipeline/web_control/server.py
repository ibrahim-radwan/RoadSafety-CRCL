#!/usr/bin/env python3
"""
web_control/server.py — start and stop capture sessions from a browser (iPad).

The Jetson rides headless on the scooter. This serves one page on the Wi-Fi
interface; the operator opens http://<jetson-ip>:8080 on the same network.

WHAT IT DOES
    Start  -> runs capture_session.py exactly as a terminal would
              (python3 -u capture_session.py --out sessions/<name> ...).
    Stop   -> sends SIGINT to the launcher: the same as Ctrl+C, so the
              tested stop order (radar first, camera/LiDAR trail, finalize)
              is unchanged.
    Status -> parsed from the launcher's own per-second output: radar
              packets/s, camera FPS and gaps, LiDAR packets/s and losses,
              plus free disk space.
    After a session ends, check_session.py runs automatically and its
    PASS/FAIL is shown and saved as <session>/check_session.txt.

LIVE VIEW (liveview.py, ../live/)
    For extrinsic alignment: the camera's left image, the LiDAR top view and
    the radar's range-azimuth map with detections, a few times a second,
    scaled down for the Wi-Fi. Nothing is saved. Live view and recording
    NEVER run at the same time: each start is refused while the other runs
    (one lock in create_app), and live view switches itself off after 10
    minutes. --live-from-session plays a recorded session instead of the
    sensors (development, tests).

ROBUSTNESS
    The launcher runs in its own session, so a crash or restart of this
    server does not stop a recording. On start-up the server looks for a
    capture_session.py that is already running and adopts it (Stop works).
    There is no authentication by design: anyone on the same Wi-Fi can
    start/stop recordings.

USAGE (conda env: flask, psutil)
    python web_control/server.py                      # 0.0.0.0:8080
    python web_control/server.py --port 8080 --sessions-root /mnt/nvme/sessions
"""
from __future__ import annotations

import argparse
import atexit
import collections
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
from flask import Flask, jsonify, request, send_file

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent
sys.path.insert(0, str(PIPELINE))
import board_health  # noqa: E402  (sysfs temperatures, no root)
import liveview  # noqa: E402
LAUNCHER = PIPELINE / "capture_session.py"
CHECKER = PIPELINE / "check_session.py"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

COUNTDOWN_RE = re.compile(r"Starting in (\d+) s")
RADAR_RE = re.compile(r"\[radar\]\s+t\+\s*(\d+)s\s+([\d,]+) pkt/s")
CAMERA_RE = re.compile(r"\[camera\] (\d+) frames \| observed ([\d.]+) FPS \| SDK ([\d.]+) FPS.*\| gaps (\d+)")
LIDAR_RE = re.compile(r"\[lidar\]\s+(\d+)s \| msop\s+(\d+)/s \| imu\s+(\d+)/s \| lost (\d+) \| queue (\d+) \|\s+([\d.]+) MB")
# Known failure signatures in the session log -> what to check in the field.
HINTS = [
    (("is not mounted - insert the SD card",),
     "SD card not found - insert the card (fully, until it clicks) and press Start again"),
    (("CONFIG_FPGA: TIMEOUT", "RESET_FPGA: no ACK", "all DCA commands"),
     "radar capture board (DCA1000) not responding - check the radar's 12 V power and its Ethernet "
     "cable, then power-cycle the radar"),
    (("radar rate low",),
     "radar data rate too low (DCA1000 throttled) - power-cycle the radar + DCA1000 (12 V off, wait 10 s, on), "
     "then start a new session"),
    (("no MSOP packets",),
     "no LiDAR data - check the LiDAR power and the USB-Ethernet adapter (LiDAR at 192.168.34.200)"),
    (("ZED SDK sees 0 cameras", "CAMERA NOT DETECTED"),
     "camera not detected - unplug and replug the ZED USB cable, then start again"),
    (("free space below", "not enough space"),
     "storage full - copy finished sessions off the SD card and delete them (or free space on the "
     "Jetson's internal disk, where the camera is staged during recording)"),
    (("board is HOT", "board too hot"),
     "the Jetson overheated - shade it / improve airflow and let it cool"),
]

# Launcher lines that mark a phase change.
PHASES = [
    ("PREFLIGHT FAILED", "failed"),
    ("Starting in ", "countdown"),
    ("starting lidar", "starting"),
    ("starting camera", "starting"),
    ("starting radar", "starting radar"),
    ("[radar]   START_RECORD: OK", "recording"),
    ("stopping radar", "stopping"),
    ("radar stopped", "stopping"),
    ("camera is verifying", "finalizing"),
    ("SESSION SUMMARY", "finished"),
]


class Session:
    """The one capture session that may be running, plus its parsed status."""

    def __init__(self, root: Path, launcher_python: str, checker_python: str, camera_stage: Path = None):
        self.root = root
        self.camera_stage = camera_stage
        self.launcher_python = launcher_python
        self.checker_python = checker_python
        self.lock = threading.Lock()
        self.proc = None            # Popen when started here
        self.pid = None             # pid when adopted after a server restart
        self.name = None
        self.guide = None            # "stage3" = calibration guide on the page; None = plain recording
        self.started = None
        self.phase = "idle"
        self.log = collections.deque(maxlen=3000)
        self.log_seq = 0
        self.sensors = {}
        self.result = None          # summary after the launcher exits
        self.has_radar, self.duration, self.n_other, self.ready = True, 0.0, 2, 0
        self.countdown_end = self.rec_start = self.rec_end = None   # operator timing
        self.check = None           # {"status": "running"|"PASS"|"FAIL", ...}

    # ── helpers ─────────────────────────────────────────────────────────
    def _append(self, line: str):
        with self.lock:
            self.log_seq += 1
            self.log.append((self.log_seq, line))
            for key, phase in PHASES:
                if key in line:
                    self.phase = phase
            now = time.time()
            m = RADAR_RE.search(line)
            if m:
                self.sensors["radar"] = {"t": now, "pkt_s": int(m.group(2).replace(",", "")),
                                         "secs": int(m.group(1))}
            m = CAMERA_RE.search(line)
            if m:
                self.sensors["camera"] = {"t": now, "frames": int(m.group(1)), "fps": float(m.group(2)),
                                          "gaps": int(m.group(4))}
            m = LIDAR_RE.search(line)
            if m:
                self.sensors["lidar"] = {"t": now, "msop_s": int(m.group(2)), "imu_s": int(m.group(3)),
                                         "lost": int(m.group(4)), "mb": float(m.group(6))}
            # Operator timing (from the log, ~0.2 s resolution; guidance only).
            m = COUNTDOWN_RE.search(line)
            if m:
                self.countdown_end = now + int(m.group(1))
            if " is recording" in line:
                self.ready += 1
            # Recording starts when the radar records (it starts last, after
            # camera and LiDAR are ready); without a radar, when every sensor is.
            if self.rec_start is None and ("[radar]   START_RECORD: OK" in line
                                           or (not self.has_radar and " is recording" in line
                                               and self.ready >= self.n_other)):
                self.rec_start = now
            if self.rec_start is not None and self.rec_end is None and any(
                    k in line for k in ("stopping radar", "radar stopped", "stopping lidar", "stopping camera")):
                self.rec_end = now
            if "stop reason :" in line:
                self.result = {"stop_reason": line.split("stop reason :", 1)[1].strip()}
            if "SESSION INCOMPLETE" in line and self.result is not None:
                self.result["incomplete"] = True

    def running(self) -> bool:
        if self.proc is not None:
            return self.proc.poll() is None
        if self.pid is not None:
            try:
                return psutil.Process(self.pid).status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                return False
        return False

    # ── control ─────────────────────────────────────────────────────────
    def start(self, name: str, note: str, secs: float, delay: float, sensors: list, guide: str = None) -> str:
        with self.lock:
            if self.running():
                return "a session is already running"
        if not NAME_RE.match(name):
            return "name: letters, digits, '-' and '_' only (max 64)"
        out = self.root / name
        if out.exists():
            return f"session '{name}' already exists - choose a new name"
        cmd = [self.launcher_python, "-u", str(LAUNCHER), "--out", str(out), "--secs", str(secs),
               "--delay", str(delay), "--note", note]
        for s in ("radar", "camera", "lidar"):
            if s not in sensors:
                cmd.append(f"--no-{s}")
        if self.camera_stage and "camera" in sensors:
            cmd += ["--camera-stage", str(self.camera_stage)]
        logs = self.root / ".launcher"
        logs.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.log.clear()
            self.sensors, self.result, self.check = {}, None, None
            self.name, self.started, self.phase, self.pid = name, time.time(), "starting", None
            self.guide = guide
            self.has_radar, self.duration = "radar" in sensors, secs
            self.n_other, self.ready = len([x for x in sensors if x != "radar"]), 0
            self.countdown_end = self.rec_start = self.rec_end = None
        # The launcher must NEVER write to a pipe owned by this server: if the
        # server died, the launcher's next line would raise BrokenPipeError and
        # kill the recording. Its stdout goes to a file instead; everything it
        # reports is also in <session>/session.log, which this server follows
        # (and simply re-reads after a restart). Own session: Ctrl+C on this
        # server never reaches it.
        stdout_path = logs / f"{name}.txt"
        with open(stdout_path, "w") as stdout:
            self.proc = subprocess.Popen(cmd, cwd=PIPELINE, stdout=stdout, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
        self._append(f"$ {' '.join(cmd)}")
        threading.Thread(target=self._follow, args=(out / "session.log", stdout_path), daemon=True).start()
        return ""

    def _follow(self, log_path: Path, stdout_path):
        """Tail session.log while the launcher runs, then finish the session."""
        fh, buf = None, ""
        while True:
            alive = self.running()          # sample BEFORE reading: nothing is lost at exit
            if fh is None and log_path.is_file():
                fh = open(log_path, encoding="utf-8", errors="replace")
            if fh is not None:
                chunk = fh.read()
                if chunk:
                    buf += chunk
                    *lines, buf = buf.split("\n")
                    for line in lines:
                        self._append(line.rstrip())
            if not alive:
                break
            time.sleep(0.2)
        if fh is not None:
            if buf:
                self._append(buf.rstrip())
            fh.close()
        rc = self.proc.returncode if self.proc is not None else None
        if rc not in (0, None) and stdout_path is not None and stdout_path.is_file():
            # Errors before the launcher opened session.log (or a traceback).
            for line in stdout_path.read_text(errors="replace").splitlines()[-15:]:
                self._append(f"[launcher] {line}")
        self._append(f"[launcher exited{'' if rc is None else f' with code {rc}'}]")
        with self.lock:
            if self.phase != "failed":
                self.phase = "finished"
            self.result = self.result or {}
            self.result["exit_code"] = rc
        self._run_check()

    def _run_check(self):
        out = self.root / self.name
        if not (out / "session.json").is_file():
            hints = self._hints()                       # e.g. a preflight failure
            with self.lock:
                self.check = {"status": "skipped", "reason": "no session.json (session did not start)",
                              "summary": hints}
                self.phase = "done"
            return
        with self.lock:
            self.phase, self.check = "checking", {"status": "running"}
        r = subprocess.run([self.checker_python, str(CHECKER), "--session", str(out)],
                           capture_output=True, text=True)
        text = r.stdout + r.stderr
        (out / "check_session.txt").write_text(text)
        verdict = "PASS" if r.returncode == 0 else "FAIL"
        lines = text.splitlines()
        at = [i for i, l in enumerate(lines) if "VERDICT" in l]
        tail = lines[at[0] + 2:] if at else lines[-15:]      # skip the "=====" under VERDICT
        hints = self._hints()
        with self.lock:
            self.check = {"status": verdict, "summary": hints + [l.strip() for l in tail if l.strip()][:20]}
            self.phase = "done"

    def _hints(self) -> list:
        """Plain-language causes for known failure signatures in the log."""
        with self.lock:
            text = "\n".join(line for _, line in self.log)
        hints = []
        for patterns, hint in HINTS:
            if any(p in text for p in patterns):
                hints.append("HINT: " + hint)
        return hints

    def stop(self) -> str:
        pid = self.proc.pid if self.proc is not None else self.pid
        if pid is None or not self.running():
            return "no session is running"
        os.kill(pid, signal.SIGINT)      # = Ctrl+C: the launcher stops everything in order
        self._append("[stop requested from the web page]")
        return ""

    def adopt_running(self):
        """After a server restart: find a capture_session.py that is still
        running and follow its session.log from the start (status rebuilds)."""
        for p in psutil.process_iter(["pid", "cmdline", "create_time"]):
            cmd = p.info.get("cmdline") or []
            if any(str(c).endswith("capture_session.py") for c in cmd) and "--out" in cmd:
                out = Path(cmd[cmd.index("--out") + 1])
                with self.lock:
                    self.pid, self.name, self.started = p.info["pid"], out.name, p.info["create_time"]
                    self.phase = "recording"
                self._append(f"[server restarted - adopted running session {out.name}, pid {self.pid}]")
                threading.Thread(target=self._follow, args=(out / "session.log", None), daemon=True).start()
                return

    def status(self) -> dict:
        with self.lock:
            now = time.time()
            sensors = {k: dict(v, stale=now - v["t"] > 3.0, age_s=round(now - v["t"], 1))
                       for k, v in self.sensors.items()}
            return {"phase": self.phase, "running": self.running(), "name": self.name, "guide": self.guide,
                    "elapsed_s": round(now - self.started) if self.started else None,
                    "sensors": sensors, "result": self.result, "check": self.check,
                    "disk_free_gb": round(shutil.disk_usage(self.root if self.root.exists() else PIPELINE).free / 1e9, 1),
                    "log_seq": self.log_seq, "log_epoch": self.started,
                    "countdown_s": (max(0, round(self.countdown_end - now))
                                    if self.countdown_end and self.phase == "countdown" else None),
                    "recording_s": (round((self.rec_end or now) - self.rec_start, 1)
                                    if self.rec_start else None),
                    "recording_active": bool(self.rec_start and not self.rec_end),
                    "duration_s": self.duration,
                    "board": board_health.snapshot(),          # live, also while idle
                    "board_warn_c": board_health.WARN_C, "board_hot_c": board_health.HOT_C}

    def log_since(self, since: int) -> list:
        with self.lock:
            return [(n, l) for n, l in self.log if n > since][-500:]


def list_sessions(root: Path, active: str = None) -> list:
    out = []
    if not root.is_dir():
        return out
    dirs = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]   # .launcher = internal
    for d in sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True):
        meta = {}
        if (d / "session.json").is_file():
            try:
                meta = json.loads((d / "session.json").read_text())
            except ValueError:
                pass
        check = "recording" if d.name == active else None
        if (d / "check_session.txt").is_file():
            check = "PASS" if "  PASS" in (d / "check_session.txt").read_text() else "FAIL"
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        out.append({"name": d.name, "started": meta.get("started_local"), "note": meta.get("note", ""),
                    "stop_reason": meta.get("stop_reason"), "sensors": meta.get("sensors", []),
                    "size_gb": round(size / 1e9, 2), "check": check})
    return out


RECORDING_SCRIPTS = ("capture_session.py", "dca_capture_radar_sync.py", "record_lidar_raw.py", "recorder.py")


def recording_processes() -> list:
    """pids of capture processes on this machine, also ones not started from this page (a terminal)."""
    pids = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        cmd = p.info.get("cmdline") or []
        if any(str(c).endswith(RECORDING_SCRIPTS) for c in cmd[1:3]):
            pids.append(p.info["pid"])
    return pids


def create_app(session: Session, live: liveview.LiveView = None) -> Flask:
    app = Flask(__name__)
    live = live or liveview.LiveView(sys.executable)
    # Recording and live view NEVER run at the same time: both starts take this lock and check the other.
    exclusive = threading.Lock()

    @app.get("/")
    def index():
        return send_file(HERE / "index.html")

    @app.get("/api/status")
    def api_status():
        st = session.status()
        st["live_on"] = live.active()
        return jsonify(st)

    @app.get("/api/log")
    def api_log():
        return jsonify(session.log_since(int(request.args.get("since", 0))))

    @app.post("/api/start")
    def api_start():
        body = request.get_json(force=True, silent=True) or {}
        try:
            secs = float(body.get("secs") or 0)
            delay = float(body.get("delay") or 0)
        except ValueError:
            return jsonify(error="duration and delay must be numbers"), 400
        sensors = [s for s in body.get("sensors", ["radar", "camera", "lidar"]) if s in ("radar", "camera", "lidar")]
        if not sensors:
            return jsonify(error="select at least one sensor"), 400
        guide = "stage3" if body.get("guide") == "stage3" else None
        with exclusive:
            if live.active():
                return jsonify(error="live view is on - stop it before recording"), 400
            err = session.start(str(body.get("name", "")).strip(), str(body.get("note", ""))[:500], secs, delay,
                                sensors, guide)
        return (jsonify(error=err), 400) if err else jsonify(ok=True)

    @app.post("/api/stop")
    def api_stop():
        err = session.stop()
        return (jsonify(error=err), 400) if err else jsonify(ok=True)

    @app.get("/api/sessions")
    def api_sessions():
        return jsonify(list_sessions(session.root, session.name if session.running() else None))

    @app.get("/api/sessions/<name>/check")
    def api_check(name):
        if not NAME_RE.match(name):
            return "bad name", 400
        p = session.root / name / "check_session.txt"
        return (p.read_text(), 200, {"Content-Type": "text/plain"}) if p.is_file() else ("no check yet", 404)

    # ── live view ───────────────────────────────────────────────────────
    @app.post("/api/live/start")
    def api_live_start():
        body = request.get_json(force=True, silent=True) or {}
        sensors = [s for s in body.get("sensors", list(liveview.SENSORS)) if s in liveview.SENSORS]
        if not sensors:
            return jsonify(error="select at least one sensor"), 400
        with exclusive:
            if session.running():
                return jsonify(error="a recording is running - live view is off while recording"), 400
            pids = recording_processes()
            if pids:
                return jsonify(error=f"capture processes are running (pid {', '.join(map(str, pids))}) - "
                                     f"live view is off while recording"), 400
            err = live.start(sensors)
        return (jsonify(error=err), 400) if err else jsonify(ok=True)

    @app.post("/api/live/stop")
    def api_live_stop():
        err = live.stop()
        return (jsonify(error=err), 400) if err else jsonify(ok=True)

    @app.post("/api/live/control")
    def api_live_control():
        try:
            return jsonify(live.control(request.get_json(force=True, silent=True) or {}))
        except (TypeError, ValueError):
            return jsonify(error="bad value"), 400

    @app.get("/api/live/state")
    def api_live_state():
        return jsonify(live.status())

    @app.get("/api/live/log")
    def api_live_log():
        return jsonify(live.log_lines())

    @app.get("/api/live/frame/<sensor>")
    def api_live_frame(sensor):
        """The sensor's newest image after ?after=<seq>, waiting up to 2 s (204 if none)."""
        if sensor not in liveview.SENSORS:
            return "bad sensor", 404
        got = live.frame(sensor, int(request.args.get("after", 0)))
        if got is None:
            return "", 204
        seq, data = got
        return data, 200, {"Content-Type": "image/jpeg", "X-Seq": str(seq), "Cache-Control": "no-store"}

    return app


def main() -> int:
    ap = argparse.ArgumentParser(description="Web control for capture sessions")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--sessions-root", type=Path, default=Path("/mnt/sdcard/sessions"),
                    help="where sessions are recorded (the SD card)")
    ap.add_argument("--camera-stage", type=Path, default=Path("/home/jetson/camera_stage"),
                    help="the camera records here (eMMC) and is moved into the session afterwards; "
                         "'none' records it straight into the session")
    ap.add_argument("--launcher-python", default="/usr/bin/python3",
                    help="interpreter for capture_session.py (stdlib only)")
    ap.add_argument("--checker-python", default="/usr/bin/python3")
    ap.add_argument("--live-python", default=sys.executable,
                    help="interpreter for the live-view workers (the conda env: torch, pyzed, LiDAR wrapper)")
    ap.add_argument("--live-from-session", type=Path, default=None,
                    help="live view plays this recorded session instead of the sensors (development, tests)")
    a = ap.parse_args()

    stage = None if str(a.camera_stage).lower() == "none" else a.camera_stage
    session = Session(a.sessions_root, a.launcher_python, a.checker_python, stage)
    session.adopt_running()
    liveview.end_stray_workers(lambda m: print(m, flush=True))
    live = liveview.LiveView(a.live_python, a.live_from_session)
    atexit.register(live.shutdown)
    app = create_app(session, live)
    print(f"Capture web control on http://{a.host}:{a.port}  (sessions: {session.root})", flush=True)
    app.run(host=a.host, port=a.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
