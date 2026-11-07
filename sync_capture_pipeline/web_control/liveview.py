"""
web_control/liveview.py - the live view's server side: starts one worker per selected sensor
(sync_capture_pipeline/live/*_live.py), keeps the newest image and numbers of each in memory for
the page, and switches everything off after MAX_SECS.

Workers are children of this server, linked by pipes (live/common.py): if the server stops or
crashes, their stdin closes and their next write fails, so they stop too. A live view can
therefore never outlive the server's lockout against recording (server.py).

Nothing is saved, except calibration captures on request (2026-11-07): "Save background" / "Save
position" make <calib-root>/<set>/<name>/ with capture.json, and the workers write their next ~2 s
of data into it (live/common.py Capture). calib_set() reads a set back for the page.
"""
from __future__ import annotations

import collections
import json
import os
import re
import signal
import struct
import subprocess
import threading
import time
from pathlib import Path

import psutil

PIPELINE = Path(__file__).resolve().parent.parent
LIVE_DIR = PIPELINE / "live"
SENSORS = ("radar", "camera", "lidar")
WORKER = {s: LIVE_DIR / f"{s}_live.py" for s in SENSORS}
MAX_SECS = 600.0
STOP_GRACE_S = 8.0           # SIGINT, then SIGKILL after this (the radar sends STOP_RECORD first)
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}$")     # calibration set and position names


def stray_workers() -> list:
    """Live-view worker processes running on this machine (any owner, e.g. left by an earlier server)."""
    found = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        cmd = p.info.get("cmdline") or []
        if any(str(c).endswith(tuple(f"live/{s}_live.py" for s in SENSORS)) for c in cmd):
            found.append(p)
    return found


class Worker:
    def __init__(self, sensor: str, cmd: list, log):
        self.sensor, self.log = sensor, log
        self.lock = threading.Lock()
        self.header, self.jpeg, self.seq, self.t = None, None, 0, None
        self.error = None
        self.cv = threading.Condition(self.lock)
        self.proc = subprocess.Popen(cmd, cwd=LIVE_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, start_new_session=True)
        threading.Thread(target=self._read, name=f"live-{sensor}", daemon=True).start()
        threading.Thread(target=self._read_log, name=f"live-{sensor}-log", daemon=True).start()

    def _read(self):
        f = self.proc.stdout
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            hl, pl = struct.unpack("<II", hdr)
            h = f.read(hl)
            data = f.read(pl) if pl else b""
            if len(h) < hl or len(data) < pl:
                break
            try:
                header = json.loads(h)
            except ValueError:
                continue
            with self.cv:
                if header.get("error"):
                    self.error = header["error"]
                self.header, self.t = header, time.time()
                if data:
                    self.jpeg, self.seq = data, header.get("seq", self.seq + 1)
                self.cv.notify_all()

    def _read_log(self):
        for line in self.proc.stderr:
            self.log(f"[{self.sensor}] {line.decode(errors='replace').rstrip()}")

    def alive(self) -> bool:
        return self.proc.poll() is None

    def control(self, msg: dict):
        try:
            self.proc.stdin.write((json.dumps(msg) + "\n").encode())
            self.proc.stdin.flush()
        except (OSError, ValueError):
            pass

    def interrupt(self):
        if self.alive():
            self.proc.send_signal(signal.SIGINT)

    def frame(self, after: int, timeout: float):
        """(seq, jpeg) newer than `after`, waiting up to timeout; None if there is none."""
        with self.cv:
            self.cv.wait_for(lambda: self.seq > after or not self.alive(), timeout)
            return (self.seq, self.jpeg) if self.seq > after and self.jpeg else None


class LiveView:
    """At most one live view; the lockout against recording is the server's (server.py)."""

    def __init__(self, python: str, from_session: Path = None, max_secs: float = MAX_SECS, calib_root: Path = None):
        self.python, self.from_session, self.max_secs = python, from_session, float(max_secs)
        self.calib_root = Path(calib_root) if calib_root else None     # calibration captures go here
        self.lock = threading.Lock()
        self.workers: dict = {}
        self.started = self.deadline = None
        self.stop_reason = None
        self.control_state = {"range_m": 20.0, "radar_map": "static", "reflector_only": False}
        self._bg_n = 0                  # background take/clear tokens (one-shot, see control)
        self._cap_n = 0                 # calibration capture tokens (one-shot, see capture)
        self.log = collections.deque(maxlen=400)
        self._stopping = False

    def _log(self, line: str):
        self.log.append(f"{time.strftime('%H:%M:%S')} {line}")

    def active(self) -> bool:
        """True while any worker of the live view still runs (stopping included)."""
        with self.lock:
            return any(w.alive() for w in self.workers.values())

    def start(self, sensors: list) -> str:
        with self.lock:
            if any(w.alive() for w in self.workers.values()):
                return "live view is already on"
            stray = stray_workers()
            if stray:
                return (f"live-view processes from an earlier run are still running (pid "
                        f"{', '.join(str(p.pid) for p in stray)}) - wait a few seconds and retry")
            self.log.clear()
            self.workers, self.stop_reason, self._stopping = {}, None, False
            self.started = time.time()
            self.deadline = self.started + self.max_secs
            for s in sensors:
                cmd = [self.python, "-u", str(WORKER[s]), "--max-secs", str(self.max_secs + 15),
                       "--range-m", str(self.control_state["range_m"])]
                if self.from_session:
                    cmd += ["--from-session", str(self.from_session)]
                self._log(f"$ {' '.join(cmd)}")
                self.workers[s] = Worker(s, cmd, self._log)
                self.workers[s].control(self.control_state)
        threading.Thread(target=self._watch, name="live-watch", daemon=True).start()
        return ""

    def _watch(self):
        """Auto-off at the deadline; notes when every worker has exited."""
        while True:
            with self.lock:
                workers = dict(self.workers)
                deadline = self.deadline
            if not any(w.alive() for w in workers.values()):
                break
            if deadline and time.time() >= deadline and not self._stopping:
                self.stop(f"switched off after {self.max_secs / 60:g} minutes (time limit)")
            time.sleep(0.5)
        for s, w in workers.items():
            rc = w.proc.returncode
            self._log(f"[{s} exited with code {rc}]")
        with self.lock:
            self.stop_reason = self.stop_reason or "all live-view sensors stopped"

    def stop(self, reason: str = "stopped from the web page") -> str:
        with self.lock:
            workers = dict(self.workers)
            if not any(w.alive() for w in workers.values()):
                return "live view is not on"
            if self._stopping:
                return ""
            self._stopping, self.stop_reason = True, reason
        self._log(f"[{reason}]")
        for w in workers.values():
            w.interrupt()

        def reap():
            t_end = time.time() + STOP_GRACE_S
            for w in workers.values():
                try:
                    w.proc.wait(max(0.1, t_end - time.time()))
                except subprocess.TimeoutExpired:
                    self._log(f"[{w.sensor}: no exit after {STOP_GRACE_S:.0f} s - killed]")
                    w.proc.kill()
                    w.proc.wait()
        threading.Thread(target=reap, name="live-reap", daemon=True).start()
        return ""

    def control(self, msg: dict):
        ok = {}
        if "range_m" in msg:
            ok["range_m"] = float(min(80.0, max(5.0, float(msg["range_m"]))))
        if msg.get("radar_map") in ("static", "moving"):
            ok["radar_map"] = msg["radar_map"]
        if "reflector_only" in msg:
            ok["reflector_only"] = bool(msg["reflector_only"])
        if msg.get("calib_set") and self.calib_root is not None and NAME_RE.match(str(msg["calib_set"])):
            # the workers use this set's saved background (if any) while it is the page's set
            ok["use_background"] = str(self.calib_root / str(msg["calib_set"]) / "background")
        once = {}
        if msg.get("background") in ("take", "clear"):
            # one-shot: sent to the running workers only, never kept in control_state, so a new
            # live view starts without a background. The workers hold it in memory; nothing is saved.
            self._bg_n += 1
            once["background"] = f"{msg['background']}:{self._bg_n}"
        with self.lock:
            self.control_state.update(ok)
            workers = list(self.workers.values())
        for w in workers:
            w.control({**ok, **once})
        return {**ok, **once}

    def capture(self, kind: str, set_name: str, name: str = None, r_lo: float = 1.0, r_hi: float = 15.0) -> dict:
        """A calibration capture: makes <calib_root>/<set>/<name>/ (name "background" for a
        background) with capture.json and sends the workers a one-shot token to write their next
        ~2 s of data into it. Returns {"dir"} or {"error"}."""
        if kind not in ("background", "position"):
            return {"error": "kind must be background or position"}
        if self.calib_root is None:
            return {"error": "no calibration folder configured"}
        name = "background" if kind == "background" else (name or "").strip()
        set_name = (set_name or "").strip()
        for what, v in (("set", set_name), ("position", name)):
            if not NAME_RE.match(v):
                return {"error": f"{what} name: letters, digits and _ . + - only (up to 64), not '{v}'"}
        if kind == "position" and name == "background":
            return {"error": "'background' is reserved for the empty scene"}
        with self.lock:
            workers = {s: w for s, w in self.workers.items() if w.alive() and s in ("radar", "lidar", "camera")}
        if not workers:
            return {"error": "live view is not on"}
        d = self.calib_root / set_name / name
        if d.exists() and kind == "position":
            return {"error": f"position '{name}' already exists in set '{set_name}' - use a new name"}
        if d.exists():                                  # a new background replaces the old one
            import shutil
            shutil.rmtree(d)
        d.mkdir(parents=True)
        lo, hi = float(r_lo), float(r_hi)
        info = {"kind": kind, "set": set_name, "name": name, "time_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "sensors": sorted(workers), "r_lo": lo, "r_hi": hi,
                "source": str(self.from_session) if self.from_session else "live",
                "note": "live-view calibration capture; process with calib_reflector.py --set <set folder>"}
        (d / "capture.json").write_text(json.dumps(info, indent=1) + "\n")
        self._cap_n += 1
        msg = {"capture": {"token": self._cap_n, "kind": kind, "dir": str(d), "r_lo": lo, "r_hi": hi}}
        for w in workers.values():
            w.control(msg)
        self._log(f"[calibration capture {set_name}/{name} ({kind}) -> {d}]")
        return {"dir": str(d), "token": self._cap_n}

    def calib_sets(self) -> list:
        if self.calib_root is None or not self.calib_root.is_dir():
            return []
        sets = [p for p in self.calib_root.iterdir() if p.is_dir() and NAME_RE.match(p.name)]
        return [p.name for p in sorted(sets, key=lambda p: p.stat().st_mtime, reverse=True)]

    def calib_set(self, set_name: str) -> dict:
        """A set's captures with each sensor's reflector summary, and the LiDAR yaw estimates."""
        if self.calib_root is None or not NAME_RE.match(set_name or ""):
            return {"error": "bad set name"}
        root = self.calib_root / set_name
        out = {"set": set_name, "dir": str(root), "positions": [], "background": None}
        if not root.is_dir():
            return out

        def read(p):
            try:
                return json.loads(p.read_text())
            except (OSError, ValueError):
                return None
        for d in sorted((q for q in root.iterdir() if (q / "capture.json").is_file()), key=lambda q: q.stat().st_mtime):
            cap = read(d / "capture.json") or {}
            row = {"name": d.name, "kind": cap.get("kind"), "time": cap.get("time_local"),
                   "saved": {s: (d / s / "meta.json").is_file() for s in cap.get("sensors", [])}}
            for s in ("radar", "lidar"):
                r = read(d / s / "reflector.json")
                if r is not None:
                    row[s] = {k: r.get(k) for k in ("found", "x", "y", "z", "range", "az_deg", "rise_db", "present",
                                                     "n_per_sweep", "own_x", "own_y", "own_az_deg", "background")}
            if cap.get("kind") == "background":
                out["background"] = row
            else:
                out["positions"].append(row)
        # LiDAR yaw from the radar (positions near its boresight only, the most robust; the radar azimuth
        # off-axis is good since the measured array layout, 2026-11-07); calib_reflector.py also uses the camera
        both = [r for r in out["positions"] if (r.get("radar") or {}).get("found") and (r.get("lidar") or {}).get("found")
                and r["lidar"].get("own_az_deg") is not None]
        bore = [r for r in both if abs(r["radar"]["az_deg"]) <= 5.0]
        if bore:
            d = [r["radar"]["az_deg"] - r["lidar"]["own_az_deg"] for r in bore]
            out["yaw_from_radar_boresight_deg"] = {"mean": round(float(sum(d) / len(d)), 2), "n": len(d)}
        return out

    def frame(self, sensor: str, after: int, timeout: float = 2.0):
        with self.lock:
            w = self.workers.get(sensor)
        return None if w is None else w.frame(after, timeout)

    def status(self) -> dict:
        now = time.time()
        with self.lock:
            on = any(w.alive() for w in self.workers.values())
            sensors = {}
            for s, w in self.workers.items():
                with w.lock:
                    sensors[s] = {"alive": w.alive(), "seq": w.seq, "error": w.error,
                                  "age_s": None if w.t is None else round(now - w.t, 1),
                                  "header": w.header}
            return {"on": on, "stopping": on and self._stopping, "sensors": sensors,
                    "remaining_s": max(0, round(self.deadline - now)) if on and self.deadline else None,
                    "max_s": self.max_secs, "stop_reason": None if on else self.stop_reason,
                    "control": dict(self.control_state),
                    "source": str(self.from_session) if self.from_session else None}

    def log_lines(self) -> list:
        return list(self.log)

    def shutdown(self):
        """Server exit: stop the workers (they would also stop on their own when stdin closes)."""
        with self.lock:
            workers = list(self.workers.values())
        for w in workers:
            w.interrupt()
        for w in workers:
            try:
                w.proc.wait(STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                w.proc.kill()


def end_stray_workers(log=print):
    """At server start-up: a live view cannot survive its server, but make sure (SIGINT, then kill)."""
    stray = stray_workers()
    for p in stray:
        log(f"ending a leftover live-view worker, pid {p.pid}")
        try:
            os.kill(p.pid, signal.SIGINT)
        except OSError:
            pass
    gone, alive = psutil.wait_procs(stray, timeout=STOP_GRACE_S)
    for p in alive:
        p.kill()
