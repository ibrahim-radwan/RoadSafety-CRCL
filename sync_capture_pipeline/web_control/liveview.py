"""
web_control/liveview.py - the live view's server side: starts one worker per selected sensor
(sync_capture_pipeline/live/*_live.py), keeps the newest image and numbers of each in memory for
the page, and switches everything off after MAX_SECS.

Workers are children of this server, linked by pipes (live/common.py): if the server stops or
crashes, their stdin closes and their next write fails, so they stop too. A live view can
therefore never outlive the server's lockout against recording (server.py). Nothing is saved.
"""
from __future__ import annotations

import collections
import json
import os
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

    def __init__(self, python: str, from_session: Path = None, max_secs: float = MAX_SECS):
        self.python, self.from_session, self.max_secs = python, from_session, float(max_secs)
        self.lock = threading.Lock()
        self.workers: dict = {}
        self.started = self.deadline = None
        self.stop_reason = None
        self.control_state = {"range_m": 20.0, "radar_map": "static"}
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
        with self.lock:
            self.control_state.update(ok)
            workers = list(self.workers.values())
        for w in workers:
            w.control(ok)
        return ok

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
