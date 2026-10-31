"""
live/common.py - what the three live-view workers share: the link to the web server, pacing,
JPEG encoding and the bird's-eye view geometry.

A worker is a child process of web_control/server.py (web_control/liveview.py starts it):

    stdout  binary messages to the server: <u32 header length><u32 payload length><JSON header>
            <payload, a JPEG or empty>. Only the newest message per sensor is kept, in memory.
    stdin   control lines from the server (JSON objects, e.g. {"range_m": 20}).
    stderr  log lines (shown on the page under "Live view log").

Live view never saves anything. A worker stops on SIGINT/SIGTERM, at its own --max-secs deadline,
when the server closes its stdin (server stopped or crashed) and when a write to the server fails
(broken pipe): it can never outlive the server, which is what enforces "never at the same time as
a recording".
"""
from __future__ import annotations

import json
import os
import signal
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent
ROOT = PIPELINE.parent
for p in (PIPELINE, PIPELINE / "replay"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

MAX_SECS = 600.0                 # live view switches itself off after 10 minutes
# The bird's-eye views (radar and LiDAR) share one extent so that radar detections can be drawn over
# the LiDAR image: x forward 0..R, y (left +) -R*ASPECT/2..+R*ASPECT/2.
ASPECT = 1.5


def add_args(ap):
    ap.add_argument("--from-session", type=Path, default=None,
                    help="play a recorded session in real time instead of the sensor (development and tests)")
    ap.add_argument("--hz", type=float, default=4.0, help="updates per second sent to the page")
    ap.add_argument("--max-secs", type=float, default=MAX_SECS + 15,
                    help="stop by itself after this long (the server stops live view at 600 s)")
    ap.add_argument("--range-m", type=float, default=20.0, help="initial forward range of the view")


def bev_extent(range_m: float) -> tuple:
    """(x0, x1, y0, y1) of the shared bird's-eye extent."""
    R = float(range_m)
    return 0.0, R, -R * ASPECT / 2, R * ASPECT / 2


class Link:
    """The worker's side of the server link (see the module docstring)."""

    def __init__(self, sensor: str, max_secs: float, control: dict = None):
        self.sensor = sensor
        # fd 1 belongs to the messages. Everything printed, by Python or by C++ (rs_driver, ZED SDK),
        # goes to stderr instead, so it cannot corrupt the stream.
        self._out = os.fdopen(os.dup(1), "wb", buffering=0)
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        self._wlock = threading.Lock()
        self.stop = threading.Event()
        self.deadline = time.monotonic() + max_secs
        self.control = dict(control or {})
        self.seq = 0
        signal.signal(signal.SIGINT, lambda *_: self.stop.set())
        signal.signal(signal.SIGTERM, lambda *_: self.stop.set())
        threading.Thread(target=self._read_control, name="live-control", daemon=True).start()

    def _read_control(self):
        for line in sys.stdin:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if isinstance(msg, dict):
                self.control.update(msg)
        self.stop.set()                       # stdin closed: the server is gone (or stopped us)

    def running(self) -> bool:
        if time.monotonic() >= self.deadline and not self.stop.is_set():
            log(f"{self.sensor}: {self.deadline:.0f} s deadline reached - stopping")
            self.stop.set()
        return not self.stop.is_set()

    def send(self, header: dict, payload: bytes = b"") -> None:
        self.seq += 1
        h = json.dumps(dict(header, sensor=self.sensor, seq=self.seq), separators=(",", ":")).encode()
        try:
            with self._wlock:
                self._out.write(struct.pack("<II", len(h), len(payload)) + h + payload)
        except OSError:                        # broken pipe: the server is gone
            self.stop.set()

    def error(self, text: str) -> None:
        """A fatal problem, shown on the page."""
        log(f"{self.sensor}: ERROR: {text}")
        self.send({"error": text})


def log(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


class Pacer:
    """At most hz updates per second; wait() sleeps until the next slot (or until stop)."""

    def __init__(self, hz: float, stop: threading.Event):
        self.period, self.stop, self.next = 1.0 / max(0.2, hz), stop, time.monotonic()

    def wait(self):
        now = time.monotonic()
        if now < self.next:
            self.stop.wait(self.next - now)
        self.next = max(self.next + self.period, time.monotonic())


class Rate:
    """Events per second over the last few seconds."""

    def __init__(self, window_s: float = 3.0):
        self.window, self.t = window_s, []

    def tick(self) -> None:
        now = time.monotonic()
        self.t = [x for x in self.t if now - x < self.window] + [now]

    @property
    def hz(self) -> float:
        return 0.0 if len(self.t) < 2 else (len(self.t) - 1) / max(self.t[-1] - self.t[0], 1e-6)


def jpeg(rgb: np.ndarray, quality: int = 75) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(rgb[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def rounded(a, nd: int = 2) -> list:
    """An array as nested lists of rounded floats, for the JSON header."""
    return np.round(np.asarray(a, float), nd).tolist()
