"""Live view (web_control/liveview.py, live/*_live.py) without sensors:

  * the lockout: a live view cannot start while a recording runs, and a recording cannot start while
    the live view is on (server endpoints, with stand-ins for both);
  * on a reference session (--from-session): every worker sends images and numbers, the live view
    switches itself off at its time limit, every worker exits cleanly, nothing is written into the
    session, and a worker stops by itself when its server goes away.

The sensor inputs themselves (DCA1000, LiDAR sockets, live ZED) need the rig and are not tested here.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PIPELINE), str(PIPELINE / "web_control")]
pytest.importorskip("flask")
pytest.importorskip("psutil")
import liveview  # noqa: E402
import server  # noqa: E402

S = Path(os.environ.get("REPORT_TEST_SESSIONS", "/mnt/sdcard/sessions/freewalk_ref_02").split(",")[0])
need_session = pytest.mark.skipif(not (S / "radar" / "frames.csv").is_file(), reason="no reference session")


class FakeSession:
    def __init__(self, running=False):
        self.is_running, self.started = running, []

    def running(self):
        return self.is_running

    def start(self, *args):
        self.started.append(args)
        return ""


class FakeLive:
    def __init__(self, on=False):
        self.on, self.started = on, []

    def active(self):
        return self.on

    def start(self, sensors):
        self.started.append(sensors)
        return ""


def _client(sess, live, monkeypatch):
    monkeypatch.setattr(server, "recording_processes", lambda: [])
    return server.create_app(sess, live).test_client()


def test_no_live_view_while_recording(monkeypatch):
    sess, live = FakeSession(running=True), FakeLive()
    r = _client(sess, live, monkeypatch).post("/api/live/start", json={"sensors": ["radar"]})
    assert r.status_code == 400 and "recording" in r.get_json()["error"]
    assert live.started == []


def test_no_live_view_while_capture_processes_run(monkeypatch):
    sess, live = FakeSession(), FakeLive()
    c = _client(sess, live, monkeypatch)
    monkeypatch.setattr(server, "recording_processes", lambda: [4242])      # e.g. started from a terminal
    r = c.post("/api/live/start", json={"sensors": ["radar"]})
    assert r.status_code == 400 and "4242" in r.get_json()["error"]
    assert live.started == []


def test_no_recording_while_live_view(monkeypatch):
    sess, live = FakeSession(), FakeLive(on=True)
    r = _client(sess, live, monkeypatch).post("/api/start", json={"name": "x", "sensors": ["lidar"]})
    assert r.status_code == 400 and "live view" in r.get_json()["error"]
    assert sess.started == []


def test_both_allowed_when_idle(monkeypatch):
    sess, live = FakeSession(), FakeLive()
    c = _client(sess, live, monkeypatch)
    assert c.post("/api/live/start", json={"sensors": ["radar", "bogus"]}).status_code == 200
    assert live.started == [["radar"]]
    assert c.post("/api/start", json={"name": "x", "sensors": ["lidar"]}).status_code == 200
    assert len(sess.started) == 1


def _snapshot(root: Path) -> dict:
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*")}


@need_session
def test_live_view_from_session_and_auto_off():
    pytest.importorskip("torch")
    sensors = ["radar", "lidar"]
    try:
        import pyzed.sl  # noqa: F401
        sensors.append("camera")
    except ImportError:
        pass
    before = _snapshot(S)
    lv = liveview.LiveView(sys.executable, S, max_secs=40)
    assert lv.start(sensors) == ""
    try:
        got = {}
        t_end = time.monotonic() + 60
        while time.monotonic() < t_end and len(got) < len(sensors):
            st = lv.status()
            for s in sensors:
                assert not st["sensors"][s]["error"], st["sensors"][s]["error"]
                if s not in got and st["sensors"][s]["seq"] >= 3:
                    got[s] = st["sensors"][s]["header"]
            time.sleep(0.5)
        assert sorted(got) == sorted(sensors), f"no images from {set(sensors) - set(got)}"
        for s in sensors:
            seq, jpg = lv.frame(s, 0)
            assert jpg[:2] == b"\xff\xd8" and 5_000 < len(jpg) < 150_000
        assert got["radar"]["extent"] == [0.0, 20.0, -15.0, 15.0]
        assert {"moving", "static"} <= set(got["radar"]) and "bright" in got["lidar"]
        lv.control({"range_m": 40, "radar_map": "moving"})
        t_end = time.monotonic() + 10
        while time.monotonic() < t_end:
            h = lv.status()["sensors"]["radar"]["header"]
            if h.get("map") == "moving" and h["extent"][1] == 40.0:
                break
            time.sleep(0.3)
        else:
            raise AssertionError("radar did not follow the control message")
        # the time limit: switched off by the server, every worker exits by itself
        t_end = time.monotonic() + 40 + liveview.STOP_GRACE_S + 10
        while lv.active() and time.monotonic() < t_end:
            time.sleep(0.5)
        st = lv.status()
        assert not st["on"] and "time limit" in st["stop_reason"]
        assert all(w.proc.returncode == 0 for w in lv.workers.values()), lv.log_lines()[-20:]
        assert liveview.stray_workers() == []
    finally:
        lv.shutdown()
    assert _snapshot(S) == before, "live view changed files in the session"


@need_session
def test_worker_stops_when_server_goes_away():
    pytest.importorskip("torch")
    p = subprocess.Popen([sys.executable, "-u", str(liveview.WORKER["radar"]), "--from-session", str(S)],
                         cwd=liveview.LIVE_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL)
    try:
        hl, pl = struct.unpack("<II", p.stdout.read(8))             # its first message
        assert json.loads(p.stdout.read(hl))["sensor"] == "radar"
        p.stdin.close()                                              # the server is gone
        p.stdout.close()
        assert p.wait(timeout=15) == 0
    finally:
        if p.poll() is None:
            p.kill()
