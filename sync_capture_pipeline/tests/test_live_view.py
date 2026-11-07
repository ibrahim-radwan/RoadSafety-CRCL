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


def test_json_has_no_nan(monkeypatch):
    """Browsers reject NaN / Infinity in JSON: the page would show nothing. The server writes null."""
    live = FakeLive()
    live.status = lambda: {"sensors": {"camera": {"header": {"fps": float("nan"), "x": [1.0, float("inf")]}}}}
    r = _client(FakeSession(), live, monkeypatch).get("/api/live/state")
    assert r.status_code == 200
    assert b"NaN" not in r.data and b"Infinity" not in r.data
    assert json.loads(r.data)["sensors"]["camera"]["header"] == {"fps": None, "x": [1.0, None]}


def test_failure_answers_in_json(monkeypatch, capsys):
    """A failing endpoint answers 500 with the error in JSON (the page shows it) and logs the traceback."""
    sess = FakeSession()

    def broken():
        raise RuntimeError("sensor table broke")
    sess.status = broken
    r = _client(sess, FakeLive(), monkeypatch).get("/api/status")
    assert r.status_code == 500
    assert "RuntimeError: sensor table broke" in r.get_json()["error"]
    assert "Traceback" in capsys.readouterr().err
    r = _client(sess, FakeLive(), monkeypatch).get("/api/no-such-thing")
    assert r.status_code == 404 and "404" in r.get_json()["error"]


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
        # "Take background": held by the workers in memory (the file snapshot below proves nothing
        # is written), then the reflector pick switches to what changed
        bg_workers = [s for s in ("radar", "lidar") if s in sensors]
        assert lv.control({"background": "take"})["background"].startswith("take:")
        assert "background" not in lv.control_state, "a background must not carry over to a new live view"
        t_end = time.monotonic() + 25
        while time.monotonic() < t_end:
            hs = {s: lv.status()["sensors"][s]["header"] for s in bg_workers}
            if all(h.get("background", {}).get("state") == "ready" for h in hs.values()):
                break
            time.sleep(0.3)
        else:
            raise AssertionError(f"background not ready: {[h.get('background') for h in hs.values()]}")
        assert all(isinstance(h.get("changed"), list) for h in hs.values())
        lv.control({"background": "clear"})
        t_end = time.monotonic() + 10
        while time.monotonic() < t_end:
            if all(lv.status()["sensors"][s]["header"].get("background", {}).get("state") == "off"
                   for s in bg_workers):
                break
            time.sleep(0.3)
        else:
            raise AssertionError("background not cleared")
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


def test_capture_names_are_checked(tmp_path):
    lv = liveview.LiveView(sys.executable, None, calib_root=tmp_path)
    assert "set name" in lv.capture("background", "../x")["error"]
    assert "position name" in lv.capture("position", "calib_1", "a/b")["error"]
    assert "reserved" in lv.capture("position", "calib_1", "background")["error"]
    assert "not on" in lv.capture("position", "calib_1", "5m_left")["error"]
    assert "kind" in lv.capture("delete", "calib_1", "x")["error"]
    assert list(tmp_path.iterdir()) == []                        # nothing made for a refused capture


def test_calibration_folder_is_not_a_session(tmp_path):
    (tmp_path / server.CALIB_DIR / "calib_1").mkdir(parents=True)
    (tmp_path / "run_1").mkdir()
    assert [s["name"] for s in server.list_sessions(tmp_path)] == ["run_1"]


@need_session
def test_calibration_capture_writes_only_its_folder(tmp_path):
    """Save background, then Save position (the same session plays both, so "found" is not
    asserted): every worker writes its part into calibration/<set>/<name>/, the set reads back,
    calib_reflector.py reads the captures, and nothing in the played session changes."""
    pytest.importorskip("torch")
    sensors = ["radar", "lidar"]
    before = _snapshot(S)
    lv = liveview.LiveView(sys.executable, S, max_secs=90, calib_root=tmp_path)
    assert lv.start(sensors) == ""
    try:
        t_end = time.monotonic() + 60
        while time.monotonic() < t_end and not all(lv.status()["sensors"][s]["seq"] >= 3 for s in sensors):
            time.sleep(0.5)
        for kind, name in (("background", None), ("position", "p1")):
            r = lv.capture(kind, "calib_t", name, 1.0, 15.0)
            assert "dir" in r, r
            t_end = time.monotonic() + 40
            while time.monotonic() < t_end:
                caps = [lv.status()["sensors"][s]["header"].get("capture", {}) for s in sensors]
                if all(c.get("token") == r["token"] and c.get("state") in ("saved", "error") for c in caps):
                    break
                time.sleep(0.3)
            assert all(c.get("state") == "saved" for c in caps), caps
        assert lv.capture("position", "calib_t", "p1")["error"].startswith("position 'p1' already exists")
        d = tmp_path / "calib_t"
        for f in ("background/radar/background_band.npy", "background/lidar/background_voxels.npy",
                  "p1/radar/frames.npy", "p1/radar/reflector.json", "p1/lidar/sweeps.npz", "p1/lidar/reflector.json",
                  "p1/capture.json"):
            assert (d / f).is_file(), f
        st = lv.calib_set("calib_t")
        assert st["background"] and [p["name"] for p in st["positions"]] == ["p1"]
        assert {"radar", "lidar"} <= set(st["positions"][0])
        lv.control({"reflector_only": True})
        t_end = time.monotonic() + 10
        while time.monotonic() < t_end and not all(lv.status()["sensors"][s]["header"].get("reflector_only")
                                                   for s in sensors):
            time.sleep(0.3)
        assert all(lv.status()["sensors"][s]["header"].get("reflector_only") for s in sensors)
    finally:
        lv.stop()
        t_end = time.monotonic() + liveview.STOP_GRACE_S + 5
        while lv.active() and time.monotonic() < t_end:
            time.sleep(0.3)
        lv.shutdown()
    r = subprocess.run([sys.executable, str(PIPELINE / "calib_reflector.py"), "--set", str(tmp_path / "calib_t"),
                        "--no-figure", "--no-camera"], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert (tmp_path / "calib_t" / "reflector_summary.json").is_file()
    assert _snapshot(S) == before, "a calibration capture changed files in the played session"
