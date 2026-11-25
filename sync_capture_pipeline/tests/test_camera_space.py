"""Camera space on the staging disk (no sensors needed):

  * capture_session.stage_status: free space, road minutes, leftovers (not the session recording now);
  * the web server refuses Start below the reserve and for a duration longer than the space;
  * "Move to SD card": a leftover moves into its session on the SD card (recorded in session.json);
    leftovers without a session, or whose session already has a camera folder, are not moved.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PIPELINE), str(PIPELINE / "web_control")]
import capture_session as cs  # noqa: E402

server = pytest.importorskip("server")


def setup(tmp_path: Path):
    root, stage = tmp_path / "sd", tmp_path / "stage"
    for name in ("run_B", "run_D"):
        (root / name).mkdir(parents=True)
        (stage / name / "camera").mkdir(parents=True)
        (stage / name / "camera" / "camera.svo2").write_bytes(b"\1" * 2_000_000)
    (root / "run_B" / "session.json").write_text(json.dumps({"camera_staged": {"error": "SD card error"}}))
    (root / "run_D" / "camera").mkdir()                      # a partial copy already on the SD card
    (stage / "orphan" / "camera").mkdir(parents=True)
    (stage / "orphan" / "camera" / "x").write_bytes(b"\1")
    (stage / "empty").mkdir()                                # empty folders are not leftovers
    (stage / "now" / "camera").mkdir(parents=True)
    (stage / "now" / "camera" / "x").write_bytes(b"\1")
    return root, stage


def test_stage_status(tmp_path):
    _, stage = setup(tmp_path)
    st = cs.stage_status(stage, active="now", reserve_gb=1.0)
    assert [x["name"] for x in st["leftovers"]] == ["orphan", "run_B", "run_D"]
    assert st["minutes"] == pytest.approx(max(0.0, st["free_gb"] - 1.0) / cs.GB_PER_HOUR["camera"] * 60, abs=0.2)
    assert cs.stage_status(stage, reserve_gb=1e6)["minutes"] == 0


def test_start_refused_without_space(tmp_path, monkeypatch):
    root, stage = setup(tmp_path)
    c = server.create_app(server.Session(root, sys.executable, sys.executable, stage)).test_client()
    sp = c.get("/api/status").json["camera_space"]
    r = c.post("/api/start", json={"name": "x1", "secs": (sp["minutes"] + 5) * 60, "sensors": ["camera"]})
    assert r.status_code == 400 and "more than the" in r.json["error"]
    orig = cs.stage_status
    monkeypatch.setattr(cs, "stage_status", lambda st, active=None: orig(st, active, reserve_gb=1e6))
    r = c.post("/api/start", json={"name": "x2", "secs": 0, "sensors": ["camera"]})
    assert r.status_code == 400 and "not enough camera space" in r.json["error"]
    assert "move the leftover camera data" in r.json["error"]
    assert not (root / "x2").exists()


def test_move_leftover(tmp_path):
    root, stage = setup(tmp_path)
    c = server.create_app(server.Session(root, sys.executable, sys.executable, stage)).test_client()
    why = {x["name"]: (x["movable"], x["why"]) for x in c.get("/api/status").json["camera_space"]["leftovers"]}
    assert why["run_B"][0] and not why["orphan"][0] and not why["run_D"][0]
    assert "no session" in c.post("/api/stage/move", json={"name": "orphan"}).json["error"]
    assert "already has a camera folder" in c.post("/api/stage/move", json={"name": "run_D"}).json["error"]
    assert c.post("/api/stage/move", json={"name": "run_B"}).json == {"ok": True}
    for _ in range(100):
        mv = c.get("/api/status").json["stage_move"]
        if mv["state"] != "moving":
            break
        time.sleep(0.05)
    assert mv["state"] == "done", mv
    assert (root / "run_B" / "camera" / "camera.svo2").stat().st_size == 2_000_000
    assert not (stage / "run_B").exists() and (stage / "run_D").exists()
    meta = json.loads((root / "run_B" / "session.json").read_text())
    assert meta["camera_staged"] == {"error": "SD card error"} and meta["camera_moved_later"]["gb"] == 0.002
    names = [x["name"] for x in c.get("/api/status").json["camera_space"]["leftovers"]]
    assert "run_B" not in names
