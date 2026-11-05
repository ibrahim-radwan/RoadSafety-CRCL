"""LiDAR -> vehicle extrinsic (perception/core/extrinsic.py): selection by mounting date and by a
session's alignment.json, and the point transform. No recorded data needed."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE))
from perception.core import extrinsic as ex  # noqa: E402


def _file(tmp_path):
    f = tmp_path / "ext.json"
    f.write_text(json.dumps({"mountings": [
        {"id": "b", "valid_from": "2026-11-03T00:00:00", "yaw_deg": 40.0, "tx_m": 0.1, "ty_m": -0.2},
        {"id": "a", "valid_from": "2026-10-20T00:00:00", "yaw_deg": 5.0, "tx_m": 0.0, "ty_m": 0.0}]}))
    return f


def test_selection_by_date(tmp_path):
    f = _file(tmp_path)
    assert ex.extrinsic_for_date("2026-10-11T09:56:13+1100", f)["id"] == "identity"
    assert ex.extrinsic_for_date("2026-10-25T12:00:00+1100", f)["id"] == "a"
    assert ex.extrinsic_for_date("2026-11-03T16:37:21+1100", f)["id"] == "b"
    assert ex.current_extrinsic(f)["id"] == "b"
    assert ex.extrinsic_for_date("2026-11-03", ex.EXTRINSIC_FILE.parent / "missing.json")["id"] == "identity"


def test_session_uses_alignment_record_first(tmp_path):
    f = _file(tmp_path)
    s = tmp_path / "sess"
    (s / "sync").mkdir(parents=True)
    (s / "session.json").write_text(json.dumps({"started_local": "2026-11-03T16:37:21+1100"}))
    assert ex.extrinsic_for_session(s, f)["id"] == "b"
    (s / "sync" / "alignment.json").write_text(json.dumps({"lidar_extrinsic": dict(ex.IDENTITY)}))
    assert ex.extrinsic_for_session(s, f)["id"] == "identity"       # re-aligned with --lidar-extrinsic none


def test_identity_returns_the_same_points():
    p = np.random.default_rng(0).normal(size=(50, 6)).astype(np.float32)
    assert ex.lidar_points_to_vehicle(p, dict(ex.IDENTITY)) is p


def test_transform_rotates_shifts_and_keeps_no_returns():
    ext = {"id": "t", "yaw_deg": 90.0, "tx_m": 1.0, "ty_m": 0.0}
    p = np.array([[2.0, 0.0, 0.5, np.sqrt(4.25), 100, 0.01],     # 2 m ahead of the LiDAR
                  [0.0, 0.0, 0.0, 0.0, 0, 0.02]], np.float32)     # no return
    q = ex.lidar_points_to_vehicle(p, ext)
    assert np.allclose(q[0, :3], [1.0, 2.0, 0.5], atol=1e-5)    # turned 90 deg left, shifted 1 m
    assert np.isclose(q[0, 3], np.sqrt(1 + 4 + 0.25), atol=1e-5)
    assert np.all(q[1, :4] == 0)
    assert np.all(q[:, 4:] == p[:, 4:]) and np.all(p[0, :2] == [2.0, 0.0])   # input untouched


def test_repo_file_is_valid():
    e = ex.extrinsic_for_date("2026-11-03T18:44:12+1100")
    assert e["id"] != "identity" and 30 < e["yaw_deg"] < 50
    assert ex.extrinsic_for_date("2026-10-11T13:00:00+1100")["id"] == "identity"   # reference sessions
