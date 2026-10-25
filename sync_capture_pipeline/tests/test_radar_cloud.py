"""Tests for the pre-computed filtered radar point cloud (radar_cloud.py).

Checks every session that has sync/radar_cloud_filtered.npz:
  - one index entry per radar frame, points consistent with the index;
  - the pipeline-frame xyz is exactly the team's xyz converted
    (forward = team y, left = -team x, z = team z);
  - points lie inside the radar's range window.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pytest

DEFAULT = ["/mnt/sdcard/sessions/freewalk_ref_02", "/mnt/sdcard/sessions/stage3_02"]
SESSIONS = [Path(s) for s in os.environ.get("REPORT_TEST_SESSIONS", ",".join(DEFAULT)).split(",") if s]
SESSIONS = [s for s in SESSIONS if (s / "sync" / "radar_cloud_filtered.npz").is_file()]
pytestmark = pytest.mark.skipif(not SESSIONS, reason="no session with radar_cloud_filtered.npz")


@pytest.mark.parametrize("S", SESSIONS, ids=[s.name for s in SESSIONS])
def test_index_matches_frames(S):
    d = np.load(S / "sync" / "radar_cloud_filtered.npz")
    rows = list(csv.DictReader((S / "sync" / "radar_timeline.csv").open()))
    assert len(d["start"]) == len(rows) + 1
    assert d["start"][0] == 0 and d["start"][-1] == len(d["xyz"]) == len(d["team_xyz"])
    assert np.all(np.diff(d["start"]) >= 0)
    assert [str(f) for f in d["frame_file"]] == [r["filename"] for r in rows]


@pytest.mark.parametrize("S", SESSIONS, ids=[s.name for s in SESSIONS])
def test_conversion_to_pipeline_frame(S):
    d = np.load(S / "sync" / "radar_cloud_filtered.npz")
    t, x = d["team_xyz"], d["xyz"]
    np.testing.assert_allclose(x[:, 0], t[:, 1])          # forward
    np.testing.assert_allclose(x[:, 1], -t[:, 0])         # left (team x is right-positive)
    np.testing.assert_allclose(x[:, 2], t[:, 2])
    np.testing.assert_allclose(d["doppler"], t[:, 4])


@pytest.mark.parametrize("S", SESSIONS, ids=[s.name for s in SESSIONS])
def test_points_inside_range_window(S):
    d = np.load(S / "sync" / "radar_cloud_filtered.npz")
    r = d["team_xyz"][:, 5]
    assert (r > 0).all() and (r < 80).all()
