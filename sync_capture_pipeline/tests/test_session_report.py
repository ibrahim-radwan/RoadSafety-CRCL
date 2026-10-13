"""Tests for the HTML session report (make_report.py / report/).

The report is descriptive, so the tests check that it is complete and
truthful rather than judging the data:
  - it builds for every kept session, including the DCA-fault session;
  - every figure renders to a valid PNG;
  - the page is self-contained (no external references);
  - its numbers agree with check_session.txt and sync/alignment.json.

Run (conda env):
    ~/miniconda3/envs/rel_1_py_3_8/bin/python -m pytest -v sync_capture_pipeline/tests
Sessions: $REPORT_TEST_SESSIONS (comma-separated paths) or the default list below.
Collected data is cached in <session>/sync/report_data.pkl; the first run (or
REPORT_TEST_REGENERATE=1) collects from the raw data, which takes minutes per session.
"""
from __future__ import annotations

import base64
import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE))
from report import collect, figures as F, render  # noqa: E402

DEFAULT = ["/mnt/sdcard/sessions/freewalk_ref_02", "/mnt/sdcard/sessions/freewalk_ref_01",
           "/mnt/sdcard/sessions/stage3_01", "/mnt/sdcard/sessions/stage3_02", "/mnt/sdcard/sessions/stage3_03"]
SESSIONS = [Path(s) for s in os.environ.get("REPORT_TEST_SESSIONS", ",".join(DEFAULT)).split(",") if s]
SESSIONS = [s for s in SESSIONS if (s / "session.json").is_file()]
REGEN = os.environ.get("REPORT_TEST_REGENERATE") == "1"

# figure functions; OPTIONAL may legitimately be absent (e.g. no single LiDAR target)
FIGS = ["timeline", "intervals", "pairing", "velocity_check", "radar_range_time", "radar_doppler_target",
        "radar_adc", "radar_profiles", "lidar_maps", "lidar_stats", "camera_thumbs", "camera_quality",
        "camera_imu", "range_overlay", "snapshots"]
OPTIONAL = {"radar_doppler_target", "camera_imu"}

_cache = {}


def data(S: Path) -> dict:
    if S in _cache:
        return _cache[S]
    if not (S / "sync" / "radar_timeline.csv").is_file():
        subprocess.run([sys.executable, str(PIPELINE / "align_session.py"), "--session", str(S)], check=True,
                       stdout=subprocess.DEVNULL)
    pkl = S / "sync" / "report_data.pkl"
    if pkl.is_file() and not REGEN:
        D = pickle.loads(pkl.read_bytes())
    else:
        D = collect.collect(S)
        pkl.write_bytes(pickle.dumps(D, protocol=4))
    _cache[S] = D
    return D


def check_value(S: Path, pattern: str):
    text = (S / "check_session.txt").read_text()
    m = re.search(pattern, text)
    return int(m.group(1).replace(",", "")) if m else None


pytestmark = pytest.mark.skipif(not SESSIONS, reason="no sessions available (SD card not mounted?)")
ids = [s.name for s in SESSIONS]


@pytest.mark.parametrize("S", SESSIONS, ids=ids)
def test_page_builds_and_is_self_contained(S):
    page = render.build(data(S))
    for sec in ("overview", "timeline", "sync", "radar", "lidar", "camera", "cross"):
        assert f'id="{sec}"' in page, f"section {sec} missing"
    assert not re.search(r'(src|href)="https?://', page), "external reference in a self-contained report"
    assert "<li>" in page                                     # observations present
    assert "PASS" not in page.split("Observations")[1].split("</ul>")[0], "observations must not give a verdict"


@pytest.mark.parametrize("S", SESSIONS, ids=ids)
@pytest.mark.parametrize("name", FIGS)
def test_each_figure_renders(S, name):
    b64 = getattr(F, name)(data(S))
    if b64 is None:
        assert name in OPTIONAL, f"{name} returned nothing"
        return
    raw = base64.b64decode(b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n" and len(raw) > 5000, f"{name}: not a real PNG"


@pytest.mark.parametrize("S", SESSIONS, ids=ids)
def test_numbers_match_check_session(S):
    D = data(S)
    n_radar = check_value(S, r"frames\s+:\s+([\d,]+) in manifest")
    if n_radar is not None:
        assert len(D["radar"]["t"]) == n_radar
    n_cam = check_value(S, r"frames\s+:\s+([\d,]+) recorded")
    if n_cam is not None:
        assert len(D["camera"]["frame_index"]) == n_cam
    dropped = check_value(S, r"dropped\s+:\s+([\d,]+) frames")
    if dropped is not None:
        assert int(D["camera"]["missing"].sum()) == dropped


@pytest.mark.parametrize("S", SESSIONS, ids=ids)
def test_numbers_match_alignment(S):
    D = data(S)
    al = json.loads((S / "sync" / "alignment.json").read_text())
    assert len(D["radar"]["t"]) == al["radar"]["frames"]
    assert len(D["camera"]["frame_index"]) == al["camera"]["frames"]
    # the report decodes the LiDAR itself; it must see the same sweeps as align_session
    assert abs(len(D["lidar"]["sweep_t"]) - al["lidar"]["sweeps"]) <= 2
    assert D["align"]["offsets"] == al["offsets"]


@pytest.mark.parametrize("S", SESSIONS, ids=ids)
def test_observations_are_factual(S):
    D = data(S)
    obs = render.observations(D)
    assert any("Radar raw data" in o for o in obs)
    assert any(o.startswith("Camera:") for o in obs)
    assert any(o.startswith("LiDAR:") for o in obs)
    pps = D["log"]["radar_pps"]
    if len(pps) > 3:
        line = next(o for o in obs if o.startswith("Radar packet rate"))
        assert f"{int(pps[2:, 1].min()):,}" in line


def test_dca_fault_session_is_visible():
    """freewalk_ref_01: the DCA1000 throttled to ~2,970 packets/s. The report must
    show it (low packet rate in the observations), not hide or crash on it."""
    S = Path("/mnt/sdcard/sessions/freewalk_ref_01")
    if S not in SESSIONS:
        pytest.skip("freewalk_ref_01 not available")
    line = next(o for o in render.observations(data(S)) if o.startswith("Radar packet rate"))
    low = int(re.search(r"rate ([\d,]+)", line).group(1).replace(",", ""))
    assert low < 5000, line
