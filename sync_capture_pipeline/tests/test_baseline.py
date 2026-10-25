"""Regression against the frozen golden frames (perception/baseline.py golden).

For every session with <session>/sync/baseline/golden_ours.npz: our radar detector, run again on
the golden frames, must reproduce the stored range-Doppler and range-azimuth maps and detections.
This guards the restructuring of the library: a refactor must not change the output. A deliberate
algorithm change re-creates the golden files (and re-scores with baseline.py compare).

Equivalence tests for each stage ported from the old chain compare against golden_team_gpu.npz /
golden_team_cpu.npz and are added with the port.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE))
pytest.importorskip("torch")

DEFAULT = ["/mnt/sdcard/sessions/freewalk_ref_02", "/mnt/sdcard/sessions/stage3_02"]
SESSIONS = [Path(s) for s in os.environ.get("REPORT_TEST_SESSIONS", ",".join(DEFAULT)).split(",") if s]
SESSIONS = [s for s in SESSIONS if (s / "sync" / "baseline" / "golden_ours.npz").is_file()]
pytestmark = pytest.mark.skipif(not SESSIONS, reason="no session with sync/baseline/golden_ours.npz")


@pytest.mark.parametrize("S", SESSIONS, ids=[s.name for s in SESSIONS])
def test_ours_reproduces_golden(S):
    import radar_firmware
    from perception.radar import RX_PHASE_FILE, RadarDetector, rx_phase_for
    from perception.baseline import file_hash
    g = np.load(S / "sync" / "baseline" / "golden_ours.npz")
    meta = json.loads((S / "sync" / "baseline" / "golden.json").read_text())
    if meta["rx_phase_file_md5"] != file_hash(RX_PHASE_FILE):
        pytest.skip("calibration/radar_rx_phase.json changed since the golden files were made: re-create them")
    variant, _ = radar_firmware.session_variant(S)
    det = RadarDetector(radar_firmware.cfg_for_variant(variant), rx_phase_deg=rx_phase_for(variant))
    rows = list(csv.DictReader((S / "sync" / "radar_timeline.csv").open()))
    for i, k in enumerate(g["frames"]):
        o = det.process(np.fromfile(S / "radar" / rows[int(k)]["filename"], dtype=np.int16))
        np.testing.assert_allclose(o["rd"].cpu().numpy(), g["rd"][i], atol=0.05)    # dB; GPU float order
        np.testing.assert_allclose(o["ra"].cpu().numpy(), g["ra"][i], atol=0.05)
        d = o["detections"]
        ref = g["det"][g["det_start"][i]:g["det_start"][i + 1]]
        assert len(d) == len(ref), f"frame {k}: {len(d)} detections, golden {len(ref)}"
        np.testing.assert_allclose(d, ref, atol=0.05)
