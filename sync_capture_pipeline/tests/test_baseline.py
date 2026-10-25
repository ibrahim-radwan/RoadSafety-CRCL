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


@pytest.mark.parametrize("path", ["gpu", "cpu"])
@pytest.mark.parametrize("S", SESSIONS, ids=[s.name for s in SESSIONS])
def test_team_chain_reproduces_golden(S, path, tmp_path):
    """The old chain's process_frame (until it is retired) must give exactly its golden outputs on
    both of its paths: the merge moves code around it (config, reader) without changing it."""
    import subprocess
    gp = S / "sync" / "baseline" / f"golden_team_{path}.npz"
    if not gp.is_file():
        pytest.skip(f"no {gp.name}")
    g = np.load(gp)
    if path == "gpu":
        pytest.importorskip("cupy")
    out = tmp_path / f"team_{path}.npz"
    subprocess.run([sys.executable, str(PIPELINE / "perception" / "baseline.py"), "_team", "--session", str(S),
                    "--path", path, "--frames", ",".join(str(int(k)) for k in g["frames"]), "--out", str(out)],
                   check=True, capture_output=True)
    n = np.load(out)
    for f in ("RD_raw", "RD_mti", "RA", "RE", "rd_det", "ra_det", "rd_thresh", "ra_thresh", "xyz", "xyz_start"):
        np.testing.assert_allclose(n[f], g[f], rtol=0, atol=1e-4, err_msg=f)


@pytest.mark.parametrize("path", ["gpu", "cpu"])
@pytest.mark.parametrize("S", SESSIONS, ids=[s.name for s in SESSIONS])
def test_radar_chain_reproduces_team_golden(S, path):
    """perception/radar_chain.py (the old chain's front end in PyTorch) must give the old chain's
    golden maps and range-Doppler detections: path "gpu" = CASO-CFAR, "cpu" = OS-CFAR.

    The CPU path's range-azimuth map is compared on the range bins with detections: in the others
    the focus mask falls back to the top 3 of nearly equal noise bins, and a 1e-4 dB arithmetic
    difference picks a different one (the old chain's own two paths differ there; the port agrees
    with the GPU path everywhere)."""
    from perception.core.calibration import load_radar_config
    from perception.core.session import Session
    from perception.radar_chain import RadarChain
    gp = S / "sync" / "baseline" / f"golden_team_{path}.npz"
    if not gp.is_file():
        pytest.skip(f"no {gp.name}")
    g = np.load(gp)
    sess = Session(S)
    cfg, corr = load_radar_config(sess.cfg_path)
    chain = RadarChain(cfg, corr, cfar="caso" if path == "gpu" else "os")
    for i, k in enumerate(g["frames"]):
        o = {f: v.cpu().numpy() for f, v in chain.process(sess.radar.raw(int(k))).items()}
        for f in ("RD_raw", "RD_mti", "rd_thresh", "RE"):
            np.testing.assert_allclose(o[f], g[f][i], atol=0.01, err_msg=f"{f}, frame {k}")
        rows = slice(None) if path == "gpu" else o["rd_det"].any(axis=0)
        ra, ga = o["RA"][rows], g["RA"][i][rows]
        np.testing.assert_allclose(ra, ga, atol=0.05, err_msg=f"RA, frame {k}")
        assert (o["rd_det"] == g["rd_det"][i]).all(), f"rd_det, frame {k}"
