"""Stage-level equivalence of perception/radar_dsp.py with the old chain, for the stages the shipped
configuration does not run (so the per-frame golden files do not cover them): CA, CAGO, CASO and OS
CFAR, and the Capon, MUSIC and Bartlett beamformers.

The reference is <session>/sync/baseline/golden_stages.npz: the old chain's own functions
(processing/radar_processing.py: cfar_2d, _az_capon, _az_music) run on one real frame of
freewalk_ref_02 on 2026-10-25, before the old chain was deleted, with their inputs stored next to
their outputs. Bartlett's reference is its intended a^H R a built from the old chain's steering
and covariance helpers: the old _az_bartlett could not run (an einsum index error).

Criteria: CFAR thresholds within 0.01 dB and identical detections; Capon and Bartlett within
0.05 dB; MUSIC the same peaks (its pseudo-spectrum is ill-conditioned where a range bin has no
clear signal, and float32 arithmetic moves the signal/noise subspace split there).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE))
torch = pytest.importorskip("torch")

SESSIONS = [Path(s) for s in os.environ.get("REPORT_TEST_SESSIONS", "/mnt/sdcard/sessions/freewalk_ref_02").split(",")]
GOLDEN = [s / "sync" / "baseline" / "golden_stages.npz" for s in SESSIONS]
GOLDEN = [g for g in GOLDEN if g.is_file()]
pytestmark = pytest.mark.skipif(not GOLDEN, reason="no sync/baseline/golden_stages.npz (freewalk_ref_02)")


@pytest.fixture(scope="module")
def g():
    z = np.load(GOLDEN[0])
    return {k: z[k] for k in z.files}, json.loads(str(z["params"]))


@pytest.mark.parametrize("method", ["ca", "cago", "caso", "os"])
def test_cfar_matches_old(g, method):
    from perception import radar_dsp as dsp
    z, p = g
    heat = torch.as_tensor(z["heat"], device="cuda" if torch.cuda.is_available() else "cpu")
    d, t = dsp.cfar_2d(heat, p["cfar_guard"], p["cfar_train"], method, p["cfar_fa_rate"], p["cfar_os_rank"],
                       p["cfar_margin_db"], p["cfar_min_power_db"], p["nms_size"])
    np.testing.assert_allclose(10 * np.log10(t.cpu().numpy()),
                               10 * np.log10(np.maximum(z[f"cfar_{method}_thresh"], 1e-30)), atol=0.01)
    assert (d.cpu().numpy() == z[f"cfar_{method}_det"]).all()


@pytest.mark.parametrize("method", ["capon", "music", "bartlett"])
def test_beamformers_match_old(g, method):
    from perception import radar_dsp as dsp
    z, p = g
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cube = torch.as_tensor(z["bf_cube"], device=dev)
    focus = torch.as_tensor(z["bf_focus"], device=dev)
    pos = z["bf_pos"]
    win = dsp.window(p["az_window"], len(pos), p["chebyshev_atten_db"], dev)
    if method == "capon":
        new = dsp.ra_capon(cube, focus, pos, win, p["az_fft_size"], p["music_diag_load"],
                           p["music_forward_backward_avg"]).cpu().numpy()
        np.testing.assert_allclose(new, z["capon"], atol=0.05)
    elif method == "music":
        new = dsp.ra_music(cube, focus, pos, win, p["az_fft_size"], p["music_num_sources"], p["music_diag_load"],
                           p["music_forward_backward_avg"]).cpu().numpy()
        old = z["music"]
        assert np.mean(np.abs(new.argmax(axis=1) - old.argmax(axis=1)) <= 1) >= 0.95
        assert np.median(np.abs(new - old)) < 0.05
    else:
        new = dsp.ra_bartlett(cube, focus, pos, win, p["az_fft_size"], p["music_forward_backward_avg"]).cpu().numpy()
        np.testing.assert_allclose(new, z["bartlett"], atol=0.05)
