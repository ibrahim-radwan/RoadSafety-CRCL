"""Stage-level equivalence of perception/radar_dsp.py with the old chain's numpy functions, for
the stages the shipped configuration does not run (so the golden frames do not cover them):
CA and CAGO CFAR, and the Capon, MUSIC and Bartlett beamformers. Inputs are real frames.

Skips when the old chain (cascaded_radar_capture_pipeline/processing) is no longer in the tree or
the reference session is absent; the ports were verified while both existed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

PIPELINE = Path(__file__).resolve().parents[1]
TEAM = PIPELINE.parent / "cascaded_radar_capture_pipeline"
sys.path[:0] = [str(PIPELINE), str(TEAM)]
torch = pytest.importorskip("torch")

S = Path(os.environ.get("REPORT_TEST_SESSIONS", "/mnt/sdcard/sessions/freewalk_ref_02").split(",")[0])
pytestmark = pytest.mark.skipif(not (TEAM / "processing" / "radar_processing.py").is_file()
                                or not (S / "sync" / "radar_timeline.csv").is_file(),
                                reason="needs the old chain and a reference session")


@pytest.fixture(scope="module")
def frame():
    from perception.core.calibration import load_radar_config
    from perception.core.session import Session
    from perception.radar_chain import RadarChain
    sess = Session(S)
    cfg, corr = load_radar_config(sess.cfg_path)
    chain = RadarChain(cfg, corr, cfar="caso")
    k = len(sess.radar) // 3
    return cfg, chain, chain.process(sess.radar.raw(k))


@pytest.mark.parametrize("method", ["ca", "cago", "caso", "os"])
def test_cfar_matches_old(frame, method):
    from processing.radar_processing import cfar_2d as old
    from perception import radar_dsp as dsp
    cfg, chain, o = frame
    heat = 10.0 ** (o["RD_mti"] / 10.0)
    cfg.cfar_method = method
    try:
        d_old, t_old = old(heat.cpu().numpy(), cfg)
    finally:
        cfg.cfar_method = "os"
    d, t = dsp.cfar_2d(heat, cfg.cfar_guard, cfg.cfar_train, method, cfg.cfar_fa_rate, cfg.cfar_os_rank,
                       cfg.cfar_margin_db, cfg.cfar_min_power_db, cfg.nms_size)
    np.testing.assert_allclose(10 * np.log10(t.cpu().numpy()), 10 * np.log10(np.maximum(t_old, 1e-30)), atol=0.01)
    assert (d.cpu().numpy() == d_old).all()


@pytest.mark.parametrize("method", ["capon", "music", "bartlett"])
def test_beamformers_match_old(frame, method):
    """Capon: equal within 0.05 dB. MUSIC: the same peaks (its pseudo-spectrum is ill-conditioned
    where a range bin has no clear signal: the signal/noise subspace split is arbitrary there, and
    float32 arithmetic moves it; the old code itself runs in complex64). Bartlett: the old
    _az_bartlett cannot run (its einsum reads the [azimuth, channel] steering matrix as [channel,
    azimuth] and raises ValueError), so the reference is its intended a^H R a, built from the old
    chain's own steering and covariance helpers."""
    import processing.radar_processing as rp
    from perception import radar_dsp as dsp
    cfg, chain, o = frame
    cube, pos = dsp.unique_aperture(o["rd_mti_channels"], chain.az_pos)
    rows = np.flatnonzero(chain.keep_r.cpu().numpy())[:40]           # 40 range bins: the old code loops in Python
    focus = o["focus"][:, rows]
    sub = cube[:, rows]
    sub_np, focus_np = sub.cpu().numpy(), focus.cpu().numpy()
    if method == "capon":
        old = rp._az_capon(sub_np, cfg, pos.astype(np.float32), focus_np)
        new = dsp.ra_capon(sub, focus, pos, chain.win_a, cfg.az_fft_size, cfg.music_diag_load,
                           cfg.music_forward_backward_avg).cpu().numpy()
        np.testing.assert_allclose(new, old, atol=0.05)
    elif method == "music":
        old = rp._az_music(sub_np, cfg, pos.astype(np.float32), focus_np)
        new = dsp.ra_music(sub, focus, pos, chain.win_a, cfg.az_fft_size, cfg.music_num_sources,
                           cfg.music_diag_load, cfg.music_forward_backward_avg).cpu().numpy()
        assert np.mean(np.abs(new.argmax(axis=1) - old.argmax(axis=1)) <= 1) >= 0.95
        assert np.median(np.abs(new - old)) < 0.05
    else:
        taper = rp._az_window_weights(cfg, len(pos))
        sv = (rp._steering_matrix(pos, cfg.az_fft_size) * taper[None, :]).astype(np.complex64)   # [az, P]
        old = np.zeros((len(rows), cfg.az_fft_size), np.float32)
        for i in range(len(rows)):
            X = (rp._range_snapshots(sub_np, focus_np, i).T * taper[:, None]).astype(np.complex64)
            R = rp._covariance_from_snapshots(X, diag_load=0.0, fb_avg=cfg.music_forward_backward_avg)
            old[i] = 20.0 * np.log10(np.maximum(np.einsum("ki,ij,kj->k", sv.conj(), R, sv).real, 1e-20) + 1e-9)
        new = dsp.ra_bartlett(sub, focus, pos, chain.win_a, cfg.az_fft_size, cfg.music_forward_backward_avg)
        np.testing.assert_allclose(new.cpu().numpy(), old, atol=0.05)
