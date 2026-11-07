"""The measured radar array layout (calibration/radar_array.json, 2026-11-07), on synthetic frames
(no session needed): a point target placed with the measured layout and the inter-chip phase must
come out of RadarDetector at its angle, at every angle (the team's table split the beam and erred
5-16 deg off boresight), and the coherent refinement must be precise.

    the layout file       contiguous chip groups, chirps 3 and 4 elevated, rx_phase_for -> None
    moving target         detections at -40..+40 deg within 0.3 deg (refined) / 1.5 deg (incoherent)
    stationary reflector  refine_azimuth on a background-subtracted band within 0.2 deg
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE))
torch = pytest.importorskip("torch")

from perception.core.calibration import array_for, array_layout, rx_phase_for   # noqa: E402
from perception.core.conventions import AZIMUTH_SIGN                              # noqa: E402

VARIANT = "r075m_v2_8fps"


def _detector(**kw):
    import radar_firmware
    from perception.radar import RadarDetector
    cfg = radar_firmware.cfg_for_variant(VARIANT)
    if not Path(cfg).is_file():
        pytest.skip("no r075m_v2_8fps .cfg")
    return RadarDetector(cfg, device="cuda" if torch.cuda.is_available() else "cpu", **kw)


def _frame(det, az_world: float, rbin: int, v: float, amp: float = 2000.0, noise: float = 3.0, seed: int = 0):
    """A raw int16 frame with one point target at world azimuth az_world (+ left), range bin rbin,
    radial velocity v, laid out with the MEASURED array and the variant's inter-chip phase."""
    lay, ic = array_layout(), array_for(VARIANT)
    rng = np.random.default_rng(seed)
    s = np.sin(np.radians(AZIMUTH_SIGN * az_world))                     # processing sin (+ right)
    t_slot = (det.cfg.idle_time_us + det.cfg.ramp_end_us) * 1e-6
    fd = 2.0 * v / det.cfg.lambda_m
    n = np.arange(det.n_s)
    x = np.zeros((det.n_loops, det.n_tx, det.n_rx, det.n_s), complex)
    for t in range(det.n_tx):
        tp = lay["chirp_tx_pos"][t]
        for r in range(det.n_rx):
            chip = lay["rx_chip"][r]
            if tp is None:
                ph = 1.234 * (t + 1) + 0.3 * r                                  # elevated TX: not used in azimuth
            else:
                ph = np.pi * (chip * lay["chip_offset"] + tp + lay["rx_pos"][r]) * s
            if chip == 1:
                ph += ic["slave_phase_rad_per_m"] * rbin * det.range_res + np.radians(ic["slave_phase_deg"])
            for lp in range(det.n_loops):
                g = lp * det.n_tx + t
                x[lp, t, r] = np.exp(1j * (2 * np.pi * rbin * n / det.n_s + ph + 2 * np.pi * fd * g * t_slot))
    x = amp * x + noise * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    iq = np.stack([x.imag, x.real] if det.iq_swap else [x.real, x.imag], axis=-1)
    return np.round(iq).astype(np.int16).reshape(-1)


def test_layout_file():
    lay = array_layout()
    assert lay is not None, "calibration/radar_array.json has no layout"
    txp = lay["chirp_tx_pos"]
    assert [i for i, p in enumerate(txp) if p is None] == [3, 4]
    G = max(p for p in txp if p is not None) + max(lay["rx_pos"]) + 1
    pos = {p + q for p in txp if p is not None for q in lay["rx_pos"]}
    assert pos == set(range(G)), "each chip's group must be contiguous"
    assert lay["chip_offset"] >= G                                        # the groups do not overlap
    assert array_for(VARIANT) is not None
    assert rx_phase_for(VARIANT) is None                                  # in-situ phase belongs to the team layout


@pytest.mark.parametrize("az", [-40.0, -20.0, -9.0, 0.0, 4.0, 9.0, 20.0, 40.0])
def test_moving_target_azimuth(az):
    det = _detector(cuda_graph=False)
    v = float(det.dop_axis[det.n_loops // 2 + 8])
    d = det.process(_frame(det, az, 20, v))["detections"]
    assert len(d), "target not detected"
    from perception.radar import C
    q = d[np.argmax(d[:, C["snr_db"]])]
    assert abs(q[C["range"]] - 20 * det.range_res) < 0.2
    assert abs(q[C["az_deg"]] - az) < 0.3, f"refined azimuth {q[C['az_deg']]:.2f} for {az}"
    inc = _detector(cuda_graph=False, inter_chip=False)                   # incoherent only
    d2 = inc.process(_frame(inc, az, 20, v))["detections"]
    q2 = d2[np.argmax(d2[:, C["snr_db"]])]
    assert abs(q2[C["az_deg"]] - az) < 1.5, f"incoherent azimuth {q2[C['az_deg']]:.2f} for {az}"


def test_map_has_low_sidelobes():
    """One target: the range-azimuth map's strongest bin outside the main lobe is >= 15 dB down (the
    team table's FFT across the gap had 1-3 dB)."""
    det = _detector(cuda_graph=False)
    v = float(det.dop_axis[det.n_loops // 2 + 8])
    out = det.process(_frame(det, 12.0, 20, v, noise=0.0))
    p = out["p_az"].cpu().numpy()[det.n_loops // 2 + 8, 20]
    p = p[det.fov_sl]
    k = int(np.argmax(p))
    L, R = k, k
    while L > 0 and p[L - 1] < p[L]:
        L -= 1
    while R < len(p) - 1 and p[R + 1] < p[R]:
        R += 1
    side = np.r_[p[:L], p[R + 1:]]
    assert 10 * np.log10(p[k] / side.max()) >= 15.0


def test_refine_azimuth_on_a_static_reflector():
    det = _detector(cuda_graph=False)
    for az in (-13.1, 0.0, 9.0):
        bg = det.static_band(_frame(det, 0.0, 17, 0.0, amp=0.0, seed=1))
        bands = [det.static_band(_frame(det, az, 17, 0.0, seed=k)) - bg for k in range(4)]
        m = np.mean([det.static_map_of(b) for b in bands], axis=0)
        r_i, a_i = np.unravel_index(int(np.argmax(m)), m.shape)
        az0 = float(det.static_az_deg[a_i])
        az1, ok = det.refine_azimuth(bands, r_i * det.range_res, az0)
        assert ok and abs(az1 - az) < 0.2, f"{az}: incoherent {az0:.2f}, refined {az1:.2f}"


def test_background_subtraction():
    """set_background: a static scene (reflector A) is taken out coherently, so a new reflector B is the
    strongest stationary detection, at its angle; CUDA graph == eager; use_background False restores the
    plain map; refine_azimuths (batched) == refine_azimuth (one by one)."""
    from perception.radar import C
    dets = [_detector(cuda_graph=False)]
    if torch.cuda.is_available():
        dets.append(_detector())
    a_az, a_bin, b_az, b_bin = -25.0, 30, 9.0, 17
    bg = [_frame(dets[0], a_az, a_bin, 0.0, seed=k) for k in range(4)]
    both = (_frame(dets[0], a_az, a_bin, 0.0, seed=9).astype(np.int32)
            + _frame(dets[0], b_az, b_bin, 0.0, amp=600.0, seed=10)).astype(np.int16)
    maps = []
    for det in dets:
        plain = det.process(both, moving_only=False, static=True)["static_detections"]
        assert abs(plain[np.argmax(plain[:, C["power_db"]]), C["range"]] - a_bin * det.range_res) < 0.2
        det.set_background(bg)
        out = det.process(both, moving_only=False, static=True)
        maps.append(out["static_map"].cpu().numpy())
        d = out["static_detections"]
        q = d[np.argmax(d[:, C["power_db"]])]
        assert abs(q[C["range"]] - b_bin * det.range_res) < 0.2, "the background reflector is still the strongest"
        assert abs(q[C["az_deg"]] - b_az) < 2.0
        az1, ok1 = det.refine_azimuth([out["static_band"]], q[C["range"]], q[C["az_deg"]])
        azs, oks = det.refine_azimuths([out["static_band"]], [q[C["range"]], q[C["range"]]], [q[C["az_deg"]]] * 2)
        assert ok1 and abs(az1 - b_az) < 0.3 and np.allclose(azs, az1) and oks.all()
        det.use_background = False
        again = det.process(both, moving_only=False, static=True)["static_detections"]
        assert np.allclose(again, plain)
    if len(maps) == 2:
        assert np.abs(maps[0] - maps[1]).max() < 1e-3
