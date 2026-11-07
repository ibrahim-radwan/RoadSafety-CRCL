"""The corner-reflector tools (2026-11-07), on synthetic data (no session needed):

  perception/core/reflector.py   LiDAR change -> the persistent reflector, not a passer-by; radar
                                 change peak with sub-cell interpolation; the 2D rigid fit
  RadarDetector.static_band / static_map_of   = process(static=True)["static_map"]
  RadarDetector.detect_static_map             = process(static=True)["static_detections"] (no interpolation)
  replay LidarSource._fill_gap   fills exactly the sector a short RS-Fairy sweep lacks, and nothing else
"""
from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest

PIPELINE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PIPELINE), str(PIPELINE / "replay")]
from perception.core import reflector as rf  # noqa: E402

RNG = np.random.default_rng(3)


def _pts(xyz, intensity=50.0):
    xyz = np.asarray(xyz, np.float32)
    r = np.linalg.norm(xyz, axis=1)
    return np.c_[xyz, r, np.full(len(xyz), intensity, np.float32), np.zeros(len(xyz), np.float32)].astype(np.float32)


def _wall_sweep():
    """A wall at x = 8 m and a floor, ~2000 returns."""
    y, z = RNG.uniform(-5, 5, 1500), RNG.uniform(-1.0, 2.0, 1500)
    fx, fy = RNG.uniform(1, 8, 600), RNG.uniform(-5, 5, 600)
    return _pts(np.r_[np.c_[np.full(1500, 8.0), y, z], np.c_[fx, fy, np.full(600, -1.1)]])


def test_lidar_reflector_is_the_persistent_change():
    bg = rf.background_voxels([_wall_sweep() for _ in range(5)])
    sweeps = []
    for i in range(20):
        refl = _pts(np.array([4.0, 1.0, -0.3]) + RNG.normal(0, 0.03, (8, 3)))       # in every sweep
        walker = _pts(np.array([3.0, -2.0 + 0.2 * i, 0.0]) + RNG.normal(0, 0.1, (30, 3)))   # moves
        sweeps.append(np.r_[_wall_sweep(), refl, walker])
    res = rf.lidar_reflector(sweeps, bg, 1.0, 10.0)
    assert res["found"]
    assert abs(res["x"] - 4.0) < 0.1 and abs(res["y"] - 1.0) < 0.1 and abs(res["z"] + 0.3) < 0.1
    assert res["present"] == 1.0 and 5 <= res["n_per_sweep"] <= 8
    assert all(len(p) <= 10 for p in res["points"])             # the reflector only, not the walker


def test_lidar_no_reflector_in_the_background_itself():
    sw = [_wall_sweep() for _ in range(10)]
    bg = rf.background_voxels(sw[:5])
    assert not rf.lidar_reflector(sw[5:], bg, 1.0, 10.0)["found"]


def test_radar_reflector_interpolated_peak():
    r_axis = np.arange(64) * 0.3
    az = np.linspace(-60, 60, 121)
    ref = np.full((64, 121), 100.0)
    rr, aa = np.meshgrid(r_axis, az, indexing="ij")
    maps = []
    for _ in range(8):
        m = ref + RNG.normal(0, 0.5, ref.shape)
        m = 10 * np.log10(10 ** (m / 10) + 10 ** (14.0 - ((rr - 5.08) / 0.25) ** 2 - ((aa - 10.4) / 3.0) ** 2))
        maps.append(m + 100 - 14)                                      # a +~40 dB bump at 5.08 m, +10.4 deg
    res = rf.radar_reflector(maps, ref, r_axis, az, 1.0, 15.0)
    assert res["found"] and res["present"] == 1.0
    assert abs(res["range"] - 5.08) < 0.08 and abs(res["az_deg"] - 10.4) < 0.4
    assert len(res["mean_points"]) >= 1


def test_fit_rigid_2d_recovers_yaw_and_shift():
    src = RNG.uniform(-5, 5, (10, 2))
    yaw = np.radians(37.2)
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    dst = src @ R.T + [-0.03, -0.01]
    f = rf.fit_rigid_2d(src, dst)
    assert abs(f["yaw_deg"] - 37.2) < 1e-6 and abs(f["tx_m"] + 0.03) < 1e-6 and f["max_m"] < 1e-9


def _detector():
    torch = pytest.importorskip("torch")
    import radar_firmware
    from perception.radar import RadarDetector
    cfg = radar_firmware.cfg_for_variant("r075m_v2_8fps")
    if not Path(cfg).is_file():
        pytest.skip("no r075m_v2_8fps .cfg")
    return torch, RadarDetector(cfg, device="cuda" if torch.cuda.is_available() else "cpu")


def test_static_band_map_equals_process():
    torch, det = _detector()
    raw = RNG.integers(-200, 200, det.n_loops * det.n_tx * det.n_rx * det.n_s * 2, dtype=np.int16)
    out = det.process(raw, moving_only=False, detect=False, static=True)
    m = out["static_map"].float().cpu().numpy()
    m2 = det.static_map_of(det.static_band(raw))
    assert m2.shape == m.shape and np.abs(m2 - m).max() < 1e-2
    d0 = out["static_detections"]
    d1 = det.detect_static_map(m, interpolate=False)
    assert d1.shape == d0.shape and np.allclose(d1, d0, atol=1e-3)


def test_coherent_subtraction_removes_the_background():
    torch, det = _detector()
    raw = RNG.integers(-200, 200, det.n_loops * det.n_tx * det.n_rx * det.n_s * 2, dtype=np.int16)
    b = det.static_band(raw)
    assert det.static_map_of(b - b).max() < -50                 # 10 log10(1e-6) = -60 dB floor


def test_replay_fill_gap_only_fills_a_short_sweep():
    import sources

    def sweep(gap: bool):
        az = np.radians(np.arange(-180, 180, 0.25))
        el = np.radians(np.arange(-10, 30, 1.0))
        A, E = np.meshgrid(az, el)
        keep = ~((np.degrees(A) > -16.5) & (np.degrees(A) < 0)) if gap else np.ones_like(A, bool)
        r = 6.0
        xyz = np.c_[(r * np.cos(E) * np.cos(A))[keep], (r * np.cos(E) * np.sin(A))[keep], (r * np.sin(E))[keep]]
        return _pts(xyz)
    fake = types.SimpleNamespace(ext={"yaw_deg": 0.0, "tx_m": 0.0, "ty_m": 0.0}, slock=threading.Lock(), sweeps=[],
                                 GAP_AZ=sources.LidarSource.GAP_AZ)
    fake._own_az = types.MethodType(sources.LidarSource._own_az, fake)
    fill = types.MethodType(sources.LidarSource._fill_gap, fake)
    full, short = sweep(False), sweep(True)
    assert fill((0.0, full))[1] == 0                             # nothing before it: nothing to fill from
    fake.sweeps.append((0.0, full))
    p, n = fill((0.1, short))
    missing = len(full) - len(short)
    assert 0.95 * missing <= n <= missing                       # (cells half-empty at the gap's edges stay)
    fake.sweeps.append((0.1, short))
    assert fill((0.2, full))[1] == 0                             # a full sweep after a short one: untouched


def test_reflector_from_new_matches_and_replay_readout():
    """lidar_reflector_from_new (the replay's path: returns already found new) gives what lidar_reflector
    gives; the replay's readout rows hold both sensors' reflector and radar - LiDAR."""
    rng = np.random.default_rng(3)
    bg = [np.c_[rng.uniform(1, 20, (4000, 2)) * [1, 0.5], rng.uniform(-1, 1, 4000)] for _ in range(3)]
    bg = [np.c_[b, np.linalg.norm(b, axis=1), np.full(len(b), 30.0)].astype(np.float32) for b in bg]
    ref = rf.background_voxels(bg)
    sweeps = []
    for k in range(6):
        refl = np.c_[5.0 + rng.normal(0, 0.03, 8), -1.8 + rng.normal(0, 0.03, 8), rng.normal(-0.3, 0.05, 8)]
        refl = np.c_[refl, np.linalg.norm(refl, axis=1), np.full(8, 80.0)].astype(np.float32)
        sweeps.append(np.r_[bg[k % 3], refl])
    a = rf.lidar_reflector(sweeps, ref, 3, 7)
    b = rf.lidar_reflector_from_new([p[rf.new_points(p, ref)] for p in sweeps], 3, 7)
    assert a["found"] and b["found"] and abs(a["x"] - b["x"]) < 1e-6 and abs(a["y"] - b["y"]) < 1e-6
    sys.path.insert(0, str(PIPELINE / "replay"))
    from replay_session import reflector_readout
    rad = {"reflector": dict(range=5.45, az_deg=-21.0, x=5.09, y=-1.95, power_db=140.0, refined=True)}
    rows = reflector_readout({"reflector": b, "background": (8, 100)}, rad)
    assert [r[0] for r in rows] == ["LiDAR", "radar", "Δ"]
    assert abs(rows[2][3] - (5.45 - b["range"])) < 1e-9 and abs(rows[2][4] - (-21.0 - b["az_deg"])) < 1e-9
    assert reflector_readout(None, rad)[0][0] == "radar" and len(reflector_readout(None, None)) == 0
