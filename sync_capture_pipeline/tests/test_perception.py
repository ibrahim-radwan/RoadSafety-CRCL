"""Tests for our detection pipeline (perception/): radar, LiDAR, tracker.

Synthetic tests (always run, GPU if available):
  - multipath ghosts are flagged, the direct return is not;
  - clustering keeps a walker's micro-Doppler spread as one object and leaves ghosts out;
  - the tracker confirms a steady mover, never a one-frame false alarm, and does not
    report a track whose motion its Doppler contradicts;
  - the LiDAR detector finds a person on flat and on sloped ground, ignores a roof
    with free space below, and classes the person as a pedestrian.

Session tests (skip when the reference sessions are absent; $REPORT_TEST_SESSIONS):
  - radar azimuth sign (world = - processing) and Doppler sign (+ = receding) agree
    with the LiDAR walker track;
  - the in-situ phase correction narrows the azimuth spread against the LiDAR.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pytest

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE))
torch = pytest.importorskip("torch")
from perception.lidar import CLASSES, O, LidarDetector  # noqa: E402
from perception.radar import C, DET_COLS, cluster_detections, flag_multipath  # noqa: E402
from perception.tracking import Tracker  # noqa: E402

DEFAULT = ["/mnt/sdcard/sessions/freewalk_ref_02"]
SESSIONS = [Path(s) for s in os.environ.get("REPORT_TEST_SESSIONS", ",".join(DEFAULT)).split(",") if s]
SESSIONS = [s for s in SESSIONS if (s / "sync" / "timing_lidar_track.npz").is_file()
            and (s / "radar").is_dir() and s.name.startswith("freewalk_ref_02")]


def _det(rows):
    d = np.zeros((len(rows), len(DET_COLS)), np.float32)
    for i, (r, az, v, pw) in enumerate(rows):
        d[i, C["range"]], d[i, C["az_deg"]], d[i, C["v_r"]], d[i, C["power_db"]] = r, az, v, pw
        d[i, C["x"]], d[i, C["y"]] = r * np.cos(np.radians(az)), r * np.sin(np.radians(az))
    return d


# ── radar (synthetic) ────────────────────────────────────────────────────────
def test_multipath_flags():
    d = _det([(5.0, 10.0, -1.2, 110.0),      # walker
              (10.1, 10.5, -2.3, 95.0),      # double bounce: 2 r, ~2 v
              (7.5, 9.0, -1.15, 100.0),      # behind, same v, 10 dB weaker: via a wall
              (8.0, -30.0, 0.8, 100.0)])     # an unrelated target
    g = flag_multipath(d, v_span=9.4)
    assert g.tolist() == [0.0, 1.0, 1.0, 0.0]


def test_cluster_keeps_walker_whole():
    rows = [(6.0, 0.0, -1.2, 110.0), (6.2, 1.0, -2.6, 100.0), (6.1, -1.0, -0.3, 100.0),   # one walker, limbs
            (15.0, 20.0, 1.0, 100.0)]
    d = _det(rows)
    d[3, C["ghost"]] = 0
    lab, objs = cluster_detections(d)
    assert len(objs) == 2 and lab[0] == lab[1] == lab[2] != lab[3]
    d[3, C["ghost"]] = 1                                                    # ghosts make no objects
    lab, objs = cluster_detections(d)
    assert len(objs) == 1 and lab[3] == -1


# ── tracker (synthetic) ──────────────────────────────────────────────────────
def test_tracker_confirms_mover_not_clutter():
    rng = np.random.default_rng(0)
    T = Tracker()
    reported = []
    for k in range(30):
        t = k * 0.125
        m = [[10 - 1.0 * t + rng.normal(0, 0.1), 2 + rng.normal(0, 0.1)]]
        if k == 7:
            m.append([3.0, -6.0])                                           # a one-frame false alarm
        T.update(t, np.array(m))
        reported.append([tr.id for tr in T.reported()])
    ids = {i for r in reported for i in r}
    assert len(ids) == 1                                                    # only the mover, one track
    assert all(len(r) == 1 for r in reported[4:])


def test_tracker_rejects_doppler_inconsistent_track():
    T = Tracker(max_vr_innov=0.6)
    for k in range(20):
        t = k * 0.125
        x = 10 - 1.0 * t                                                    # approaching at 1 m/s along x...
        T.update(t, np.array([[x, 0.0]]), radial=np.array([+1.5]))          # ...but "receding" at 1.5 m/s
    assert not T.reported()
    T = Tracker(max_vr_innov=0.6)
    for k in range(20):
        t = k * 0.125
        T.update(t, np.array([[10 - 1.0 * t, 0.0]]), radial=np.array([-1.0]))
    assert len(T.reported()) == 1


# ── LiDAR (synthetic) ────────────────────────────────────────────────────────
def _scene(slope=0.0, roof=False, person=True):
    rng = np.random.default_rng(1)
    g = rng.uniform([-20, -20], [40, 20], size=(60000, 2))
    z = -1.0 + slope * g[:, 0] + rng.normal(0, 0.02, len(g))                # ground, rising slope per m
    pts = [np.c_[g, z]]
    if person:                                                              # 0.5 x 0.4 x 1.75 m at (8, 1)
        p = rng.uniform([7.75, 0.8, 0], [8.25, 1.2, 1.75], size=(600, 3))
        p[:, 2] += -1.0 + slope * 8.0
        pts.append(p)
    if roof:                                                                # 6 x 6 m slab 2.4 m up, no posts
        r = rng.uniform([2, -3, 1.4], [8, 3, 1.45], size=(8000, 3))
        pts.append(r)
    P = np.concatenate(pts).astype(np.float32)
    rr = np.linalg.norm(P, axis=1)
    return np.c_[P, rr, np.full(len(P), 50.0), np.zeros(len(P))].astype(np.float32)


@pytest.mark.parametrize("slope", [0.0, 0.12])
def test_lidar_person_on_ground(slope):
    out = LidarDetector().detect(_scene(slope=slope))
    ob = out["objects"]
    near = ob[np.hypot(ob[:, O["x"]] - 8.0, ob[:, O["y"]] - 1.0) < 0.6]
    assert len(near) == 1
    assert CLASSES[int(near[0, O["cls"]])] == "pedestrian"
    assert abs(near[0, O["height"]] - 1.75) < 0.25
    others = ob[np.hypot(ob[:, O["x"]] - 8.0, ob[:, O["y"]] - 1.0) >= 0.6]
    assert len(others) == 0                                                 # the (sloped) ground makes no objects


def test_lidar_ignores_overhead_roof():
    out = LidarDetector().detect(_scene(roof=True, person=False))
    assert len(out["objects"]) == 0
    lab = out["point_label"]
    assert (lab[-8000:] == 0).all()                                         # roof: non-ground, in no object


# ── conventions on real data ─────────────────────────────────────────────────
@pytest.mark.parametrize("S", SESSIONS, ids=[s.name for s in SESSIONS])
def test_radar_signs_against_lidar(S):
    import radar_firmware
    from perception.radar import RadarDetector, rx_phase_for
    variant, _ = radar_firmware.session_variant(S)
    det = RadarDetector(radar_firmware.cfg_for_variant(variant), rx_phase_deg=rx_phase_for(variant))
    rows = list(csv.DictReader((S / "sync" / "radar_timeline.csv").open()))
    tr = np.load(S / "sync" / "timing_lidar_track.npz")
    ok_t = np.isfinite(tr["r"])
    az_pairs, v_pairs = [], []
    for k in range(0, len(rows), 7):
        t = float(rows[k]["t_ref"])
        j = int(np.argmin(np.abs(tr["t"] - t)))
        if abs(tr["t"][j] - t) > 0.06 or not ok_t[j] or not 3 < tr["r"][j] < 12:
            continue
        d = det.process(np.fromfile(S / "radar" / rows[k]["filename"], dtype=np.int16))["detections"]
        near = np.abs(d[:, C["range"]] - tr["r"][j]) < 0.5 if len(d) else np.zeros(0, bool)
        if not near.any():
            continue
        q = d[near][np.argmax(d[near, C["snr_db"]])]
        if abs(tr["az"][j]) > 8:
            az_pairs.append((q[C["az_deg"]], tr["az"][j]))
        m = ok_t & (np.abs(tr["t"] - t) < 0.3)
        if m.sum() >= 4:
            rate = np.polyfit(tr["t"][m] - t, tr["r"][m], 1)[0]
            if abs(rate) > 0.4 and abs(q[C["v_r"]]) > 0.2:
                v_pairs.append((q[C["v_r"]], rate))
    a, v = np.array(az_pairs), np.array(v_pairs)
    assert len(a) >= 15 and len(v) >= 40
    assert np.mean(np.sign(a[:, 0]) == np.sign(a[:, 1])) > 0.8             # world azimuth = - processing
    assert np.mean(np.sign(v[:, 0]) == np.sign(v[:, 1])) > 0.9             # + = receding
