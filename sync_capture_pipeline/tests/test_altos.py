"""The Altos V4 radar in the pipeline, on synthetic packets (no recording or sensor needed):

  * altos_v4.FrameAssembler: multi-packet frames, a lost frame, a radar restart (frame ids from 0);
  * align_session.altos_capture_times: the fitted capture times remove arrival jitter and split
    at a restart;
  * a session with a synthetic TI radar and an Altos: check_session passes, align_session writes the
    Altos timeline, points and pairs.csv columns, with pairing errors within half an Altos frame.
"""
from __future__ import annotations

import csv
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

PIPELINE = Path(__file__).resolve().parents[1]
ALTOS = PIPELINE.parent / "altos_standalone_capture"
sys.path[:0] = [str(PIPELINE), str(ALTOS)]
import altos_v4 as av  # noqa: E402
import align_session as al  # noqa: E402
from record_altos_raw import PcapWriter  # noqa: E402

PERIOD = 1 / 15


def packets(frame_id, n_points, t_sec=1741188000):
    """The UDP payloads of one frame of n_points points (72 per packet)."""
    pts = b"".join(struct.pack("<5f", 2.0 + 0.01 * k, 0.0, 0.1, -0.05, 100.0) for k in range(n_points))
    out = []
    for off in range(0, max(len(pts), 1), 72 * 20):
        chunk = pts[off:off + 72 * 20]
        out.append(struct.pack("<IIQIBBHII", av.MAGIC, 0, t_sec, frame_id, 0, 0, 0, len(pts), off) + chunk)
    return out


def test_assembler_multipacket_loss_restart():
    asm = av.FrameAssembler()
    done = []
    t = 0
    for fid in [100, 101, 103, 104, 0, 1, 2]:          # 102 lost; restart after 104
        for p in packets(fid, 150):                     # 3 packets per frame
            t += 1
            done += asm.feed(p, t)
    done += asm.flush()
    assert [f.frame_id for f in done] == [100, 101, 103, 104, 0, 1, 2]
    assert all(f.complete and f.n_points == 150 for f in done)
    assert asm.missing_ids == 1 and asm.restarts == 1 and asm.incomplete == 0


def test_capture_times_remove_jitter_and_split_at_restart():
    rng = np.random.default_rng(1)
    n = 900
    fid = np.r_[np.arange(5000, 5000 + n // 2), np.arange(0, n - n // 2)]     # a restart half-way
    true = 100.0 + np.arange(n) * PERIOD * (1 + 300e-6)                       # rate 300 ppm off nominal
    true[n // 2:] += 2.0                                                      # the restart's pause
    arrival = true + 0.012 + rng.exponential(0.002, n)                        # latency + jitter (only adds)
    t_fit, stretch, info = al.altos_capture_times(fid, arrival)
    assert info["stretches"] == 2 and stretch[0] == 0 and stretch[-1] == 1
    err = (t_fit - true) - 0.012                                              # constant latency stays in
    assert np.abs(err).max() < 0.002                                          # jitter removed to < 2 ms
    assert np.std(arrival - true) > 1.5 * np.std(t_fit - true)


def make_session(root: Path) -> Path:
    S = root / "s"
    (S / "altos").mkdir(parents=True)
    (S / "radar").mkdir()
    w = PcapWriter(S / "altos" / "altos.pcap", "224.1.2.4")
    t0 = 1000.0
    n_pk = 0
    for k in range(300):                                                      # 20 s of Altos frames
        for j, p in enumerate(packets(k, 100)):
            w.write(int((t0 + k * PERIOD + 0.0002 * j) * 1e9), "192.168.3.10", 39272, 4040, p)
            n_pk += 1
    w.close()
    (S / "altos" / "metadata.json").write_text(json.dumps({"complete": True, "packets": n_pk,
                                                           "radar_settings": {"install": [{"profile": "150m_1"}]}}))
    with open(S / "radar" / "frames.csv", "w", newline="") as f:              # TI radar: 8 fps inside the Altos span
        c = csv.writer(f)
        c.writerow(["idx", "filename", "t_burst_monotonic", "t_first_monotonic", "t_last_monotonic", "lost_bytes",
                    "burst_offset_bytes"])
        for k in range(120):
            tb = t0 + 2.0 + k * 0.125
            c.writerow([k, f"frame_{k:06d}.bin", f"{tb:.6f}", f"{tb:.6f}", f"{tb + 0.09:.6f}", 0, 0])
    (S / "session.json").write_text(json.dumps({"sensors": ["altos", "radar"], "exit_codes": {"altos": 0, "radar": 0},
                                                "stop_reason": "radar finished", "started_local": "2026-11-17T10:00:00+1100"}))
    return S


def test_session_check_and_align(tmp_path):
    S = make_session(tmp_path)
    r = subprocess.run([sys.executable, str(PIPELINE / "check_session.py"), "--session", str(S)],
                       capture_output=True, text=True)
    assert "[5 Altos V4 radar]" in r.stdout and "lost 0, incomplete 0" in r.stdout, r.stdout
    assert "radar frames fall outside altos" not in r.stdout
    r = subprocess.run([sys.executable, str(PIPELINE / "align_session.py"), "--session", str(S), "--offsets", "none",
                        "--lidar-extrinsic", "none"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    tl = list(csv.DictReader(open(S / "sync" / "altos_timeline.csv")))
    assert len(tl) == 300 and all(int(x["points"]) == 100 for x in tl)
    d = np.load(S / "sync" / "altos_points.npz")
    assert d["points"].shape == (30000, 8) and d["valid"].all() and d["frame"].max() == 299
    pairs = list(csv.DictReader(open(S / "sync" / "pairs.csv")))
    err = np.array([float(p["altos_error_ms"]) for p in pairs])
    assert len(pairs) == 120 and np.abs(err).max() <= 1000 * PERIOD / 2 + 0.5
    rep = json.loads((S / "sync" / "alignment.json").read_text())
    assert rep["altos"]["frames_lost"] == 0 and not rep["altos_offset_calibrated"]


def test_range_timing_recovers_a_known_lag():
    """report.collect.range_timing on a synthetic walk (towards and away, 1.2 m/s): an Altos-like range
    track 64 ms late, 0.2 m further and noisy, against a LiDAR-like track at 10 Hz."""
    pytest = __import__("pytest")
    pytest.importorskip("matplotlib")
    from report.collect import range_timing
    rng = np.random.default_rng(3)

    def walk(t):                                         # 3 m <-> 11 m, 1.2 m/s, triangle wave
        ph = (t * 1.2) % 16.0
        return 3.0 + np.where(ph < 8.0, ph, 16.0 - ph)
    tl = np.arange(0, 120, 0.1)
    lid = {"track_t": tl, "track_r": walk(tl) + rng.normal(0, 0.02, len(tl))}
    ta = np.arange(0, 120, 1 / 15)
    r = walk(ta - 0.064) + 0.2 + rng.normal(0, 0.05, len(ta))
    out = range_timing(ta, r, lid, n_boot=30)
    assert out["status"] == "ok"
    assert abs(out["tau_ms"] + 64) <= 6 and out["sigma_ms"] < 6
    assert abs(out["range_diff_m"] - 0.2) < 0.03
