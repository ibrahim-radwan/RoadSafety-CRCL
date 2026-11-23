"""Session notes (check_session.py --quick) and the full check, on a synthetic session (no sensors needed):

  * --quick reads only the recorders' own reports: it passes with a LiDAR PCAP it would fail to read,
    summarises the camera gaps and says the SVO2 decode check is still to run;
  * the full check fails until verify_svo.py's result says the SVO2 was decoded, then passes;
  * a camera stopped early (full disk) whose SVO2 verified afterwards is a warning, not a failure;
  * a clean "storage limit" stop is a warning; radar frames misaligned by a steady offset (the DCA
    started mid-frame) fail with that explanation; frames outside camera coverage are counted.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE / "tests"))
from test_altos import make_session  # noqa: E402

FRAME_BYTES = 256 * 64 * 48 * 4
T0 = 1000.0                                       # the synthetic session's CLOCK_MONOTONIC start


def add_radar_bins(S: Path) -> None:
    for row in csv.DictReader(open(S / "radar" / "frames.csv")):
        with open(S / "radar" / row["filename"], "wb") as f:      # sparse: right size, no disk
            f.truncate(FRAME_BYTES)


def add_camera(S: Path, gaps=((300, 1), (420, 5)), status=None) -> None:
    """30 fps camera covering the radar, with dropped frames before the given frame indices."""
    c = S / "camera"
    (c / "capture" / "timing").mkdir(parents=True)
    (c / "calibration").mkdir()
    (c / "calibration" / "SN1.conf").write_text("x")
    (c / "camera.svo2").write_bytes(b"\0" * 1000)
    wall = 1_700_000_000_000_000_000 - int(T0 * 1e9)             # realtime - monotonic
    missing = dict(gaps)
    rows, t, prev = [], int((T0 + 0.5) * 1e9), None
    for k in range(600):                                          # 20 s
        t += 33_333_333 * (1 + missing.get(k, 0))
        rows.append((k, t + wall, t + wall + 1000, t + 1000, prev))
        prev = t + wall
    with open(c / "capture" / "frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "timestamp_ns", "host_receipt_ns", "host_receipt_mono_ns"])
        w.writerows(r[:4] for r in rows)
    with open(c / "capture" / "timing" / "camera_gaps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "timestamp_ns", "previous_timestamp_ns", "frame_interval_ns",
                    "estimated_missing_frames"])
        for k, m in gaps:
            w.writerow([k, rows[k][1], rows[k][4], rows[k][1] - rows[k][4], m])
    st = {"complete": True, "error": None, "recorded_frames": 600, "observed_fps": 29.7,
          "estimated_missing_camera_frames": sum(m for _, m in gaps),
          "finalized_svo_frame_count_verified": True, "finalized_svo_timestamps_verified": False,
          "svo_decode_check": "deferred: verify_svo.py"}
    st.update(status or {})
    (c / "capture_status.json").write_text(json.dumps(st))


def add_lidar(S: Path) -> None:
    (S / "lidar").mkdir()
    (S / "lidar" / "lidar.pcap").write_bytes(b"not a pcap")       # --quick must not read it
    (S / "lidar" / "metadata.json").write_text(json.dumps({
        "complete": True, "error": None, "duration_s": 22.0, "packets": {"msop": 70000, "difop": 22},
        "msop_lost": 0, "queue_drops": 0, "kernel_udp_drops": {},
        "first_msop": {"monotonic_ns": int(T0 * 1e9)}, "last_msop": {"monotonic_ns": int((T0 + 22) * 1e9)}}))


def full_session(tmp_path: Path, **camera) -> Path:
    S = make_session(tmp_path)                                    # TI radar (frames.csv) + Altos
    add_radar_bins(S)
    add_camera(S, **camera)
    add_lidar(S)
    meta = json.loads((S / "session.json").read_text())
    meta["sensors"] = ["lidar", "camera", "altos", "radar"]
    meta["exit_codes"] = {n: 0 for n in meta["sensors"]}
    (S / "session.json").write_text(json.dumps(meta))
    # the Altos recorder's own counters and ledger, as record_altos_raw.py writes them
    am = json.loads((S / "altos" / "metadata.json").read_text())
    am.update({"frames": 300, "frames_lost": 0, "frames_incomplete": 0, "radar_restarts": 0})
    (S / "altos" / "metadata.json").write_text(json.dumps(am))
    with open(S / "altos" / "frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "t_first_monotonic_ns", "t_last_monotonic_ns", "radar_time_s", "points",
                    "packets", "complete"])
        for k in range(300):
            w.writerow([k, int((T0 + k / 15) * 1e9), int((T0 + k / 15) * 1e9) + 1000, 0, 100, 7, 1])
    return S


def check(S: Path, *extra) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(PIPELINE / "check_session.py"), "--session", str(S), *extra],
                          capture_output=True, text=True)


def verdict(out: str) -> str:
    return out.split("VERDICT", 1)[1]


def test_quick_reads_only_the_recorders_reports(tmp_path):
    S = full_session(tmp_path)
    r = check(S, "--quick")
    assert r.returncode == 0, r.stdout
    assert "SESSION NOTES" in r.stdout and "  PASS - no problem reported" in r.stdout
    assert "not re-read: full check" in r.stdout                          # LiDAR and Altos
    assert "decode check: not run yet" in r.stdout
    # gaps: one single, one of 5 frames (200 ms) - times from the first radar frame
    assert "gaps        : 2 (1 of one frame, 1 longer); longest 200 ms" in r.stdout, r.stdout
    assert "verify_svo.py --session" in r.stdout                          # how to run the full check


def test_full_check_needs_the_svo2_decoded(tmp_path):
    S = full_session(tmp_path)
    r = check(S)
    assert r.returncode == 1
    assert "lidar.pcap unreadable" in r.stdout                            # the full check reads the PCAP
    assert "SVO2 not decoded and verified yet" in verdict(r.stdout)
    (S / "camera" / "svo_verification.json").write_text(json.dumps({"ok": True, "frames_decoded": 600}))
    r = check(S)
    assert "SVO2 not decoded" not in r.stdout
    assert "decoded and timestamps verified=True (verify_svo.py)" in r.stdout
    (S / "camera" / "svo_verification.json").write_text(json.dumps({"ok": False, "error": "frame 12"}))
    assert "SVO2 verification failed (verify_svo.py): frame 12" in verdict(check(S, "--quick").stdout)


def test_camera_stopped_early_but_verified_is_a_warning(tmp_path):
    S = full_session(tmp_path, status={"complete": False, "error": "Minimum free space reached; recording stopped",
                                       "finalized_svo_frame_count_verified": False})
    r = check(S, "--quick")
    assert "camera recording incomplete" in verdict(r.stdout)
    (S / "camera" / "svo_verification.json").write_text(json.dumps({"ok": True, "frames_decoded": 600}))
    v = verdict(check(S, "--quick").stdout)
    assert "camera recording incomplete" not in v
    assert "camera stopped early (Minimum free space reached" in v and "usable" in v


def test_storage_limit_stop_is_a_warning(tmp_path):
    S = full_session(tmp_path)
    meta = json.loads((S / "session.json").read_text())
    meta["stop_reason"] = "storage limit: camera staging disk /home/jetson/camera_stage below 3.0 GB free"
    (S / "session.json").write_text(json.dumps(meta))
    r = check(S, "--quick")
    assert r.returncode == 0, r.stdout
    assert "session stopped early, cleanly: storage limit" in verdict(r.stdout)


def test_misaligned_radar_and_coverage_counts(tmp_path):
    S = full_session(tmp_path)
    p = S / "radar" / "frames.csv"
    rows = list(csv.DictReader(open(p)))
    for k, row in enumerate(rows):
        row["burst_offset_bytes"] = "0" if k == 0 else str(-327_000 - 16 * (k % 3))   # steady, as in "check"
    tail = [dict(rows[-1], idx=str(len(rows) + j), filename=f"frame_{len(rows) + j:06d}.bin",
                 t_burst_monotonic=f"{T0 + 21 + 0.125 * j:.6f}") for j in range(3)]  # past the camera's end
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows + tail)
    add_radar_bins(S)
    v = verdict(check(S, "--quick").stdout)
    assert "radar frames misaligned with transmission bursts" in v and "DCA started capturing mid-frame" in v
    assert "outside camera coverage - those frames cannot be paired (0 at the start, 3 at the end, of 123)" in v


def test_camera_gap_times_are_from_the_first_radar_frame(tmp_path):
    S = full_session(tmp_path, gaps=((300, 1),))
    out = check(S, "--quick").stdout
    line = next(x for x in out.splitlines() if x.strip().startswith("gaps"))
    # radar starts at T0 + 2 s; frame 300 (one frame missing before it) is at T0 + 0.5 + 302/30 s
    t = float(line.split("at t = ")[1].split(" s")[0])
    assert abs(t - (0.5 + 302 / 30 - 2.0)) < 0.05, line
    assert os.path.getsize(S / "radar" / "frame_000000.bin") == FRAME_BYTES
