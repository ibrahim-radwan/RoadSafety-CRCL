#!/usr/bin/env python3
"""
align_session.py — put every sensor of a capture_session.py session on one
clock (CLOCK_MONOTONIC), then pair camera and LiDAR data to each radar frame.

See SYNC_DESIGN_3_SENSORS.md sections 3-4. In short:

RADAR   t_burst (host arrival of each frame's first packet) is fitted with a
        straight line over the frame index. The radar's crystal makes the
        period very regular, so the fit removes network/scheduling jitter.

CAMERA  The ZED SDK capture time is on the wall clock. Each frame row also
        holds host_receipt_ns / host_receipt_mono_ns taken back-to-back, so
            t_mono = timestamp_ns - (host_receipt_ns - host_receipt_mono_ns)
        converts every frame exactly. A jump in that difference means the
        wall clock was stepped during the session; it is reported.

LIDAR   1. Clock fit: every MSOP packet carries the LiDAR timestamp and was
           stamped with CLOCK_MONOTONIC on arrival (lidar.pcap).
        2. Every point carries its own time. Points are rotated into the
           radar frame (--lidar-rpy-deg) and those inside the radar's field
           of view are kept. A sweep's pairing time is when it scanned the
           radar BORESIGHT; the FOV time span is reported alongside.

FITS    Arrival delay only ever ADDS. Each fit takes its slope by least
        squares and its intercept from the LOWER envelope of the residuals
        (5th percentile), i.e. the least-delayed arrivals. What remains is
        a constant per-sensor latency - exactly what Stage 3 measures and
        --offsets removes. Until Stage 3 has been run, pairs are mechanically
        correct but carry that unknown constant.

OUTPUT  <session>/sync/
    radar_timeline.csv   per radar frame: arrival, fitted and reference time
    camera_timeline.csv  per camera frame: capture time on CLOCK_MONOTONIC
    lidar_timeline.csv   per sweep: boresight time, FOV span, point counts
    pairs.csv            one row per radar frame: matched camera frame and
                         LiDAR sweep, each with its signed time error
    alignment.json       fit parameters, quality figures, offsets used

USAGE (conda env: needs numpy and the RoboSense wrapper)
    python align_session.py --session ./sessions/run01
    python align_session.py --session ./sessions/run01 --offsets stage3_offsets.json
"""
from __future__ import annotations

import argparse
import os
import csv
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

# One BLAS thread: the per-sweep work is small, and a 12-thread pool only
# spins against the decoder thread.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_OFFSETS = HERE / "calibration" / "stage3_offsets.json"      # Stage 3 result in use
ROOT = HERE.parent
LIDAR_DIR = ROOT / "lidar_standalone_capture"
WRAPPER_DIR = LIDAR_DIR / "rs_driver" / "python_wrapper_2"

MSOP_ID = b"\x55\xaa\x05\x5a"
MSOP_TS_OFFSET = 20
ENVELOPE_PCT = 5.0
# A pair whose nearest frame is further than GAP_FACTOR x half a period fell
# inside a gap in that sensor (dropped frames); 20% margin covers jitter.
GAP_FACTOR = 1.2

# Radar field of view for r075m (cascade_r075m.cfg aoaFovCfg): azimuth
# +/-85 deg, elevation +/-5 deg. Only the DIRECTION matters for timing - the
# time a direction is scanned does not depend on range - so no range gate.
FOV_AZ_DEG = 85.0
FOV_EL_DEG = 5.0


# ── helpers ─────────────────────────────────────────────────────────────────
def load_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, header: list, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def stats_ms(x) -> dict:
    x = np.asarray(x, dtype=float) * 1e3
    x = x[np.isfinite(x)]
    if not x.size:
        return {}
    return {"median": round(float(np.median(x)), 3), "p01": round(float(np.percentile(x, 1)), 3),
            "p99": round(float(np.percentile(x, 99)), 3), "min": round(float(x.min()), 3),
            "max": round(float(x.max()), 3)}


def envelope_fit(x, y, reject_ms=20.0):
    """y ~ a*x + b with b on the lower envelope. Large outliers (e.g. a
    stalled reader) are dropped before the final slope fit."""
    x0 = x[0]
    xs = x - x0
    a, b = np.polyfit(xs, y, 1)
    for _ in range(3):
        r = y - (a * xs + b)
        keep = np.abs(r - np.median(r)) < reject_ms / 1e3
        if keep.sum() < max(10, len(x) // 2):
            return None                 # not a line: e.g. a DCA backlog (radar) - caller decides
        a, b = np.polyfit(xs[keep], y[keep], 1)
    r = y - (a * xs + b)
    b += np.percentile(r[keep], ENVELOPE_PCT)
    resid = y - (a * xs + b)
    return a, b, x0, resid, keep


def rotation(roll, pitch, yaw):
    r, p, y = np.radians([roll, pitch, yaw])
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def nearest(sorted_t, t):
    """Index of the value in sorted_t nearest to t (NaNs must be removed)."""
    i = np.searchsorted(sorted_t, t)
    if i == 0:
        return 0
    if i == len(sorted_t):
        return len(sorted_t) - 1
    return i if sorted_t[i] - t < t - sorted_t[i - 1] else i - 1


# ── radar ───────────────────────────────────────────────────────────────────
def radar_timeline(session: Path, offset_s: float):
    rows = load_csv(session / "radar" / "frames.csv")
    if "lost_bytes" not in rows[0]:
        raise SystemExit("radar/frames.csv predates the 3-sensor receiver (no lost_bytes): "
                         "its t_burst is one frame late - re-record with the current pipeline")
    k = np.array([int(r["idx"]) for r in rows], dtype=float)
    t = np.array([float(r["t_burst_monotonic"]) for r in rows])
    fit = envelope_fit(k, t)
    source = "arrival times (envelope fit)"
    if fit is None:
        # Arrival times are not linear in the frame index: the DCA1000 delivered
        # frames late and unevenly (backlog, see check_session). The radar itself
        # free-runs on its crystal, so frame k was measured at t0 + k * period;
        # the first frame arrives before any backlog builds up.
        import radar_firmware
        period = float(radar_firmware.firmware_map()[radar_firmware.session_variant(session)[0]]
                       ["frame_periodicity_ms"]) / 1e3
        k0, a, b = k[0], period, t[0]
        resid = t - (a * (k - k0) + b)
        keep = np.zeros(len(k), bool)
        source = "NOMINAL frame period from the first frame (arrival times unusable: DCA1000 backlog)"
        print(f"  WARNING radar: arrival times are not a line - using {source}")
    else:
        a, b, k0, resid, keep = fit
    t_fit = a * (k - k0) + b
    out = [(int(kk), r["filename"], f"{tt:.6f}", f"{tf:.6f}", f"{tf + offset_s:.6f}",
            f"{rr * 1e3:.3f}", int(not ok), r.get("lost_bytes", ""))
           for kk, r, tt, tf, rr, ok in zip(k, rows, t, t_fit, resid, keep)]
    write_csv(session / "sync" / "radar_timeline.csv",
              ["idx", "filename", "t_burst_arrival", "t_fit", "t_ref", "residual_ms", "outlier",
               "lost_bytes"], out)
    report = {"frames": len(rows), "timing_source": source, "period_ms": round(a * 1e3, 6), "fps": round(1 / a, 4),
              "residual_ms": stats_ms(resid), "outliers": int((~keep).sum()),
              "frames_with_loss": int(sum(1 for r in rows if int(r.get("lost_bytes") or 0) > 0))}
    return t_fit + offset_s, [r["filename"] for r in rows], [int(r.get("lost_bytes") or 0) for r in rows], report


# ── camera ──────────────────────────────────────────────────────────────────
def camera_timeline(session: Path, offset_s: float):
    rows = load_csv(session / "camera" / "capture" / "frames.csv")
    if "host_receipt_mono_ns" not in rows[0]:
        raise SystemExit("camera/capture/frames.csv has no host_receipt_mono_ns - recorded "
                         "with a ZED recorder older than the 3-sensor changes")
    idx = np.array([int(r["frame_index"]) for r in rows])
    sdk = np.array([int(r["timestamp_ns"]) for r in rows], dtype=np.int64)
    host = np.array([int(r["host_receipt_ns"]) for r in rows], dtype=np.int64)
    mono = np.array([int(r["host_receipt_mono_ns"]) for r in rows], dtype=np.int64)
    wall_minus_mono = host - mono                       # ns, per frame
    t = (sdk - wall_minus_mono) / 1e9                   # capture time on CLOCK_MONOTONIC
    latency = (mono / 1e9) - t                          # capture -> host receipt
    steps = np.where(np.abs(np.diff(wall_minus_mono)) > 1_000_000)[0]   # > 1 ms
    out = [(int(i), f"{tt:.6f}", f"{tt + offset_s:.6f}", f"{l * 1e3:.3f}")
           for i, tt, l in zip(idx, t, latency)]
    write_csv(session / "sync" / "camera_timeline.csv",
              ["frame_index", "t_capture", "t_ref", "sdk_to_receipt_ms"], out)
    gaps = load_csv(session / "camera" / "capture" / "timing" / "camera_gaps.csv") \
        if (session / "camera" / "capture" / "timing" / "camera_gaps.csv").is_file() else []
    report = {"frames": len(rows),
              "fps": round((len(t) - 1) / (t[-1] - t[0]), 4) if len(t) > 1 else None,
              "sdk_to_receipt_latency_ms": stats_ms(latency),
              "wall_clock_steps": int(len(steps)),
              "wall_minus_mono_drift_ms": round(float(wall_minus_mono[-1] - wall_minus_mono[0]) / 1e6, 3),
              # A gap event can span several frames (e.g. one 167 ms gap = 4 frames).
              "gap_events": len(gaps),
              "dropped_frames_estimated": sum(int(g.get("estimated_missing_frames") or 0) for g in gaps)}
    return t + offset_s, idx, report


# ── LiDAR ───────────────────────────────────────────────────────────────────
def lidar_clock_fit(session: Path):
    spec = importlib.util.spec_from_file_location("check_lidar_session", LIDAR_DIR / "check_lidar_session.py")
    cls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cls)
    meta = json.loads((session / "lidar" / "metadata.json").read_text())
    msop = meta["ports"]["msop"]
    host, lid = [], []
    for t_ns, dport, payload in cls.read_pcap(session / "lidar" / "lidar.pcap"):
        if dport == msop and payload[:4] == MSOP_ID:
            host.append(t_ns)
            lid.append(int.from_bytes(payload[MSOP_TS_OFFSET:MSOP_TS_OFFSET + 6], "big")
                       + int.from_bytes(payload[MSOP_TS_OFFSET + 6:MSOP_TS_OFFSET + 10], "big") * 1e-6)
    host = np.array(host, dtype=np.int64) / 1e9
    lid = np.array(lid)
    fit = envelope_fit(lid, host)
    if fit is None:
        raise SystemExit("LiDAR packet clock does not map linearly to the host clock - check lidar/clock_map.csv "
                         "and the pcap (clock reset or a stalled recorder?)")
    a, b, l0, resid, keep = fit
    report = {"packets": int(len(lid)), "span_lidar_s": round(float(lid[-1] - lid[0]), 3),
              "drift_ppm": round((a - 1) * 1e6, 3),
              "receipt_delay_above_envelope_ms": stats_ms(resid), "outliers": int((~keep).sum())}
    return (lambda tl: a * (np.asarray(tl) - l0) + b), report


def lidar_sweeps(session: Path, rot, boresight_deg: float, rate: float, split_angle: float):
    sys.path.insert(0, str(WRAPPER_DIR))
    from lidar import Lidar

    sweeps = []
    lock = threading.Lock()
    cos_bore = np.cos(np.radians(boresight_deg))
    tan_bore = np.tan(np.radians(boresight_deg))
    sin_el = np.sin(np.radians(FOV_EL_DEG))
    tan_az = np.tan(np.radians(FOV_AZ_DEG))     # FOV_AZ_DEG < 90

    def on_cloud(points, ts, seq):
        # Runs on the decoder's thread and must keep up with the decode rate,
        # so direction tests use plain arithmetic, not trigonometry per point.
        p = np.asarray(points)
        if ts <= 0 or p.shape[1] < 6:
            return                          # first partial sweep: no timestamp
        x = rot[0, 0] * p[:, 0] + rot[0, 1] * p[:, 1] + rot[0, 2] * p[:, 2]   # LiDAR -> radar frame
        y = rot[1, 0] * p[:, 0] + rot[1, 1] * p[:, 1] + rot[1, 2] * p[:, 2]
        z = rot[2, 0] * p[:, 0] + rot[2, 1] * p[:, 1] + rot[2, 2] * p[:, 2]
        r = p[:, 3]                         # range does not change under rotation
        ok = r > 0
        fov = ok & (np.abs(z) <= sin_el * r) & (x > 0) & (np.abs(y) <= tan_az * x)
        bore = ok & (x >= cos_bore * r)     # within boresight_deg of radar +x
        if bore.sum() < 20:                 # nothing ahead returned: use the
            bore = fov & (np.abs(y) <= tan_bore * x)    # azimuth window only
        dt = p[:, 5].astype(np.float64)
        row = (seq, ts, len(p), int(fov.sum()), int(bore.sum()),
               ts + float(np.median(dt[bore])) if bore.any() else np.nan,
               ts + float(dt[fov].min()) if fov.any() else np.nan,
               ts + float(dt[fov].max()) if fov.any() else np.nan)
        with lock:
            sweeps.append(row)

    # Decode with the sweep boundary BEHIND the vehicle: rs_driver's default
    # (0 deg) splits the forward field of view across two sweeps 100 ms
    # apart. The PCAP holds raw packets, so this applies to old sessions too.
    cfg = yaml.safe_load((session / "lidar" / "config_used.yaml").read_text())
    cfg["lidar"]["split_angle"] = float(split_angle)
    decode_cfg = session / "sync" / "lidar_decode_config.yaml"
    decode_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
    lidar = Lidar(str(decode_cfg))
    lidar.on_pointcloud(on_cloud)
    lidar.start(pcap_path=session / "lidar" / "lidar.pcap", pcap_rate=rate)
    while not lidar.input_finished():
        time.sleep(0.2)
    time.sleep(1.0)
    lidar.stop()
    sweeps.sort(key=lambda r: r[1])
    return sweeps


def lidar_timeline(session: Path, offset_s: float, rot, boresight_deg: float, rate: float,
                   split_angle: float):
    to_mono, fit = lidar_clock_fit(session)
    sweeps = lidar_sweeps(session, rot, boresight_deg, rate, split_angle)
    bore = to_mono([s[5] for s in sweeps])
    f0 = to_mono([s[6] for s in sweeps])
    f1 = to_mono([s[7] for s in sweeps])
    out = [(i, s[0], f"{s[1]:.6f}", s[2], s[3], s[4], f"{b:.6f}", f"{b + offset_s:.6f}",
            f"{a:.6f}", f"{c:.6f}", f"{(c - a) * 1e3:.2f}")
           for i, (s, b, a, c) in enumerate(zip(sweeps, bore, f0, f1))]
    write_csv(session / "sync" / "lidar_timeline.csv",
              ["sweep", "seq", "cloud_ts_lidar", "points", "fov_points", "boresight_points",
               "t_boresight", "t_ref", "t_fov_first", "t_fov_last", "fov_span_ms"], out)
    span = (f1 - f0)
    # Completeness: the decoder drops data silently if it is driven faster
    # than it can decode, so compare against what the packets span.
    pts = np.array([s[2] for s in sweeps])
    seqs = [s[0] for s in sweeps]
    expected = fit["span_lidar_s"] * 10.0
    report = dict(fit, sweeps=len(sweeps),
                  sweeps_expected_10hz=round(expected, 1),
                  seq_gaps=sum(1 for a, b in zip(seqs, seqs[1:]) if b != a + 1),
                  points_per_sweep={"min": int(pts.min()), "median": int(np.median(pts)),
                                    "max": int(pts.max())} if len(pts) else {},
                  sweeps_partial=int((pts < 0.9 * np.median(pts)).sum()) if len(pts) else 0,
                  sweeps_without_boresight=int(np.isnan(bore).sum()),
                  fov_span_ms=stats_ms(span),
                  sweep_period_ms=stats_ms(np.diff(bore[np.isfinite(bore)])))
    return bore + offset_s, span, report


# ── pairing ─────────────────────────────────────────────────────────────────
def pair(session, t_radar, radar_files, radar_lost, cam_t, cam_idx, lid_t, lid_span):
    rows, cam_err, lid_err = [], [], []
    cam_ok = np.isfinite(cam_t)
    ct, ci = cam_t[cam_ok], cam_idx[cam_ok]
    order = np.argsort(ct)
    ct, ci = ct[order], ci[order]
    lid_ok = np.where(np.isfinite(lid_t))[0] if lid_t is not None else np.array([], int)
    lt = lid_t[lid_ok] if lid_t is not None else np.array([])
    cam_period = np.median(np.diff(ct)) if len(ct) > 1 else np.nan
    lid_period = np.median(np.diff(lt)) if len(lt) > 1 else np.nan
    for k, tr in enumerate(t_radar):
        flags = []
        c_idx = c_t = c_e = ""
        if len(ct):
            j = nearest(ct, tr)
            c_idx, c_t, e = int(ci[j]), ct[j], ct[j] - tr
            if not (ct[0] - cam_period / 2 <= tr <= ct[-1] + cam_period / 2):
                flags.append("outside_camera")
            else:
                cam_err.append(e)
                if abs(e) > GAP_FACTOR * cam_period / 2:
                    flags.append("camera_gap")      # radar frame fell inside dropped camera frames
            c_e = f"{e * 1e3:.2f}"
            c_t = f"{c_t:.6f}"
        l_idx = l_t = l_e = l_span = ""
        if len(lt):
            j = nearest(lt, tr)
            e = lt[j] - tr
            l_idx, l_t, l_span = int(lid_ok[j]), f"{lt[j]:.6f}", f"{lid_span[lid_ok[j]] * 1e3:.2f}"
            if not (lt[0] - lid_period / 2 <= tr <= lt[-1] + lid_period / 2):
                flags.append("outside_lidar")
            else:
                lid_err.append(e)
                if abs(e) > GAP_FACTOR * lid_period / 2:
                    flags.append("lidar_gap")       # a missing sweep around this radar frame
            l_e = f"{e * 1e3:.2f}"
        if radar_lost[k]:
            flags.append("radar_lost_bytes")
        rows.append((k, radar_files[k], f"{tr:.6f}", c_idx, c_t, c_e, l_idx, l_t, l_e, l_span,
                     radar_lost[k], ";".join(flags)))
    write_csv(session / "sync" / "pairs.csv",
              ["radar_idx", "radar_file", "t_radar", "camera_frame_index", "t_camera",
               "camera_error_ms", "lidar_sweep", "t_lidar_boresight", "lidar_error_ms",
               "lidar_fov_span_ms", "radar_lost_bytes", "flags"], rows)
    return {"pairs": len(rows),
            "camera_error_ms": stats_ms(cam_err), "camera_period_ms": round(float(cam_period) * 1e3, 3),
            "lidar_error_ms": stats_ms(lid_err),
            "lidar_period_ms": round(float(lid_period) * 1e3, 3) if np.isfinite(lid_period) else None,
            "flagged": sum(1 for r in rows if r[-1]),
            "camera_gap_pairs": sum(1 for r in rows if "camera_gap" in r[-1]),
            "lidar_gap_pairs": sum(1 for r in rows if "lidar_gap" in r[-1])}


# ── main ────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Align a 3-sensor session and pair frames to radar frames")
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--offsets", default=None,
                    help="Stage 3 result: JSON {radar_ms, camera_ms, lidar_ms} added to each sensor's time. "
                         f"Default: {DEFAULT_OFFSETS} if it exists; 'none' = uncalibrated")
    ap.add_argument("--lidar-rpy-deg", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                    metavar=("ROLL", "PITCH", "YAW"),
                    help="rotation LiDAR -> radar frame (radar: x forward, y left, z up). "
                         "Default: mounted facing the same way as the radar.")
    ap.add_argument("--boresight-deg", type=float, default=5.0,
                    help="half-angle around radar +x whose scan time is the sweep's pairing time")
    ap.add_argument("--split-angle", type=float, default=180.0,
                    help="LiDAR sweep boundary for decoding (deg, LiDAR frame); 180 = behind")
    ap.add_argument("--decode-rate", type=float, default=3.0,
                    help="LiDAR PCAP decode speed, multiple of real time. rs_driver drops data "
                         "above ~4x on this Jetson; the report counts decoded vs expected sweeps.")
    a = ap.parse_args()

    s = a.session
    meta = json.loads((s / "session.json").read_text())
    sensors = meta.get("sensors", [])
    if "radar" not in sensors:
        raise SystemExit("session has no radar: radar frames are the pairing anchor")
    offsets = {"radar_ms": 0.0, "camera_ms": 0.0, "lidar_ms": 0.0}
    off_path = None
    if a.offsets is None:
        off_path = DEFAULT_OFFSETS if DEFAULT_OFFSETS.is_file() else None
    elif str(a.offsets).lower() != "none":
        off_path = Path(a.offsets)
    calibrated = off_path is not None
    if calibrated:
        cal = json.loads(off_path.read_text())
        offsets.update({k: float(cal[k]) for k in offsets if k in cal})
        want = (cal.get("valid_for") or {}).get("radar_firmware")
        have = (meta.get("radar_firmware") or {}).get("variant")
        if want and have != want:
            print(f"  WARNING: offsets in {off_path} were measured with radar firmware {want}, "
                  f"this session used {have or 'unknown'} - re-run Stage 3 for it")
    (s / "sync").mkdir(exist_ok=True)

    print(f"Session: {s}")
    print(f"  sensors: {', '.join(sensors)}")
    print(f"  offsets: {offsets} {'(Stage 3)' if calibrated else '(NOT CALIBRATED - Stage 3 not applied)'}")

    report = {"session": str(s), "offsets": offsets, "offsets_calibrated": calibrated,
              "offsets_file": str(off_path) if off_path else None,
              "lidar_split_angle_deg": a.split_angle,
              "lidar_rpy_deg": list(a.lidar_rpy_deg), "boresight_deg": a.boresight_deg,
              "envelope_percentile": ENVELOPE_PCT}

    t_radar, files, lost, report["radar"] = radar_timeline(s, offsets["radar_ms"] / 1e3)
    print(f"\nRadar : {report['radar']['frames']} frames, period {report['radar']['period_ms']:.3f} ms "
          f"({report['radar']['fps']:.3f} fps), fit residual p99 "
          f"{report['radar']['residual_ms'].get('p99')} ms, outliers {report['radar']['outliers']}")

    cam_t, cam_idx = np.array([]), np.array([], int)
    if "camera" in sensors:
        cam_t, cam_idx, report["camera"] = camera_timeline(s, offsets["camera_ms"] / 1e3)
        c = report["camera"]
        print(f"Camera: {c['frames']} frames, {c['fps']:.3f} fps, SDK->receipt "
              f"{c['sdk_to_receipt_latency_ms'].get('median')} ms (median), "
              f"wall-clock steps {c['wall_clock_steps']}, dropped {c['dropped_frames_estimated']} frames "
              f"in {c['gap_events']} gaps")

    lid_t, lid_span = None, None
    if "lidar" in sensors:
        rot = rotation(*a.lidar_rpy_deg)
        lid_t, lid_span, report["lidar"] = lidar_timeline(s, offsets["lidar_ms"] / 1e3, rot,
                                                          a.boresight_deg, a.decode_rate,
                                                          a.split_angle)
        l = report["lidar"]
        print(f"LiDAR : {l['sweeps']} sweeps (expected ~{l['sweeps_expected_10hz']:.0f}, seq gaps "
              f"{l['seq_gaps']}, partial {l['sweeps_partial']}), clock drift {l['drift_ppm']} ppm, receipt delay p99 "
              f"{l['receipt_delay_above_envelope_ms'].get('p99')} ms, FOV span "
              f"{l['fov_span_ms'].get('median')} ms, sweeps without boresight points "
              f"{l['sweeps_without_boresight']}")

    report["pairing"] = pair(s, t_radar, files, lost, cam_t, cam_idx, lid_t, lid_span)
    p = report["pairing"]
    print(f"\nPairs : {p['pairs']} radar frames, flagged {p['flagged']} "
          f"(in a camera gap {p['camera_gap_pairs']}, in a LiDAR gap {p['lidar_gap_pairs']})")
    if p["camera_error_ms"]:
        print(f"  camera error ms: {p['camera_error_ms']}  (bound: half a frame = "
              f"{p['camera_period_ms'] / 2:.1f} ms)")
    if p["lidar_error_ms"]:
        print(f"  LiDAR  error ms: {p['lidar_error_ms']}  (bound: half a sweep = "
              f"{(p['lidar_period_ms'] or 0) / 2:.1f} ms)")

    (s / "sync" / "alignment.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {s / 'sync'}/: radar_timeline.csv camera_timeline.csv lidar_timeline.csv "
          f"pairs.csv alignment.json")
    if not calibrated:
        print("Pairs carry each sensor's unmeasured constant latency until Stage 3 offsets are applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
