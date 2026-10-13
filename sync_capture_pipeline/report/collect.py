"""Collect everything the session report shows, reading each sensor's data once.

Order matters: LiDAR first (its target track drives the radar Doppler-at-target
and the snapshot moments), then radar, then camera. All times are seconds on
CLOCK_MONOTONIC after align_session (t_ref); the report plots them relative to
the first radar frame.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "replay"))
import align_session as al                 # noqa: E402  (sets OPENBLAS threads)
import measure_sync_offset as m            # noqa: E402
import radar_firmware                      # noqa: E402

N_SNAPSHOTS = 6
SNAP_WINDOW_S = 4.0            # search +/- this around each nominal snapshot time
LIDAR_STATS_EVERY = 5          # accumulate LiDAR distributions on every 5th sweep
RADAR_STATIC_EVERY = 25        # static (non-moving-only) range profile on every 25th frame
RADAR_HIST_EVERY = 10          # raw amplitude histogram on every 10th frame
CAMERA_STATS_HZ = 5.0          # camera brightness statistics rate
RT_MAX_RANGE_M = 40.0          # range-time map extent
BEV_X = np.arange(0.0, 25.0001, 0.125)
BEV_Y = np.arange(-12.0, 12.0001, 0.125)


def _csv(path: Path) -> list:
    return list(csv.DictReader(path.open())) if path.is_file() else []


def log(msg: str):
    print(f"  {msg}", flush=True)


# ── session-level metadata, logs ────────────────────────────────────────────
def collect_meta(S: Path) -> dict:
    meta = json.loads((S / "session.json").read_text())
    align = json.loads((S / "sync" / "alignment.json").read_text())
    variant, how = radar_firmware.session_variant(S)
    fw = dict(radar_firmware.firmware_map().get(variant, {}))
    check = (S / "check_session.txt").read_text() if (S / "check_session.txt").is_file() else ""
    verdict = re.search(r"^\s+(PASS|FAIL)\b.*$", check, re.M)
    warn = re.findall(r"^\s+- (.+)$", check.split("VERDICT")[-1], re.M) if "VERDICT" in check else []
    return {"session": S.name, "path": str(S), "meta": meta, "align": align, "fw_variant": variant,
            "fw_source": how, "fw": fw, "check_line": verdict.group(0).strip() if verdict else None,
            "check_items": warn}


def parse_log(S: Path) -> dict:
    """Per-second radar packet counts and 5-s board samples from session.log."""
    radar_pps, board = [], []
    rx = re.compile(r"\[radar\]\s+t\+\s*(\d+)s\s+([\d,]+) pkt/s")
    bx = re.compile(r"\[system\] tj ([\d.]+) C \| cpu ([\d.]+) C \| gpu ([\d.]+) C \| board ([\d.]+) C \| fan (\d+)%")
    for line in (S / "session.log").read_text(errors="replace").splitlines():
        a = rx.search(line)
        if a:
            radar_pps.append((int(a.group(1)), int(a.group(2).replace(",", ""))))
            continue
        b = bx.search(line)
        if b:
            board.append(tuple(float(v) for v in b.groups()))
    return {"radar_pps": np.array(radar_pps, dtype=float).reshape(-1, 2),
            "board": np.array(board, dtype=float).reshape(-1, 5)}


# ── LiDAR ───────────────────────────────────────────────────────────────────
def collect_lidar(S: Path, t0: float, duration: float, rmin: float, rmax: float, rate: float = 3.0) -> dict:
    sys.path.insert(0, str(al.WRAPPER_DIR))
    import yaml
    from lidar import Lidar

    to_mono, fit = al.lidar_clock_fit(S)
    # inverse clock map (linear), to select snapshot sweeps by lidar time
    a_ = (to_mono(1000.0) - to_mono(0.0)) / 1000.0
    b_ = float(to_mono(0.0))
    to_lidar = lambda tm: (np.asarray(tm) - b_) / a_          # noqa: E731
    nominal = t0 + duration * (np.arange(N_SNAPSHOTS) + 0.5) / N_SNAPSHOTS
    snap_lo, snap_hi = to_lidar(nominal - SNAP_WINDOW_S - 0.5), to_lidar(nominal + SNAP_WINDOW_S + 0.5)

    tan_az, sin_el = np.tan(np.radians(30.0)), np.sin(np.radians(15.0))
    r0, r1 = max(0.0, rmin - 1.0), rmax + 1.0
    lock = threading.Lock()
    sweeps_target, per_sweep, snaps = [], [], {}
    acc = {"bev": np.zeros((400, 400)), "range": np.zeros(400), "intensity": np.zeros(256),
           "az_range": np.zeros((180, 120)), "n_acc": 0}
    counter = [0]

    def on_cloud(points, ts, seq):
        p = np.asarray(points)
        if ts <= 0 or p.ndim != 2 or p.shape[1] < 6:
            return
        x, y, z, r, inten = p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4]
        mtz = (r >= r0) & (r <= r1) & (x > 0) & (np.abs(y) <= tan_az * x) & (np.abs(z) <= sin_el * r)
        keys = m._voxel_keys(np.floor(x[mtz] / m.VOXEL_M).astype(np.int64),
                             np.floor(y[mtz] / m.VOXEL_M).astype(np.int64),
                             np.floor(z[mtz] / m.VOXEL_M).astype(np.int64))
        item = (ts, keys, x[mtz].astype(np.float32), y[mtz].astype(np.float32), r[mtz].astype(np.float32),
                ts + p[mtz, 5].astype(np.float64))
        # The RS-Fairy returns a fixed grid: directions with no return come back
        # as zero-range points. Only valid returns count in the statistics.
        valid = np.isfinite(r) & (r > 0.05)
        nv = int(valid.sum())
        near = float(np.mean(r[valid] < 1.0)) if nv else np.nan
        with lock:
            counter[0] += 1
            k = counter[0]
            sweeps_target.append(item)
            per_sweep.append((ts, nv, near, len(p)))
            if k % LIDAR_STATS_EVERY == 0:
                xv, yv, rv = x[valid], y[valid], r[valid]
                acc["n_acc"] += 1
                acc["bev"] += np.histogram2d(xv, yv, bins=400, range=[[-40, 40], [-40, 40]])[0]
                acc["range"] += np.histogram(rv, bins=400, range=(0, 100))[0]
                acc["intensity"] += np.histogram(inten[valid], bins=256, range=(0, 256))[0]
                acc["az_range"] += np.histogram2d(np.degrees(np.arctan2(yv, xv)), rv, bins=[180, 120],
                                                  range=[[-180, 180], [0, 60]])[0]
            for j in range(N_SNAPSHOTS):
                if snap_lo[j] <= ts <= snap_hi[j]:
                    f = valid & (x > -2) & (x < 26) & (np.abs(y) < 13) & (z > -2.5) & (z < 3)
                    snaps[ts] = np.c_[x[f], y[f], z[f]][::2].astype(np.float16)     # every 2nd point

    cfg = yaml.safe_load((S / "lidar" / "config_used.yaml").read_text())
    cfg["lidar"]["split_angle"] = 180.0
    dcfg = S / "sync" / "report_lidar_decode.yaml"
    dcfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
    lid = Lidar(str(dcfg))
    lid.on_pointcloud(on_cloud)
    with contextlib.redirect_stdout(io.StringIO()):
        lid.start(pcap_path=S / "lidar" / "lidar.pcap", pcap_rate=rate)
    while not lid.input_finished():
        time.sleep(0.2)
    time.sleep(1.0)
    with contextlib.redirect_stdout(io.StringIO()):
        lid.stop()
    sweeps_target.sort(key=lambda w: w[0])
    per_sweep.sort()
    tl, lr, la, tinfo = m.lidar_target_from_sweeps(sweeps_target, to_mono, fit, rmin, rmax, 20, 0.3, 10.0)
    np.savez(S / "sync" / "timing_lidar_track.npz", t=tl, r=lr, az=la, range_pct=10.0, rmin=rmin, rmax=rmax)
    ps = np.array(per_sweep, dtype=float)
    stats = _csv(S / "lidar" / "stats.csv")
    return {
        "sweep_t": to_mono(ps[:, 0]), "sweep_points": ps[:, 1], "sweep_near_frac": ps[:, 2],
        "sweep_grid_points": ps[:, 3],
        "acc": acc, "track_t": tl, "track_r": lr, "track_az": la, "track_info": tinfo, "clock_fit": fit,
        "snap_sweeps": {float(to_mono(k)): v for k, v in snaps.items()},
        "stats_t": np.array([int(r["t_monotonic_ns"]) / 1e9 for r in stats]),
        "stats_msop": np.array([int(r["msop"]) for r in stats]),
        "stats_lost": np.array([int(r["msop_lost"]) for r in stats]),
        "timeline": _csv(S / "sync" / "lidar_timeline.csv"),
    }


def choose_snapshots(t0: float, duration: float, lid: dict, radar_t: np.ndarray) -> list:
    """One moment per sixth of the session: the radar frame closest to the
    nominal time at which the LiDAR has the target (falls back to nominal)."""
    tl, lr = lid["track_t"], lid["track_r"]
    out = []
    for j in range(N_SNAPSHOTS):
        nom = t0 + duration * (j + 0.5) / N_SNAPSHOTS
        ok = np.isfinite(lr) & (np.abs(tl - nom) <= SNAP_WINDOW_S)
        t_pick = tl[ok][np.argmin(np.abs(tl[ok] - nom))] if ok.any() else nom
        out.append(int(np.argmin(np.abs(radar_t - t_pick))))
    return out


# ── radar ───────────────────────────────────────────────────────────────────
def collect_radar(S: Path, lid: dict, snap_idx: list) -> dict:
    import torch  # noqa: F401  (GPU processor)
    with contextlib.redirect_stdout(io.StringIO()):
        import sources
        proc = sources.RadarProcessor(BEV_X, BEV_Y, radar_firmware.cfg_for_session(S))
    rows = _csv(S / "sync" / "radar_timeline.csv")
    t = np.array([float(r["t_ref"]) for r in rows])
    t_arr = np.array([float(r["t_burst_arrival"]) for r in rows])
    files = [r["filename"] for r in rows]
    n = len(files)
    rng_axis = np.asarray(proc.range_axis)
    v_axis = np.asarray(proc.dop_axis)
    kmax = int(np.searchsorted(rng_axis, RT_MAX_RANGE_M))
    rt = np.full((n, kmax), np.nan, dtype=np.float32)
    ch_rms = np.full((n, 48), np.nan, dtype=np.float32)
    iq = np.full((n, 2), np.nan)                 # I/Q power ratio, I-Q correlation
    peak = np.zeros(n, dtype=np.int32)
    clipped = np.zeros(n, dtype=np.int64)
    dop_target = np.full((n, len(v_axis)), np.nan, dtype=np.float32)
    vel = np.full(n, np.nan)
    amp_hist = np.zeros(32769, dtype=np.int64)
    static_acc, n_static = np.zeros(len(rng_axis)), 0
    const, ref = None, None
    snaps = {}
    tl, lr = lid["track_t"], lid["track_r"]
    okl = np.isfinite(lr)
    tli, lri = tl[okl], lr[okl]
    for i, fn in enumerate(files):
        raw = np.fromfile(S / "radar" / fn, dtype="<i2")
        if raw.size != 1572864:
            continue
        if ref is None:
            ref, const = raw.copy(), np.ones(raw.size, bool)
        else:
            const &= raw == ref
        a = np.abs(raw.astype(np.int32))
        peak[i] = int(a.max())
        clipped[i] = int(np.count_nonzero(a >= 32767))
        if i % RADAR_HIST_EVERY == 0:
            amp_hist += np.bincount(a, minlength=32769)[:32769]
        x = raw.reshape(64, 6, 2, 4, 256, 2).astype(np.float32)
        I, Q = x[..., 0], x[..., 1]
        I = I - I.mean(axis=-1, keepdims=True)
        Q = Q - Q.mean(axis=-1, keepdims=True)
        pw = (I * I + Q * Q).mean(axis=(0, 4))           # [tx, dev, rx]
        ch_rms[i] = np.sqrt(pw).reshape(48)
        pI, pQ = float((I * I).mean()), float((Q * Q).mean())
        iq[i] = (pI / pQ if pQ > 0 else np.nan, float((I * Q).mean()) / np.sqrt(pI * pQ) if pI * pQ > 0 else np.nan)
        rd, bev = proc.process(raw)                       # moving-only; rd [doppler, range] dB
        lin = 10 ** (rd.astype(np.float64) / 10)
        rt[i] = 10 * np.log10(lin[:, :kmax].sum(axis=0) + 1e-12)
        if len(tli) > 1 and tli[0] <= t[i] <= tli[-1]:
            j = np.searchsorted(tli, t[i])
            if 0 < j < len(tli) and tli[j] - tli[j - 1] <= 0.3:
                r0 = np.interp(t[i], tli, lri)
                rb = (rng_axis > r0 - 0.4) & (rng_axis < r0 + 0.8)
                if rb.any():
                    prof = lin[:, rb].sum(axis=1)
                    dop_target[i] = 10 * np.log10(prof + 1e-12)
                    pr = np.clip(prof - np.median(prof), 0, None)
                    pr[np.abs(v_axis) < 0.15] = 0
                    if pr.sum() > 0:
                        vel[i] = float((pr * v_axis).sum() / pr.sum())
        if i % RADAR_STATIC_EVERY == 0:
            rds, _ = proc.process(raw, moving_only=False)
            static_acc += (10 ** (rds.astype(np.float64) / 10)).sum(axis=0)
            n_static += 1
        if i in snap_idx:
            snaps[i] = {"rd": rd.copy(), "bev": bev.copy()}
        if i % 500 == 0:
            log(f"radar {i}/{n}")
    # dominant moving target range per frame (for the radar-vs-LiDAR overlay)
    rlin = 10 ** (rt.astype(np.float64) / 10)
    floor_bins = np.nanpercentile(rlin, 20, axis=0)
    fg = np.clip(rlin - floor_bins, 0, None)
    k0 = int(np.searchsorted(rng_axis, radar_firmware.firmware_map().get(
        radar_firmware.session_variant(S)[0], {}).get("hpf_floor_m", 1.0)))
    k1 = int(np.searchsorted(rng_axis, 25.0))
    dom = np.full(n, np.nan)
    for i in range(n):
        seg = fg[i, k0:k1]
        if np.isfinite(seg).all() and seg.max() > 0:
            j = int(np.argmax(seg))
            snr = 10 * np.log10((rlin[i, k0 + j] + 1e-12) / (floor_bins[k0 + j] + 1e-12))
            if snr >= 6 and 0 < j < len(seg) - 1:
                dom[i] = rng_axis[k0 + j]
    return {"t": t, "t_arrival": t_arr, "files": files, "range_axis": rng_axis[:kmax], "v_axis": v_axis,
            "rt": rt, "ch_rms": ch_rms, "iq": iq, "peak": peak, "clipped": clipped, "amp_hist": amp_hist,
            "static_profile": static_acc / max(1, n_static), "static_range_axis": rng_axis,
            "stale_fraction": float(const.mean()) if const is not None else np.nan,
            "dop_target": dop_target, "vel": vel, "dominant_range": dom, "snaps": snaps,
            "bev_x": BEV_X, "bev_y": BEV_Y, "mirror_azimuth": bool(proc.mirror_azimuth),
            "az_axis": np.asarray(proc.az_axis)}


# ── camera ──────────────────────────────────────────────────────────────────
def collect_camera(S: Path, t0: float, snap_cam_idx: list) -> dict:
    import pyzed.sl as sl
    frames = _csv(S / "camera" / "capture" / "frames.csv")
    tl = {int(r["frame_index"]): float(r["t_ref"]) for r in _csv(S / "sync" / "camera_timeline.csv")}
    idx = np.array([int(r["frame_index"]) for r in frames])
    t_rx = np.array([int(r["host_receipt_mono_ns"]) / 1e9 for r in frames])
    missing = np.array([int(r["estimated_missing_frames_before"] or 0) for r in frames])
    t_ref = np.array([tl.get(int(i), np.nan) for i in idx])
    n_total = len(frames)
    stride = max(1, int(round(30.0 / CAMERA_STATS_HZ)))
    n_thumb = 10
    thumb_pos = set(int(v) for v in np.linspace(0, n_total - 1, n_thumb + 2)[1:-1])
    want_snap = set(int(v) for v in snap_cam_idx if v >= 0)

    init = sl.InitParameters()
    init.set_from_svo_file(str(S / "camera" / "camera.svo2"))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.optional_settings_path = str(S / "camera" / "calibration")
    cam = sl.Camera()
    if cam.open(init) != sl.ERROR_CODE.SUCCESS:
        raise SystemExit("cannot open SVO2")
    rt_ = sl.RuntimeParameters()
    mat = sl.Mat()
    small, big = sl.Resolution(320, 180), sl.Resolution(640, 360)
    bright, thumbs, snaps = [], {}, {}
    pos = 0
    while True:
        code = cam.grab(rt_)
        if code == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break
        if int(code.value) > 0:
            continue
        pos = cam.get_svo_position()
        if pos % stride == 0:
            cam.retrieve_image(mat, sl.VIEW.LEFT, sl.MEM.CPU, small)
            g = mat.get_data()[..., :3].astype(np.float32).mean(axis=2)
            bright.append((pos, float(g.mean()), float(np.percentile(g, 5)), float(np.percentile(g, 95)),
                           float((g >= 250).mean()), float((g <= 5).mean())))
        if pos in thumb_pos or pos in want_snap:
            cam.retrieve_image(mat, sl.VIEW.LEFT, sl.MEM.CPU, big)
            im = mat.get_data()[..., [2, 1, 0]].copy()           # BGRA -> RGB
            if pos in thumb_pos:
                thumbs[pos] = im[::2, ::2]
            if pos in want_snap:
                snaps[pos] = im
    cam.close()
    b = np.array(bright, dtype=float).reshape(-1, 6)
    bt = np.array([tl.get(int(p), np.nan) for p in b[:, 0]])
    imu = _csv(S / "camera" / "capture" / "imu" / "imu_raw.csv")
    imu_t = np.array([int(r["host_receipt_mono_ns"]) / 1e9 for r in imu])
    acc = np.array([[float(r["linear_acc_m_s2_x"]), float(r["linear_acc_m_s2_y"]), float(r["linear_acc_m_s2_z"])]
                    for r in imu])
    gyr = np.array([[float(r["angular_vel_rad_s_x"]), float(r["angular_vel_rad_s_y"]), float(r["angular_vel_rad_s_z"])]
                    for r in imu])
    calib = json.loads((S / "camera" / "calibration" / "camera.json").read_text())
    return {"frame_index": idx, "t_rx": t_rx, "t_ref": t_ref, "missing": missing,
            "bright_t": bt, "bright": b[:, 1:], "thumbs": {tl.get(k, np.nan): v for k, v in sorted(thumbs.items())},
            "snaps": snaps, "imu_t": imu_t, "imu_acc": acc, "imu_gyr": gyr, "calib": calib,
            "status": json.loads((S / "camera" / "capture_status.json").read_text())
            if (S / "camera" / "capture_status.json").is_file() else {}}


# ── timing check (velocity) ─────────────────────────────────────────────────
def velocity_timing(radar: dict, lid: dict, window: float = 30.0) -> list:
    from scipy.signal import savgol_filter
    tl, lr = lid["track_t"], lid["track_r"]
    ok = np.isfinite(lr)
    t, r = tl[ok], lr[ok]
    rate = np.full(len(t), np.nan)
    breaks = np.r_[0, np.nonzero(np.diff(t) > 0.3)[0] + 1, len(t)]
    for a_, b_ in zip(breaks[:-1], breaks[1:]):
        if b_ - a_ >= 7:
            rate[a_:b_] = np.gradient(savgol_filter(r[a_:b_], 7, 2), t[a_:b_])
    tr, rv = radar["t"], radar["vel"]
    t0, out = tr[0], []
    for w0 in np.arange(0, tr[-1] - t0, window):
        rs = (tr - t0 >= w0) & (tr - t0 < w0 + window) & np.isfinite(rv)
        ls = (t - t0 >= w0 - 1) & (t - t0 < w0 + window + 1) & np.isfinite(rate)
        row = {"start": float(w0), "end": float(w0 + window)}
        speed = float(np.sqrt(np.nanmean(rate[ls] ** 2))) if ls.sum() else 0.0
        if rs.sum() < 60 or ls.sum() < 60 or speed < 0.4:
            out.append(dict(row, status="not enough radial motion"))
            continue
        fit = m.best_shift(tr[rs], rv[rs], t[ls], rate[ls], scale=True, min_samples=40)
        if fit is None or fit["at_search_edge"]:
            out.append(dict(row, status="no fit"))
            continue
        sig = m.bootstrap_sigma(tr[rs], rv[rs], t[ls], rate[ls], True, 40, fit["tau_s"])
        out.append(dict(row, status="ok", residual_ms=fit["tau_s"] * 1e3, sigma_ms=sig * 1e3,
                        scale=fit["alpha"], rms=fit["rms"], speed=speed))
    return out


def collect(S: Path) -> dict:
    t_start = time.monotonic()
    log("metadata and logs ...")
    D = collect_meta(S)
    D["log"] = parse_log(S)
    rrows = _csv(S / "sync" / "radar_timeline.csv")
    radar_t = np.array([float(r["t_ref"]) for r in rrows])
    t0, duration = float(radar_t[0]), float(radar_t[-1] - radar_t[0])
    D["t0"], D["duration"] = t0, duration
    rmin = float(D["fw"].get("hpf_floor_m", 2.62))
    log("LiDAR: decoding the pcap once (about 1/3 of the recording time) ...")
    D["lidar"] = collect_lidar(S, t0, duration, rmin, 25.0)
    D["snap_radar_idx"] = choose_snapshots(t0, duration, D["lidar"], radar_t)
    pairs = _csv(S / "sync" / "pairs.csv")
    D["pairs"] = pairs
    snap_cam = [int(pairs[i]["camera_frame_index"]) if pairs[i]["camera_frame_index"] not in ("", None) else -1
                for i in D["snap_radar_idx"]]
    log(f"radar: {len(radar_t)} frames (GPU) ...")
    D["radar"] = collect_radar(S, D["lidar"], D["snap_radar_idx"])
    log("camera: replaying the SVO2 ...")
    D["camera"] = collect_camera(S, t0, snap_cam)
    D["snap_cam_idx"] = snap_cam
    log("timing check (radar Doppler vs LiDAR range rate) ...")
    D["vel_timing"] = velocity_timing(D["radar"], D["lidar"])
    D["collect_s"] = time.monotonic() - t_start
    return D
