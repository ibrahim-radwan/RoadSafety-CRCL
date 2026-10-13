#!/usr/bin/env python3
"""
measure_sync_offset.py — Stage 3: measure the fixed radar / camera / LiDAR
time offsets from a calibration recording. (Two-sensor version: legacy/.)

WHY THIS IS NEEDED
    align_session.py puts every sensor on CLOCK_MONOTONIC, but each sensor
    still carries a constant latency that timestamps cannot reveal: radar
    acquisition -> LVDS -> DCA -> Ethernet, camera exposure -> USB -> SDK.
    Nearest-neighbour pairing hides such a constant (it just picks another
    frame). Only an event every sensor observes can measure it.

THE RECORDING (one person, ~60 s, capture_session.py as usual)
    A flat metal plate held square to the radar, at radar height, 11.5-14 m
    away (the radar's high-pass filter suppresses returns inside ~10.5 m):
      1. walk between 11.5 m and 14 m with sharp turnarounds   -> range
      2. side-steps of ~1 m at ~12.5 m with sharp stops         -> azimuth
      3. flip the plate square <-> edge-on at irregular times   -> cross-check
    Nothing else moving. Record 2-3 times; the results must agree.

HOW IT MEASURES (LiDAR is the reference: per-point times, tight clock fit)
    radar  - range FFT of every raw frame; static background removed; the
             target's range tracked inside the window.
    LiDAR  - the target's points in the same window: median range, azimuth
             and per-point time, per sweep.
    camera - SVO2 replayed; moving pixels against a median background; their
             horizontal centroid converted to an angle with the intrinsics.
    Then, for shifts tau over +/-1 s, the radar range at t+tau is compared
    with the LiDAR range (allowing a constant range bias), and the camera
    angle with the LiDAR azimuth (allowing scale + bias). The best tau is the
    offset. Sharp turnarounds and stops are what make tau unambiguous: at
    constant speed a time shift and a range bias look the same.

    Sign: offsets are ADDED to each sensor's time by align_session.py
    (--offsets), so radar_ms = tau where radar(t) matches LiDAR(t + tau).

SELF-CHECKS (an offset is only reported if they pass)
    - radar and LiDAR agree on the target's range (bias < 1 m, |alpha-1| small)
    - enough valid samples in each track; a clear cost minimum
    - each third of the recording gives the same offset (spread reported)

USAGE (conda env: numpy, pyzed, opencv, matplotlib)
    python measure_sync_offset.py --session ./sessions/stage3_01
    python measure_sync_offset.py --session ./sessions/stage3_0{1,2,3} --out stage3_offsets.json
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import align_session as al          # same folder; also sets OPENBLAS_NUM_THREADS=1
import radar_firmware
import numpy as np

HERE = Path(__file__).resolve().parent
C = 299_792_458.0

# r075m / r075m_v2 raw frame: chirps(384) x devices(2) x rx/dev(4) x samples(256) x (I,Q) int16
N_CHIRPS, N_RX, N_SAMPLES = 384, 8, 256
SHIFT_RANGE_S, SHIFT_STEP_S = 1.0, 0.002
MAX_INTERP_GAP_S = 0.3


# ── radar ───────────────────────────────────────────────────────────────────
def radar_range_res(cfg_path: Path) -> float:
    for line in cfg_path.read_text(errors="replace").splitlines():
        tok = line.split("%")[0].split()
        if tok and tok[0] == "profileCfg":
            slope_hz_s = float(tok[8]) * 1e12          # MHz/us -> Hz/s
            n, fs = int(tok[10]), float(tok[11]) * 1e3   # samples, ksps -> sps
            return C / (2 * slope_hz_s * n / fs)
    raise SystemExit(f"profileCfg not found in {cfg_path}")


N_LOOPS = 64                         # chirp = loop * 6 + tx


def radar_profiles(session: Path, files: list) -> np.ndarray:
    """MOVING-only range power profile per frame, both spectrum halves.

    Static clutter is removed inside each frame by subtracting, per TX/RX
    channel and sample, the mean over the 64 chirp loops (zero-Doppler
    removal). This keeps the walker even when they stay in one range span for
    most of the session, which a time-median background would absorb. It
    also cancels frame-constant stale words exactly (pre-v2 firmware)."""
    win = np.hanning(N_SAMPLES).astype(np.float32)
    out = np.empty((len(files), N_SAMPLES), dtype=np.float64)
    for i, fn in enumerate(files):
        raw = np.fromfile(session / "radar" / fn, dtype=np.int16).reshape(
            N_LOOPS, N_CHIRPS // N_LOOPS, N_RX, N_SAMPLES, 2)
        iq = raw[..., 0].astype(np.float32) + 1j * raw[..., 1].astype(np.float32)
        iq -= iq.mean(axis=0, keepdims=True)            # static clutter, per channel and sample
        spec = np.fft.fft(iq * win, axis=-1)
        out[i] = (np.abs(spec) ** 2).mean(axis=(0, 1, 2))
    return out


def radar_track(session: Path, t_radar, files, rmin, rmax, range_res, min_snr_db):
    prof = radar_profiles(session, files)            # already moving-only
    bg = np.percentile(prof, 20, axis=0)             # per-bin floor: noise + residual clutter
    fg = np.clip(prof - bg, 0, None)
    k0, k1 = int(np.floor(rmin / range_res)), int(np.ceil(rmax / range_res))
    pos = fg[:, k0:k1 + 1].sum()
    neg = fg[:, N_SAMPLES - k1:N_SAMPLES - k0 + 1].sum()
    swapped = neg > pos                              # Q,I order mirrors the spectrum
    if swapped:
        fg, prof, bg = fg[:, ::-1], prof[:, ::-1], bg[::-1]
        fg, prof, bg = np.roll(fg, 1, 1), np.roll(prof, 1, 1), np.roll(bg, 1, 1)
    rng = np.full(len(files), np.nan)
    amp = np.full(len(files), np.nan)
    for i in range(len(files)):
        seg = fg[i, k0:k1 + 1]
        j = int(np.argmax(seg))
        k = k0 + j
        snr = 10 * np.log10((prof[i, k] + 1e-12) / (bg[k] + 1e-12))
        amp[i] = 10 * np.log10(prof[i, k] + 1e-12)
        if snr < min_snr_db or j in (0, len(seg) - 1):
            continue
        y0, y1, y2 = np.log(fg[i, k - 1:k + 2] + 1e-12)          # parabolic peak interpolation
        d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) if (y0 - 2 * y1 + y2) != 0 else 0.0
        rng[i] = (k + np.clip(d, -0.5, 0.5)) * range_res
    info = {"iq_order": "Q,I (swapped)" if swapped else "I,Q",
            "halves_foreground_energy_ratio": round(float(max(pos, neg) / max(min(pos, neg), 1e-12)), 2),
            "frames": len(files), "frames_with_target": int(np.isfinite(rng).sum()),
            "range_res_m": round(range_res, 4)}
    return np.asarray(t_radar), rng, amp, info


# ── LiDAR ───────────────────────────────────────────────────────────────────
VOXEL_M = 0.25            # background map cell (LiDAR target extraction)
_VBITS, _VOFF = 21, 1 << 20


def _voxel_keys(ix, iy, iz):
    return ((ix + _VOFF) << (2 * _VBITS)) | ((iy + _VOFF) << _VBITS) | (iz + _VOFF)


def lidar_track(session: Path, rot, rmin, rmax, az_half, el_half, rate, min_points, bg_frac=0.3, range_pct=50.0):
    """Track the MOVING target only. Outdoors the target window also holds
    static scenery (walls, bins, fences, ground), which a plain median would
    track instead of the person. Pass 1 decodes every sweep and keeps the
    points in the target zone with their 25 cm voxel; pass 2 marks voxels
    occupied in >= bg_frac of the sweeps (dilated by one voxel) as background
    and tracks the remaining points, keeping the cluster within 1 m of their
    centre."""
    to_mono, fit = al.lidar_clock_fit(session)
    sys.path.insert(0, str(al.WRAPPER_DIR))
    import yaml
    from lidar import Lidar

    sweeps = []                    # (ts, keys, x, y, r, t_point) per sweep
    lock = threading.Lock()
    tan_az, sin_el = np.tan(np.radians(az_half)), np.sin(np.radians(el_half))
    r0, r1 = max(0.0, rmin - 1.0), rmax + 1.0          # margin: the target may cross the edges

    def on_cloud(points, ts, seq):
        p = np.asarray(points)
        if ts <= 0 or p.shape[1] < 6:
            return
        x = rot[0, 0] * p[:, 0] + rot[0, 1] * p[:, 1] + rot[0, 2] * p[:, 2]
        y = rot[1, 0] * p[:, 0] + rot[1, 1] * p[:, 1] + rot[1, 2] * p[:, 2]
        z = rot[2, 0] * p[:, 0] + rot[2, 1] * p[:, 1] + rot[2, 2] * p[:, 2]
        r = p[:, 3]
        m = (r >= r0) & (r <= r1) & (x > 0) & (np.abs(y) <= tan_az * x) & (np.abs(z) <= sin_el * r)
        keys = _voxel_keys(np.floor(x[m] / VOXEL_M).astype(np.int64), np.floor(y[m] / VOXEL_M).astype(np.int64),
                           np.floor(z[m] / VOXEL_M).astype(np.int64))
        item = (ts, keys, x[m].astype(np.float32), y[m].astype(np.float32), r[m].astype(np.float32),
                ts + p[m, 5].astype(np.float64))
        with lock:
            sweeps.append(item)

    cfg = yaml.safe_load((session / "lidar" / "config_used.yaml").read_text())
    cfg["lidar"]["split_angle"] = 180.0
    dcfg = session / "sync" / "stage3_lidar_decode.yaml"
    dcfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
    lid = Lidar(str(dcfg))
    lid.on_pointcloud(on_cloud)
    lid.start(pcap_path=session / "lidar" / "lidar.pcap", pcap_rate=rate)
    while not lid.input_finished():
        time.sleep(0.2)
    time.sleep(1.0)
    lid.stop()
    sweeps.sort(key=lambda w: w[0])

    return lidar_target_from_sweeps(sweeps, to_mono, fit, rmin, rmax, min_points, bg_frac, range_pct)


def lidar_target_from_sweeps(sweeps, to_mono, fit, rmin, rmax, min_points, bg_frac=0.3, range_pct=10.0):
    """Pass 2 of lidar_track, on already-decoded target-zone points:
    sweeps = [(ts_lidar, voxel_keys, x, y, r, t_point_lidar), ...] sorted by ts.
    Shared with the session report, which decodes the pcap only once."""
    # Pass 2: static background = voxels occupied in >= bg_frac of sweeps, dilated by 1.
    uniq, counts = np.unique(np.concatenate([np.unique(w[1]) for w in sweeps]), return_counts=True)
    bg = uniq[counts >= bg_frac * len(sweeps)]
    shifts = np.array([(dx << (2 * _VBITS)) + (dy << _VBITS) + dz
                       for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)], dtype=np.int64)
    bg_dil = np.unique((bg[:, None] + shifts[None, :]).ravel())

    rows = []
    for ts, keys, x, y, r, tp in sweeps:
        fg = ~np.isin(keys, bg_dil, assume_unique=False) & (r >= rmin) & (r <= rmax)
        if fg.sum() >= min_points:
            cx, cy = np.median(x[fg]), np.median(y[fg])
            near = fg & ((x - cx) ** 2 + (y - cy) ** 2 <= 1.0)          # the cluster around the target
            if near.sum() >= min_points:
                rows.append((float(np.median(tp[near])), float(np.percentile(r[near], range_pct)),
                             float(np.degrees(np.arctan2(np.median(y[near]), np.median(x[near])))), int(near.sum())))
                continue
        rows.append((ts + 0.05, np.nan, np.nan, int(fg.sum())))
    t = to_mono([w[0] for w in rows])
    rng = np.array([w[1] for w in rows])
    az = np.array([w[2] for w in rows])
    info = {"sweeps": len(rows), "sweeps_with_target": int(np.isfinite(rng).sum()),
            "median_points_on_target": int(np.median([w[3] for w in rows if np.isfinite(w[1])]))
            if np.isfinite(rng).any() else 0,
            "background_voxels": int(len(bg)), "clock_drift_ppm": fit["drift_ppm"]}
    return np.asarray(t), rng, az, info


# ── camera ──────────────────────────────────────────────────────────────────
def camera_track(session: Path, cam_t, cam_idx, width, min_frac, az_band=None, min_blob_px=40):
    """Lateral angle of the walker in the camera.

    Moving pixels (vs a median background) are searched only inside a window
    where the walker can be: the azimuth band the LiDAR saw the target in
    (az_band, deg, already widened by the caller) and -20..+20 deg of
    elevation. The largest connected blob in that window is the target, so
    trees, shadows or glare elsewhere in the image cannot pull the estimate.
    A person at 14 m is only a few hundred pixels at 480 px width, hence an
    absolute blob size (min_blob_px) rather than an image fraction."""
    import pyzed.sl as sl
    from scipy import ndimage
    t_by_idx = dict(zip(cam_idx.tolist(), cam_t.tolist()))
    calib = json.loads((session / "camera" / "calibration" / "camera.json").read_text())
    left = calib["rectified"]["left"]
    scale = width / left["width"]
    fx, cx = left["fx"] * scale, left["cx"] * scale
    fy, cy = left.get("fy", left["fx"]) * scale, left.get("cy", left["height"] / 2) * scale
    height = int(round(left["height"] * scale))
    # Region of interest (image right = radar/LiDAR -y, so column falls as azimuth rises).
    c0, c1 = 0, width
    if az_band is not None:
        c0 = int(np.clip(cx - fx * np.tan(np.radians(az_band[1])), 0, width - 1))
        c1 = int(np.clip(cx - fx * np.tan(np.radians(az_band[0])), c0 + 1, width))
    r0 = int(np.clip(cy - fy * np.tan(np.radians(20.0)), 0, height - 1))
    r1 = int(np.clip(cy + fy * np.tan(np.radians(20.0)), r0 + 1, height))

    init = sl.InitParameters()
    init.set_from_svo_file(str(session / "camera" / "camera.svo2"))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.optional_settings_path = str(session / "camera" / "calibration")
    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"cannot open SVO2: {status}")
    rt = sl.RuntimeParameters()
    mat = sl.Mat()
    res = sl.Resolution(width, height)
    frames, idxs = [], []
    try:
        while True:
            code = cam.grab(rt)
            if code == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if int(code.value) > 0:
                continue
            cam.retrieve_image(mat, sl.VIEW.LEFT, sl.MEM.CPU, res)
            bgra = mat.get_data()
            gray = (0.114 * bgra[..., 0] + 0.587 * bgra[..., 1] + 0.299 * bgra[..., 2]).astype(np.float32)
            frames.append(gray)
            idxs.append(cam.get_svo_position())
    finally:
        mat.free()
        cam.close()
    stack = np.stack(frames)
    bg = np.median(stack[:: max(1, len(stack) // 60)], axis=0)
    thr = max(20.0, 4 * float(np.median(np.abs(stack[len(stack) // 2] - bg))))
    xs = np.arange(width, dtype=np.float32)
    t, ang, energy = [], [], []
    prev = None
    for idx, g in zip(idxs, stack):
        if idx not in t_by_idx:
            prev = g
            continue
        roi = np.abs(g[r0:r1, c0:c1] - bg[r0:r1, c0:c1]) > thr
        a = np.nan
        if roi.sum() >= min_blob_px:
            lab, n = ndimage.label(roi, structure=np.ones((3, 3)))
            sizes = np.bincount(lab.ravel())[1:]
            k = int(np.argmax(sizes)) + 1
            if sizes[k - 1] >= min_blob_px:
                xc = c0 + float(np.nonzero(lab == k)[1].mean())
                a = float(np.degrees(np.arctan2(-(xc - cx), fx)))      # image right = radar -y
        t.append(t_by_idx[idx])
        ang.append(a)
        energy.append(float(np.abs(g - prev).mean()) if prev is not None else np.nan)
        prev = g
    info = {"frames_decoded": len(stack), "frames_with_target": int(np.isfinite(ang).sum()),
            "foreground_threshold": round(thr, 1), "image_width": width,
            "roi_px": [c0, r0, c1, r1], "az_band_deg": None if az_band is None else [round(float(v), 1) for v in az_band]}
    return np.asarray(t), np.asarray(ang), np.asarray(energy), info


# ── offset search ───────────────────────────────────────────────────────────
def interp_with_gaps(t_query, t_ref, v_ref):
    ok = np.isfinite(v_ref)
    tr, vr = t_ref[ok], v_ref[ok]
    out = np.full(len(t_query), np.nan)
    if len(tr) < 2:
        return out
    i = np.clip(np.searchsorted(tr, t_query), 1, len(tr) - 1)
    gap = tr[i] - tr[i - 1]
    inside = (t_query >= tr[0]) & (t_query <= tr[-1]) & (gap <= MAX_INTERP_GAP_S)
    out[inside] = np.interp(t_query[inside], tr, vr)
    return out


def best_shift(t_a, v_a, t_ref, v_ref, scale: bool, min_samples: int, taus=None):
    """tau minimising the residual of v_a(t) ~ alpha*v_ref(t+tau) + beta.
    alpha is fixed to 1 unless scale=True."""
    ok = np.isfinite(v_a)
    t_a, v_a = t_a[ok], v_a[ok]
    if taus is None:
        taus = np.arange(-SHIFT_RANGE_S, SHIFT_RANGE_S + SHIFT_STEP_S / 2, SHIFT_STEP_S)
    cost = np.full(len(taus), np.nan)
    params = [None] * len(taus)
    for i, tau in enumerate(taus):
        ref = interp_with_gaps(t_a + tau, t_ref, v_ref)
        m = np.isfinite(ref)
        if m.sum() < min_samples:
            continue
        if scale:
            A = np.c_[ref[m], np.ones(m.sum())]
            (alpha, beta), *_ = np.linalg.lstsq(A, v_a[m], rcond=None)
        else:
            alpha, beta = 1.0, float(np.median(v_a[m] - ref[m]))
        r = v_a[m] - (alpha * ref[m] + beta)
        # Trimmed RMS: the worst 10% (radar locking onto clutter, a camera
        # frame with someone else moving) must not dominate the cost.
        r2 = np.sort(r ** 2)[: max(1, int(0.9 * len(r)))]
        cost[i] = float(np.sqrt(np.mean(r2)))
        params[i] = (float(alpha), float(beta), int(m.sum()))
    if not np.isfinite(cost).any():
        return None
    i = int(np.nanargmin(cost))
    tau = taus[i]
    if 0 < i < len(taus) - 1 and np.isfinite(cost[i - 1]) and np.isfinite(cost[i + 1]):
        c0, c1, c2 = cost[i - 1:i + 2]
        den = c0 - 2 * c1 + c2
        if den > 0:
            tau += SHIFT_STEP_S * 0.5 * (c0 - c2) / den
    edge = i in (0, len(taus) - 1)
    alpha, beta, n = params[i]
    # Contrast: how much worse the cost is +/-50 ms away (flat = ambiguous).
    j = int(round(0.05 / SHIFT_STEP_S))
    side = [cost[k] for k in (i - j, i + j) if 0 <= k < len(cost) and np.isfinite(cost[k])]
    contrast = (min(side) / cost[i]) if side and cost[i] > 0 else np.nan
    return {"tau_s": float(tau), "rms": cost[i], "alpha": alpha, "beta": beta, "samples": n,
            "at_search_edge": edge, "contrast_50ms": float(contrast), "taus": taus, "cost": cost}


def bootstrap_sigma(t_a, v_a, t_ref, v_ref, scale, min_samples, center, n=30, seed=0):
    """Spread of tau when the samples are resampled with replacement: the
    uncertainty of this session's estimate. Searched near the full-data tau."""
    ok = np.where(np.isfinite(v_a))[0]
    rng = np.random.default_rng(seed)
    taus = np.arange(center - 0.2, center + 0.2 + SHIFT_STEP_S / 2, SHIFT_STEP_S)
    out = []
    for _ in range(n):
        pick = np.sort(rng.choice(ok, size=len(ok), replace=True))
        r = best_shift(t_a[pick], v_a[pick], t_ref, v_ref, scale, min_samples, taus)
        if r and not r["at_search_edge"]:
            out.append(r["tau_s"])
    return float(np.std(out)) if len(out) >= n // 2 else float("inf")


def measure_one(session: Path, a) -> dict:
    print(f"\n=== {session}")
    (session / "sync").mkdir(exist_ok=True)
    variant, how = radar_firmware.session_variant(session)
    cfg = a.radar_cfg or radar_firmware.cfg_for_variant(variant)
    rmin = a.range_min if a.range_min is not None else radar_firmware.firmware_map()[variant]["hpf_floor_m"]
    print(f"  radar firmware {variant} ({how}); cfg {cfg.name}; target window {rmin:.2f}-{a.range_max:.2f} m")
    rr = radar_range_res(cfg)
    t_r, files, _, _ = al.radar_timeline(session, 0.0)
    cam_t, cam_idx, _ = al.camera_timeline(session, 0.0)
    rot = al.rotation(*a.lidar_rpy_deg)

    print("  radar : range FFT of every frame ...")
    tr, r_rng, r_amp, r_info = radar_track(session, t_r, files, rmin, a.range_max, rr, a.min_snr_db)
    print(f"          {r_info}")
    print("  LiDAR : decoding sweeps ...")
    tl, l_rng, l_az, l_info = lidar_track(session, rot, rmin, a.range_max, a.az_half, a.el_half,
                                          a.decode_rate, a.min_lidar_points, range_pct=a.lidar_range_pct)
    print(f"          {l_info}")
    print("  camera: replaying SVO2 ...")
    band = None
    if np.isfinite(l_az).sum() > 20:
        lo, hi = np.nanpercentile(l_az, [2, 98])
        band = (lo - a.camera_az_margin, hi + a.camera_az_margin)
    tc, c_ang, c_energy, c_info = camera_track(session, cam_t, cam_idx, a.camera_width, a.min_camera_frac, band)
    print(f"          {c_info}")

    res = {"session": str(session), "radar": r_info, "lidar": l_info, "camera": c_info, "checks": []}
    radar_fit = best_shift(tr, r_rng, tl, l_rng, scale=False, min_samples=20)
    cam_fit = best_shift(tc, c_ang, tl, l_az, scale=True, min_samples=60)

    if radar_fit:
        bias = radar_fit["beta"]
        sigma = bootstrap_sigma(tr, r_rng, tl, l_rng, False, 20, radar_fit["tau_s"])
        ok = (not radar_fit["at_search_edge"] and abs(bias) < 1.0 and sigma <= a.max_radar_sigma_ms / 1e3)
        res["radar_vs_lidar"] = {k: v for k, v in radar_fit.items() if k not in ("taus", "cost")}
        res["radar_vs_lidar"]["sigma_ms"] = round(sigma * 1e3, 2)
        res["radar_sigma_ms"] = round(sigma * 1e3, 2)
        res["radar_ms"] = round(radar_fit["tau_s"] * 1e3, 2) if ok else None
        if not ok:
            res["checks"].append(f"radar offset rejected: edge={radar_fit['at_search_edge']} "
                                 f"range bias {bias:+.2f} m (limit 1 m), uncertainty {sigma * 1e3:.1f} ms "
                                 f"(limit {a.max_radar_sigma_ms} ms)")
    else:
        res["radar_ms"] = None
        res["checks"].append("radar offset: too few frames with the target seen by both radar and LiDAR")

    if cam_fit:
        sigma_c = bootstrap_sigma(tc, c_ang, tl, l_az, True, 60, cam_fit["tau_s"])
        ok = (not cam_fit["at_search_edge"] and 0.5 < cam_fit["alpha"] < 2.0
              and sigma_c <= a.max_camera_sigma_ms / 1e3)
        res["camera_vs_lidar"] = {k: v for k, v in cam_fit.items() if k not in ("taus", "cost")}
        res["camera_vs_lidar"]["sigma_ms"] = round(sigma_c * 1e3, 2)
        res["camera_sigma_ms"] = round(sigma_c * 1e3, 2)
        res["camera_ms"] = round(cam_fit["tau_s"] * 1e3, 2) if ok else None
        if not ok:
            res["checks"].append(f"camera offset rejected: edge={cam_fit['at_search_edge']} "
                                 f"scale {cam_fit['alpha']:.2f} (0.5-2), uncertainty {sigma_c * 1e3:.1f} ms "
                                 f"(limit {a.max_camera_sigma_ms} ms)")
    else:
        res["camera_ms"] = None
        res["checks"].append("camera offset: too few frames with the target seen by both camera and LiDAR")

    # Flips: radar amplitude steps vs camera motion energy (independent radar-camera check).
    if res.get("radar_ms") is not None and res.get("camera_ms") is not None:
        d_amp = np.abs(np.gradient(r_amp))
        best = (-1.0, None)
        for tau in np.arange(-SHIFT_RANGE_S, SHIFT_RANGE_S, SHIFT_STEP_S * 5):
            e = interp_with_gaps(tr + tau, tc, c_energy)
            m = np.isfinite(e) & np.isfinite(d_amp)
            if m.sum() > 20:
                rho = np.corrcoef(d_amp[m], e[m])[0, 1]
                if rho > best[0]:
                    best = (rho, tau)
        if best[1] is not None:
            implied = (best[1] * 1e3) + res["camera_ms"]
            res["flips_crosscheck"] = {"radar_ms_implied": round(implied, 1), "correlation": round(best[0], 3),
                                       "note": "coarse (5x step, one radar frame); agreement within ~1 radar "
                                               "frame supports the main estimate"}

    plot(session, tr, r_rng, tl, l_rng, l_az, tc, c_ang, radar_fit, cam_fit)
    (session / "sync" / "stage3_result.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    print(f"  -> radar_ms {res['radar_ms']} (+/- {res.get('radar_sigma_ms')})  camera_ms {res['camera_ms']} "
          f"(+/- {res.get('camera_sigma_ms')})  (LiDAR reference = 0)")
    for c in res["checks"]:
        print(f"     ! {c}")
    return res


def plot(session, tr, r_rng, tl, l_rng, l_az, tc, c_ang, radar_fit, cam_fit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t0 = np.nanmin(tl) if len(tl) else 0.0
    fig, ax = plt.subplots(3, 1, figsize=(12, 10))
    ax[0].plot(tl - t0, l_rng, ".", ms=2, label="LiDAR range")
    if radar_fit:
        ax[0].plot(tr + radar_fit["tau_s"] - t0, r_rng - radar_fit["beta"], "o", ms=4, mfc="none",
                   label=f"radar range, shifted {radar_fit['tau_s'] * 1e3:+.1f} ms, bias {radar_fit['beta']:+.2f} m")
    ax[0].set_ylabel("range (m)")
    ax[0].legend()
    ax[1].plot(tl - t0, l_az, ".", ms=2, label="LiDAR azimuth")
    if cam_fit:
        ax[1].plot(tc + cam_fit["tau_s"] - t0, (c_ang - cam_fit["beta"]) / cam_fit["alpha"], ".", ms=2,
                   label=f"camera angle, shifted {cam_fit['tau_s'] * 1e3:+.1f} ms")
    ax[1].set_ylabel("azimuth (deg)")
    ax[1].legend()
    for fit, name in ((radar_fit, "radar"), (cam_fit, "camera")):
        if fit:
            ax[2].plot(fit["taus"] * 1e3, fit["cost"] / np.nanmin(fit["cost"]), label=name)
    ax[2].set_xlabel("shift tau (ms)")
    ax[2].set_ylabel("residual / minimum")
    ax[2].set_ylim(0.9, 4)
    ax[2].legend()
    ax[0].set_title(f"Stage 3 - {session.name}")
    fig.tight_layout()
    fig.savefig(session / "sync" / "stage3.png", dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 3: measure fixed radar/camera/LiDAR offsets")
    ap.add_argument("--session", type=Path, nargs="+", required=True, help="one or more calibration sessions")
    ap.add_argument("--out", type=Path, help="write the combined offsets here (for align_session.py --offsets)")
    ap.add_argument("--radar-cfg", type=Path, default=None,
                    help="override; default is the .cfg of the firmware recorded in session.json")
    ap.add_argument("--range-min", type=float, default=None,
                    help="target window start (m); default is the firmware's near floor (r075m 10.5, r075m_v2 2.62)")
    ap.add_argument("--range-max", type=float, default=15.0, help="target window end (m)")
    ap.add_argument("--az-half", type=float, default=30.0, help="LiDAR target wedge half-width (deg)")
    ap.add_argument("--el-half", type=float, default=15.0, help="LiDAR target wedge half-height (deg)")
    ap.add_argument("--lidar-rpy-deg", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--min-snr-db", type=float, default=6.0, help="radar target above static background")
    ap.add_argument("--min-lidar-points", type=int, default=20)
    ap.add_argument("--min-camera-frac", type=float, default=0.005, help="moving fraction of the image")
    ap.add_argument("--lidar-range-pct", type=float, default=10.0,
                    help="LiDAR target range = this percentile of the target's point ranges "
                         "(50 = median of body+plate; ~10 = nearest surface, the plate, like the radar)")
    ap.add_argument("--camera-width", type=int, default=480, help="decode width (px) for the camera track")
    ap.add_argument("--camera-az-margin", type=float, default=8.0,
                    help="camera search window = LiDAR target azimuth range +/- this (deg)")
    ap.add_argument("--decode-rate", type=float, default=3.0)
    ap.add_argument("--max-radar-sigma-ms", type=float, default=20.0,
                    help="reject a session whose radar offset uncertainty (bootstrap) exceeds this")
    ap.add_argument("--max-camera-sigma-ms", type=float, default=10.0)
    a = ap.parse_args()

    results = [measure_one(s, a) for s in a.session]
    print("\n" + "=" * 72)
    print("  STAGE 3 RESULT (LiDAR = reference, offsets ADDED to each sensor's time)")
    print("=" * 72)
    combined = {"lidar_ms": 0.0}
    ok = True
    for key in ("radar_ms", "camera_ms"):
        skey = key.replace("_ms", "_sigma_ms")
        good = [r for r in results if r.get(key) is not None]
        for r in results:
            print(f"  {key:9s} {Path(r['session']).name:<24} {r.get(key)}  +/- {r.get(skey)}")
        if not good:
            print(f"  {key}: NO VALID ESTIMATE")
            ok = False
            continue
        # Inverse-variance weighted mean: a clean session counts more.
        vals = np.array([r[key] for r in good])
        sig = np.maximum(np.array([r[skey] for r in good]), 0.5)
        w = 1 / sig ** 2
        mean, err = float((w * vals).sum() / w.sum()), float(1 / np.sqrt(w.sum()))
        combined[key] = round(mean, 2)
        combined[skey] = round(err, 2)
        spread = float(np.ptp(vals)) if len(vals) > 1 else None
        print(f"  {key}: {mean:+.2f} +/- {err:.2f} ms from {len(vals)} session(s)"
              + (f", spread {spread:.2f} ms" if spread is not None else " (record 2-3 to check agreement)"))
        # Sessions should agree within their own uncertainties.
        if spread is not None and spread > 3 * float(np.sqrt((sig ** 2).max() + (sig ** 2).min())):
            print(f"  ! {key} sessions disagree ({spread:.1f} ms) beyond their uncertainties - check the plots")
            ok = False
    combined["sessions"] = [r["session"] for r in results]
    combined["measured_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if a.out and ok:
        # Record what these offsets are valid for, so align_session can warn on a mismatch.
        fws = {radar_firmware.session_variant(Path(x))[0] for x in a.session}
        combined["valid_for"] = {
            "radar_firmware": fws.pop() if len(fws) == 1 else sorted(fws),
            "note": "offsets hold while radar firmware/frame period, DCA packet delay, camera mode "
                    "and LiDAR config are unchanged"}
        combined["method"] = f"measure_sync_offset.py, LiDAR reference, --lidar-range-pct {a.lidar_range_pct:g}"
        a.out.write_text(json.dumps(combined, indent=2) + "\n")
        print(f"\n  wrote {a.out}  ->  python align_session.py --session <run> --offsets {a.out}")
    elif a.out:
        print(f"\n  {a.out} NOT written: resolve the issues above first")
    print("  plots: <session>/sync/stage3.png")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
