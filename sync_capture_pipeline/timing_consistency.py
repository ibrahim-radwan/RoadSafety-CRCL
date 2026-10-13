#!/usr/bin/env python3
"""
timing_consistency.py - does a session's radar timing still agree with the LiDAR
after the Stage 3 offsets are applied?

Stage 3 measures the offsets once, on a dedicated plate routine. This tool
re-measures the radar-vs-LiDAR residual on ordinary recordings: any stretch
where a single person moves toward or away from the sensors. The session is
cut into windows (default 30 s); in each window with enough radial motion
the radar range track is fitted to the LiDAR range track exactly as Stage 3
does (measure_sync_offset.best_shift / bootstrap_sigma, LiDAR on the nearest
surface). The radar times used are align_session's t_ref, which already
include the Stage 3 radar offset, so a consistent session gives residuals
scattered around 0.

    python timing_consistency.py --session /mnt/sdcard/sessions/freewalk_ref_02
    python timing_consistency.py --session <s> --window 30 --json out.json

Needs align_session.py to have been run (sync/radar_timeline.csv).
The LiDAR decode is cached in <session>/sync/timing_lidar_track.npz.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

import align_session as al          # sets OPENBLAS_NUM_THREADS=1
import measure_sync_offset as m
import radar_firmware

MIN_RADIAL_SPEED = 0.4      # m/s RMS of the LiDAR range rate inside a window
MAX_GATE_M = 1.5            # radar samples further than this from the LiDAR range are another target
MIN_SAMPLES = 60            # radar frames with the target, per window (7.5 s at 8 fps)


def lidar_series(session: Path, rmin: float, rmax: float, range_pct: float, rate: float):
    cache = session / "sync" / "timing_lidar_track.npz"
    if cache.exists():
        d = np.load(cache)
        if float(d["range_pct"]) == range_pct and float(d["rmin"]) == rmin and float(d["rmax"]) == rmax:
            return d["t"], d["r"], d["az"]
    tl, lr, la, _ = m.lidar_track(session, al.rotation(0, 0, 0), rmin, rmax, 30.0, 15.0, rate, 20,
                                  range_pct=range_pct)
    np.savez(cache, t=tl, r=lr, az=la, range_pct=range_pct, rmin=rmin, rmax=rmax)
    return tl, lr, la


def analyse(session: Path, window: float, rmin: float | None, rmax: float, range_pct: float, rate: float):
    rows = list(csv.DictReader((session / "sync" / "radar_timeline.csv").open()))
    if not rows:
        raise SystemExit(f"{session}: no sync/radar_timeline.csv - run align_session.py first")
    tr = np.array([float(r["t_ref"]) for r in rows])
    files = [r["filename"] for r in rows]
    variant, _ = radar_firmware.session_variant(session)
    if rmin is None:
        rmin = float(radar_firmware.firmware_map()[variant]["hpf_floor_m"])
    rr = m.radar_range_res(radar_firmware.cfg_for_variant(variant))
    print(f"  radar: {len(files)} frames, firmware {variant}, window {rmin:.2f}-{rmax:.1f} m ...", flush=True)
    _, r_rng, _, _ = m.radar_track(session, tr, files, rmin, rmax, rr, 6.0)
    print("  LiDAR: target track (nearest surface) ...", flush=True)
    tl, lr, _ = lidar_series(session, rmin, rmax, range_pct, rate)

    ok = np.isfinite(lr)
    t0 = tr[0]
    out = []
    for w0 in np.arange(0.0, tr[-1] - t0, window):
        rs = (tr - t0 >= w0) & (tr - t0 < w0 + window)
        ls = ok & (tl - t0 >= w0 - 1.0) & (tl - t0 < w0 + window + 1.0)
        row = {"start_s": round(float(w0), 1), "end_s": round(float(w0 + window), 1)}
        if ls.sum() < 20:
            out.append(dict(row, status="no LiDAR target"))
            continue
        # radial motion strength from the LiDAR track
        v = np.diff(lr[ls]) / np.diff(tl[ls])
        speed = float(np.sqrt(np.mean(np.clip(v, -3, 3) ** 2)))
        # gate radar samples to the LiDAR target (drops other movers / clutter locks)
        li = np.interp(tr[rs], tl[ls], lr[ls])
        rv = r_rng[rs].copy()
        rv[np.abs(rv - li) > MAX_GATE_M] = np.nan
        n = int(np.isfinite(rv).sum())
        row.update(radial_speed_mps=round(speed, 2), radar_samples=n,
                   lidar_range_m=[round(float(np.nanmin(lr[ls])), 1), round(float(np.nanmax(lr[ls])), 1)])
        if speed < MIN_RADIAL_SPEED or n < MIN_SAMPLES:
            out.append(dict(row, status="too little radial motion" if speed < MIN_RADIAL_SPEED
                            else "too few radar samples"))
            continue
        fit = m.best_shift(tr[rs], rv, tl[ls], lr[ls], scale=False, min_samples=MIN_SAMPLES // 2)
        if fit is None or fit["at_search_edge"]:
            out.append(dict(row, status="no fit"))
            continue
        sig = m.bootstrap_sigma(tr[rs], rv, tl[ls], lr[ls], False, MIN_SAMPLES // 2, fit["tau_s"])
        out.append(dict(row, status="ok", residual_ms=round(fit["tau_s"] * 1e3, 1),
                        sigma_ms=round(sig * 1e3, 1), range_bias_m=round(fit["beta"], 3),
                        rms_m=round(fit["rms"], 3)))
    return out


def summarise(rows):
    good = [r for r in rows if r.get("status") == "ok" and np.isfinite(r["sigma_ms"]) and r["sigma_ms"] < 15]
    if not good:
        return {"windows_used": 0}
    res = np.array([r["residual_ms"] for r in good])
    sig = np.array([r["sigma_ms"] for r in good])
    w = 1 / np.maximum(sig, 1.0) ** 2
    mean = float((w * res).sum() / w.sum())
    chi2 = float(np.sum(((res - mean) / np.maximum(sig, 1.0)) ** 2) / max(1, len(res) - 1))
    return {"windows_used": len(good), "weighted_mean_ms": round(mean, 1),
            "sigma_of_mean_ms": round(float(1 / np.sqrt(w.sum())), 1),
            "median_ms": round(float(np.median(res)), 1), "spread_ms": round(float(res.max() - res.min()), 1),
            "reduced_chi2": round(chi2, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--session", type=Path, nargs="+", required=True)
    ap.add_argument("--window", type=float, default=30.0, help="window length (s)")
    ap.add_argument("--range-min", type=float, default=None, help="default: the radar firmware's near floor")
    ap.add_argument("--range-max", type=float, default=25.0)
    ap.add_argument("--lidar-range-pct", type=float, default=10.0)
    ap.add_argument("--decode-rate", type=float, default=3.0)
    ap.add_argument("--json", type=Path, help="write all window results here")
    a = ap.parse_args()

    report = {}
    for s in a.session:
        print(f"\n=== {s}")
        rows = analyse(s, a.window, a.range_min, a.range_max, a.lidar_range_pct, a.decode_rate)
        summ = summarise(rows)
        report[str(s)] = {"windows": rows, "summary": summ}
        print(f"  {'window':>13} {'status':<26} {'residual':>9} {'sigma':>6} {'speed':>6} {'range m':>12} {'n':>4}")
        for r in rows:
            res = f"{r['residual_ms']:+.1f}" if "residual_ms" in r else ""
            sg = f"{r['sigma_ms']:.1f}" if "sigma_ms" in r else ""
            sp = f"{r['radial_speed_mps']:.2f}" if "radial_speed_mps" in r else ""
            rg = f"{r['lidar_range_m'][0]}-{r['lidar_range_m'][1]}" if "lidar_range_m" in r else ""
            print(f"  {r['start_s']:6.0f}-{r['end_s']:<6.0f} {r['status']:<26} {res:>9} {sg:>6} {sp:>6} {rg:>12} "
                  f"{r.get('radar_samples', ''):>4}")
        print(f"  SUMMARY: {summ}")
    if a.json:
        a.json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
