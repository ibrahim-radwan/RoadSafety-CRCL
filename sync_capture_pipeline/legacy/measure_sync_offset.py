"""
measure_sync_offset.py — Stage 3: measure the physical radar/camera offset.

WHY THIS IS NEEDED
    Both sensors timestamp when the HOST saw data, not when the world was
    sampled. Radar: acquisition -> LVDS -> DCA buffering -> Ethernet.
    Camera: exposure -> USB -> librealsense. Those latencies differ, and the
    difference is a FIXED bias that shifts every pair identically.

    Nearest-neighbour pairing cannot reveal it. It always selects the
    CLOSEST camera frame, so any constant offset silently changes WHICH
    frame is chosen while the residual stays symmetric about zero -
    simulated offsets of 0/50/100/200 ms all produce a median residual under
    1 ms while shifting the selection by 0/1/2/3 frames.

    The only way to measure it is an event both sensors observe.

HOW IT WORKS
    A sharp motion (hand clap in frame, metal plate dropped through the
    beam) produces a spike in BOTH streams:

      camera : mean |pixel difference| between consecutive frames
      radar  : mean |sample difference| between consecutive raw frames.
               Static clutter cancels between frames, so this is effectively
               a crude frame-to-frame MTI and needs no DSP at all.

    Find the peak in each, compare timestamps:

        offset = t_radar_peak - t_camera_peak

    Positive means the radar timestamp is LATER than the camera's for the
    same physical event, so radar timestamps should be reduced by that much
    before pairing:

        python check_session.py --session <s> --offset-ms <offset>

REQUIREMENTS
    numpy, opencv-python. Run in the camera conda env.

USAGE
    python measure_sync_offset.py --session ./sessions/sync01
    python measure_sync_offset.py --session ./sessions/sync01 --full-frames
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


FRAME_BYTES = 256 * 64 * 48 * 4


def load_csv(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def radar_motion(rdir: Path, rows: list[dict], prefix_bytes: int) -> tuple:
    """Frame-to-frame absolute difference of the raw ADC stream.

    Reading only a PREFIX of each frame keeps this fast (119 x 3.1 MB is
    374 MB otherwise). A gross motion event is visible in any representative
    slice of the chirps; --full-frames reads everything if the prefix proves
    too noisy.
    """
    ts, sig, prev = [], [], None
    for r in rows:
        fp = rdir / r["filename"]
        if not fp.exists():
            continue
        n = prefix_bytes if prefix_bytes else -1
        raw = np.fromfile(fp, dtype=np.int16, count=(n // 2) if n > 0 else -1)
        if raw.size == 0:
            continue
        cur = raw.astype(np.float32)
        if prev is not None and prev.size == cur.size:
            sig.append(float(np.mean(np.abs(cur - prev))))
            ts.append(float(r.get("t_burst_monotonic")
                            or r.get("t_first_monotonic")))
        prev = cur
    return np.array(ts), np.array(sig)


def camera_motion(cdir: Path, rows: list[dict]) -> tuple:
    ts, sig, prev = [], [], None
    for r in rows:
        fp = cdir / r["filename"]
        if not fp.exists():
            continue
        img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        cur = img.astype(np.float32)
        if prev is not None and prev.shape == cur.shape:
            sig.append(float(np.mean(np.abs(cur - prev))))
            ts.append(float(r["t_arrival_monotonic"]))
        prev = cur
    return np.array(ts), np.array(sig)



def xcorr_offset(rt, rs, ct, cs, max_lag=1.0, grid=0.005):
    """Cross-correlate the two motion signals on a common time grid.

    Peak-to-peak matching of a single impulse CANNOT resolve offsets below
    one radar frame period: at 4 fps, offsets of 0/50/120 ms all fall in
    the same radar frame and argmax returns the same index. Verified -
    injected offsets of 0, 50, 120 and -80 ms all recovered as 0 ms.

    Cross-correlation uses EVERY sample rather than one peak, so the
    estimate is limited by the interpolation grid and the amount of motion,
    not by the radar frame period. A parabolic fit at the correlation
    maximum gives sub-grid resolution.

    Requires SUSTAINED VARIED MOTION, not a single event.
    """
    t0 = max(rt[0], ct[0]); t1 = min(rt[-1], ct[-1])
    if t1 - t0 < 5.0:
        return None, 0.0, 0.0
    g = np.arange(t0, t1, grid)

    def norm(t, y):
        v = np.interp(g, t, y); v = v - v.mean()
        sd = v.std()
        return v / sd if sd > 0 else v

    a_, b_ = norm(rt, rs), norm(ct, cs)
    n_lag = int(max_lag / grid)
    lags = np.arange(-n_lag, n_lag + 1)
    cc = np.array([float(np.mean(np.roll(a_, -int(L)) * b_)) for L in lags])
    k = int(np.argmax(cc))
    peak_lag = lags[k] * grid
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k-1], cc[k], cc[k+1]
        den = (y0 - 2*y1 + y2)
        if den != 0:
            peak_lag += grid * (0.5 * (y0 - y2) / den)
    return float(peak_lag), float(cc[k]), float(cc[k])


def peak(ts: np.ndarray, sig: np.ndarray, label: str):
    """Return (t_peak, prominence). Prominence guards against reporting a
    peak when there was no event - a flat signal has a 'maximum' too."""
    if sig.size < 5:
        print(f"  {label}: too few frames ({sig.size})")
        return None, 0.0
    i = int(np.argmax(sig))
    med = float(np.median(sig))
    mad = float(np.median(np.abs(sig - med))) or 1e-9
    prom = (sig[i] - med) / mad
    print(f"  {label:<8} peak at t={ts[i]:.3f}  value={sig[i]:.2f}  "
          f"median={med:.2f}  prominence={prom:.1f} MAD")
    return float(ts[i]), prom


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure radar/camera sync offset")
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--full-frames", action="store_true",
                    help="read whole radar frames instead of a 256 KB prefix")
    ap.add_argument("--min-prominence", type=float, default=6.0,
                    help="reject a peak below this many MAD above median")
    a = ap.parse_args()

    rad_rows = load_csv(a.session / "radar" / "frames.csv")
    cam_rows = load_csv(a.session / "camera" / "frames.csv")
    print("=" * 74)
    print(f"  SYNC OFFSET MEASUREMENT: {a.session}")
    print("=" * 74)
    print(f"  radar frames {len(rad_rows)}   camera frames {len(cam_rows)}")
    print(f"  radar read   {'full frames' if a.full_frames else '256 KB prefix'}\n")

    rt, rs = radar_motion(a.session / "radar", rad_rows,
                          0 if a.full_frames else 256 * 1024)
    ct, cs = camera_motion(a.session / "camera", cam_rows)

    print("  motion peaks")
    r_pk, r_prom = peak(rt, rs, "radar")
    c_pk, c_prom = peak(ct, cs, "camera")

    print("\n" + "=" * 74)
    if r_pk is None or c_pk is None:
        print("  Could not locate peaks in both streams.")
        return 1

    weak = [n for n, p in (("radar", r_prom), ("camera", c_prom))
            if p < a.min_prominence]
    if weak:
        print("  NO CLEAR EVENT")
        print("=" * 74)
        print(f"  Weak peak in: {', '.join(weak)} "
              f"(< {a.min_prominence:.0f} MAD above median)")
        print("""
  A flat signal still has a maximum, so a low-prominence peak means there
  was probably no sharp event - not that the offset is zero. Re-record with
  a more decisive one: a metal plate or tray dropped through the beam
  inside the camera's field of view works better than a hand clap, because
  it is both a strong radar reflector and a large pixel change.

  Keep the rest of the scene still for a few seconds either side so the
  event stands alone.""")
        return 1

    # Cross-correlation is the PRIMARY estimate; peak matching is only a
    # coarse sanity check because it is quantised to the radar frame period.
    xc_lag, xc_val, xc_conf = xcorr_offset(rt, rs, ct, cs)
    print("  cross-correlation (primary estimate)")
    if xc_lag is None:
        print("    overlap too short (<5 s) - falling back to peak matching")
    else:
        print(f"    lag = {xc_lag*1000:+.1f} ms   r = {xc_val:.3f}   "
              f"peak correlation r = {xc_conf:.2f}")
        if xc_conf < 0.30:
            print("    WEAK: correlation peak is not distinct - there was")
            print("    probably not enough sustained motion to correlate.")

    # r >= 0.30 between two independently-derived motion signals is a
    # real match; below that there was not enough common motion.
    use_xc = xc_lag is not None and xc_conf >= 0.30
    off = xc_lag if use_xc else (r_pk - c_pk)
    method = ("cross-correlation" if use_xc else
              "peak matching (COARSE - quantised to the radar frame period)")
    print(f"\n  method used: {method}")
    print("  RESULT")
    print("=" * 74)
    print(f"  radar peak  t = {r_pk:.3f}")
    print(f"  camera peak t = {c_pk:.3f}")
    print(f"\n  offset = t_radar - t_camera = {off*1000:+.1f} ms")

    cam_T = float(np.median(np.diff(ct))) if ct.size > 1 else 1 / 15
    frames = off / cam_T
    print(f"  camera period {cam_T*1000:.1f} ms  ->  {frames:+.2f} camera frames")
    print(f"  subject motion at 1.4 m/s: {abs(off)*1.4*100:.1f} cm")

    print("\n  APPLY IT")
    print(f"      python check_session.py --session <s> --offset-ms {off*1000:.1f}")
    print("""
  RECORDING PROTOCOL
    Cross-correlation needs SUSTAINED VARIED MOTION, not a single event:
    20-30 s of walking in and out of the beam, waving, changing
    direction. Every frame then contributes to the estimate.

    A single clap is NOT sufficient. Peak matching alone is quantised to
    the radar frame period (250 ms at 4 fps), and offsets smaller than
    that - the interesting range - read as zero.

  CAVEATS
    Confirm the peak correlation r is comfortably above 0.30 before
    using the number. Repeat on 2-3 recordings and check they agree.

    The offset is a property of THIS rig and configuration. Re-measure
    if the frame rate, cabling or mounting changes.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
