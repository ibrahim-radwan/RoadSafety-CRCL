#!/usr/bin/env python3
"""Radar data-integrity check: stale buffer words + range-time map.

Usage:  python check_radar_frames.py <session_dir> [--png rt.png]

Re-implements the key test of the firmware investigation's
check_radar_session.py (2026-10-10). The r075m firmware booted from QSPI left
~50% of every chirp as stale DSS-L3 words that are bit-identical in every
frame; r075m_v2 / r075m_v2_8fps fixed it. A healthy stream gives ~0%.

Frame layout: int16 LE [loop 64][tx 6][dev 2][rx 4][sample 256][I, Q], I first.
Range-time uses per-frame clutter removal (mean over loops) and a Hann
range FFT, so only moving targets show.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

N_LOOP, N_TX, N_DEV, N_RX, N_S = 64, 6, 2, 4, 256
FRAME_I16 = N_LOOP * N_TX * N_DEV * N_RX * N_S * 2
RANGE_RES_M = 299792458.0 / (2 * 19.9878e12 * N_S / 10e6)     # r075m family


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--png", type=Path)
    ap.add_argument("--max-range", type=float, default=40.0)
    a = ap.parse_args()

    files = sorted((a.session / "radar").glob("frame_*.bin"))
    if len(files) < 3:
        sys.exit("need at least 3 frames")
    bad = sum(f.stat().st_size != FRAME_I16 * 2 for f in files)
    t = None
    man = a.session / "radar" / "frames.csv"
    if man.exists():
        rows = list(csv.DictReader(man.open()))
        t = np.array([float(r["t_burst_monotonic"]) for r in rows])
        t -= t[0]
    print(f"frames: {len(files)}   wrong size: {bad}"
          + (f"   median interval {np.median(np.diff(t)) * 1e3:.1f} ms" if t is not None else ""))

    f0 = np.fromfile(files[0], dtype="<i2")
    const = np.ones(FRAME_I16, bool)
    win = np.hanning(N_S).astype(np.float32)
    n_bins = int(a.max_range / RANGE_RES_M)
    rt = np.empty((len(files), n_bins))
    peak = 0
    for i, f in enumerate(files):
        x = np.fromfile(f, dtype="<i2")
        if i:
            const &= x == f0
        peak = max(peak, int(np.abs(x).max()))
        x = x.reshape(N_LOOP, N_TX * N_DEV * N_RX, N_S, 2).astype(np.float32)
        iq = x[..., 0] + 1j * x[..., 1]
        iq -= iq.mean(axis=0, keepdims=True)                        # static clutter
        sp = np.abs(np.fft.fft(iq * win, axis=-1)[..., :n_bins]) ** 2
        rt[i] = sp.mean(axis=(0, 1))
    frac = const.mean()
    print(f"identical in every frame: {100 * frac:.2f} %   (healthy ~0 %; r075m defect ~50 %)")
    print(f"peak |sample|: {peak}   (clips at 32767)")
    print("VERDICT:", "STALE WORDS PRESENT - data degraded" if frac > 0.01 else "no stale words")

    db = 10 * np.log10(rt / np.median(rt, axis=0))
    k = db[:, int(2.0 / RANGE_RES_M):].argmax(axis=1) + int(2.0 / RANGE_RES_M)
    print(f"strongest moving return: range {RANGE_RES_M * k.min():.1f}-{RANGE_RES_M * k.max():.1f} m, "
          f"median {RANGE_RES_M * np.median(k):.1f} m")
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tt = t if t is not None else np.arange(len(files))
        plt.figure(figsize=(12, 5))
        plt.pcolormesh(tt, np.arange(n_bins) * RANGE_RES_M, np.clip(db, 0, 30).T, cmap="Greys", shading="auto")
        plt.xlabel("time (s)" if t is not None else "frame")
        plt.ylabel("range (m)")
        plt.title(f"{a.session.name}: moving returns (dB over median)")
        plt.tight_layout()
        plt.savefig(a.png, dpi=100)
        print(f"wrote {a.png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
