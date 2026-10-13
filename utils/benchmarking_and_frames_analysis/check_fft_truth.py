"""
check_fft_truth.py — is the GPU FFT wrong, or is it just float32 rounding?

THE QUESTION
    check_parity_stages.py showed every FFT stage differs between GPU and CPU
    while every non-FFT stage (power reduce, CFAR, focus mask) agrees exactly.
    Comparing GPU against CPU cannot say WHICH is right — both are float32.

THE METHOD
    Compute the same transform in float64 as ground truth and measure how far
    each backend sits from it.

        both ~equally close   -> both float32-correct; the difference is
                                 rounding, and the pipeline is simply
                                 threshold-sensitive on a sparse cloud
        one clearly further   -> that backend has a real defect

WHY THE EARLIER 'rel' NUMBER WAS NOT CONCLUSIVE
    It was max(|a-b|/|b|) over bins down to 1e-6 of peak. Near a spectral null
    |b| is tiny, so float32-level ABSOLUTE error yields a huge RELATIVE error.
    The metric was dominated by the weakest bins. This script reports
    peak-relative and L2-relative error instead, plus error restricted to bins
    within 40 dB of the peak - the ones CFAR can actually act on.

USAGE
    python check_fft_truth.py --folder <session frames> --cfg <cfg> [--frame 50]
"""
from __future__ import annotations
import argparse, glob, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np


def imports():
    try:
        from processing.radar_config import (parse_cfg_file, parse_calibration,
                                              apply_raw_iq_order_override)
        from processing.radar_reader import read_frame
        from processing import radar_processing as rp
        from processing import gpu_accel as ga
    except ImportError:
        from radar_config import (parse_cfg_file, parse_calibration,       # type: ignore
                                  apply_raw_iq_order_override)
        from radar_reader import read_frame                                # type: ignore
        import radar_processing as rp                                      # type: ignore
        import gpu_accel as ga                                             # type: ignore
    return (parse_cfg_file, parse_calibration, apply_raw_iq_order_override,
            read_frame, rp, ga)


def score(name, arr, truth):
    """Distance from the float64 reference, using dynamic-range-safe metrics."""
    a = np.asarray(arr, np.complex128)
    t = np.asarray(truth, np.complex128)
    d = np.abs(a - t)
    m = np.abs(t)
    peak = m.max() if m.size else 1.0
    l2 = np.linalg.norm(d) / max(np.linalg.norm(m), 1e-30)
    pk = d.max() / max(peak, 1e-30)
    near = m > peak * 1e-2                      # within 40 dB of peak
    rel40 = (d[near] / m[near]).max() if near.any() else 0.0
    print(f'    {name:<12} L2rel={l2:.3e}   peakrel={pk:.3e}   rel@40dB={rel40:.3e}')
    return l2, pk, rel40


def verdict(tag, g, c):
    """g, c are (l2, peak, rel40) tuples for GPU and CPU."""
    print()
    ratio = g[0] / c[0] if c[0] > 0 else float('inf')
    eps32 = np.finfo(np.float32).eps
    print(f'    float32 epsilon = {eps32:.2e}')
    print(f'    GPU/CPU L2 error ratio = {ratio:.2f}')
    if max(g[0], c[0]) < 1e-4:
        if 0.2 < ratio < 5.0:
            print(f'    => BOTH are float32-accurate and comparably so.')
            print(f'       {tag}: the GPU/CPU difference is ROUNDING, not a defect.')
            return 'benign'
        worse = 'GPU' if ratio > 1 else 'CPU'
        print(f'    => Both are small but {worse} is {max(ratio,1/ratio):.1f}x worse.')
        print(f'       {tag}: suspicious, inspect the {worse} implementation.')
        return 'skewed'
    worse = 'GPU' if g[0] > c[0] else 'CPU'
    print(f'    => Error is LARGE ({max(g[0],c[0]):.2e}). {worse} is further from truth.')
    print(f'       {tag}: likely a real defect in the {worse} path.')
    return 'defect'


def main():
    ap = argparse.ArgumentParser(description='FFT accuracy vs a float64 reference')
    ap.add_argument('--folder', required=True)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--frame', type=int, default=50)
    a = ap.parse_args()

    (parse_cfg_file, parse_calibration, apply_raw_iq_order_override,
     read_frame, rp, ga) = imports()
    if not getattr(ga, 'GPU_AVAILABLE', False):
        sys.exit('[ERROR] GPU_AVAILABLE is False.')
    import cupy as cp

    files = sorted(glob.glob(os.path.join(a.folder, '*.bin')))
    if not files:
        sys.exit(f'[ERROR] no .bin in {a.folder}')
    idx = min(a.frame, len(files) - 1)

    cfg = parse_cfg_file(a.cfg)
    corr = parse_calibration(a.cfg, cfg.num_tx, cfg.num_rx)
    cfg = apply_raw_iq_order_override(cfg, 'i-first')
    cfg.quiet_mode = True

    cube = read_frame(files[idx], cfg, 0)
    if corr is not None and cfg.apply_calibration:
        cube = rp.apply_calibration(cube, corr)

    print('=' * 78)
    print('  FFT ACCURACY vs float64 GROUND TRUTH')
    print('=' * 78)
    print(f'  frame : {os.path.basename(files[idx])}')
    print()

    results = {}

    # ── S2 range FFT ─────────────────────────────────────────────────────────
    print('  S2 RANGE FFT')
    g = np.asarray(ga.range_fft_gpu(cube, cfg))
    c = np.asarray(rp.range_fft(cube, cfg))

    # float64 reference: replicate range_fft in double precision.
    # Take a representative slice to keep memory sane.
    sl = (slice(None), slice(0, 8), slice(None), slice(None)) if cube.ndim == 4 \
         else (slice(0, 8),)
    try:
        n_s = cube.shape[-1]
        win = np.hanning(n_s).astype(np.float64)
        sub = np.asarray(cube[sl], np.complex128)
        ref = np.fft.fft(sub * win, axis=-1)
        # match the pipeline's output slicing by comparing common shape
        gs = np.asarray(g[sl], np.complex128)
        cs = np.asarray(c[sl], np.complex128)
        if ref.shape == gs.shape == cs.shape:
            print(f'    reference: float64 windowed FFT, slice {gs.shape}')
            results['S2'] = (score('GPU', gs, ref), score('CPU', cs, ref))
            verdict('S2 range FFT', *results['S2'])
        else:
            print(f'    shape mismatch (ref {ref.shape} vs {gs.shape}) -')
            print(f'    falling back to GPU-vs-CPU peak-relative only')
            d = np.abs(np.asarray(g, np.complex128) - np.asarray(c, np.complex128))
            m = np.abs(np.asarray(c, np.complex128))
            print(f'    peakrel = {d.max()/max(m.max(),1e-30):.3e}   '
                  f'L2rel = {np.linalg.norm(d)/max(np.linalg.norm(m),1e-30):.3e}')
    except Exception as e:                                   # noqa: BLE001
        print(f'    reference failed: {type(e).__name__}: {e}')

    # ── Isolated cuFFT vs numpy on synthetic data ────────────────────────────
    print()
    print('  ISOLATED FFT TEST  (synthetic, removes all pipeline variables)')
    rng = np.random.default_rng(0)
    x64 = (rng.normal(size=(64, 256, 48)) + 1j * rng.normal(size=(64, 256, 48)))
    w64 = np.hanning(48)
    ref = np.fft.fftshift(np.fft.fft(x64 * w64, n=256, axis=-1), axes=-1)

    x32 = x64.astype(np.complex64)
    w32 = w64.astype(np.float32)
    cpu32 = np.fft.fftshift(np.fft.fft(x32 * w32, n=256, axis=-1), axes=-1)
    xg = cp.asarray(x32); wg = cp.asarray(w32)
    gpu32 = cp.asnumpy(cp.fft.fftshift(cp.fft.fft(xg * wg, n=256, axis=-1), axes=-1))

    r = (score('GPU', gpu32, ref), score('CPU', cpu32, ref))
    v = verdict('isolated cuFFT vs pocketfft', *r)

    print()
    print('=' * 78)
    print('  CONCLUSION')
    print('=' * 78)
    if v == 'benign':
        print("""  The GPU and CPU FFTs are equally accurate against float64 truth.
  Their mutual difference is float32 rounding, NOT a defect.

  Consequence: the 6-of-30 frames with different point counts come from
  CFAR decisions flipping for detections that sit within rounding distance
  of their threshold - amplified by the very sparse cloud (1.4-2.4 pts/frame).

  This is a REPRODUCIBILITY question, not a correctness one:
    - keep CPU        -> bit-reproducible, ~66x slower on the FFT
    - move to GPU     -> much faster, but detection counts shift slightly
                         and every CPU-derived metric must be re-measured
    - reduce sensitivity -> raise cfar_margin_db so marginal detections are
                         not decided by rounding; costs sensitivity

  Whichever is chosen, do NOT mix results from the two backends.""")
    else:
        print("""  One backend is measurably further from float64 truth. Treat this as a
  real defect and fix it before using the GPU path for any result.""")


if __name__ == '__main__':
    main()
