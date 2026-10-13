"""
check_parity_stages.py — isolate WHICH DSP stage diverges between GPU and CPU.

check_parity.py showed the two backends produce different point clouds:
6 of 30 frames had different point counts, always GPU < CPU, with position
differences up to 0.144 m (2.2 range bins) and power differences up to 2.72 dB.
Those are bin-level decisions, not float rounding.

This runs ONE real frame through the pipeline and compares each GPU function
against its CPU counterpart on IDENTICAL INPUT, so the first stage to diverge
is the culprit. Later stages inherit upstream differences, so read the FIRST
failure, not the last.

Known non-suspect: cfar_2d_gpu returns _cfar_2d_cpu when cfg.cfar_method ==
'os', which is the pipeline default. Both paths run identical CFAR.

USAGE
    python check_parity_stages.py --folder <session frames> --cfg <cfg> [--frame 50]
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


def cmp(name, a, b, tol=1e-4):
    """Compare two arrays. Returns True if they agree.

    Handles COMPLEX arrays correctly: an earlier version used
    np.asarray(x, np.float64), which silently discards the imaginary part
    and understates the error for every FFT stage.

    Reports RELATIVE error as well as absolute, because absolute error alone
    cannot distinguish float32 rounding from an algorithmic difference:

        rel ~1e-6 (float32 eps)  -> benign rounding
        rel >> 1e-6              -> different algorithm/window/normalisation
        near-constant ratio      -> a scale factor, i.e. a real bug
    """
    if a is None or b is None:
        print(f'  {name:<24} SKIP (one side None)')
        return True
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        print(f'  {name:<24} SHAPE DIFFERS  gpu{a.shape} vs cpu{b.shape}')
        return False

    if a.dtype == bool or b.dtype == bool:
        n = int((a != b).sum())
        print(f'  {name:<24} {"OK" if n==0 else "DIFFERS":<9} '
              f'{n}/{a.size} cells ({100*n/max(a.size,1):.2f}%)')
        return n == 0

    if np.iscomplexobj(a) or np.iscomplexobj(b):
        ca = np.asarray(a, np.complex128)
        cb = np.asarray(b, np.complex128)
        both = np.isfinite(ca) & np.isfinite(cb)
        d = np.abs(ca[both] - cb[both])          # true complex magnitude
        scale = np.abs(cb[both])
    else:
        fa = np.asarray(a, np.float64)
        fb = np.asarray(b, np.float64)
        both = np.isfinite(fa) & np.isfinite(fb)
        d = np.abs(fa[both] - fb[both])
        scale = np.abs(fb[both])

    if d.size == 0:
        print(f'  {name:<24} SKIP (no finite values)')
        return True

    mx = float(d.max())
    big = scale > (scale.max() * 1e-6 if scale.max() > 0 else 1)
    rel = float((d[big] / scale[big]).max()) if big.any() else 0.0
    nz = int((d > tol).sum())

    # A constant ratio across large values indicates a scale-factor bug
    # rather than rounding.
    ratio_hint = ''
    if big.sum() > 100:
        r = (np.abs(ca[both][big]) / scale[big]) if np.iscomplexobj(a) \
            else (np.abs(fa[both][big]) / scale[big])
        rmed = float(np.median(r))
        if abs(rmed - 1.0) > 1e-4:
            ratio_hint = f'  SCALE={rmed:.6f}'

    verdict = 'OK' if rel < 1e-4 else 'DIFFERS'
    print(f'  {name:<24} {verdict:<9} absmax={mx:.2e}  rel={rel:.2e}  '
          f'>{tol:.0e}: {nz}/{d.size}{ratio_hint}')
    return rel < 1e-4


def main():
    ap = argparse.ArgumentParser(description='Isolate the diverging DSP stage')
    ap.add_argument('--folder', required=True)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--frame', type=int, default=50)
    a = ap.parse_args()

    (parse_cfg_file, parse_calibration, apply_raw_iq_order_override,
     read_frame, rp, ga) = imports()

    if not getattr(ga, 'GPU_AVAILABLE', False):
        sys.exit('[ERROR] GPU_AVAILABLE is False - nothing to compare.')

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
    print('  STAGE-BY-STAGE GPU vs CPU PARITY')
    print('=' * 78)
    print(f'  frame  : {os.path.basename(files[idx])}')
    print(f'  method : cfar_method={getattr(cfg,"cfar_method","?")} '
          f'(OS falls back to CPU on both paths)')
    print()
    print('  Read the FIRST DIFFERS line - later stages inherit upstream drift.')
    print('  ' + '-' * 74)

    first_bad = None

    # ── S2 range FFT ─────────────────────────────────────────────────────────
    g = ga.range_fft_gpu(cube, cfg)
    c = rp.range_fft(cube, cfg)
    if not cmp('S2 range FFT', g, c) and first_bad is None:
        first_bad = 'S2 range FFT'

    # Use the CPU result downstream so each stage is tested on IDENTICAL input.
    range_cube = c

    # ── S3 virtual arrays (CPU only, no GPU variant) ─────────────────────────
    ch_az, ch_el, _ = cfg.build_channel_table()
    va_az, va_el = rp.form_virtual_arrays(range_cube, cfg, ch_az, ch_el)

    # ── S4 doppler FFT + MTI ─────────────────────────────────────────────────
    # Mirror process_frame exactly: remove_clutter=cfg.clutter_removal,
    # use_window=True, then zero_dc_notch.
    try:
        mti = cfg.clutter_removal
        g = ga.doppler_fft_gpu(va_az, cfg, remove_clutter=mti, use_window=True)
        c = rp.doppler_fft(va_az, cfg, remove_clutter=mti, use_window=True)
        if not cmp('S4 doppler FFT + MTI', g, c) and first_bad is None:
            first_bad = 'S4 doppler FFT'
        virt_rd = rp.zero_dc_notch(c, cfg)
    except Exception as e:                                   # noqa: BLE001
        print(f'  S4 doppler FFT             ERROR {type(e).__name__}: {e}')
        virt_rd = None

    if virt_rd is None:
        print('\n  cannot continue without S4 output')
        sys.exit(2)

    # ── S5 power reduce ──────────────────────────────────────────────────────
    try:
        g = ga.power_db_reduce_gpu(virt_rd, axis=-1)
        c = rp._power_db_reduce(virt_rd, axis=-1)
        if not cmp('S5 power dB reduce', g, c) and first_bad is None:
            first_bad = 'S5 power reduce'
        RD = c
    except Exception as e:                                   # noqa: BLE001
        print(f'  S5 power dB reduce         ERROR {type(e).__name__}: {e}')
        RD = None

    # ── S6 CFAR ──────────────────────────────────────────────────────────────
    rd_det = None
    if RD is not None:
        try:
            heat = np.power(10.0, RD / 10.0).astype(np.float32)
            gd, _ = ga.cfar_2d_gpu(heat, cfg)
            cd, _ = rp.cfar_2d(heat, cfg)
            if not cmp('S6 CFAR detections', gd, cd) and first_bad is None:
                first_bad = 'S6 CFAR'
            rd_det = cd
        except Exception as e:                               # noqa: BLE001
            print(f'  S6 CFAR                    ERROR {type(e).__name__}: {e}')

    # ── focus mask — newly load-bearing since Stage 2.2 ──────────────────────
    if RD is not None:
        try:
            g = ga.ra_focus_mask_gpu(RD, rd_det, cfg)
            c = rp._ra_focus_mask_from_rd(RD, rd_det, cfg)
            if not cmp('RA focus mask', g, c) and first_bad is None:
                first_bad = 'RA focus mask'
        except Exception as e:                               # noqa: BLE001
            print(f'  RA focus mask              ERROR {type(e).__name__}: {e}')

    # ── S7 azimuth FFT ───────────────────────────────────────────────────────
    try:
        g = ga.az_fft_gpu(virt_rd, cfg)
        c = rp._az_fft(virt_rd, cfg)
        if not cmp('S7 azimuth FFT', g, c) and first_bad is None:
            first_bad = 'S7 azimuth FFT'
    except Exception as e:                                   # noqa: BLE001
        print(f'  S7 azimuth FFT             ERROR {type(e).__name__}: {e}')

    print('  ' + '-' * 74)
    print()
    print('=' * 78)
    print('  RESULT')
    print('=' * 78)
    if first_bad is None:
        print('  All compared stages agree within tolerance.')
        print('  The divergence is downstream of these - likely in azimuth peak')
        print('  selection or the xyz extraction gates. Re-run check_parity.py')
        print('  and inspect a specific differing frame.')
    else:
        print(f'  FIRST DIVERGING STAGE: {first_bad}')
        print()
        print('  Fix that stage before anything else - every later difference')
        print('  may simply be inherited from it.')
        print()
        print('  Decide deliberately WHICH implementation is correct. Every result')
        print('  in this project came from the CPU path, so CPU is the de-facto')
        print('  reference; the GPU variant should be made to match it unless the')
        print('  CPU version is itself shown to be wrong.')


if __name__ == '__main__':
    main()
