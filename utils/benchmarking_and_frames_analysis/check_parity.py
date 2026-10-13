"""
check_parity.py — GPU vs CPU numerical parity (audit Stage 5.4).

WHY THIS MATTERS NOW
    P0-7 revealed that a cosmetic `.decode()` had silently forced every DSP
    stage onto the CPU. So EVERY result in this project — all 13 sessions,
    every Phase 2-6 metric, every conclusion in the audit — was produced by
    the CPU path.

    With the GPU re-enabled, the pipeline switches to eight different
    implementations at once. The original plan required a GPU/CPU parity test
    and it was never written. It is now load-bearing: if the GPU disagrees,
    prior results do not carry forward.

WHAT IT DOES
    Runs the SAME frames through both paths and compares the point cloud.
    GPU is forced off by monkey-patching gpu_accel.GPU_AVAILABLE, which is
    exactly the switch radar_processing keys on.

USAGE
    python check_parity.py --folder <session frames folder> --cfg <cfg> [--n 30]

INTERPRETING
    Small float differences are EXPECTED: GPU and CPU reduce in different
    orders. What must NOT change is the DECISION structure — how many points
    survive CFAR and gating. A different cloud_pts means a different pipeline.
"""
from __future__ import annotations
import argparse, glob, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np


def load_pipeline():
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


def main():
    ap = argparse.ArgumentParser(description='GPU vs CPU numerical parity')
    ap.add_argument('--folder', required=True)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--n', type=int, default=30, help='frames to test')
    ap.add_argument('--tol', type=float, default=1e-3, help='position tolerance (m)')
    a = ap.parse_args()

    (parse_cfg_file, parse_calibration, apply_raw_iq_order_override,
     read_frame, rp, ga) = load_pipeline()

    print('=' * 78)
    print('  GPU vs CPU NUMERICAL PARITY')
    print('=' * 78)
    if not getattr(ga, 'GPU_AVAILABLE', False):
        sys.exit('[ERROR] GPU_AVAILABLE is False - nothing to compare. '
                 'Run check_gpu.py first.')

    files = sorted(glob.glob(os.path.join(a.folder, '*.bin')))
    if not files:
        sys.exit(f'[ERROR] no .bin in {a.folder}')

    # Sample across the session rather than the first N frames, so the
    # comparison covers varied geometry rather than one walk segment.
    step = max(1, len(files) // a.n)
    sel = files[::step][:a.n]
    print(f'  folder : {a.folder}')
    print(f'  frames : {len(sel)} sampled from {len(files)}')

    cfg = parse_cfg_file(a.cfg)
    corr = parse_calibration(a.cfg, cfg.num_tx, cfg.num_rx)
    cfg = apply_raw_iq_order_override(cfg, 'i-first')
    cfg.quiet_mode = True

    def run(force_cpu):
        saved = ga.GPU_AVAILABLE
        if force_cpu:
            ga.GPU_AVAILABLE = False
        out = []
        try:
            for f in sel:
                r = rp.process_frame(read_frame(f, cfg, 0), cfg, corr, verbose=False)
                xyz = r.xyz_cloud
                out.append(xyz if xyz is not None else np.empty((0, 7), np.float32))
        finally:
            ga.GPU_AVAILABLE = saved
        return out

    print('  running GPU path ...')
    g = run(force_cpu=False)
    print('  running CPU path ...')
    c = run(force_cpu=True)

    print()
    print(f'  {"frame":>6}{"gpu_pts":>9}{"cpu_pts":>9}{"max dx":>10}{"max dpwr":>10}   status')
    print('  ' + '-' * 60)

    n_count_diff = 0
    worst_pos = worst_pwr = 0.0
    for i, (gg, cc) in enumerate(zip(g, c)):
        if len(gg) != len(cc):
            n_count_diff += 1
            print(f'  {i:>6}{len(gg):>9}{len(cc):>9}{"-":>10}{"-":>10}   COUNT DIFFERS')
            continue
        if len(gg) == 0:
            continue
        # sort both by power so ordering does not create false differences
        gg = gg[np.argsort(-gg[:, 3])]
        cc = cc[np.argsort(-cc[:, 3])]
        dpos = float(np.abs(gg[:, :3] - cc[:, :3]).max())
        dpwr = float(np.abs(gg[:, 3] - cc[:, 3]).max())
        worst_pos = max(worst_pos, dpos)
        worst_pwr = max(worst_pwr, dpwr)
        if i < 10 or dpos > a.tol:
            flag = 'ok' if dpos <= a.tol else 'POSITION DIFFERS'
            print(f'  {i:>6}{len(gg):>9}{len(cc):>9}{dpos:>10.2e}{dpwr:>10.2e}   {flag}')

    print()
    print('=' * 78)
    print('  RESULT')
    print('=' * 78)
    print(f'  frames compared           : {len(g)}')
    print(f'  frames with DIFFERENT n_pts: {n_count_diff}')
    print(f'  worst position difference : {worst_pos:.3e} m')
    print(f'  worst power difference    : {worst_pwr:.3e} dB')
    print()
    if n_count_diff:
        print('  FAIL - the two paths produce different NUMBERS OF POINTS.')
        print('  That is a decision-level divergence, not float noise: CFAR or a')
        print('  gate is resolving differently. Prior CPU-derived results do NOT')
        print('  carry forward to the GPU path. Investigate before proceeding.')
        sys.exit(1)
    if worst_pos > a.tol:
        print(f'  MARGINAL - point counts match but positions differ by up to')
        print(f'  {worst_pos:.2e} m (tolerance {a.tol:.0e}). Check whether this')
        print('  exceeds the range/azimuth bin size before accepting.')
        sys.exit(1)
    print('  PASS - identical point counts, positions within tolerance.')
    print('  The GPU path reproduces the CPU path. Prior results carry forward.')
    sys.exit(0)


if __name__ == '__main__':
    main()
