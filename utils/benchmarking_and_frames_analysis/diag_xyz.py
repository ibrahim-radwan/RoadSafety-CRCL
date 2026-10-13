"""
diag_xyz.py  — Body-shape geometry diagnostic for AWR2243-2X pipeline.

WHAT THIS MEASURES AND WHY
───────────────────────────
The goal is to answer: does the processed point cloud contain body-shape
geometry, and if so, what does that geometry actually look like?

Cloud count alone does not answer this. 10 points all at the same x,y position
are not body-shaped. 4 points spread 40cm apart in x ARE a shoulder silhouette.
We need to measure the actual spatial distribution of xyz coordinates.

FRAME SELECTION (why these frames, not the highest-cloud-count ones)
─────────────────────────────────────────────────────────────────────
The original diag_xyz.py selected by cloud count. That is wrong because:
  - High cloud count can come from range-depth spread (many y positions,
    same x) — not body width
  - High cloud count frames may be at far range where lateral resolution
    is coarser
  - We need frames that prove specific properties

This script uses four selection criteria:
  A) Single RD bin, cloud > 1: proves az multi-peak is resolving LATERAL
     width (not range depth), because there is only one range bin
  B) High cloud/RD ratio >= 2: frames where multiple az peaks per range
     bin were found — the clearest signal of lateral resolution
  C) One representative frame per walk phase: ensures all ranges and
     movement directions are sampled
  D) ALL detected frames: full statistical distributions

WHAT TO LOOK FOR IN THE OUTPUT
────────────────────────────────
  X spread (lateral):  > 0.30m → body-width information present
                       < 0.10m → only 1 az peak per cluster, no width
  Y spread (range):    0.10–0.35m → correct body depth (30cm body)
                       > 0.50m → depth scatter, not real body depth
  Z values:            near 0 → radar at same height as target centre
                       negative → radar elevated above target COM
                       (adjust --radar-height to correct)
  Doppler:             near zero for most frames → subject walking
                       tangentially (not toward/away) — valid, not a bug
  cloud/RD ratio:      > 1.5 in some frames → az multi-peak firing
                       always == 1.0 → az multi-peak never fires

Usage:
    python diag_xyz.py                          # diagnostic set (15 frames)
    python diag_xyz.py --all                    # all 342 detected frames
    python diag_xyz.py --frames 55 104 293 398  # specific frames
    python diag_xyz.py --radar-height 0.8       # correct z by 0.8m offset
"""
import sys, glob, os, argparse
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from processing.radar_config     import (parse_cfg_file, parse_calibration,
                                          apply_raw_iq_order_override)
from processing.radar_reader     import read_frame
from processing.radar_processing import process_frame

CFG_PATH   = os.path.join(os.path.dirname(os.path.dirname(PROJECT_ROOT)),
                           'cascaded_radar_capture_pipeline', 'firmware',
                           'r017m', 'cascade_r017m.cfg')
FRAMES_DIR = os.path.join(PROJECT_ROOT, 'capture', 'frames')

# ── Frame selection ───────────────────────────────────────────────────────────
# Criterion A: single RD bin, cloud > 1 (proves lateral az resolution)
FRAMES_A = [9, 24, 48, 49, 50]

# Criterion B: high cloud/RD ratio >= 2 (multi-peak firing)
FRAMES_B = [301, 33, 54, 261, 284, 293]

# Criterion C: one representative per walk phase (range coverage)
FRAMES_C = [0, 24, 55, 104, 126, 182, 253, 293]

DIAGNOSTIC_SET = sorted(set(FRAMES_A + FRAMES_B + FRAMES_C))


def run(target_frames, radar_height=0.0, verbose_frames=None):
    cfg        = parse_cfg_file(CFG_PATH)
    correction = parse_calibration(CFG_PATH, cfg.num_tx, cfg.num_rx)
    cfg        = apply_raw_iq_order_override(cfg, 'i-first')
    cfg.cfar_margin_db = 5.0
    cfg.quiet_mode     = True

    bin_files = sorted(glob.glob(os.path.join(FRAMES_DIR, '*.bin')))
    if not bin_files:
        print(f"[ERROR] No .bin files in {FRAMES_DIR}")
        return

    print("=" * 72)
    print("  BODY-SHAPE GEOMETRY DIAGNOSTIC")
    print(f"  {len(target_frames)} frames  |  radar_height offset = {radar_height:.2f} m")
    print("=" * 72)
    print()
    print(f"  {'fr':>5}  {'pts':>4}  {'RD':>4}  "
          f"{'x_min':>7}  {'x_max':>7}  {'x_span':>7}  "
          f"{'y_min':>7}  {'y_max':>7}  {'y_span':>7}  "
          f"{'z_mean':>7}  {'dop_max':>8}  {'az/RD':>6}")
    print(f"  {'-'*70}")

    all_rows = []
    per_frame = []

    for fn in sorted(target_frames):
        if fn >= len(bin_files):
            continue
        try:
            cube   = read_frame(bin_files[fn], cfg, 0)
            result = process_frame(cube, cfg, correction, verbose=False)
            xyz    = result.xyz_cloud

            if xyz is None or len(xyz) == 0:
                print(f"  {fn:>5}  {'—':>4}  {'—':>4}  no detection")
                continue

            x   = xyz[:,0];  y   = xyz[:,1];  z = xyz[:,2] + radar_height
            pwr = xyz[:,3];  dop = xyz[:,4];  rng = xyz[:,5]

            rd_count = int(result.rd_det.sum()) if result.rd_det is not None else 0
            ratio    = len(xyz) / max(rd_count, 1)
            x_span   = float(x.max() - x.min())
            y_span   = float(y.max() - y.min())

            print(f"  {fn:>5}  {len(xyz):>4}  {rd_count:>4}  "
                  f"{x.min():>+7.3f}  {x.max():>+7.3f}  {x_span:>7.3f}  "
                  f"{y.min():>7.3f}  {y.max():>7.3f}  {y_span:>7.3f}  "
                  f"{z.mean():>+7.3f}  {abs(dop).max():>8.3f}  {ratio:>6.2f}")

            all_rows.append((x, y, z, dop, rng))
            per_frame.append({
                'fn': fn, 'pts': len(xyz), 'rd': rd_count,
                'x_span': x_span, 'y_span': y_span,
                'z_mean': float(z.mean()), 'dop_max': float(abs(dop).max()),
                'ratio': ratio,
            })

            # Verbose point-by-point for explicitly requested frames
            if verbose_frames and fn in verbose_frames:
                print(f"    Frame {fn} — all {len(xyz)} points:")
                for i, row in enumerate(xyz):
                    zc = row[2] + radar_height
                    print(f"      pt{i:2d}: x={row[0]:+.3f}m  y={row[1]:.3f}m  "
                          f"z={zc:+.3f}m  dop={row[4]:+.4f}m/s  "
                          f"rng={row[5]:.3f}m  az={row[6]:+.2f}°  pwr={row[3]:.1f}dB")

        except Exception as e:
            print(f"  {fn:>5}  ERROR: {e}")

    if not per_frame:
        print("\n  No detections in selected frames.")
        return

    # ── Aggregate statistics ──────────────────────────────────────────────────
    all_x   = np.concatenate([r[0] for r in all_rows])
    all_y   = np.concatenate([r[1] for r in all_rows])
    all_z   = np.concatenate([r[2] for r in all_rows])
    all_dop = np.concatenate([r[3] for r in all_rows])

    print()
    print("─" * 72)
    print("  AGGREGATE STATISTICS")
    print("─" * 72)
    print(f"  Frames with detections : {len(per_frame)}")
    print(f"  Total cloud points     : {sum(r['pts'] for r in per_frame)}")
    print()
    print(f"  X lateral  : "
          f"mean={all_x.mean():+.3f}m  std={all_x.std():.3f}m  "
          f"range=[{all_x.min():+.3f}, {all_x.max():+.3f}]m")
    print(f"  Y forward  : "
          f"mean={all_y.mean():.3f}m   std={all_y.std():.3f}m  "
          f"range=[{all_y.min():.3f}, {all_y.max():.3f}]m")
    print(f"  Z elevation: "
          f"mean={all_z.mean():+.3f}m  std={all_z.std():.3f}m  "
          f"range=[{all_z.min():+.3f}, {all_z.max():+.3f}]m")
    print(f"  Doppler    : "
          f"mean={all_dop.mean():+.4f}  std={all_dop.std():.4f}  "
          f"range=[{all_dop.min():+.4f}, {all_dop.max():+.4f}]m/s")

    x_spans = [r['x_span'] for r in per_frame]
    y_spans = [r['y_span'] for r in per_frame]
    ratios  = [r['ratio']  for r in per_frame]

    print()
    print("─" * 72)
    print("  PER-FRAME DISTRIBUTION SUMMARY")
    print("─" * 72)
    print(f"  X span per frame : mean={np.mean(x_spans):.3f}m  "
          f"median={np.median(x_spans):.3f}m  "
          f"max={np.max(x_spans):.3f}m")
    print(f"  Y span per frame : mean={np.mean(y_spans):.3f}m  "
          f"median={np.median(y_spans):.3f}m  "
          f"max={np.max(y_spans):.3f}m")
    print(f"  az/RD ratio      : mean={np.mean(ratios):.2f}  "
          f"max={np.max(ratios):.2f}  "
          f"(>1.5 = multi-peak firing)")
    print(f"  Shoulder width   : 0.45m  "
          f"| frames with x_span > 0.30m: "
          f"{sum(1 for s in x_spans if s > 0.30)} / {len(x_spans)}")
    print(f"  Body depth       : 0.30m  "
          f"| frames with y_span < 0.40m: "
          f"{sum(1 for s in y_spans if s < 0.40)} / {len(y_spans)}")

    print()
    print("─" * 72)
    print("  BODY-SHAPE VERDICT")
    print("─" * 72)
    frac_wide   = sum(1 for s in x_spans if s > 0.20) / len(x_spans)
    frac_deep   = sum(1 for s in y_spans if s < 0.40) / len(y_spans)
    frac_multi  = sum(1 for r in ratios if r > 1.5)   / len(ratios)
    z_neg       = (all_z < -0.1).mean()

    if frac_wide >= 0.5:
        print(f"  ✓ LATERAL SPREAD: {frac_wide*100:.0f}% of frames have x_span > 20cm")
        print(f"    Body-width information IS present in the cloud")
    else:
        print(f"  ✗ LATERAL SPREAD: only {frac_wide*100:.0f}% of frames have x_span > 20cm")
        print(f"    Body-width information is MISSING — step 2 (prom=6dB, margin=3dB) needed")

    if frac_multi >= 0.2:
        print(f"  ✓ AZ MULTI-PEAK:  {frac_multi*100:.0f}% of frames have az/RD ratio > 1.5")
    else:
        print(f"  ✗ AZ MULTI-PEAK:  {frac_multi*100:.0f}% — multi-peak barely firing")
        print(f"    Reduce az_peak_min_prom_db from 10 to 6")

    if z_neg > 0.3:
        print(f"  ✗ ELEVATION: {z_neg*100:.0f}% of points have z < -0.1m")
        print(f"    Radar is mounted elevated. Use --radar-height to correct.")
        print(f"    Suggested: --radar-height {abs(all_z.mean()):.2f}")
    else:
        print(f"  ✓ ELEVATION: z values centred near {all_z.mean():+.3f}m (±{all_z.std():.3f}m)")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Body-shape geometry diagnostic')
    p.add_argument('--all',          action='store_true',
                   help='Run all 342 detected frames (slow)')
    p.add_argument('--frames',       nargs='+', type=int, default=None,
                   help='Specific frame numbers to process')
    p.add_argument('--radar-height', type=float, default=0.0, dest='radar_height',
                   help='Radar mounting height in metres (shifts z by this value)')
    p.add_argument('--verbose',      nargs='*', type=int, default=None,
                   help='Print all point coordinates for these frame numbers')
    args = p.parse_args()

    if args.frames:
        target = args.frames
    elif args.all:
        import re as _re
        import os as _os
        _txt_path = _os.path.join(PROJECT_ROOT, 'capture', 'mod5b_bodyshape.txt')
        if _os.path.exists(_txt_path):
            _txt = open(_txt_path).read()
            target = [int(m.group(1)) for m in
                      _re.finditer(r'frame_(\d+)\.bin.*?cloud:([1-9]\d*)',_txt)
                      if int(m.group(2)) > 0]
        else:
            target = list(range(400))
    else:
        target = DIAGNOSTIC_SET

    verbose_frames = set(args.verbose) if args.verbose is not None else None
    run(target, radar_height=args.radar_height, verbose_frames=verbose_frames)
