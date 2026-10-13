"""
benchmark_pipeline.py — Per-stage latency benchmark for the AWR2243-2X pipeline.

Measures processing time at each DSP stage across all 400 recorded frames and
produces a stage-by-stage breakdown comparing:

  A) Current pipeline  (Mods 1-4 applied)
  B) Baseline az path  (full-cube az FFT — simulates pre-Mod4 cost for Stage 7/8)

Usage (from project root, activate conda env first):
    python -m processing.benchmark_pipeline ^
        --folder capture\frames ^
        --cfg firmware\r017m\cascade_r017m.cfg ^
        --cfar-margin 5

Output:
    benchmark_results.txt  — machine-readable TSV (frame × stage timings)
    benchmark_summary.txt  — human-readable summary table + verdict
"""

from __future__ import annotations
import argparse, os, sys, time, glob
from collections import defaultdict
from pathlib import Path
import numpy as np

# ── project imports ───────────────────────────────────────────────────────────
try:
    from processing.radar_config     import RadarConfig
    from processing.radar_reader     import read_frame, count_frames
    from processing.radar_processing import (
        process_frame,
        _az_fft, _win,
        range_fft, doppler_fft, elevation_fft,
        apply_calibration, zero_dc_notch,
        _power_db_reduce, cfar_2d,
        _cross_channel_consistency_mask,
        _prune_rd_sidelobes, _range_keep_mask,
        extract_xyz, form_virtual_arrays,
        _collapse_virtual_channels,
        _collapse_ra_from_fft,
        _ra_focus_mask_from_rd,
    )
    from processing.radar_config     import parse_cfg_file, parse_calibration, \
                                            apply_raw_iq_order_override
except ImportError as e:
    sys.exit(f"[BENCH] Import error — run from project root with conda env active.\n  {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _t() -> float:
    return time.perf_counter()

def _ms(start: float, end: float) -> float:
    return (end - start) * 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# Instrumented per-stage timer
# ─────────────────────────────────────────────────────────────────────────────
STAGE_NAMES = [
    "S1_calibration",
    "S2_range_fft",
    "S3_virtual_array",
    "S4_doppler_fft",
    "S5_rd_power",
    "S6_rd_cfar",
    "S6b_consistency",
    "S7_az_fft_full",        # full-cube az FFT (display RA map)
    "S8_pertarget_az_fft",   # per-target 1D az FFTs only
    "S8_baseline_az_fft",    # counterfactual: full-cube for detection (pre-Mod4)
    "S9_elevation_fft",
    "S10_extract_xyz",
    "TOTAL",
]


def _timed_process_frame(cube, cfg, correction, gpu_fns):
    """
    Re-implements the core of process_frame with per-stage timing.
    Returns (result_dict, timings_dict).
    Mirrors the exact computation path of process_frame post-Mod4.
    """
    t = {}
    N_r = cfg.n_range

    # Unpack GPU fn aliases
    _range_fft_fn   = gpu_fns['range_fft']
    _doppler_fft_fn = gpu_fns['doppler_fft']
    _az_fft_fn      = gpu_fns['az_fft']
    _power_db_fn    = gpu_fns['power_db']
    _elev_fft_fn    = gpu_fns['elev_fft']
    _cfar_2d_fn     = gpu_fns['cfar_2d']
    _focus_mask_fn  = gpu_fns['focus_mask']
    _collapse_ra_fn = gpu_fns['collapse_ra']

    t0_total = _t()

    # S1 Calibration
    t0 = _t()
    if correction is not None and cfg.apply_calibration:
        cube = apply_calibration(cube, correction)
    t['S1_calibration'] = _ms(t0, _t())

    # S2 Range FFT
    t0 = _t()
    range_cube = _range_fft_fn(cube, cfg)
    t['S2_range_fft'] = _ms(t0, _t())

    # S3 Virtual array
    t0 = _t()
    ch_az, ch_el, _ = cfg.build_channel_table()
    va_az, va_el = form_virtual_arrays(range_cube, cfg, ch_az, ch_el)
    from processing.radar_processing import _elevation_pair_indices
    bot_idx, top_idx = _elevation_pair_indices(ch_az, ch_el)
    t['S3_virtual_array'] = _ms(t0, _t())

    # S4 Doppler FFT + MTI + notch
    t0 = _t()
    rd_raw = _doppler_fft_fn(va_az, cfg, remove_clutter=False, use_window=True)
    rd_mti = _doppler_fft_fn(va_az, cfg, remove_clutter=cfg.clutter_removal, use_window=True)
    if bot_idx.size > 0 and top_idx.size > 0:
        rd_el_top = _doppler_fft_fn(va_el[:, :, top_idx], cfg,
                                    remove_clutter=cfg.clutter_removal, use_window=True)
        rd_el_bot = _doppler_fft_fn(va_az[:, :, bot_idx], cfg,
                                    remove_clutter=cfg.clutter_removal, use_window=True)
    else:
        rd_el_top = np.empty((cfg.doppler_fft_size or cfg.num_loops, N_r, 0), dtype=np.complex64)
        rd_el_bot = np.empty_like(rd_el_top)
    rd_mti = zero_dc_notch(rd_mti, cfg)
    rd_el_top = zero_dc_notch(rd_el_top, cfg)
    rd_el_bot = zero_dc_notch(rd_el_bot, cfg)
    t['S4_doppler_fft'] = _ms(t0, _t())

    # S5 RD power map (non-coherent sum — Mod1)
    t0 = _t()
    RD_mti = _power_db_fn(rd_mti[:, :N_r, :], axis=-1)
    from processing.radar_processing import _suppress_doppler_stripes
    RD_mti = _suppress_doppler_stripes(RD_mti, cfg)
    t['S5_rd_power'] = _ms(t0, _t())

    # S6 RD CFAR
    t0 = _t()
    if getattr(cfg, 'skip_cfar', False):
        rd_det = np.zeros(RD_mti.shape, dtype=bool)
        rd_thresh = np.full(RD_mti.shape, np.nan, dtype=np.float32)
    else:
        RD_mti_linear = 10.0 ** (RD_mti / 10.0)
        rd_det, rd_thresh_linear = _cfar_2d_fn(RD_mti_linear, cfg, force_caso=True)
        rd_thresh = 10.0 * np.log10(np.maximum(rd_thresh_linear, 1e-12))
        rd_det = _prune_rd_sidelobes(rd_det, RD_mti, cfg)
        rd_det &= _range_keep_mask(cfg, RD_mti.shape[1])[np.newaxis, :]
    t['S6_rd_cfar'] = _ms(t0, _t())

    # S6b Consistency mask (Mod2)
    t0 = _t()
    _k = int(getattr(cfg, 'consistency_k', 2))
    if _k > 0 and not getattr(cfg, 'skip_cfar', False):
        rd_det = _cross_channel_consistency_mask(rd_det, rd_mti[:, :N_r, :], _k)
    t['S6b_consistency'] = _ms(t0, _t())

    # S7 Full-cube az FFT (display RA map — always runs)
    t0 = _t()
    virt_pos   = rd_mti[:, :N_r, :]
    az_pos_arr = np.array([c["az"] for c in ch_az], dtype=np.float32)
    ra_focus_mask = _focus_mask_fn(RD_mti, rd_det, cfg)
    if getattr(cfg, "music_use_unique_aperture", True):
        virt_pos, az_pos_arr = _collapse_virtual_channels(virt_pos, az_pos_arr)
    az_cube = _az_fft_fn(virt_pos, cfg)
    RA = _collapse_ra_fn(az_cube, RD_mti, rd_det, cfg)[:N_r, :]
    t['S7_az_fft_full'] = _ms(t0, _t())

    # S8 Per-target 1D az FFT (Mod4 — detection path)
    t0 = _t()
    _n_va = virt_pos.shape[-1]
    _az_w = _win(cfg.az_window, _n_va, cfg.chebyshev_atten_db).astype(np.float32)
    ra_det = np.zeros(RA.shape, dtype=bool)
    n_confirmed = 0
    if not getattr(cfg, 'skip_cfar', False) and rd_det is not None and rd_det.sum() > 0:
        rd_confirmed_bins = np.unique(np.where(rd_det)[1])
        n_confirmed = len(rd_confirmed_bins)
        range_mask = _range_keep_mask(cfg, RA.shape[0])
        for r_i in rd_confirmed_bins:
            if r_i >= virt_pos.shape[1] or r_i >= RA.shape[0]: continue
            if not range_mask[r_i]: continue
            det_dops = np.flatnonzero(rd_det[:, r_i]) if r_i < rd_det.shape[1] else np.array([], dtype=int)
            if det_dops.size > 0:
                expand_idx = np.unique(np.clip(
                    (det_dops[:,None]+np.array([-1,0,1])).ravel(),
                    0, virt_pos.shape[0]-1))
                snap = virt_pos[expand_idx, r_i, :]
            else:
                col  = np.sum(np.abs(virt_pos[:,r_i,:])**2, axis=-1)
                snap = virt_pos[np.argmax(col):np.argmax(col)+1, r_i, :]
            sv = np.mean(snap, axis=0)
            az_spec = np.fft.fftshift(np.fft.fft(sv * _az_w, n=cfg.az_fft_size))
            ra_det[r_i, int(np.argmax(np.abs(az_spec)**2))] = True
    t['S8_pertarget_az_fft'] = _ms(t0, _t())

    # S8 BASELINE counterfactual: what would a second full-cube az FFT cost?
    # (pre-Mod4: argmax on RA[r_i,:] from a redundant full-cube call)
    t0 = _t()
    _az_cube_baseline = _az_fft_fn(virt_pos, cfg)   # the call Mod4 eliminated
    if not getattr(cfg, 'skip_cfar', False) and rd_det is not None and rd_det.sum() > 0:
        _RA_baseline = _collapse_ra_fn(_az_cube_baseline, RD_mti, rd_det, cfg)[:N_r, :]
        _ra_det_baseline = np.zeros(RA.shape, dtype=bool)
        _range_mask = _range_keep_mask(cfg, RA.shape[0])
        for r_i in np.unique(np.where(rd_det)[1]):
            if r_i >= _RA_baseline.shape[0] or not _range_mask[r_i]: continue
            _ra_det_baseline[r_i, int(np.argmax(_RA_baseline[r_i, :]))] = True
    t['S8_baseline_az_fft'] = _ms(t0, _t())

    # S9 Elevation FFT
    t0 = _t()
    RE = _elev_fft_fn(rd_el_top[:, :N_r, :], rd_el_bot[:, :N_r, :], cfg)
    t['S9_elevation_fft'] = _ms(t0, _t())

    # S10 xyz cloud
    t0 = _t()
    xyz = extract_xyz(ra_det, RA, RE, cfg.doppler_axis, rd_mti, rd_det, cfg)
    t['S10_extract_xyz'] = _ms(t0, _t())

    t['TOTAL'] = _ms(t0_total, _t())

    meta = {
        'rd_det_count': int(rd_det.sum()) if rd_det is not None else 0,
        'n_confirmed_bins': n_confirmed,
        'cloud_pts': len(xyz),
        'n_range': N_r,
        'n_az': cfg.az_fft_size,
    }
    return t, meta


# ─────────────────────────────────────────────────────────────────────────────
# GPU fn loader (mirrors process_frame's import block)
# ─────────────────────────────────────────────────────────────────────────────
def _load_gpu_fns():
    try:
        from processing.gpu_accel import (
            GPU_AVAILABLE,
            range_fft_gpu           as _rfft,
            doppler_fft_gpu         as _dfft,
            az_fft_gpu              as _azfft,
            power_db_reduce_gpu     as _pdb,
            elevation_fft_gpu       as _elfft,
            cfar_2d_gpu             as _cfar,
            ra_focus_mask_gpu       as _fmask,
            collapse_ra_from_fft_gpu as _cra,
        )
        gpu_ok = GPU_AVAILABLE
    except ImportError:
        from processing.radar_processing import (
            range_fft as _rfft, doppler_fft as _dfft,
            _az_fft as _azfft, _power_db_reduce as _pdb,
            elevation_fft as _elfft, cfar_2d as _cfar,
            _ra_focus_mask_from_rd as _fmask,
            _collapse_ra_from_fft as _cra,
        )
        gpu_ok = False
    return dict(range_fft=_rfft, doppler_fft=_dfft, az_fft=_azfft,
                power_db=_pdb, elev_fft=_elfft, cfar_2d=_cfar,
                focus_mask=_fmask, collapse_ra=_cra), gpu_ok


# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────
def _stats(vals):
    a = np.array(vals, dtype=np.float64)
    return dict(mean=float(np.mean(a)), median=float(np.median(a)),
                p10=float(np.percentile(a,10)), p90=float(np.percentile(a,90)),
                p99=float(np.percentile(a,99)), mn=float(np.min(a)), mx=float(np.max(a)),
                std=float(np.std(a)))


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark loop
# ─────────────────────────────────────────────────────────────────────────────
def run_benchmark(folder, cfg, correction, warmup=5, out_dir="."):
    bin_files = sorted(glob.glob(os.path.join(folder, "*.bin")))
    if not bin_files:
        sys.exit(f"[BENCH] No .bin files in {folder}")
    print(f"[BENCH] {len(bin_files)} frames found")

    gpu_fns, gpu_ok = _load_gpu_fns()
    print(f"[BENCH] GPU acceleration: {'YES' if gpu_ok else 'NO — CPU path'}")

    # GPU warm-up: process a few frames to fill CuPy memory pool
    print(f"[BENCH] Warming up ({warmup} frames)...")
    for bp in bin_files[:warmup]:
        cube = read_frame(bp, cfg, 0) if os.path.isfile(bp) else None
        if cube is not None:
            try:
                _timed_process_frame(cube, cfg, correction, gpu_fns)
            except Exception:
                pass
    print(f"[BENCH] Warm-up done. Starting timed run...")

    all_timings = defaultdict(list)
    all_meta    = []
    errors = 0

    for i, bp in enumerate(bin_files):
        if not os.path.isfile(bp): continue
        try:
            cube = read_frame(bp, cfg, 0)
            t, meta = _timed_process_frame(cube, cfg, correction, gpu_fns)
            for k, v in t.items():
                all_timings[k].append(v)
            all_meta.append(meta)
            if (i+1) % 50 == 0:
                avg_total = np.mean(all_timings['TOTAL'])
                print(f"  [{i+1:>3}/{len(bin_files)}]  avg so far: {avg_total:.1f} ms/frame")
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [WARN] frame {i}: {e}")

    n_frames = len(all_meta)
    print(f"[BENCH] Processed {n_frames} frames  ({errors} errors)")

    # ── Compute saving: Mod4 per-target vs baseline full-cube ─────────────────
    # S8_pertarget_az_fft  = new cost (Mod4)
    # S8_baseline_az_fft   = counterfactual cost (pre-Mod4 second full-cube call)
    # Saving per frame     = S8_baseline - S8_pertarget
    saving = [b - p for b, p in zip(all_timings['S8_baseline_az_fft'],
                                     all_timings['S8_pertarget_az_fft'])]

    # Confirmed bins per frame
    conf_bins = [m['n_confirmed_bins'] for m in all_meta]
    cloud_pts = [m['cloud_pts']        for m in all_meta]

    # ── Build report ──────────────────────────────────────────────────────────
    lines = []
    W = 70

    lines.append("=" * W)
    lines.append("  PIPELINE LATENCY BENCHMARK — AWR2243-2X (post-Mod1–4)")
    lines.append(f"  {n_frames} frames  |  GPU: {'YES' if gpu_ok else 'NO (CPU)'}  |  "
                 f"cfar_margin={cfg.cfar_margin_db} dB")
    lines.append("=" * W)

    # Per-stage table
    stages_display = [
        ("S1  Calibration",       "S1_calibration"),
        ("S2  Range FFT",         "S2_range_fft"),
        ("S3  Virtual array",     "S3_virtual_array"),
        ("S4  Doppler FFT + MTI", "S4_doppler_fft"),
        ("S5  RD power (sum)",    "S5_rd_power"),
        ("S6  RD CFAR",           "S6_rd_cfar"),
        ("S6b Consistency mask",  "S6b_consistency"),
        ("S7  Az FFT full-cube",  "S7_az_fft_full"),
        ("S8  Per-target az FFT", "S8_pertarget_az_fft"),
        ("S9  Elevation FFT",     "S9_elevation_fft"),
        ("S10 Extract xyz",       "S10_extract_xyz"),
    ]

    lines.append("")
    lines.append(f"  {'Stage':<28} {'mean ms':>8}  {'median':>8}  {'p90':>8}  {'p99':>8}  {'% total':>8}")
    lines.append(f"  {'-'*66}")
    mean_total = np.mean(all_timings['TOTAL'])
    for label, key in stages_display:
        if key not in all_timings: continue
        s = _stats(all_timings[key])
        pct = 100 * s['mean'] / mean_total if mean_total > 0 else 0
        lines.append(f"  {label:<28} {s['mean']:>8.2f}  {s['median']:>8.2f}  "
                     f"{s['p90']:>8.2f}  {s['p99']:>8.2f}  {pct:>7.1f}%")
    lines.append(f"  {'-'*66}")
    st = _stats(all_timings['TOTAL'])
    lines.append(f"  {'TOTAL':<28} {st['mean']:>8.2f}  {st['median']:>8.2f}  "
                 f"{st['p90']:>8.2f}  {st['p99']:>8.2f}  {'100.0%':>8}")

    # Mod4 saving block
    lines.append("")
    lines.append("─" * W)
    lines.append("  MOD 4 SAVING — per-target az FFT vs full-cube (baseline)")
    lines.append("─" * W)
    sv = _stats(saving)
    pt = _stats(all_timings['S8_pertarget_az_fft'])
    bl = _stats(all_timings['S8_baseline_az_fft'])
    cb = _stats(conf_bins)
    lines.append(f"  Avg confirmed range bins per frame   : {cb['mean']:>6.1f}  "
                 f"(out of {cfg.n_range} total = {100*cb['mean']/cfg.n_range:.1f}%)")
    lines.append(f"  Baseline full-cube az FFT cost        : {bl['mean']:>6.2f} ms avg")
    lines.append(f"  Per-target az FFT cost (Mod4)         : {pt['mean']:>6.2f} ms avg")
    lines.append(f"  Saving per frame                      : {sv['mean']:>6.2f} ms avg  "
                 f"({100*sv['mean']/bl['mean']:.0f}% reduction)" if bl['mean']>0 else "  n/a")
    lines.append(f"  Saving at p90                         : {sv['p90']:>6.2f} ms")
    lines.append(f"  Effective az FFT speedup factor       : "
                 f"{bl['mean']/pt['mean']:.1f}x" if pt['mean']>0 else "  n/a")

    # Latency distribution
    lines.append("")
    lines.append("─" * W)
    lines.append("  TOTAL LATENCY DISTRIBUTION")
    lines.append("─" * W)
    lines.append(f"  Mean   : {st['mean']:>7.2f} ms")
    lines.append(f"  Median : {st['median']:>7.2f} ms")
    lines.append(f"  p10    : {st['p10']:>7.2f} ms   (fastest 10% of frames)")
    lines.append(f"  p90    : {st['p90']:>7.2f} ms   (slowest 10% of frames)")
    lines.append(f"  p99    : {st['p99']:>7.2f} ms   (slowest 1%)")
    lines.append(f"  Min    : {st['mn']:>7.2f} ms")
    lines.append(f"  Max    : {st['mx']:>7.2f} ms")
    lines.append(f"  StdDev : {st['std']:>7.2f} ms")
    lines.append(f"  Throughput (mean): {1000/st['mean']:.2f} fps  "
                 f"(frame period = {st['mean']:.1f} ms vs hardware period = 256 ms)")

    # Bottleneck identification
    lines.append("")
    lines.append("─" * W)
    lines.append("  TOP 3 BOTTLENECK STAGES")
    lines.append("─" * W)
    stage_means = [(label, np.mean(all_timings[key]))
                   for label, key in stages_display if key in all_timings]
    stage_means.sort(key=lambda x: -x[1])
    for rank, (label, ms) in enumerate(stage_means[:3], 1):
        pct = 100 * ms / mean_total
        lines.append(f"  #{rank}  {label:<28}  {ms:>7.2f} ms  ({pct:.1f}% of total)")

    # Previous baseline reference
    lines.append("")
    lines.append("─" * W)
    lines.append("  REFERENCE — BASELINE (pre-Mod1, cfar_margin=10)")
    lines.append("─" * W)
    lines.append("  Avg frame latency (Windows/RTX 3080 Ti, from session logs):")
    lines.append("    Full pipeline (CFAR)  : ~270 ms/frame")
    lines.append("    --no-cfar mode        : ~200 ms/frame")
    lines.append("    WDDM period-3 pattern : ~380 / 280 / 185 ms cycle")
    lines.append(f"  Current avg            : {st['mean']:.1f} ms/frame")
    lines.append(f"  Change vs ~270 ms ref  : {st['mean']-270:+.1f} ms")

    lines.append("")
    lines.append("=" * W)

    report = "\n".join(lines)
    print("\n" + report)

    # ── Save files ────────────────────────────────────────────────────────────
    summary_path = os.path.join(out_dir, "benchmark_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\n[BENCH] Summary saved → {summary_path}")

    # TSV: frame × stage (for plotting)
    tsv_path = os.path.join(out_dir, "benchmark_results.txt")
    header = "frame\t" + "\t".join(k for _, k in stages_display) + "\tTOTAL\trd_det\tconfirmed_bins\tcloud_pts"
    rows = []
    n = len(all_meta)
    for i in range(n):
        vals = [str(i)]
        for _, k in stages_display:
            vals.append(f"{all_timings[k][i]:.4f}" if i < len(all_timings[k]) else "0")
        vals.append(f"{all_timings['TOTAL'][i]:.4f}")
        vals.append(str(all_meta[i]['rd_det_count']))
        vals.append(str(all_meta[i]['n_confirmed_bins']))
        vals.append(str(all_meta[i]['cloud_pts']))
        rows.append("\t".join(vals))
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write(header + "\n" + "\n".join(rows) + "\n")
    print(f"[BENCH] TSV data  saved → {tsv_path}")

    return all_timings, all_meta


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _build_parser():
    p = argparse.ArgumentParser(
        description="Per-stage pipeline latency benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--folder",     required=True,  help="Folder of .bin frame files")
    p.add_argument("--cfg",        required=False, help="Radar .cfg file path")
    p.add_argument("--cfar-margin",type=float, default=5.0, dest="cfar_margin",
                   help="CFAR margin dB (use same as production run)")
    p.add_argument("--warmup",     type=int, default=5,
                   help="Number of warm-up frames before timing")
    p.add_argument("--out-dir",    default=".", dest="out_dir",
                   help="Directory for output files")
    p.add_argument("--no-calib",   action="store_true",
                   help="Skip calibration")
    return p


def main():
    args = _build_parser().parse_args()

    if args.cfg:
        cfg        = parse_cfg_file(args.cfg)
        correction = parse_calibration(args.cfg, cfg.num_tx, cfg.num_rx,
                                       enabled=not args.no_calib)
    else:
        from processing.radar_config import shortrange_config
        print("[BENCH] No --cfg — using shortrange defaults")
        cfg        = shortrange_config()
        correction = np.ones((cfg.num_tx, cfg.num_rx), dtype=np.complex64)

    cfg = apply_raw_iq_order_override(cfg, "i-first")
    cfg.cfar_margin_db = args.cfar_margin
    cfg.quiet_mode     = True   # suppress per-frame diagnostic prints

    os.makedirs(args.out_dir, exist_ok=True)

    run_benchmark(args.folder, cfg, correction,
                  warmup=args.warmup, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
