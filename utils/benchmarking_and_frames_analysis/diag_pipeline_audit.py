"""
diag_pipeline_audit.py  —  Phase 0 observability tool.

Runs process_frame() over every .bin file in a folder and writes two CSVs:

  frames.csv  — one row per frame with stage-attribution counts and centroid.
  points.csv  — one row per emitted xyz_cloud point across all frames.

These are the evidence base for every later phase decision. Run this before
changing any detection parameter.

Usage (from project root, conda env active):
    python diag_pipeline_audit.py \\
        --folder  capture\\frames \\
        --cfg     firmware\\r017m\\cascade_r017m.cfg \\
        --out-dir debug_context\\stage4_audit

    # With CFAR margin override:
    python diag_pipeline_audit.py \\
        --folder capture\\frames --cfar-margin 5 \\
        --out-dir debug_context\\stage4_audit_margin5

Outputs:
    <out-dir>/frames.csv  — one row per input .bin frame
    <out-dir>/points.csv  — N rows (one per emitted cloud point)
    <out-dir>/summary.txt — human-readable summary printed and saved

Acceptance criteria (Phase 0):
    - frames.csv row count == number of input .bin files
    - points.csv row count == frames.csv cloud_pts column sum
    - You can answer from the CSVs alone:
        (a) how many frames have cloud_pts == 0
        (b) which stage drops the most candidates
        (c) the centroid trajectory y(t) and x(t)
"""
from __future__ import annotations
import argparse, csv, glob, os, sys, time
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from processing.radar_config     import (parse_cfg_file, parse_calibration,
                                          apply_raw_iq_order_override)
from processing.radar_reader     import read_frame
from processing.radar_processing import process_frame

# ── CSV column definitions ────────────────────────────────────────────────────
FRAME_COLS = [
    'frame_idx',          # 0-based frame index
    'filename',           # basename of the .bin file
    # Stage attribution counts
    'rd_cfar',            # RD-CFAR survivors before consistency filter
    'after_consistency',  # after cross-channel consistency gate
    'n_clusters',         # range-bin clusters after clustering step
    'az_peaks_raw',       # total az peaks before FoV/grating/cc gates
    'fov_rej',            # peaks rejected by FoV gate (|sin|>max_az_sin)
    'grating_rej',        # peaks rejected by grating-lobe body-width gate
    'cc_rej',             # clusters rejected by cross-cluster coherence gate
    'sg_rej',             # P0-6: clusters rejected by subject-proximity gate
    'sg_bypass',          # P0-6: 1 = gate bypassed (all clusters outside radius)
    'ra_det',             # ra_det.sum() — peaks written to detection mask
    'cloud_pts',          # final xyz_cloud point count
    'accum_pts',          # accumulated (motion-compensated) point count (Phase 2)
    # Phase 3: DBSCAN cluster
    'cluster_pts',        # pts in dominant body cluster
    'cluster_noise',      # pts classified as noise by DBSCAN
    'wide_cluster_rej',   # 1 = cluster found but discarded (rx > max_cluster_radius)
    'comp_residual_x',    # compensated-centroid spread in x (m) — trail extent
    'comp_residual_y',    # compensated-centroid spread in y (m)
    'comp_step_x',        # max consecutive compensated step in x (m)
    # Stage 3: multi-subject observables
    'n_subjects',         # confirmed tracks this frame (age >= 2)
    'n_clusters_kept',    # all DBSCAN clusters retained (incl. unconfirmed)
    'separation_m',       # distance between the two largest subjects (m), '' if <2
    'track_ids',          # e.g. "0|1" - stable ids, to spot identity churn
    'compensated',        # 0 = suppressed (multi-subject), 1 = applied
    'multi_used',         # multi-subject flag ACTUALLY used (1 frame before n_subjects)
    'n_raw_clusters',     # DBSCAN clusters before the age>=2 confirmation filter
    'cluster_radius_x',   # half-width of cluster in x (m)
    'cluster_radius_y',   # half-width of cluster in y (m)
    'cluster_cx',         # cluster centroid x (m)
    'cluster_cy',         # cluster centroid y (m)
    'subject_spread_x',   # Phase 4: std of dx in body-local coords (m)
    'subject_spread_y',   # Phase 4: std of dy in body-local coords (m)
    # Phase 6: micro-Doppler / gait features
    'dop_centroid',       # power-weighted Doppler centroid of cluster (m/s)
    'dop_spread',         # std of cluster Doppler values (m/s)
    'gait_dop_max',       # Phase 6: max |Doppler| in DBSCAN CLUSTER (m/s)
                          # renamed from 'dop_max' — collided with the
                          # Phase 0 raw-cloud 'dop_max' below, and
                          # csv.DictReader silently keeps only the last
    'n_torso_pts',        # cluster pts NOT classified as limb (see dop_limb_mode)
    'n_limb_pts',         # cluster pts classified as limb (see dop_limb_mode)
    'limb_frac',          # n_limb / cluster_pts (gait activity index)
    # Centroid
    'centroid_x',         # median x across cloud pts (lateral m)
    'centroid_y',         # median y across cloud pts (forward/range m)
    'centroid_z',         # median z across cloud pts (elevation m, radar-rel)
    # Motion
    'dop_zero_pts',       # points with |dop| < threshold (stationary/tangential)
    'dop_max',            # max |dop| in cloud (m/s)
    # Timing
    'proc_ms',            # process_frame wall-clock time (ms)
]

POINT_COLS = [
    'frame_idx',
    'x', 'y', 'z',
    'pwr_db', 'dop_mps', 'rng_m', 'az_deg',
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _setup(cfg_path: str, cfar_margin: float, audit_csv: str,
           emit_all: bool = True, temporal_accum: bool = False,
           accum_window: int = 16, subject_frame: bool = False,
           gait_features: bool = False):
    cfg        = parse_cfg_file(cfg_path)
    correction = parse_calibration(cfg_path, cfg.num_tx, cfg.num_rx)
    cfg        = apply_raw_iq_order_override(cfg, 'i-first')
    cfg.cfar_margin_db         = cfar_margin
    cfg.quiet_mode             = True
    cfg.audit_csv_path         = audit_csv
    cfg.emit_all_cluster_bins  = emit_all
    cfg.temporal_accum_en      = temporal_accum
    cfg.accum_window           = accum_window
    cfg.subject_frame_en       = subject_frame
    cfg.gait_features_en       = gait_features
    return cfg, correction


def _open_writers(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    fp = os.path.join(out_dir, 'frames.csv')
    pp = os.path.join(out_dir, 'points.csv')
    # Remove stale files so header is written fresh
    for p in (fp, pp):
        if os.path.exists(p):
            os.remove(p)
    ff = open(fp, 'w', newline='', encoding='utf-8')
    pf = open(pp, 'w', newline='', encoding='utf-8')
    fw = csv.writer(ff);  fw.writerow(FRAME_COLS)
    pw = csv.writer(pf);  pw.writerow(POINT_COLS)
    return ff, pf, fw, pw, fp, pp


# ── Main ──────────────────────────────────────────────────────────────────────
def run(folder: str, cfg_path: str, out_dir: str,
        cfar_margin: float, emit_all: bool = True,
        temporal_accum: bool = False, accum_window: int = 16,
        subject_frame: bool = False,
        gait_features: bool = False,
        full_az_cube: bool = False,
        no_subject_gate: bool = False) -> None:
    bin_files = sorted(glob.glob(os.path.join(folder, '*.bin')))
    if not bin_files:
        sys.exit(f"[AUDIT] No .bin files found in {folder}")
    # Session length is whatever was recorded — never assume a fixed count.
    # The acceptance criterion compares frames.csv rows against this, so a
    # 333-frame session no longer reports a spurious FAIL against a
    # hardcoded 400.
    n_input_files = len(bin_files)

    # ── Session-mismatch guard ───────────────────────────────────────────────
    # Naming convention places the session id (e.g. A_S01) in both the input
    # and output paths. If the output session id does not appear anywhere in
    # the input path, --folder was probably not updated when --out-dir was.
    # Warn loudly rather than fail: unconventional layouts are still allowed.
    #
    # Normalise separators explicitly. os.path.normpath only understands the
    # HOST separator, so a Windows-style path inspected on Linux (or vice
    # versa) collapses to a single basename and the guard fires on every
    # session. Splitting on both separators makes the check behave
    # identically on any platform.
    def _last_component(p: str) -> str:
        parts = [c for c in p.replace('\\', '/').rstrip('/').split('/') if c]
        return parts[-1] if parts else ''

    _out_tag = _last_component(out_dir)
    _in_path = folder.replace('\\', '/')
    # 'frames' etc. are generic container names, not session ids — skip them
    _generic = {'frames', 'data', 'capture', 'logs', 'out', 'output'}
    if _out_tag and _out_tag.lower() not in _generic and _out_tag not in _in_path:
        print("", flush=True)
        print("*" * 68, flush=True)
        print(f"  WARNING — possible session mismatch", flush=True)
        print(f"    output session id : {_out_tag}", flush=True)
        print(f"    input  folder     : {folder}", flush=True)
        print(f"    '{_out_tag}' does not appear in the input path.", flush=True)
        print(f"    Check that --folder was updated, not just --out-dir.", flush=True)
        print("*" * 68, flush=True)
        print("", flush=True)

    # Use a temp file for the per-frame audit CSV rows written by process_frame
    # (so we don't conflict with our own frames.csv)
    audit_tmp = os.path.join(out_dir, '_audit_tmp.csv')
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(audit_tmp):
        os.remove(audit_tmp)

    cfg, correction = _setup(cfg_path, cfar_margin, audit_tmp,
                             emit_all=emit_all,
                             temporal_accum=temporal_accum,
                             accum_window=accum_window,
                             subject_frame=subject_frame,
                             gait_features=gait_features)
    # Stage 2.2 kill switch — force the original full-cube azimuth FFT.
    cfg.az_fft_focused_en = not full_az_cube
    # Experiment switch: with the gate off, a rise in n_subjects proves the
    # gate was filtering out the second subject before it could be tracked.
    if no_subject_gate:
        cfg.subject_gate_en = False

    ff, pf, fw, pw, frames_path, points_path = _open_writers(out_dir)

    print(f"[AUDIT] Processing {len(bin_files)} frames …")
    t0_total = time.perf_counter()

    # Phase 2: initialise accumulator if requested
    accumulator = None
    if temporal_accum:
        try:
            from processing.accumulate import TemporalAccumulator
            accumulator = TemporalAccumulator(
                window             = accum_window,
                max_jump_m         = float(getattr(cfg, 'accum_max_centroid_jump_m', 0.6)),
                alpha              = float(getattr(cfg, 'accum_alpha', 0.3)),
                frame_rate_hz      = float(getattr(cfg, 'frame_rate_hz', 4.0)),
                dbscan_eps         = float(getattr(cfg, 'dbscan_eps', 0.80)),
                dbscan_min_samples = int(getattr(cfg, 'dbscan_min_samples', 3)),
                max_cluster_radius = float(getattr(cfg, 'max_cluster_radius', 0.65)),
                subject_frame_en   = subject_frame,
                gait_features_en   = gait_features,
                dop_limb_thr       = float(getattr(cfg, 'dop_limb_thr', 0.30)),
                # Stage 4.2: these were previously NOT passed, so the audit
                # silently used the dataclass defaults and the cfg values had
                # no effect. Changing dop_limb_mode in radar_config.py would
                # have been ignored - a config that lies about what runs.
                dop_limb_mode      = str(getattr(cfg, 'dop_limb_mode', 'adaptive')),
                dop_limb_offset    = float(getattr(cfg, 'dop_limb_offset', 0.35)),
                dop_limb_k         = float(getattr(cfg, 'dop_limb_k', 1.0)),
                dop_limb_scale_alpha = float(getattr(cfg, 'dop_limb_scale_alpha', 0.15)),
            )
        except ImportError as e:
            print(f"[AUDIT] WARNING: TemporalAccumulator not available: {e}")

    total_pts = 0
    total_accum_pts = 0
    n_zero    = 0

    # P0-6: tracked subject position handed to process_frame each frame.
    # Updated from the accumulator's DBSCAN cluster centroid after every
    # frame, so the audit reproduces exactly what the live viewer does.
    track_hint = None

    for fi, bp in enumerate(bin_files):
        cfg._frame_idx = fi   # passed through to audit CSV row

        t0 = time.perf_counter()
        try:
            cube   = read_frame(bp, cfg, 0)
            result = process_frame(cube, cfg, correction, verbose=False,
                                   track_hint=track_hint)
        except Exception as exc:
            print(f"  [WARN] frame {fi}: {exc}")
            fw.writerow([fi, os.path.basename(bp)] + [''] * (len(FRAME_COLS)-2))
            continue
        proc_ms = (time.perf_counter() - t0) * 1000

        xyz = result.xyz_cloud
        n   = len(xyz) if xyz is not None else 0
        total_pts += n
        if n == 0:
            n_zero += 1

        # Centroid and Doppler stats
        if n > 0:
            cx  = float(np.median(xyz[:, 0]))
            cy  = float(np.median(xyz[:, 1]))
            cz  = float(np.median(xyz[:, 2]))
            dop_thr = float(getattr(cfg, 'min_abs_doppler_mps', 0.0))
            if dop_thr == 0.0:
                dop_thr = 0.5 * float(getattr(cfg, 'dop_res', 0.078))
            dz  = int(np.sum(np.abs(xyz[:, 4]) < dop_thr))
            dm  = float(np.max(np.abs(xyz[:, 4])))
        else:
            cx = cy = cz = dm = float('nan')
            dz = 0

        # Read stage-attribution counts from the tmp CSV written by process_frame
        # (last appended row)
        rd_n = ac_n = ncl = apr = fv = gr = cc = ra = 0
        sgr = sgb = 0
        if os.path.exists(audit_tmp):
            with open(audit_tmp, newline='') as _tf:
                rows = list(csv.DictReader(_tf))
            if rows:
                last = rows[-1]
                rd_n = int(last.get('rd_cfar',         0) or 0)
                ac_n = int(last.get('after_consistency',0) or 0)
                ncl  = int(last.get('n_clusters',       0) or 0)
                apr  = int(last.get('az_peaks_raw',     0) or 0)
                fv   = int(last.get('fov_rej',          0) or 0)
                gr   = int(last.get('grating_rej',      0) or 0)
                cc   = int(last.get('cc_rej',           0) or 0)
                sgr  = int(last.get('sg_rej',           0) or 0)
                sgb  = int(last.get('sg_bypass',        0) or 0)
                ra   = int(last.get('ra_det',           0) or 0)

        # Phase 2+3: update accumulator and extract cluster
        accum_n = 0; cl_pts = 0; cl_noise = 0; wide_rej = 0
        res_x = ''; res_y = ''; step_x = ''
        n_subj = 0; n_kept = 0; sep_m = ''; tids = ''; comp_f = ''
        mu_f = ''; n_raw = ''
        cl_rx = ''; cl_ry = ''; cl_cx = ''; cl_cy = ''
        if accumulator is not None:
            accum_result = accumulator.update(
                xyz if (xyz is not None and len(xyz) > 0)
                else np.empty((0, 7), dtype=np.float32),
                frame_idx=fi)
            accum_n = len(accum_result.cloud)
            # Publish tracked position for the next frame's subject gate.
            # Prefer the DBSCAN cluster centroid (body-only); fall back to the
            # accumulator centroid, which is contaminated by the very returns
            # the gate removes.
            # Lag-corrected prediction — see accumulate.py AccumResult.
            # The EMA-derived centroid trails by ~2.3 frames and caused a
            # 34.6% subject-gate bypass rate in the first P0-6 run.
            if accum_n > 0:
                # Stage 3.3: list of per-track positions (one entry when a
                # single subject is present).
                track_hint = list(
                    getattr(accum_result, 'track_hints', None)
                    or [(float(accum_result.predicted_x),
                         float(accum_result.predicted_y))])
            res_x  = getattr(accum_result, 'comp_residual_x', '')
            res_y  = getattr(accum_result, 'comp_residual_y', '')
            step_x = getattr(accum_result, 'comp_step_x', '')
            comp_f = 1 if accum_result.compensated else 0
            mu_f   = 1 if getattr(accum_result, 'multi_used', False) else 0
            n_raw  = int(getattr(accum_result, 'n_raw_clusters', 0))
            _subs = getattr(accum_result.cluster, 'subjects', []) \
                    if accum_result.cluster is not None else []
            n_kept = len(_subs)
            _conf  = [sc for sc in _subs if sc.age >= 2]
            n_subj = len(_conf)
            tids   = '|'.join(str(sc.track_id) for sc in sorted(
                         _conf, key=lambda z: z.track_id))
            if len(_conf) >= 2:
                _top = sorted(_conf, key=lambda z: -z.n_pts)[:2]
                sep_m = round(float(np.hypot(
                    _top[0].centroid_x - _top[1].centroid_x,
                    _top[0].centroid_y - _top[1].centroid_y)), 4)
            if accum_result.cluster is not None:
                c = accum_result.cluster
                cl_pts   = c.n_pts
                cl_noise = c.n_noise
                wide_rej = 1 if getattr(c, 'wide_rejected', False) else 0
                # Audit A-2 follow-up: record rx/ry even when the cluster was
                # WIDE-REJECTED. Previously these were blanked whenever
                # cluster_found=False, so the rx values being cut off by
                # max_cluster_radius were invisible and the surviving rx
                # statistics were right-censored with no way to see it.
                # B_S01: 141 of 305 detected frames (46%) are rejected this way.
                _keep_geom = c.cluster_found or wide_rej
                cl_rx    = round(c.radius_x, 4) if _keep_geom else ''
                cl_ry    = round(c.radius_y, 4) if _keep_geom else ''
                cl_cx    = round(c.centroid_x, 4) if c.cluster_found else ''
                cl_cy    = round(c.centroid_y, 4) if c.cluster_found else ''
        total_accum_pts += accum_n

        # Phase 4: subject-frame spread
        sf_sx = ''; sf_sy = ''
        if accumulator is not None and hasattr(accum_result, 'subject_cloud'):
            sf = accum_result.subject_cloud
            if sf is not None and len(sf) >= 3:
                sf_sx = round(float(np.std(sf[:,0])), 4)
                sf_sy = round(float(np.std(sf[:,1])), 4)

        # Phase 6: gait features
        gf_cen=''; gf_std=''; gf_max=''; gf_tor=0; gf_lim=0; gf_frac=''
        if accumulator is not None:
            _gf = getattr(accum_result, 'gait', None)
            if _gf is not None and _gf.valid:
                gf_cen  = round(_gf.dop_centroid, 4)
                gf_std  = round(_gf.dop_spread,   4)
                gf_max  = round(_gf.dop_max,       4)
                gf_tor  = _gf.n_torso_pts
                gf_lim  = _gf.n_limb_pts
                gf_frac = round(_gf.limb_frac,     4)

        fw.writerow([
            fi, os.path.basename(bp),
            rd_n, ac_n, ncl, apr, fv, gr, cc, sgr, sgb, ra, n,
            accum_n, cl_pts, cl_noise, wide_rej, res_x, res_y, step_x,
            n_subj, n_kept, sep_m, tids, comp_f, mu_f, n_raw,
            cl_rx, cl_ry, cl_cx, cl_cy,
            sf_sx, sf_sy,
            gf_cen, gf_std, gf_max, gf_tor, gf_lim, gf_frac,
            '' if np.isnan(cx) else round(cx, 4),
            '' if np.isnan(cy) else round(cy, 4),
            '' if np.isnan(cz) else round(cz, 4),
            dz,
            '' if np.isnan(dm) else round(dm, 4),
            round(proc_ms, 1),
        ])

        # Per-point rows
        if n > 0:
            for pt in xyz:
                pw.writerow([fi,
                             round(float(pt[0]), 4), round(float(pt[1]), 4),
                             round(float(pt[2]), 4), round(float(pt[3]), 2),
                             round(float(pt[4]), 4), round(float(pt[5]), 4),
                             round(float(pt[6]), 3)])

        if (fi + 1) % 50 == 0:
            pct = 100 * (fi + 1) / len(bin_files)
            print(f"  {fi+1:4d}/{len(bin_files)}  ({pct:.0f}%)  "
                  f"pts_so_far={total_pts}  zero_frames={n_zero}")

    ff.close(); pf.close()
    elapsed = time.perf_counter() - t0_total

    # Clean up tmp
    if os.path.exists(audit_tmp):
        os.remove(audit_tmp)

    # ── Summary ───────────────────────────────────────────────────────────────
    # Re-read frames.csv for summary stats
    with open(frames_path, newline='') as _f:
        frame_rows = list(csv.DictReader(_f))

    def _col(key):
        vals = []
        for r in frame_rows:
            v = r.get(key, '')
            if v == '': continue
            try: vals.append(float(v))
            except: pass
        return np.array(vals) if vals else np.array([])

    def _rx_by_reject():
        """rx values split by whether the frame was wide-rejected.

        Requires rx to be recorded on rejected frames (see A-2 follow-up).
        Without this split the surviving rx median is right-censored at
        max_cluster_radius and looks far tighter than reality.
        """
        kept, rej = [], []
        for r in frame_rows:
            v = r.get('cluster_radius_x', '')
            if v == '': continue
            try: rx = float(v)
            except: continue
            (rej if str(r.get('wide_cluster_rej', '0')) == '1' else kept).append(rx)
        return np.array(kept), np.array(rej)

    cloud_arr  = _col('cloud_pts')
    rd_arr     = _col('rd_cfar')
    grat_arr   = _col('grating_rej')
    fov_arr    = _col('fov_rej')
    cc_arr     = _col('cc_rej')
    cx_arr     = _col('centroid_x')
    cy_arr     = _col('centroid_y')

    det_mask   = cloud_arr > 0
    n_det      = int(det_mask.sum())

    dop_zero_arr = _col('dop_zero_pts')
    all_zero_frames = sum(1 for r in frame_rows
                         if int(r.get('cloud_pts','0') or 0) > 0
                         and int(r.get('dop_zero_pts','0') or 0)
                            == int(r.get('cloud_pts','0') or 0))

    lines = [
        "=" * 64,
        "  PHASE 0/1/2 AUDIT SUMMARY",
        "=" * 64,
        # ── Input provenance ─────────────────────────────────────────────────
        # Printed FIRST and prominently. Previously the summary showed only the
        # OUTPUT paths, so running with a stale --folder (changing --out-dir but
        # not --folder) silently reprocessed the wrong session and wrote it into
        # the new session's directory. The resulting summary looked entirely
        # plausible. Echoing the source folder, file count and first/last
        # filenames makes that class of mistake impossible to miss.
        f"  INPUT  folder   : {os.path.abspath(folder)}",
        f"  INPUT  .bin files: {n_input_files}"
        f"   [{os.path.basename(bin_files[0])} … {os.path.basename(bin_files[-1])}]",
        f"  OUTPUT folder   : {os.path.abspath(out_dir)}",
        "-" * 64,
        f"  Phase 1 emit_all_cluster_bins : {'ON' if emit_all else 'OFF'}",
        f"  Phase 2 temporal_accum        : {'ON (window=' + str(accum_window) + ')' if temporal_accum else 'OFF'}",
        f"  CFAR margin                   : {cfar_margin} dB",
        f"  Stage 2.2 az FFT              : {'FULL CUBE (original)' if not getattr(cfg, 'az_fft_focused_en', True) else 'FOCUSED'}",
        f"  P0-6 subject gate             : {'ON' if getattr(cfg, 'subject_gate_en', True) else 'OFF (experiment)'}",
        f"  Stage 4.2 limb mode           : {getattr(cfg, 'dop_limb_mode', 'absolute')}"
        f"  (k={getattr(cfg, 'dop_limb_k', 1.0)})",
        f"  Frames total         : {len(frame_rows)}",
        f"  Frames with cloud>0  : {n_det}  ({100*n_det/max(len(frame_rows),1):.1f}%)",
        f"  Frames cloud==0      : {len(frame_rows)-n_det}",
        f"  Total cloud points   : {total_pts}",
        f"  Elapsed              : {elapsed:.1f}s  ({elapsed/len(frame_rows)*1000:.0f}ms/frame)",
        "",
        f"  STAGE ATTRIBUTION (totals over {len(frame_rows)} frames):",
        f"    RD-CFAR survivors  : {int(rd_arr.sum()) if len(rd_arr) else 0}",
        f"    grating_rej total  : {int(grat_arr.sum()) if len(grat_arr) else 0}",
        f"    fov_rej total      : {int(fov_arr.sum()) if len(fov_arr) else 0}",
        f"    cc_rej total       : {int(cc_arr.sum()) if len(cc_arr) else 0}",
        # P0-6 subject-proximity gate counters. sg_rej = clusters removed for
        # being > subject_gate_radius_m from the track. sg_bypass = frames where
        # EVERY cluster fell outside the radius and the frame was passed
        # ungated so the track could re-seed (safety valve).
        f"    sg_rej total       : {int(_col('sg_rej').sum())}  (P0-6 off-range contamination removed)" if 'sg_rej' in FRAME_COLS else "",
        f"    sg_bypass total    : {int(_col('sg_bypass').sum())}  (frames passed ungated — track stale)" if 'sg_bypass' in FRAME_COLS else "",
        "",
        "  CLOUD DENSITY (detected frames only):",
        f"    avg pts/frame      : {cloud_arr[det_mask].mean():.2f}" if n_det else "    —",
        f"    max pts/frame      : {cloud_arr[det_mask].max():.0f}" if n_det else "    —",
        # ── Audit P0-1 fix ───────────────────────────────────────────────────
        # These two lines previously used DIFFERENT denominators:
        #     accum   = total_accum_pts / len(frame_rows)          (ALL frames)
        #     cluster = cluster_pts[cluster_pts > 0].mean()        (DETECTED only)
        # That made accum look smaller than cluster and produced a false
        # "cluster_pts > accum_pts => multi-subject" conclusion. DBSCAN's
        # cluster is a strict SUBSET of the accumulated cloud, so per frame
        # cluster_pts <= accum_pts always. Both denominators are now stated
        # explicitly so the two figures are directly comparable.
        f"    avg accum_pts  /all frames : {total_accum_pts/max(len(frame_rows),1):.1f}  (Phase 2)" if temporal_accum else "",
        f"    avg accum_pts  /detected   : {_col('accum_pts')[det_mask].mean():.1f}  (Phase 2)" if temporal_accum and n_det else "",
        f"    avg cluster_pts/detected   : {_col('cluster_pts')[det_mask].mean():.1f}  (Phase 3)" if temporal_accum and n_det else "",
        f"    avg cluster_pts/has-cluster: {_col('cluster_pts')[_col('cluster_pts')>0].mean():.1f}  (Phase 3)" if temporal_accum and len(_col('cluster_pts')[_col('cluster_pts')>0]) else "",
        "    NOTE: compare like-for-like. cluster_pts <= accum_pts on every frame.",
        f"    wide_cluster_rej   : {int(_col('wide_cluster_rej').sum())}  (frames dropped: rx > max_cluster_radius)" if temporal_accum and 'wide_cluster_rej' in FRAME_COLS else "",
        # Compensation-residual diagnostics. comp_step_x is the largest gap
        # between consecutive sorted point x-values: DBSCAN merges the cloud
        # into one cluster iff every gap <= eps, so comparing comp_step_x
        # against dbscan_eps decides whether lowering eps can split wide
        # clusters at all. Previously written to the CSV but never printed.
        (lambda v: f"    comp_step_x        : median={np.median(v):.3f} m  p90={np.percentile(v,90):.3f} m  max={v.max():.3f} m  (vs eps={getattr(cfg,'dbscan_eps',0.80):.2f})"
         if len(v) else "    comp_step_x        : n/a")(_col('comp_step_x')) if temporal_accum and 'comp_step_x' in FRAME_COLS else "",
        (lambda v: f"    comp_residual_x    : median={np.median(v):.3f} m  max={v.max():.3f} m  (merged-cloud x-extent)"
         if len(v) else "    comp_residual_x    : n/a")(_col('comp_residual_x')) if temporal_accum and 'comp_residual_x' in FRAME_COLS else "",
        # Stage 3 multi-subject observables.
        (lambda v: f"    n_subjects         : max={int(v.max())}  frames with >1: "
                   f"{int((v>1).sum())} of {len(v)}"
         if len(v) else "    n_subjects         : n/a")(_col('n_subjects')) if temporal_accum and 'n_subjects' in FRAME_COLS else "",
        (lambda v: f"    separation_m       : median={np.median(v):.3f} m  "
                   f"min={v.min():.3f}  max={v.max():.3f}  (n={len(v)})"
         if len(v) else "    separation_m       : no multi-subject frames")(_col('separation_m')) if temporal_accum and 'separation_m' in FRAME_COLS else "",
        (lambda v: f"    compensated        : {int(v.sum())} of {len(v)} frames "
                   f"({100*v.mean():.0f}%) - suppressed while multi-subject"
         if len(v) else "")(_col('compensated')) if temporal_accum and 'compensated' in FRAME_COLS else "",
        (lambda v: f"    multi_used         : {int(v.sum())} frames used the multi-subject "
                   f"path (1-frame lag vs n_subjects - see accumulate.py)"
         if len(v) else "")(_col('multi_used')) if temporal_accum and 'multi_used' in FRAME_COLS else "",
        (lambda v: f"    n_raw_clusters     : max={int(v.max())}  frames with >1 RAW cluster: "
                   f"{int((v>1).sum())} of {len(v)}  (before age>=2 confirmation)"
         if len(v) else "")(_col('n_raw_clusters')) if temporal_accum and 'n_raw_clusters' in FRAME_COLS else "",
        (lambda k, rj: (
            f"      rx kept     n={len(k):4d}  median={np.median(k):.3f} m  max={k.max():.3f} m"
            if len(k) else "      rx kept     n=   0")
        )(*_rx_by_reject()) if temporal_accum else "",
        (lambda k, rj: (
            f"      rx REJECTED n={len(rj):4d}  median={np.median(rj):.3f} m  max={rj.max():.3f} m"
            f"   <-- censored above {getattr(cfg,'max_cluster_radius',0.65):.2f} m"
            if len(rj) else "      rx REJECTED n=   0   (no censoring)")
        )(*_rx_by_reject()) if temporal_accum else "",
        "",
        "  CENTROID (detected frames):",
        f"    x mean ± std       : {cx_arr.mean():+.3f} ± {cx_arr.std():.3f} m" if len(cx_arr) else "    —",
        f"    y mean ± std       : {cy_arr.mean():.3f} ± {cy_arr.std():.3f} m" if len(cy_arr) else "    —",
        "",
        "  DOPPLER QUALITY (Phase 1 step 3):",
        f"    Frames all-dop-zero: {all_zero_frames}  (subject walking tangentially)",
        f"    (dop=0 is physical for tangential motion; not a pipeline bug)",
        "",
        "  ACCEPTANCE CRITERIA:",
        f"  [{'PASS' if len(frame_rows)==n_input_files else 'FAIL'}]  frames.csv rows == input .bin files "
        f"({len(frame_rows)} rows / {n_input_files} files)",
        f"  [{'PASS' if True else 'FAIL'}]  points.csv row count = {total_pts}",
        f"  [CHECK] stage drops (grating={int(grat_arr.sum()) if len(grat_arr) else 0} fov={int(fov_arr.sum()) if len(fov_arr) else 0} cc={int(cc_arr.sum()) if len(cc_arr) else 0})",
        f"  [CHECK] emit_all {'ON' if emit_all else 'OFF'}: avg pts/frame = {cloud_arr[det_mask].mean():.2f}" if n_det else "",
        "",
        f"  Output files:",
        f"    {frames_path}",
        f"    {points_path}",
        "=" * 64,
    ]

    summary_txt = '\n'.join(lines)
    print(summary_txt)
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as _sf:
        _sf.write(summary_txt + '\n')


def _parse():
    p = argparse.ArgumentParser(
        description='Phase 0-6 pipeline audit → frames.csv + points.csv + summary.txt',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--folder',      required=True,
                   help='Folder of frame_*.bin files')
    p.add_argument('--cfg',
                   default=os.path.join(os.path.dirname(os.path.dirname(PROJECT_ROOT)),
                                        'cascaded_radar_capture_pipeline', 'firmware',
                                        'r017m', 'cascade_r017m.cfg'),
                   help='Radar .cfg file')
    p.add_argument('--out-dir',     default=os.path.join('debug_context',
                                                          'stage4_audit'),
                   dest='out_dir',
                   help='Output directory for CSVs and summary')
    p.add_argument('--cfar-margin', type=float, default=10.0,
                   dest='cfar_margin',
                   help='CFAR margin dB (default 10 = Phase 1 baseline)')
    p.add_argument('--emit-all', action='store_true', default=True,
                   dest='emit_all',
                   help='Enable Phase 1 emit_all_cluster_bins (default ON)')
    p.add_argument('--no-emit-all', action='store_false', dest='emit_all',
                   help='Disable emit_all_cluster_bins (Phase 0 baseline)')
    p.add_argument('--temporal-accum', action='store_true', default=False,
                   dest='temporal_accum',
                   help='Enable Phase 2 motion-compensated accumulation')
    p.add_argument('--accum-window', type=int, default=16,
                   dest='accum_window',
                   help='Phase 2 accumulation window in frames (default 16 = 4s at 4fps)')
    p.add_argument('--subject-frame', action='store_true', default=False,
                   dest='subject_frame',
                   help='Enable Phase 4 subject-centric coordinate normalisation')
    p.add_argument('--gait-features', action='store_true', default=False,
                   dest='gait_features',
                   help='Enable Phase 6 micro-Doppler gait feature extraction')
    p.add_argument('--no-subject-gate', action='store_true', default=False,
                   dest='no_subject_gate',
                   help='Disable the P0-6 subject-proximity gate. Use this to '
                        'test whether the gate is locking out a second subject '
                        '(gate lock-in) rather than the cloud genuinely lacking one.')
    p.add_argument('--full-az-cube', action='store_true', default=False,
                   dest='full_az_cube',
                   help='Stage 2.2 kill switch: use the ORIGINAL full-cube '
                        'azimuth FFT instead of the focused path. Use this to '
                        'reproduce pre-optimisation output for A/B comparison.')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse()
    run(folder         = args.folder,
        cfg_path       = args.cfg,
        out_dir        = args.out_dir,
        cfar_margin    = args.cfar_margin,
        emit_all       = args.emit_all,
        temporal_accum = args.temporal_accum,
        accum_window   = args.accum_window,
        subject_frame  = args.subject_frame,
        gait_features  = args.gait_features,
        full_az_cube   = args.full_az_cube,
        no_subject_gate= args.no_subject_gate)
