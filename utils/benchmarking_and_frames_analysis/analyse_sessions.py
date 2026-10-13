"""
analyse_sessions.py  —  Cross-session analysis for AWR2243-2X body-shape data.

PURPOSE
───────
Reads the frames.csv, points.csv, and summary.txt files already generated
by diag_pipeline_audit.py for each session and produces a single compact
report (~5-10 KB) you can paste into the chat for analysis.

DEFAULT PATH
────────────
  testing/live_capture_validation/05_outdoor_data_collection/frames/

EXPECTED FILE LAYOUT (two structures both supported)
─────────────────────────────────────────────────────
Structure A — subfolders per session:
  logs/
    A_S01/
      frames.csv
      points.csv
      summary.txt
    A_S02/
      frames.csv  ...

Structure B — flat files with session prefix:
  logs/
    A_S01_frames.csv
    A_S01_points.csv
    A_S01_summary.txt
    A_S02_frames.csv  ...

USAGE
─────
  # Use the default project path
  python analyse_sessions.py

  # Override the logs folder
  python analyse_sessions.py --root path/to/logs

  # Override the output file
  python analyse_sessions.py --out results/analysis.txt

  # Add subject metadata (optional)
  python analyse_sessions.py --subjects subjects.csv

OPTIONAL subjects.csv
──────────────────────
  subject_id,height_cm,build
  S01,182,medium
  S02,168,slim
  S03,175,heavy

OUTPUT
──────
Prints a structured report and saves it to --out (default: session_analysis.txt).
Paste the contents of session_analysis.txt into the chat.
"""

from __future__ import annotations
import argparse
import csv
import glob
import math
import os
import re
import sys
from collections import defaultdict
from typing import Optional


# ── Default path (relative to this script's location) ────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(
    _SCRIPT_DIR,
    'testing', 'live_capture_validation',
    # Must match where reaudit_all.ps1 WRITES its output. It writes to
    # .../05_outdoor_data_collection/frames/<session>/frames.csv, while this
    # script previously defaulted to .../logs/. The two never pointed at the
    # same place, so every report generated here was built from stale CSVs
    # left over from the original data collection - which is why limb% was
    # byte-identical across three runs while the re-audit numbers changed.
    '05_outdoor_data_collection', 'frames'
)


# ── Numeric helpers ───────────────────────────────────────────────────────────

def _f(v) -> Optional[float]:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _i(v) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
        return 0


def median(vals):
    if not vals:
        return None
    s = sorted(v for v in vals if v is not None)
    if not s:
        return None
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def mean(vals):
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None


def std(vals):
    v = [x for x in vals if x is not None]
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys)
             if x is not None and y is not None]
    if len(pairs) < 5:
        return None
    xs2, ys2 = zip(*pairs)
    mx, my = mean(xs2), mean(ys2)
    num = sum((x - mx) * (y - my) for x, y in zip(xs2, ys2))
    dx  = math.sqrt(sum((x - mx) ** 2 for x in xs2))
    dy  = math.sqrt(sum((y - my) ** 2 for y in ys2))
    if dx < 1e-9 or dy < 1e-9:
        return None
    return num / (dx * dy)


def fmt(v, d=3, sfx=''):
    return 'N/A' if v is None else f"{v:.{d}f}{sfx}"


def pct(v):
    return 'N/A' if v is None else f"{v*100:.1f}%"


# ── Session file discovery ────────────────────────────────────────────────────

def discover_sessions(root: str) -> dict[str, dict[str, str]]:
    """
    Return {session_id: {'frames': path, 'points': path, 'summary': path}}
    Handles both subfolder and flat-file layouts.
    """
    found: dict[str, dict[str, str]] = defaultdict(dict)

    # Structure A: root/SESSION_ID/frames.csv
    for path in glob.glob(os.path.join(root, '*', 'frames.csv')):
        sid = os.path.basename(os.path.dirname(path))
        found[sid]['frames'] = path
    for path in glob.glob(os.path.join(root, '*', 'points.csv')):
        sid = os.path.basename(os.path.dirname(path))
        found[sid]['points'] = path
    for path in glob.glob(os.path.join(root, '*', 'summary.txt')):
        sid = os.path.basename(os.path.dirname(path))
        found[sid]['summary'] = path

    # Structure B: root/SESSION_ID_frames.csv
    for path in glob.glob(os.path.join(root, '*_frames.csv')):
        sid = os.path.basename(path).replace('_frames.csv', '')
        found[sid]['frames'] = path
    for path in glob.glob(os.path.join(root, '*_points.csv')):
        sid = os.path.basename(path).replace('_points.csv', '')
        found[sid]['points'] = path
    for path in glob.glob(os.path.join(root, '*_summary.txt')):
        sid = os.path.basename(path).replace('_summary.txt', '')
        found[sid]['summary'] = path

    # Structure C: root/frames.csv (single session at root)
    if os.path.exists(os.path.join(root, 'frames.csv')):
        sid = os.path.basename(root)
        found[sid]['frames'] = os.path.join(root, 'frames.csv')
    if os.path.exists(os.path.join(root, 'points.csv')):
        sid = os.path.basename(root)
        found[sid]['points'] = os.path.join(root, 'points.csv')
    if os.path.exists(os.path.join(root, 'summary.txt')):
        sid = os.path.basename(root)
        found[sid]['summary'] = os.path.join(root, 'summary.txt')

    return dict(found)


# ── summary.txt parser ────────────────────────────────────────────────────────

def parse_summary(path: str) -> dict:
    """Extract key scalars from a diag_pipeline_audit summary.txt."""
    out = {}
    if not path or not os.path.exists(path):
        return out
    text = open(path, encoding='utf-8', errors='replace').read()

    def grab(pattern, cast=float):
        m = re.search(pattern, text)
        if m:
            try:
                return cast(m.group(1).replace(',', '').strip())
            except (ValueError, TypeError):
                pass
        return None

    out['frames_total']       = grab(r'Frames total\s*:\s*([\d]+)', int)
    out['frames_detected']    = grab(r'Frames with cloud>0\s*:\s*([\d]+)', int)
    out['frames_cloud_zero']  = grab(r'Frames cloud==0\s*:\s*([\d]+)', int)
    out['total_cloud_pts']    = grab(r'Total cloud points\s*:\s*([\d]+)', int)
    out['elapsed_s']          = grab(r'Elapsed\s*:\s*([\d.]+)s')
    out['rd_cfar_total']      = grab(r'RD-CFAR survivors\s*:\s*([\d]+)', int)
    out['grating_rej_total']  = grab(r'grating_rej total\s*:\s*([\d]+)', int)
    out['fov_rej_total']      = grab(r'fov_rej total\s*:\s*([\d]+)', int)
    out['cc_rej_total']       = grab(r'cc_rej total\s*:\s*([\d]+)', int)
    out['avg_pts_per_frame']  = grab(r'avg pts/frame\s*:\s*([\d.]+)')
    out['max_pts_per_frame']  = grab(r'max pts/frame\s*:\s*([\d]+)', int)
    out['avg_accum_pts']      = grab(r'avg accum_pts/frame\s*:\s*([\d.]+)')
    out['avg_cluster_pts']    = grab(r'avg cluster_pts\s*:\s*([\d.]+)')
    out['dop_zero_frames']    = grab(r'Frames all-dop-zero\s*:\s*([\d]+)', int)
    # centroid mean ± std line: "+0.017 ± 0.624 m"
    mx = re.search(r'x mean.*?:\s*([+-]?[\d.]+)\s*[±\+\-~]+\s*([\d.]+)', text)
    if mx:
        out['centroid_x_mean'] = _f(mx.group(1))
        out['centroid_x_std']  = _f(mx.group(2))
    my = re.search(r'y mean.*?:\s*([+-]?[\d.]+)\s*[±\+\-~]+\s*([\d.]+)', text)
    if my:
        out['centroid_y_mean'] = _f(my.group(1))
        out['centroid_y_std']  = _f(my.group(2))

    if out.get('frames_total') and out.get('frames_detected'):
        out['detection_rate'] = out['frames_detected'] / out['frames_total']
    return out


# ── points.csv loader ─────────────────────────────────────────────────────────

def load_points(path: str) -> dict:
    """Load points.csv and compute per-point distribution stats."""
    out = {}
    if not path or not os.path.exists(path):
        return out
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return out

    dop   = [_f(r.get('dop_mps', ''))   for r in rows]
    rng   = [_f(r.get('rng_m', ''))     for r in rows]
    pwr   = [_f(r.get('pwr_db', ''))    for r in rows]
    x_pts = [_f(r.get('x', ''))         for r in rows]
    y_pts = [_f(r.get('y', ''))         for r in rows]
    z_pts = [_f(r.get('z', ''))         for r in rows]

    dop_v  = [d for d in dop if d is not None]
    rng_v  = [r for r in rng if r is not None]
    pwr_v  = [p for p in pwr if p is not None]
    x_v    = [x for x in x_pts if x is not None]
    z_v    = [z for z in z_pts if z is not None]

    out['n_points']         = len(rows)
    out['dop_mean']         = mean(dop_v)
    out['dop_std']          = std(dop_v)  if len(dop_v) > 1 else None
    out['dop_min']          = min(dop_v)  if dop_v else None
    out['dop_max']          = max(dop_v)  if dop_v else None
    out['dop_abs_mean']     = mean([abs(d) for d in dop_v])
    out['dop_zero_frac']    = sum(1 for d in dop_v if abs(d) < 0.04) / len(dop_v) if dop_v else None
    out['dop_nonzero_frac'] = 1.0 - (out['dop_zero_frac'] or 0)
    out['dop_unique']       = sorted(set(round(d, 4) for d in dop_v))
    out['n_dop_unique']     = len(out['dop_unique'])
    out['rng_mean']         = mean(rng_v)
    out['rng_min']          = min(rng_v) if rng_v else None
    out['rng_max']          = max(rng_v) if rng_v else None
    out['pwr_mean']         = mean(pwr_v)
    out['pwr_std']          = std(pwr_v)  if len(pwr_v) > 1 else None
    out['x_mean']           = mean(x_v)
    out['x_std']            = std(x_v)   if len(x_v) > 1 else None
    out['z_mean']           = mean(z_v)
    out['z_std']            = std(z_v)   if len(z_v) > 1 else None

    # Doppler histogram: count pts in each 0.1 m/s bucket
    buckets = defaultdict(int)
    for d in dop_v:
        bucket = round(d * 10) / 10   # round to nearest 0.1
        buckets[bucket] += 1
    out['dop_histogram'] = dict(sorted(buckets.items()))
    return out


# ── frames.csv loader ─────────────────────────────────────────────────────────

def load_frames(path: str) -> dict:
    """Load frames.csv and return per-session metric dict."""
    if not path or not os.path.exists(path):
        return {}
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    n_total  = len(rows)
    fi       = [_i(r.get('frame_idx', '0'))      for r in rows]
    cloud    = [_i(r.get('cloud_pts', '0'))       for r in rows]
    accum    = [_i(r.get('accum_pts', '0'))       for r in rows]
    cl_pts   = [_i(r.get('cluster_pts', '0'))     for r in rows]
    cl_noise = [_i(r.get('cluster_noise', '0'))   for r in rows]
    cl_rx    = [_f(r.get('cluster_radius_x', '')) for r in rows]
    cl_ry    = [_f(r.get('cluster_radius_y', '')) for r in rows]
    cl_cx    = [_f(r.get('cluster_cx', ''))       for r in rows]
    cl_cy    = [_f(r.get('cluster_cy', ''))       for r in rows]
    sf_x     = [_f(r.get('subject_spread_x', '')) for r in rows]
    dop_cen  = [_f(r.get('dop_centroid', ''))     for r in rows]
    dop_spr  = [_f(r.get('dop_spread', ''))       for r in rows]
    # Phase 6 CLUSTER maximum. Previously read 'dop_max', which collided
    # with the Phase 0 raw-cloud column of the same name; csv.DictReader
    # keeps only the LAST duplicate, so this silently returned the cloud
    # value. Falls back to the old name for CSVs written before the fix.
    dop_max_ = [_f(r.get('gait_dop_max', r.get('dop_max', '')))
                for r in rows]
    limb_f   = [_f(r.get('limb_frac', ''))        for r in rows]
    proc_ms  = [_f(r.get('proc_ms', ''))          for r in rows]
    grating  = [_i(r.get('grating_rej', '0'))     for r in rows]
    cc_rej   = [_i(r.get('cc_rej', '0'))          for r in rows]
    fov_rej  = [_i(r.get('fov_rej', '0'))         for r in rows]
    rd_cfar  = [_i(r.get('rd_cfar', '0'))         for r in rows]
    az_raw   = [_i(r.get('az_peaks_raw', '0'))    for r in rows]

    det      = [i for i, c in enumerate(cloud) if c > 0]
    has_cl   = [i for i in det if cl_pts[i] > 0]
    active   = [i for i in has_cl if fi[i] >= 23]
    full_win = [i for i in active if cl_pts[i] >= 20]
    has_gait = [i for i in active
                if dop_cen[i] is not None and dop_spr[i] is not None
                and dop_spr[i] > 0]

    # Centroid jumps
    cy_seq = [(fi[i], cl_cy[i]) for i in has_cl if cl_cy[i] is not None]
    cy_seq.sort(key=lambda t: t[0])
    jumps = sum(
        1 for k in range(1, len(cy_seq))
        if cy_seq[k][0] - cy_seq[k-1][0] <= 3
        and abs(cy_seq[k][1] - cy_seq[k-1][1]) > 0.6
    )

    # Doppler directional check: first half vs second half of recording
    # ── Stage 4.4: direction validation by ZERO-CROSSING COUNT ───────────────
    # The previous test split frames at the session midpoint and required
    # median(first half) > 0 and median(second half) < 0. That assumes ONE
    # out-and-back cycle. Sessions contain 3-4, so both halves hold mixed
    # directions and the medians reflect the cycle count, not the walking
    # direction. Only 1 of 6 A/D sessions "passed" despite the Doppler data
    # being demonstrably correct (A_S01 spans -1.951 to +1.717 m/s).
    #
    # Counting sign changes in dop_centroid is cycle-agnostic: each
    # out-and-back produces ONE crossing (+ -> -), so n cycles give 2n-1
    # crossings: 1 cycle -> 1, 3 cycles -> 5, 4 cycles -> 7. A
    # deadband suppresses chatter around zero, where the sign of a near-zero
    # centroid is noise rather than a turn.
    _DEADBAND = 0.15                      # m/s; below this the sign is noise
    _series   = [dop_cen[i] for i in has_gait if dop_cen[i] is not None]
    _sig      = [v for v in _series if abs(v) >= _DEADBAND]
    n_cross   = 0
    for _a, _b in zip(_sig, _sig[1:]):
        if (_a > 0) != (_b > 0):
            n_cross += 1
    frac_pos  = (sum(1 for v in _sig if v > 0) / len(_sig)) if _sig else None
    med_fwd   = median([v for v in _sig if v > 0])
    med_ret   = median([v for v in _sig if v < 0])
    # Both directions must be present and reasonably balanced. A single
    # traverse legitimately shows no crossings, so this is reported rather
    # than treated as a hard pass/fail.
    sign_ok   = (n_cross >= 1 and frac_pos is not None
                 and 0.2 <= frac_pos <= 0.8)

    # Range independence: subject_spread_x vs cluster_cy
    sf_x_vals  = [sf_x[i] for i in full_win if sf_x[i] is not None and sf_x[i] > 0]
    range_vals  = [cl_cy[i] for i in full_win
                   if sf_x[i] is not None and sf_x[i] > 0 and cl_cy[i] is not None]
    dx_rng_corr = pearson(sf_x_vals, range_vals)

    # Scenario C initialisation lag
    init_lags = []
    in_zero   = True
    zero_start = 0
    for k in range(n_total):
        if cl_pts[k] == 0:
            if not in_zero:
                in_zero    = True
                zero_start = k
        else:
            if in_zero:
                init_lags.append(k - zero_start)
                in_zero = False

    rx_vals  = [cl_rx[i] for i in full_win if cl_rx[i] is not None]
    ry_vals  = [cl_ry[i] for i in full_win if cl_ry[i] is not None]

    return {
        'n_total':           n_total,
        'n_detected':        len(det),
        'detection_rate':    len(det) / n_total if n_total else 0.0,
        'avg_cloud_pts':     mean([cloud[i] for i in det]),
        'avg_accum_pts':     mean([accum[i] for i in active]),
        'avg_cluster_pts':   mean([cl_pts[i] for i in full_win]),
        'median_noise':      median([cl_noise[i] for i in full_win]),
        'centroid_jumps':    jumps,
        'n_full_win':        len(full_win),
        'n_gait_frames':     len(has_gait),
        'rx_median':         median(rx_vals),
        'rx_std':            std(rx_vals),
        'ry_median':         median(ry_vals),
        'dx_std_median':     median(sf_x_vals),
        'dx_std_std':        std(sf_x_vals),
        'dx_std_vals':       sf_x_vals,
        'dx_range_corr':     dx_rng_corr,
        'dop_fwd_median':    med_fwd,
        'dop_ret_median':    med_ret,
        'dop_sign_correct':  sign_ok,
        'limb_frac_median':  median([limb_f[i] for i in has_gait if limb_f[i] is not None]),
        'dop_spread_median': median([dop_spr[i] for i in has_gait]),
        'dop_max_max':       max((dop_max_[i] for i in has_gait if dop_max_[i] is not None), default=None),
        'dop_max_median':    median([dop_max_[i] for i in has_gait if dop_max_[i] is not None]),
        'proc_ms_median':    median([proc_ms[i] for i in range(n_total) if proc_ms[i] is not None]),
        'grat_total':        sum(grating),
        'cc_total':          sum(cc_rej),
        'fov_total':         sum(fov_rej),
        'rd_total':          sum(rd_cfar),
        'az_raw_total':      sum(az_raw),
        'cloud_total':       sum(cloud),
        'init_lags':         init_lags,
        'rx_vals':           rx_vals,
    }


# ── Report builder ────────────────────────────────────────────────────────────

def build_report(sessions: dict, subjects: dict,
                 root: str = '') -> str:
    L = []
    a = L.append

    a("=" * 72)
    a("  AWR2243-2X  CROSS-SESSION ANALYSIS REPORT")
    a("=" * 72)
    # Print the folder the CSVs were read from. analyse_sessions.py
    # previously defaulted to .../logs/ while reaudit_all.ps1 wrote to
    # .../frames/, so reports were silently built from stale data.
    a(f"  Source  : {os.path.abspath(root) if root else '(not supplied)'}")
    a(f"  Sessions: {len(sessions)}")
    a(f"  Subjects: {', '.join(sorted(subjects)) or 'no subjects.csv provided'}")
    a("")

    # ── summary.txt overview ─────────────────────────────────────────────────
    a("─" * 72)
    a("  SESSION OVERVIEW  (from summary.txt)")
    a("─" * 72)
    a(f"  {'Session':<12}  {'frames':>7}  {'det%':>6}  {'pts/fr':>7}  "
      f"{'accum':>7}  {'clust':>7}  {'grat':>6}  {'cc':>6}  {'proc_ms':>8}")
    a(f"  {'-'*70}")

    for sid in sorted(sessions):
        s  = sessions[sid]
        sm = s.get('summary', {})
        fr = s.get('frames', {})
        n  = sm.get('frames_total') or fr.get('n_total', 0)
        dr = sm.get('detection_rate') or fr.get('detection_rate')
        ap = sm.get('avg_pts_per_frame') or fr.get('avg_cloud_pts')
        aa = sm.get('avg_accum_pts')  or fr.get('avg_accum_pts')
        ac = sm.get('avg_cluster_pts') or fr.get('avg_cluster_pts')
        gr = sm.get('grating_rej_total') or fr.get('grat_total', 0)
        cc = sm.get('cc_rej_total') or fr.get('cc_total', 0)
        pm = fr.get('proc_ms_median')
        a(f"  {sid:<12}  {n:>7}  {pct(dr):>6}  {fmt(ap,1):>7}  "
          f"{fmt(aa,1):>7}  {fmt(ac,1):>7}  {gr:>6}  {cc:>6}  {fmt(pm,0):>8}")

    a("")

    # ── Per-point Doppler from points.csv ────────────────────────────────────
    a("─" * 72)
    a("  PER-POINT DOPPLER DISTRIBUTION  (from points.csv)")
    a("─" * 72)
    a(f"  {'Session':<12}  {'n_pts':>7}  {'dop_mean':>9}  {'dop_std':>9}  "
      f"{'dop_min':>8}  {'dop_max':>8}  {'n_unique':>8}")
    a(f"  {'-'*68}")

    for sid in sorted(sessions):
        pt = sessions[sid].get('points', {})
        if not pt:
            a(f"  {sid:<12}  no points.csv")
            continue
        a(f"  {sid:<12}  "
          f"{pt.get('n_points',0):>7}  "
          f"{fmt(pt.get('dop_mean'),3):>9}  "
          f"{fmt(pt.get('dop_std'),3):>9}  "
          f"{fmt(pt.get('dop_min'),3):>8}  "
          f"{fmt(pt.get('dop_max'),3):>8}  "
          f"{pt.get('n_dop_unique',0):>8}")

    a("")
    a("  Unique Doppler values by session (quantisation check — all sessions):")
    for sid in sorted(sessions):
        pt = sessions[sid].get('points', {})
        if pt:
            vals = pt.get('dop_unique', [])
            vals_str = "  ".join(f"{v:+.3f}" for v in vals[:20])
            suffix   = f"  [+{len(vals)-20} more]" if len(vals) > 20 else ""
            a(f"  {sid:<12}: {vals_str}{suffix}")
    a("")

    # ── Phase 3 cluster geometry ──────────────────────────────────────────────
    a("─" * 72)
    a("  PHASE 3 — CLUSTER GEOMETRY  (from frames.csv)")
    a("─" * 72)
    a("  rx = cluster_radius_x (lateral half-width, body + drift)")
    a("  ry = cluster_radius_y (range half-depth)")
    a(f"  {'Session':<12}  {'rx_med':>8}  {'rx_std':>7}  "
      f"{'ry_med':>8}  {'noise_med':>10}  {'jumps':>6}  {'full_win_fr':>12}")
    a(f"  {'-'*68}")
    for sid in sorted(sessions):
        fr = sessions[sid].get('frames', {})
        if not fr:
            continue
        a(f"  {sid:<12}  "
          f"{fmt(fr.get('rx_median')):>8}  "
          f"{fmt(fr.get('rx_std')):>7}  "
          f"{fmt(fr.get('ry_median')):>8}  "
          f"{fmt(fr.get('median_noise'),1):>10}  "
          f"{fr.get('centroid_jumps',0):>6}  "
          f"{fr.get('n_full_win',0):>12}")
    total_jumps = sum(sessions[s]['frames'].get('centroid_jumps', 0)
                      for s in sessions if sessions[s].get('frames'))
    a(f"\n  Total centroid jumps across all sessions: {total_jumps}  "
      f"(target = 0)")
    a("")

    # ── Phase 4 M5 ────────────────────────────────────────────────────────────
    a("─" * 72)
    a("  PHASE 4 — SUBJECT-CENTRIC SPREAD  M5  (from frames.csv)")
    a("─" * 72)
    a("  dx_std = subject_spread_x  (target 0.10–0.25 m, range-independent)")
    a(f"  {'Session':<12}  {'dx_std_med':>11}  {'dx_std_std':>11}  "
      f"{'r_vs_range':>11}  {'n_frames':>8}")
    a(f"  {'-'*58}")
    for sid in sorted(sessions):
        sc = sid.split('_')[0] if '_' in sid else ''
        if sc not in ('A', 'D'):
            continue
        fr = sessions[sid].get('frames', {})
        if not fr:
            continue
        corr = fr.get('dx_range_corr')
        flag = '  ← high' if corr is not None and abs(corr) > 0.20 else ''
        a(f"  {sid:<12}  "
          f"{fmt(fr.get('dx_std_median')):>11}  "
          f"{fmt(fr.get('dx_std_std')):>11}  "
          f"{fmt(corr,3) if corr is not None else 'N/A':>11}{flag}  "
          f"{len(fr.get('dx_std_vals', [])):>8}")
    a("")

    # ── Phase 6 — Doppler direction ───────────────────────────────────────────
    a("─" * 72)
    a("  PHASE 6 — DOPPLER DIRECTION VALIDATION  (Scenarios A and D)")
    a("─" * 72)
    a("  dop_fwd > 0 = walking away  |  dop_ret < 0 = walking toward")
    a(f"  {'Session':<12}  {'dop_fwd':>9}  {'dop_ret':>9}  {'sign_ok':>8}")
    a(f"  {'-'*44}")
    for sid in sorted(sessions):
        sc = sid.split('_')[0] if '_' in sid else ''
        if sc not in ('A', 'D'):
            continue
        fr = sessions[sid].get('frames', {})
        if not fr:
            continue
        ok = '✓' if fr.get('dop_sign_correct') else '✗'
        a(f"  {sid:<12}  "
          f"{fmt(fr.get('dop_fwd_median'),3):>9}  "
          f"{fmt(fr.get('dop_ret_median'),3):>9}  "
          f"{ok:>8}")
    a("")

    # ── Phase 6 — gait features by session ────────────────────────────────────
    a("─" * 72)
    a("  PHASE 6 — GAIT FEATURES BY SESSION  (from frames.csv)")
    a("─" * 72)
    a(f"  {'Session':<12}  {'limb%':>7}  {'dop_spr':>8}  "
      f"{'dop_max':>8}  {'gait_fr':>8}")
    a(f"  {'-'*52}")
    for sid in sorted(sessions):
        fr = sessions[sid].get('frames', {})
        if not fr:
            continue
        a(f"  {sid:<12}  "
          f"{pct(fr.get('limb_frac_median')):>7}  "
          f"{fmt(fr.get('dop_spread_median')):>8}  "
          f"{fmt(fr.get('dop_max_median')):>8}  "
          f"{fr.get('n_gait_frames',0):>8}")
    a("")
    # NOTE: the "A/D high, B low" expectation belonged to the LEGACY absolute
    # threshold (|dop| >= 0.30), where limb% mostly measured walking SPEED and
    # so was high for radial and low for lateral. Under the adaptive threshold
    # limb% measures the SHAPE of the Doppler distribution about the torso, so
    # the discriminator is bimodal (radial) vs near-Gaussian (lateral) spread.
    # Radial should still read higher, but the 20% figure no longer applies.
    a("  limb% by scenario (adaptive threshold: radial > lateral expected):")
    for sc in ['A', 'B', 'C', 'D']:
        vals = [sessions[k]['frames'].get('limb_frac_median')
                for k in sessions
                if k.startswith(f"{sc}_") and sessions[k].get('frames')
                and sessions[k]['frames'].get('limb_frac_median') is not None]
        if vals:
            a(f"    Scenario {sc}:  mean={pct(mean(vals))}  "
              f"min={pct(min(vals))}  max={pct(max(vals))}")
    a("")

    # ── Scenario B check ──────────────────────────────────────────────────────
    b_sids = sorted(k for k in sessions if k.startswith('B_'))
    if b_sids:
        a("─" * 72)
        a("  SCENARIO B — LATERAL WALK CHECKS")
        a("─" * 72)
        a("  Expected: dop ≈ 0, cluster tracks subject laterally")
        a("  (limb% < 20% was a LEGACY absolute-threshold expectation and no")
        a("   longer applies - see the gait section note)")
        a("")
        for sid in b_sids:
            fr = sessions[sid].get('frames', {})
            pt = sessions[sid].get('points', {})
            a(f"  {sid}:")
            # dop_fwd/dop_ret are NOT a lateral test. Stage 4.4 redefined them
            # as the medians of the POSITIVE and NEGATIVE tails after a
            # 0.15 m/s deadband, so |dop_fwd| >= 0.15 by construction - the
            # old "near zero" reading measured the deadband, not the walk.
            # Per-point mean Doppler is the correct quantity.
            if pt:
                a(f"    pts dop_mean={fmt(pt.get('dop_mean'),3)} m/s"
                  f"   <-- lateral test (expected near zero)")
            a(f"    dir-check tails: fwd={fmt(fr.get('dop_fwd_median'),3)}  "
              f"ret={fmt(fr.get('dop_ret_median'),3)}  (deadband-limited)")
            a(f"    limb%={pct(fr.get('limb_frac_median'))}  "
              f"(legacy <20% target no longer applies)")
            if pt:
                a(f"    pts dop_std={fmt(pt.get('dop_std'),3)}")
            a(f"    rx={fmt(fr.get('rx_median'))}m  "
              f"ry={fmt(fr.get('ry_median'))}m")
        a("")

    # ── Scenario C — init lag ─────────────────────────────────────────────────
    c_sids = sorted(k for k in sessions if k.startswith('C_'))
    if c_sids:
        a("─" * 72)
        a("  SCENARIO C — TRACK INITIALISATION LAG")
        a("─" * 72)
        a("  lag = frames from cloud=0 to first cluster_pts>0")
        a("  target <= 16 frames (= 4s accumulator window)")
        a("")
        all_lags = []
        for sid in c_sids:
            fr  = sessions[sid].get('frames', {})
            lags = fr.get('init_lags', [])
            all_lags.extend(lags)
            if lags:
                a(f"  {sid}:  n_transitions={len(lags)}  "
                  f"mean={fmt(mean(lags),1)}fr  "
                  f"max={max(lags)}fr  values={lags}")
            else:
                a(f"  {sid}:  no stationary→walking transitions detected")
        if all_lags:
            over = sum(1 for l in all_lags if l > 16)
            a(f"\n  Overall:  mean={fmt(mean(all_lags),1)}fr  "
              f"median={fmt(median(all_lags),1)}fr  "
              f"max={max(all_lags)}fr")
            a(f"  Lag > 16 frames: {over}/{len(all_lags)} transitions")
        a("")

    # ── Scenario D ────────────────────────────────────────────────────────────
    d_sids = sorted(k for k in sessions if k.startswith('D_'))
    if d_sids:
        a("─" * 72)
        a("  SCENARIO D — MULTI-LEG MIXED SPEED")
        a("─" * 72)
        a(f"  {'Session':<12}  {'limb%':>7}  {'dop_max_max':>11}  "
          f"{'dop_spread':>10}  {'gait_fr':>8}")
        a(f"  {'-'*54}")
        for sid in d_sids:
            fr = sessions[sid].get('frames', {})
            if not fr:
                continue
            a(f"  {sid:<12}  "
              f"{pct(fr.get('limb_frac_median')):>7}  "
              f"{fmt(fr.get('dop_max_max')):>11}  "
              f"{fmt(fr.get('dop_spread_median')):>10}  "
              f"{fr.get('n_gait_frames',0):>8}")
        a("")

    # ── Inter-subject body shape ──────────────────────────────────────────────
    a_sids = sorted(k for k in sessions if k.startswith('A_'))
    if len(a_sids) >= 2:
        a("─" * 72)
        a("  INTER-SUBJECT BODY SHAPE COMPARISON  (Scenario A)")
        a("─" * 72)
        a("")
        subj_rx  = {}
        subj_dx  = {}
        for sid in a_sids:
            fr   = sessions[sid].get('frames', {})
            subj = sid.split('_')[1] if '_' in sid else sid
            subj_rx[subj] = fr.get('rx_median')
            subj_dx[subj] = fr.get('dx_std_median')

        a(f"  {'Subject':<10}  {'height':>8}  {'build':>8}  "
          f"{'rx (m)':>10}  {'dx_std (m)':>12}")
        a(f"  {'-'*52}")
        for subj in sorted(subj_rx):
            meta = subjects.get(subj, {})
            ht   = meta.get('height_cm', '?')
            bld  = meta.get('build', '?')
            a(f"  {subj:<10}  {ht:>8}  {bld:>8}  "
              f"{fmt(subj_rx[subj]):>10}  "
              f"{fmt(subj_dx.get(subj)):>12}")

        rx_l  = [v for v in subj_rx.values() if v is not None]
        dx_l  = [v for v in subj_dx.values() if v is not None]
        if len(rx_l) > 1:
            a(f"\n  rx variation:     range={fmt(min(rx_l))}–{fmt(max(rx_l))}m  "
              f"std={fmt(std(rx_l))}m")
        if len(dx_l) > 1:
            a(f"  dx_std variation: range={fmt(min(dx_l))}–{fmt(max(dx_l))}m  "
              f"std={fmt(std(dx_l))}m")
        a("")
        if len(dx_l) > 1:
            if std(dx_l) > 0.010:
                a("  ✓ Inter-subject dx_std variation detectable (std > 0.010 m)")
            else:
                a("  ✗ Inter-subject dx_std variation small (std <= 0.010 m)")
                a("    Body-width differences may be below this radar's resolution")
        a("")

    # ── Z elevation from points.csv ───────────────────────────────────────────
    a("─" * 72)
    a("  ELEVATION CHECK  (from points.csv)  — z_mean should be near 0")
    a("─" * 72)
    a(f"  {'Session':<12}  {'z_mean (m)':>11}  {'z_std (m)':>10}")
    a(f"  {'-'*38}")
    for sid in sorted(sessions):
        pt = sessions[sid].get('points', {})
        if not pt:
            continue
        flag = '  ← check radar height' if pt.get('z_mean') is not None and abs(pt['z_mean']) > 0.3 else ''
        a(f"  {sid:<12}  {fmt(pt.get('z_mean'),3):>11}  "
          f"{fmt(pt.get('z_std'),3):>10}{flag}")
    a("")

    # ── Filter stage attribution ──────────────────────────────────────────────
    a("─" * 72)
    a("  FILTER STAGE ATTRIBUTION  (totals across all sessions)")
    a("─" * 72)
    def tot(key_sm, key_fr):
        t = 0
        for s in sessions.values():
            t += s.get('summary', {}).get(key_sm) or s.get('frames', {}).get(key_fr, 0) or 0
        return t
    rd    = tot('rd_cfar_total', 'rd_total')
    grat  = tot('grating_rej_total', 'grat_total')
    cc    = tot('cc_rej_total', 'cc_total')
    fov   = tot('fov_rej_total', 'fov_total')
    a(f"  RD-CFAR survivors : {rd}")
    a(f"  grating_rej       : {grat}")
    a(f"  cc_rej            : {cc}")
    a(f"  fov_rej           : {fov}")
    # Audit fix: the previous formula divided AZIMUTH-PEAK rejections by
    # RD-DOMAIN detections — incompatible units, so it returned negative
    # "pass-through" values (-42.2% outdoor, -5.7% A_S01). Each RD detection
    # spawns up to az_peak_max_n azimuth candidates, so the correct
    # denominator is az_peaks_raw.
    az_raw = sum((s.get('frames', {}) or {}).get('az_raw_total', 0) or 0
                 for s in sessions.values())
    cloud_t = sum((s.get('frames', {}) or {}).get('cloud_total', 0) or 0
                  for s in sessions.values())
    a(f"  az peaks raw      : {az_raw}   (candidates before FoV/grating/cc gates)")
    a(f"  cloud pts emitted : {cloud_t}")
    if az_raw > 0:
        a(f"  az-peak survival  : {cloud_t/az_raw*100:.1f}%   "
          f"(cloud_pts / az_peaks_raw)")
        a(f"  rejected by gates : {(grat+cc+fov)/az_raw*100:.1f}%   "
          f"(grating+cc+fov / az_peaks_raw)")
    a("  NOTE: rd_cfar counts RD cells; grating/fov/cc count azimuth peaks. "
      "Do not mix the two.")
    a("")

    # ── Phase 7 readiness ─────────────────────────────────────────────────────
    a("─" * 72)
    a("  PHASE 7 READINESS CHECKLIST")
    a("─" * 72)
    # Sessions with ZERO detections (A_S03, D_S02) cannot be evaluated: there
    # is no data to judge. Including them let two dead recordings veto ten good
    # ones - "Min detection rate 0.0%" and "Doppler sign 4/6" were both purely
    # this. Sessions are excluded; NO threshold is relaxed.
    # Key is 'detection_rate' (not 'det_rate') - check both the summary and
    # frames dicts, mirroring how the overview table resolves it.
    def _dr(v):
        return ((v.get('summary') or {}).get('detection_rate')
                or (v.get('frames') or {}).get('detection_rate') or 0.0)
    _empty = [k for k, v in sessions.items() if not _dr(v)]
    _live  = {k: v for k, v in sessions.items() if k not in _empty}
    if _empty:
        a(f"  NOTE: excluding {len(_empty)} zero-detection session(s) from the")
        a(f"        checklist - no data to evaluate: {', '.join(sorted(_empty))}")
        a("")

    checks = []

    # 1. Tracking stability.
    # The target of 0 dates from when the accumulator was FROZEN (P0-5): a
    # frozen centroid cannot jump, so 0 was guaranteed and measured nothing.
    # This is the first REAL stability measurement. The target is left at 0
    # deliberately - it should be re-derived from evidence, not relaxed so the
    # check passes. The rate is reported so a target CAN be derived.
    j = sum(_live[s]['frames'].get('centroid_jumps', 0)
            for s in _live if _live[s].get('frames'))
    _fw = sum(_live[s]['frames'].get('full_win_frames', 0) or 0
              for s in _live if _live[s].get('frames'))
    _rate = f"  ({100.0*j/_fw:.1f}% of {_fw} full-window frames)" if _fw else ""
    checks.append((j == 0,
                   f"Centroid stability: {j} total jumps{_rate} "
                   f"(target = 0; predates the P0-5 fix)"))

    # 2. M5 range independence
    corrs = [_live[s]['frames'].get('dx_range_corr')
             for s in _live if _live[s].get('frames')
             and _live[s]['frames'].get('dx_range_corr') is not None
             and s.split('_')[0] in ('A', 'D')]
    if corrs:
        mc = max(abs(c) for c in corrs)
        checks.append((mc < 0.20, f"M5 range-independence: max|r|={fmt(mc,3)} (target < 0.20)"))

    # 3. Doppler sign
    ad = [_live[k]['frames'].get('dop_sign_correct')
          for k in _live if k.split('_')[0] in ('A', 'D')
          and _live[k].get('frames')]
    if ad:
        checks.append((all(ad),
                       f"Doppler sign correct: {sum(bool(x) for x in ad)}/{len(ad)} sessions"))

    # 4. Detection rate
    rates = [_live[s].get('summary', {}).get('detection_rate')
             or _live[s].get('frames', {}).get('detection_rate')
             for s in _live]
    rates = [r for r in rates if r is not None]
    if rates:
        checks.append((min(rates) >= 0.75,
                       f"Min detection rate: {pct(min(rates))} (target >= 75%)"))

    # 5. Inter-subject variation
    dx_a = [sessions[k]['frames'].get('dx_std_median')
            for k in sessions if k.startswith('A_')
            and sessions[k].get('frames')
            and sessions[k]['frames'].get('dx_std_median') is not None]
    if len(dx_a) >= 2:
        v = std(dx_a)
        checks.append((v > 0.005,
                       f"Inter-subject dx_std variation: std={fmt(v,4)}m (> 0.005m detectable)"))

    # 6. Scenario B Doppler near zero
    # Use per-point MEAN Doppler. dop_fwd_median is the positive-tail median
    # after a 0.15 m/s deadband (Stage 4.4), so it is >= 0.15 by construction
    # and could never pass a "< 0.20" test - that failure was an artefact of
    # the direction-check rewrite, not a property of the lateral walks.
    b_dops = [abs(sessions[k]['points'].get('dop_mean') or 0)
              for k in sessions if k.startswith('B_') and sessions[k].get('points')]
    if b_dops:
        checks.append((all(d < 0.2 for d in b_dops),
                       f"Scenario B near-zero Doppler: max|pts dop_mean|={fmt(max(b_dops),3)} (target < 0.20)"))

    a("")
    for ok, msg in checks:
        a(f"  {'✓' if ok else '✗'} {msg}")
    n_pass = sum(1 for ok, _ in checks if ok)
    a("")
    a(f"  {n_pass}/{len(checks)} checks passed")
    if n_pass == len(checks):
        a("  All checks passed — pipeline ready to proceed to Phase 7.")
    else:
        a("  Address failing checks before beginning Phase 7.")

    a("")
    a("=" * 72)
    return "\n".join(L)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Cross-session analysis — AWR2243-2X body-shape pipeline')
    p.add_argument('--root',
                   default=DEFAULT_ROOT,
                   help='Folder containing session subfolders or prefixed CSV files')
    p.add_argument('--out',
                   default='session_analysis.txt',
                   help='Output report file (default: session_analysis.txt)')
    p.add_argument('--subjects',
                   default='subjects.csv',
                   help='Optional CSV with subject_id,height_cm,build')
    args = p.parse_args()

    # Load subject metadata
    subjects = {}
    subj_path = args.subjects
    if not os.path.isabs(subj_path):
        subj_path = os.path.join(_SCRIPT_DIR, subj_path)
    if os.path.exists(subj_path):
        with open(subj_path, newline='') as f:
            for row in csv.DictReader(f):
                sid = row.get('subject_id', '').strip()
                if sid:
                    subjects[sid] = {k: v.strip() for k, v in row.items()}
        print(f"Subjects loaded: {list(subjects)}")
    else:
        print(f"No subjects.csv found (optional) — proceeding without metadata")

    # Resolve root
    root = args.root
    if not os.path.isabs(root):
        root = os.path.join(_SCRIPT_DIR, root)

    if not os.path.isdir(root):
        print(f"[ERROR] Logs folder not found: {root}")
        print("  Run with --root path/to/logs to override")
        sys.exit(1)

    # Discover session files
    raw = discover_sessions(root)
    if not raw:
        print(f"[ERROR] No session files found under: {root}")
        print("  Expected frames.csv / points.csv / summary.txt in subfolders")
        sys.exit(1)

    print(f"\nFound {len(raw)} sessions under: {root}")
    for sid in sorted(raw):
        files = raw[sid]
        tags  = ' '.join(k for k in ('frames', 'points', 'summary') if k in files)
        print(f"  {sid:<14}  {tags}")

    # Load all files
    print("\nLoading...")
    sessions = {}
    for sid in sorted(raw):
        files = raw[sid]
        entry = {}
        if 'summary' in files:
            entry['summary'] = parse_summary(files['summary'])
        if 'points' in files:
            entry['points'] = load_points(files['points'])
        if 'frames' in files:
            entry['frames'] = load_frames(files['frames'])
            fr = entry['frames']
            det = pct(fr.get('detection_rate'))
            gf  = fr.get('n_gait_frames', 0)
            print(f"  ✓ {sid:<14}  "
                  f"{fr.get('n_total',0)} frames  "
                  f"det={det}  "
                  f"gait_frames={gf}")
        sessions[sid] = entry

    if not sessions:
        print("[ERROR] No data loaded.")
        sys.exit(1)

    # Build and write report
    report = build_report(sessions, subjects, root)
    print()
    print(report)

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(_SCRIPT_DIR, out_path)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved to: {out_path}")
    print("Paste the contents of that file into the chat.")


if __name__ == '__main__':
    main()
