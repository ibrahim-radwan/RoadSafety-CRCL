"""
analyse_gate.py — measure what the ghost gate is actually seeing.

WHY
───
Post-P0-5 the accumulator holds only ~52% of its 16-frame window, i.e. it
re-seeds every ~8 frames. Whether that is genuine track breaks or an
over-tight `max_jump` cannot be decided by argument — it needs the
frame-to-frame centroid displacement distribution.

That distribution is computable from `centroid_x` / `centroid_y`, which
have been columns in frames.csv since Phase 0. NO RE-RUN IS NEEDED.

If the newer `comp_step_x` / `comp_residual_x` columns are present (added
2026-08-02) they are reported too — those decide whether lowering
`dbscan_eps` can do anything at all.

USAGE
─────
    python analyse_gate.py
    python analyse_gate.py --root <folder-containing-session-subfolders>
    python analyse_gate.py --max-jump 0.6 --eps 0.80

OUTPUT
──────
    gate_analysis.txt   (paste into the chat)
"""
from __future__ import annotations
import argparse, csv, glob, math, os, sys

# ── path defaults mirroring reaudit_all.ps1 ──────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(
    _HERE, 'testing', 'live_capture_validation',
    '05_outdoor_data_collection', 'frames')
OVERRIDES = {
    'outdoor_long': os.path.join(
        _HERE, 'testing', 'live_capture_validation',
        '06_outdoor_multiple_subjects'),
}


def _f(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _i(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def pct_of(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round(q / 100.0 * (len(s) - 1)))))
    return s[k]


def median(vals):
    return pct_of(vals, 50)


def mean(vals):
    return sum(vals) / len(vals) if vals else None


def fmt(v, d=3):
    return '  -  ' if v is None else f'{v:.{d}f}'


def analyse(csv_path, max_jump, eps):
    with open(csv_path, newline='', encoding='utf-8-sig') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None

    have_resid = 'comp_step_x' in rows[0]

    fi   = [_i(r.get('frame_idx', 0))        for r in rows]
    cx   = [_f(r.get('centroid_x', ''))      for r in rows]
    cy   = [_f(r.get('centroid_y', ''))      for r in rows]
    npts = [_i(r.get('cloud_pts', 0))        for r in rows]

    # frame-to-frame displacement of the RAW per-frame centroid.
    # This is exactly the quantity the ghost gate thresholds.
    dy, dxy, gaps = [], [], []
    for k in range(1, len(rows)):
        if cy[k] is None or cy[k - 1] is None:
            continue
        if fi[k] - fi[k - 1] != 1:      # only consecutive frames
            continue
        d_y = abs(cy[k] - cy[k - 1])
        dy.append(d_y)
        if cx[k] is not None and cx[k - 1] is not None:
            dxy.append(math.hypot(cx[k] - cx[k - 1], d_y))

    step  = [_f(r.get('comp_step_x', ''))     for r in rows] if have_resid else []
    resid = [_f(r.get('comp_residual_x', '')) for r in rows] if have_resid else []
    step  = [v for v in step  if v is not None and v > 0]
    resid = [v for v in resid if v is not None and v > 0]

    trips = sum(1 for v in dy if v > max_jump)
    return {
        'n_rows'      : len(rows),
        'n_pairs'     : len(dy),
        'dy_med'      : median(dy),
        'dy_p90'      : pct_of(dy, 90),
        'dy_p99'      : pct_of(dy, 99),
        'dy_max'      : max(dy) if dy else None,
        'trips'       : trips,
        'trip_pct'    : 100.0 * trips / len(dy) if dy else None,
        'pts_mean'    : mean([n for n in npts if n > 0]),
        'have_resid'  : have_resid,
        'step_med'    : median(step),
        'step_p90'    : pct_of(step, 90),
        'step_max'    : max(step) if step else None,
        'step_gt_eps' : (100.0 * sum(1 for v in step if v > eps) / len(step)) if step else None,
        'resid_med'   : median(resid),
        'resid_max'   : max(resid) if resid else None,
    }


def main():
    ap = argparse.ArgumentParser(description='Ghost-gate / residual analysis from existing frames.csv')
    ap.add_argument('--root', default=DEFAULT_ROOT)
    ap.add_argument('--out', default='gate_analysis.txt')
    ap.add_argument('--max-jump', type=float, default=0.6, dest='max_jump')
    ap.add_argument('--eps', type=float, default=0.80)
    a = ap.parse_args()

    found = {}
    for p in sorted(glob.glob(os.path.join(a.root, '*', 'frames.csv'))):
        found[os.path.basename(os.path.dirname(p))] = p
    for name, folder in OVERRIDES.items():
        p = os.path.join(folder, 'frames.csv')
        if os.path.exists(p):
            found[name] = p

    if not found:
        print(f'[ERROR] no frames.csv under {a.root}')
        sys.exit(1)

    L = []
    add = L.append
    add('=' * 92)
    add('  GHOST-GATE / COMPENSATION-RESIDUAL ANALYSIS')
    add(f'  max_jump = {a.max_jump} m     dbscan_eps = {a.eps} m')
    add('  computed from EXISTING frames.csv — no pipeline re-run')
    add('=' * 92)

    res = {}
    for name, path in found.items():
        try:
            r = analyse(path, a.max_jump, a.eps)
            if r:
                res[name] = r
        except Exception as exc:            # noqa: BLE001
            add(f'  {name}: ERROR {exc}')

    add('')
    add('  SECTION 1 — frame-to-frame centroid displacement (what the gate thresholds)')
    add('  ' + '-' * 88)
    add(f"  {'session':<14}{'pairs':>7}{'pts/f':>7}{'median':>9}{'p90':>8}{'p99':>8}{'max':>8}"
        f"{'>gate':>7}{'trip%':>8}")
    add('  ' + '-' * 88)
    for k in sorted(res):
        r = res[k]
        add(f"  {k:<14}{r['n_pairs']:>7}{fmt(r['pts_mean'],2):>7}{fmt(r['dy_med']):>9}"
            f"{fmt(r['dy_p90']):>8}{fmt(r['dy_p99']):>8}{fmt(r['dy_max']):>8}"
            f"{r['trips']:>7}{fmt(r['trip_pct'],1):>8}")

    all_trip = [r['trip_pct'] for r in res.values() if r['trip_pct'] is not None]
    add('')
    add(f"  Mean gate-trip rate: {fmt(mean(all_trip),1)}%")
    add('')
    add('  READING THIS:')
    add('    p90 well below max_jump  -> gate is correctly sized; trips are genuine breaks.')
    add('    p90 near/above max_jump  -> gate fires on ordinary motion+noise, not ghosts.')
    add('    Compare median against expected motion: 1.2 m/s at 4 fps = 0.30 m/frame.')

    add('')
    add('  SECTION 2 — compensation residual (decides whether lowering eps can help)')
    add('  ' + '-' * 88)
    any_resid = any(r['have_resid'] for r in res.values())
    if not any_resid:
        add('  comp_step_x / comp_residual_x NOT present in these CSVs.')
        add('  Deploy the updated accumulate.py + diag_pipeline_audit.py and re-run')
        add('  reaudit_all.ps1 to populate them. Section 1 above is unaffected.')
    else:
        add(f"  {'session':<14}{'step_med':>10}{'step_p90':>10}{'step_max':>10}"
            f"{'%>eps':>8}{'resid_med':>11}{'resid_max':>11}")
        add('  ' + '-' * 88)
        for k in sorted(res):
            r = res[k]
            if not r['have_resid']:
                add(f'  {k:<14}  (column absent)')
                continue
            add(f"  {k:<14}{fmt(r['step_med']):>10}{fmt(r['step_p90']):>10}"
                f"{fmt(r['step_max']):>10}{fmt(r['step_gt_eps'],1):>8}"
                f"{fmt(r['resid_med']):>11}{fmt(r['resid_max']):>11}")
        add('')
        add('  READING THIS:')
        add(f"    step_x is the largest gap between consecutive sorted point x-values.")
        add(f"    DBSCAN merges the cloud into ONE cluster iff every gap <= eps ({a.eps} m).")
        add('    If step_p90 << eps, lowering eps will NOT split the trail — the cloud is')
        add('    dense and eps is irrelevant. Fix compensation instead.')
        add('    If step_p90 ~ eps, lowering eps WILL break chains and reduce rx.')

    add('')
    add('=' * 92)
    add('  DECISION THIS ENABLES')
    add('=' * 92)
    add('  Q1  Is max_jump too tight?      -> Section 1: is p90 close to 0.6 m?')
    add('  Q2  Can eps reduce wide rx?     -> Section 2: is step_p90 close to eps?')
    add('  Q3  Is it motion or noise?      -> Section 1 median vs 0.30 m expected motion')

    out = a.out if os.path.isabs(a.out) else os.path.join(_HERE, a.out)
    txt = '\n'.join(L)
    print(txt)
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write(txt)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
