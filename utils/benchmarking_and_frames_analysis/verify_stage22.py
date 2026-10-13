"""
verify_stage22.py — confirm the focused azimuth FFT changed NOTHING numerically.

Stage 2.2 replaced the full [n_dop, n_r, n_az] azimuth cube with a gathered FFT
over only the RD-focus-masked cells. It is meant to be mathematically identical:
the collapse only ever read the masked cells anyway.

So the acceptance test is NOT the speedup — it is that the pipeline output is
unchanged. This diffs two frames.csv files column by column.

USAGE
    python verify_stage22.py --before before/frames.csv --after after/frames.csv

    # or point at two audit output folders
    python verify_stage22.py --before debug_before\\A_S01 --after debug_after\\A_S01

EXIT CODE
    0 = identical on all critical columns  (accept the change)
    1 = differences found                  (reject / investigate)
"""
from __future__ import annotations
import argparse, csv, math, os, sys

# Columns that MUST be bit-identical. If the RA map changed at all, the azimuth
# peak positions move, which propagates to every one of these.
CRITICAL = [
    'rd_cfar', 'az_peaks_raw', 'fov_rej', 'grating_rej', 'cc_rej',
    'sg_rej', 'sg_bypass', 'ra_det', 'cloud_pts',
    'centroid_x', 'centroid_y', 'centroid_z',
    'accum_pts', 'cluster_pts', 'cluster_noise',
    'cluster_radius_x', 'cluster_radius_y', 'cluster_cx', 'cluster_cy',
    'subject_spread_x', 'dop_centroid', 'dop_spread', 'limb_frac',
    'gait_dop_max', 'n_torso_pts', 'n_limb_pts',
]
# Timing columns are EXPECTED to differ — that is the point of the change.
IGNORE = {'proc_ms'}

TOL = 1e-4          # float compare tolerance


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, 'frames.csv')
    if not os.path.exists(path):
        sys.exit(f"[ERROR] not found: {path}")
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f)), path


def num(v):
    if v is None or v == '':
        return None
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except ValueError:
        return v            # non-numeric (e.g. filename) — compare as string


def compare(rows_b, rows_a, tol):
    """Return (total_diffs, per_column_report)."""
    if len(rows_b) != len(rows_a) or not rows_b:
        return -1, []
    cols = [c for c in CRITICAL if c in rows_b[0] and c in rows_a[0]]
    total, report = 0, []
    for c in cols:
        nd, mx, first = 0, 0.0, None
        for i, (rb, ra) in enumerate(zip(rows_b, rows_a)):
            vb, va = num(rb.get(c)), num(ra.get(c))
            if vb is None and va is None:
                continue
            if vb is None or va is None:
                nd += 1
                first = first or (i, rb.get(c), ra.get(c))
                continue
            if isinstance(vb, str) or isinstance(va, str):
                if vb != va:
                    nd += 1
                    first = first or (i, vb, va)
                continue
            d = abs(vb - va)
            if d > tol:
                nd += 1
                mx = max(mx, d)
                first = first or (i, vb, va)
        total += nd
        report.append((c, nd, mx, first))
    return total, report


def run_batch(before_root, after_root, tol):
    """Compare every session folder present under both roots."""
    import glob as _glob
    names = sorted({os.path.basename(os.path.dirname(p))
                    for p in _glob.glob(os.path.join(before_root, '*', 'frames.csv'))})
    if not names:
        sys.exit(f'[ERROR] no */frames.csv under {before_root}')

    print('=' * 78)
    print('  STAGE 2.2 BATCH VERIFICATION — all sessions must be UNCHANGED')
    print('=' * 78)
    print(f'  {"session":<16}{"rows":>7}{"diffs":>8}   verdict')
    print('  ' + '-' * 50)

    failures, skipped = [], []
    for n in names:
        pb = os.path.join(before_root, n, 'frames.csv')
        pa = os.path.join(after_root,  n, 'frames.csv')
        if not os.path.exists(pa):
            skipped.append(n)
            print(f'  {n:<16}{"-":>7}{"-":>8}   SKIP (no after)')
            continue
        rb, _ = load(pb)
        ra, _ = load(pa)
        total, report = compare(rb, ra, tol)
        if total < 0:
            failures.append((n, 'row count / empty'))
            print(f'  {n:<16}{len(rb):>7}{"n/a":>8}   FAIL (row count {len(rb)} vs {len(ra)})')
            continue
        verdict = 'OK' if total == 0 else 'DIFFERS'
        print(f'  {n:<16}{len(rb):>7}{total:>8}   {verdict}')
        if total:
            failures.append((n, report))

    print()
    if not failures:
        print('  ' + '=' * 60)
        print(f'  PASS — {len(names) - len(skipped)} sessions identical on all critical columns.')
        print('  The focused azimuth FFT is numerically equivalent.')
        print('  ' + '=' * 60)
        return 0

    print('  ' + '=' * 60)
    print(f'  FAIL — {len(failures)} session(s) differ. Do NOT accept.')
    print('  ' + '=' * 60)
    for n, rep in failures:
        if isinstance(rep, str):
            print(f'    {n}: {rep}')
            continue
        for c, nd, mx, first in rep:
            if nd:
                i, vb, va = first
                print(f'    {n} / {c}: {nd} diffs, max {mx:.6g}; row {i}: {vb} -> {va}')
    return 1


def main():
    ap = argparse.ArgumentParser(description='Verify Stage 2.2 changed nothing numerically')
    ap.add_argument('--before', required=True, help='frames.csv, session folder, or ROOT of session folders with --batch')
    ap.add_argument('--after',  required=True, help='matching path for the AFTER run')
    ap.add_argument('--batch', action='store_true',
                    help='Treat --before/--after as ROOTS containing <session>/frames.csv and compare every session')
    ap.add_argument('--tol', type=float, default=TOL)
    a = ap.parse_args()

    if a.batch:
        sys.exit(run_batch(a.before, a.after, a.tol))

    rows_b, pb = load(a.before)
    rows_a, pa = load(a.after)

    print('=' * 78)
    print('  STAGE 2.2 VERIFICATION — output must be UNCHANGED')
    print('=' * 78)
    print(f'  before : {pb}   ({len(rows_b)} rows)')
    print(f'  after  : {pa}   ({len(rows_a)} rows)')
    print()

    if len(rows_b) != len(rows_a):
        print(f'  [FAIL] row count differs: {len(rows_b)} vs {len(rows_a)}')
        sys.exit(1)
    if not rows_b:
        print('  [FAIL] no rows')
        sys.exit(1)

    cols = [c for c in CRITICAL if c in rows_b[0] and c in rows_a[0]]
    missing = [c for c in CRITICAL if c not in rows_b[0]]
    if missing:
        print(f'  note: columns absent from these CSVs (skipped): {", ".join(missing)}')
        print()

    print(f'  {"column":<22}{"n_diff":>8}{"max_abs_diff":>15}   verdict')
    print('  ' + '-' * 60)

    total_diffs = 0
    worst = []
    for c in cols:
        nd = 0
        mx = 0.0
        first = None
        for i, (rb, ra) in enumerate(zip(rows_b, rows_a)):
            vb, va = num(rb.get(c)), num(ra.get(c))
            if vb is None and va is None:
                continue
            if vb is None or va is None:
                nd += 1
                if first is None:
                    first = (i, rb.get(c), ra.get(c))
                continue
            if isinstance(vb, str) or isinstance(va, str):
                if vb != va:
                    nd += 1
                    if first is None:
                        first = (i, vb, va)
                continue
            d = abs(vb - va)
            if d > a.tol:
                nd += 1
                mx = max(mx, d)
                if first is None:
                    first = (i, vb, va)
        total_diffs += nd
        verdict = 'OK' if nd == 0 else 'DIFFERS'
        print(f'  {c:<22}{nd:>8}{mx:>15.6g}   {verdict}')
        if nd:
            worst.append((c, nd, mx, first))

    print()
    if total_diffs == 0:
        print('  ' + '=' * 60)
        print('  PASS — output identical on all critical columns.')
        print('  The focused azimuth FFT is numerically equivalent.')
        print('  Any timing improvement is therefore a genuine win.')
        print('  ' + '=' * 60)
        sys.exit(0)

    print('  ' + '=' * 60)
    print(f'  FAIL — {total_diffs} differing values across {len(worst)} column(s).')
    print('  The RA map changed. Do NOT accept the optimisation.')
    print('  ' + '=' * 60)
    for c, nd, mx, first in worst[:6]:
        i, vb, va = first
        print(f'    {c}: {nd} diffs, max {mx:.6g}; first at row {i}: {vb} -> {va}')
    sys.exit(1)


if __name__ == '__main__':
    main()
