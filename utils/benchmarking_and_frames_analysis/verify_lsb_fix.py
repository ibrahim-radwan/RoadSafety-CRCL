"""
verify_lsb_fix.py — confirm the slope-LSB correction landed, and only that.

WHAT IT CHECKS
    The slope appears in processing ONLY as the range-axis scale:

        range_res = c / (2 * slope * N / fs)        so  range_res ~ 1/slope
        rng_m     = bin_index * range_res

    Detection happens in BIN space. The FFT, CFAR and all bin-level decisions
    are untouched by a slope edit — only the metre label on each bin changes.
    So every rng_m must scale by slope_old/slope_new.

    Expected for SHORTRANGE 44.41 -> 44.6588:  ratio = 0.994429  (-0.5571%)

WHY THE RATIO IS NOT BIT-IDENTICAL  (and why v1 of this script was wrong)
    points.csv stores rng_m ROUNDED to a few decimal places — 4 in practice.
    Each value therefore carries a rounding error e in +/-0.5e-4 m, and

        ratio = (r + e_a)/(r + e_b)  ~  (r_a/r_b) * (1 + (e_a - e_b)/r)

    so the ratio inherits a scatter of (e_a - e_b)/r. At 5 m that is ~2e-5 —
    which was exactly the fixed tolerance v1 applied to the spread, making
    the test unmeetable before any real error existed. v1 checked the mean
    against theory but never asked whether its spread criterion was
    physically achievable.

    The fix: scatter from rounding falls INVERSELY with range. Multiplying
    each point's deviation by its own range removes the range dependence and
    leaves a quantity bounded by max|e_a - e_b| = one quantisation step. That
    bound is absolute and parameter-free, so it is a real test rather than a
    tuned threshold — and the 1/r signature becomes positive evidence that
    the residual IS rounding, not a processing difference.

USAGE
    python verify_lsb_fix.py --before debug_context\\lsb_before\\points.csv ^
                             --after  debug_context\\lsb_after\\points.csv

    # non-shortrange variants:
    python verify_lsb_fix.py --before ... --after ... --slope-old 8.883 --slope-new 8.9318
"""
from __future__ import annotations
import argparse, collections, csv, math, os, sys


def decimals(path, col='rng_m', sample=400):
    """Infer the stored precision of a column from its raw text."""
    best = 0
    with open(path, newline='', encoding='utf-8-sig') as f:
        for i, r in enumerate(csv.DictReader(f)):
            if i >= sample:
                break
            s = (r.get(col) or '').strip()
            if '.' in s:
                best = max(best, len(s.split('.', 1)[1].rstrip()))
    return best or 6


def load(path):
    if not os.path.exists(path):
        sys.exit(f"[ERROR] not found: {path}")
    by = collections.defaultdict(list)
    with open(path, newline='', encoding='utf-8-sig') as f:
        rd = csv.DictReader(f)
        if 'rng_m' not in (rd.fieldnames or []):
            sys.exit(f"[ERROR] {path} has no 'rng_m'. Columns: {rd.fieldnames}")
        for r in rd:
            try:
                by[int(r['frame_idx'])].append(float(r['rng_m']))
            except (TypeError, ValueError):
                pass
    for k in by:
        by[k].sort()
    return by


def main():
    ap = argparse.ArgumentParser(description='Verify the slope-LSB correction')
    ap.add_argument('--before', required=True)
    ap.add_argument('--after', required=True)
    ap.add_argument('--slope-old', type=float, default=44.41)
    ap.add_argument('--slope-new', type=float, default=44.6588)
    ap.add_argument('--mean-tol', type=float, default=1e-5,
                    help='allowed error on the MEAN ratio (default 1e-5)')
    a = ap.parse_args()

    expected = a.slope_old / a.slope_new
    dp = max(decimals(a.before), decimals(a.after))
    step = 10.0 ** (-dp)
    bound = step                      # max|e_a - e_b| for two values each +/-step/2

    print('=' * 78)
    print('  SLOPE-LSB CORRECTION CHECK')
    print('=' * 78)
    print(f'  before   : {a.before}')
    print(f'  after    : {a.after}')
    print(f'  slope    : {a.slope_old} -> {a.slope_new} MHz/us')
    print(f'  expected : every rng_m x {expected:.6f}  ({100*(expected-1):+.4f}%)')
    print(f'  rng_m stored to {dp} dp -> rounding floor +/-{step/2:.1e} m per value')
    print()

    B, A = load(a.before), load(a.after)
    common = sorted(set(B) & set(A))
    if not common:
        sys.exit('[ERROR] no frame_idx in common - same session?')

    same = [f for f in common if len(B[f]) == len(A[f])]
    diff = [f for f in common if len(B[f]) != len(A[f])]

    print(f'  frames before / after / common : {len(B)} / {len(A)} / {len(common)}')
    print(f'  same point count               : {len(same)}')
    print(f'  changed point count            : {len(diff)}'
          f'{"  (metre-gate boundary flips - expected)" if diff else ""}')
    if diff:
        print('    ' + ', '.join(str(x) for x in diff[:12]) + (' ...' if len(diff) > 12 else ''))
    print()

    pts = []                          # (rng_before, ratio)
    for fi in same:
        for rb, ra in zip(B[fi], A[fi]):
            if rb > 1e-9:
                pts.append((rb, ra / rb))
    if not pts:
        sys.exit('[ERROR] no comparable points')

    ratios = [r for _, r in pts]
    mean = sum(ratios) / len(ratios)
    sd = math.sqrt(sum((r - mean) ** 2 for r in ratios) / len(ratios))

    # range-normalised residual: (ratio - expected) * r  must sit inside +/-step
    resid = [(r_ - expected) * rng for rng, r_ in pts]
    worst = max(abs(x) for x in resid)
    rmin = min(rng for rng, _ in pts)
    rmax = max(rng for rng, _ in pts)

    # does the scatter fall as 1/r, as rounding requires?
    inv = [1.0 / rng for rng, _ in pts]
    dev = [abs(r_ - mean) for _, r_ in pts]
    mi, md = sum(inv) / len(inv), sum(dev) / len(dev)
    num = sum((i - mi) * (d - md) for i, d in zip(inv, dev))
    den = math.sqrt(sum((i - mi) ** 2 for i in inv) * sum((d - md) ** 2 for d in dev))
    corr = num / den if den else 0.0

    print('  ' + '-' * 74)
    print(f'  points compared   : {len(pts)}       range span {rmin:.2f} - {rmax:.2f} m')
    print(f'  ratio mean        : {mean:.8f}')
    print(f'  ratio std         : {sd:.2e}   (predicted from rounding: '
          f'{step/2/math.sqrt(3)*math.sqrt(2)/(sum(r for r,_ in pts)/len(pts)):.2e})')
    print(f'  mean vs expected  : {abs(mean-expected):.2e}  ({100*(mean/expected-1):+.5f}%)')
    print()
    print(f'  range-normalised residual |(ratio-expected) x r|')
    print(f'    worst           : {worst:.2e} m')
    print(f'    allowed         : {bound:.2e} m   (one quantisation step)')
    print(f'  corr(|dev|, 1/r)  : {corr:+.3f}   (positive = scatter is rounding)')
    print('  ' + '-' * 74)
    print()

    ok_mean = abs(mean - expected) < a.mean_tol
    ok_resid = worst <= bound

    print('=' * 78)
    print('  VERDICT')
    print('=' * 78)
    if ok_mean and ok_resid:
        print('  PASS. The mean ratio is the expected slope ratio, and every point')
        print('  sits within one CSV quantisation step of it once range is')
        print('  accounted for. The correction landed and changed nothing except')
        print('  the range-axis label.')
        if corr > 0.1:
            print(f'  The residual scatter falls as 1/range (corr {corr:+.2f}), which is')
            print('  the signature of decimal rounding rather than a real difference.')
        if diff:
            print(f'  ({len(diff)} frame(s) changed point count - expected where a point')
            print('   sat on a metre-expressed gate boundary.)')
        sys.exit(0)

    if not ok_mean:
        print(f'  FAIL - ratio {mean:.8f}, expected {expected:.8f}.')
        implied = a.slope_old / mean if mean else float('nan')
        print(f'  The "after" run behaves as though the slope were {implied:.4f} MHz/us.')
        if abs(implied - a.slope_old) < 1e-3:
            print('  That is the OLD value - the run did not read the edited file.')
            print('  The audit script defaults to firmware/r017m/cascade_r017m.cfg.')
            print('  Re-run with an explicit --cfg path to the file you actually changed.')
        else:
            print('  Check the digits written to the cfg: 925 x 0.0482797623 = 44.6588.')
    if not ok_resid:
        print(f'  FAIL - range-normalised residual {worst:.2e} m exceeds the {bound:.2e} m')
        print('  quantisation step, so the discrepancy is larger than rounding can')
        print('  explain. Something other than the slope differs between the runs:')
        print('  different frames, another cfg field, or a code change.')
    sys.exit(1)


if __name__ == '__main__':
    main()
