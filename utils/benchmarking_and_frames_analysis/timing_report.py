"""
timing_report.py — Stage 2.2 proc_ms A/B from CSVs already on disk.

Reads verify/fullcube/<session>/frames.csv and verify/focused/<session>/frames.csv
and reports the per-frame processing time for each path. Nothing is re-run.

Kept as a FILE rather than an inline here-string: piping a script into
`python -` from PowerShell requires `@'...'@ | python -`, and getting that
wrong silently drops into the interactive REPL instead of failing.

USAGE
    python timing_report.py
    python timing_report.py --root verify
"""
from __future__ import annotations
import argparse, csv, os, statistics as st, sys


def proc_ms(path):
    if not os.path.exists(path):
        return []
    vals = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            v = r.get('proc_ms', '')
            try:
                x = float(v)
                if x > 0:
                    vals.append(x)
            except (TypeError, ValueError):
                pass
    return vals


def detected(path):
    """Frames with cloud_pts > 0 — mask density scales with detections."""
    if not os.path.exists(path):
        return 0, 0
    n = d = 0
    with open(path, newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            n += 1
            try:
                if int(r.get('cloud_pts', 0) or 0) > 0:
                    d += 1
            except ValueError:
                pass
    return d, n


def pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round(q / 100.0 * (len(s) - 1)))))
    return s[k]


def main():
    ap = argparse.ArgumentParser(description='Stage 2.2 proc_ms A/B')
    ap.add_argument('--root', default='verify')
    a = ap.parse_args()

    fc = os.path.join(a.root, 'fullcube')
    fo = os.path.join(a.root, 'focused')
    if not os.path.isdir(fc) or not os.path.isdir(fo):
        sys.exit(f'[ERROR] expected {fc} and {fo}')

    names = sorted(os.listdir(fc))

    print('=' * 84)
    print('  STAGE 2.2 — proc_ms A/B  (identical input, same binary)')
    print('=' * 84)
    print(f"  {'session':<15}{'det%':>7}{'full med':>10}{'foc med':>9}"
          f"{'saved':>8}{'speedup':>9}{'full p90':>10}{'foc p90':>9}")
    print('  ' + '-' * 80)

    tot_f = tot_o = 0.0
    rows = 0
    per = []
    for n in names:
        pa = os.path.join(fc, n, 'frames.csv')
        pb = os.path.join(fo, n, 'frames.csv')
        A, B = proc_ms(pa), proc_ms(pb)
        if not A or not B:
            print(f'  {n:<15}  (missing proc_ms)')
            continue
        d, tot = detected(pa)
        dr = 100.0 * d / tot if tot else 0.0
        ma, mb = st.median(A), st.median(B)
        tot_f += ma
        tot_o += mb
        rows += 1
        per.append((n, dr, ma, mb))
        print(f'  {n:<15}{dr:>6.1f}%{ma:>10.1f}{mb:>9.1f}'
              f'{ma-mb:>8.1f}{ma/mb:>8.2f}x{pct(A,90):>10.1f}{pct(B,90):>9.1f}')

    if not rows:
        sys.exit('[ERROR] no comparable sessions')

    mf, mo = tot_f / rows, tot_o / rows
    print('  ' + '-' * 80)
    print(f'  {"MEAN":<15}{"":>7}{mf:>10.1f}{mo:>9.1f}{mf-mo:>8.1f}{mf/mo:>8.2f}x')
    print()
    print(f'  Saving: {mf-mo:.1f} ms/frame  ({(1-mo/mf)*100:.1f}%) across {rows} sessions')

    # Derive S7's real cost from the measured saving rather than assuming
    # the reference benchmark applies. benchmark_summary.txt reported
    # 68.4 ms mean / 26.13 ms S7, but it ran with 0 confirmed range bins and
    # cfar_margin 5.0, so it is NOT a valid baseline for these sessions.
    print()
    print('  ' + '=' * 78)
    print('  INTERPRETATION')
    print('  ' + '=' * 78)
    red = 5.6      # measured mask density 17.9% -> 1/0.179
    s7_full = (mf - mo) / (1 - 1 / red)
    print(f'  Only S7 (azimuth FFT) changed; all other stages are untouched.')
    print(f'  Measured mask density 17.9% implies a {red:.1f}x work reduction on S7.')
    print(f'    saving = S7_full x (1 - 1/{red:.1f})')
    print(f'    => S7_full     ~ {s7_full:6.1f} ms  ({s7_full/mf*100:.0f}% of the {mf:.0f} ms frame)')
    print(f'    => S7_focused  ~ {s7_full/red:6.1f} ms')
    print()
    if s7_full > 45:
        print(f'  WARNING: {s7_full:.0f} ms for a [64,256,256] FFT is far above the')
        print(f'  ~26 ms GPU figure in benchmark_summary.txt. The audit path may be')
        print(f'  running the azimuth FFT on CPU. Run:  python check_gpu.py')

    # Outlier detection: a disturbed run inflates or deflates one session.
    print()
    print('  ' + '=' * 78)
    print('  OUTLIER CHECK  (p90/median should be ~1.0-1.15 for a clean run)')
    print('  ' + '=' * 78)
    clean, dirty = [], []
    for n, dr, ma, mb in per:
        A = proc_ms(os.path.join(fc, n, 'frames.csv'))
        r = pct(A, 90) / ma if ma else 0
        (dirty if r > 1.3 else clean).append((n, r, ma, mb))
    for n, r, ma, mb in dirty:
        print(f'  {n:<15} p90/med = {r:.2f}  <-- run disturbed, exclude from means')
    if not dirty:
        print('  none - all runs clean')
    if dirty and clean:
        cs = sum(a / b for _, _, a, b in clean) / len(clean)
        cv = sum(a - b for _, _, a, b in clean) / len(clean)
        print()
        print(f'  Excluding {len(dirty)} disturbed run(s):')
        print(f'    mean saving  {cv:.1f} ms/frame     mean speedup {cs:.2f}x')

    print()
    print('  ' + '=' * 78)
    print('  SCENE DEPENDENCE')
    print('  ' + '=' * 78)
    zero = [p for p in per if p[1] < 1.0 and p[0] not in [d[0] for d in dirty]]
    live = [p for p in per if p[1] >= 1.0 and p[0] not in [d[0] for d in dirty]]
    if zero and live:
        zs = sum(x[2] / x[3] for x in zero) / len(zero)
        ls = sum(x[2] / x[3] for x in live) / len(live)
        print(f'    zero-detection ({len(zero)}): mean speedup {zs:.2f}x')
        print(f'    detecting      ({len(live)}): mean speedup {ls:.2f}x')
        print(f'    difference {abs(zs-ls):.2f}x -> '
              f'{"scene-dependent" if abs(zs-ls) > 0.3 else "scene-INdependent"}')
    else:
        print('    insufficient clean sessions in one category to compare')


if __name__ == '__main__':
    main()
