#!/usr/bin/env python3
"""
make_report.py - self-contained HTML report of a three-sensor session.

    python make_report.py --session /mnt/sdcard/sessions/<name>
    python make_report.py --session <s> --reuse        # re-render from the cached data (figures/text only)

Writes <session>/session_report.html (figures embedded; opens offline).
Descriptive only: no pass/fail - recording completeness is check_session.py's job.
Runs align_session.py first if the session has not been aligned. Reads every
sensor's data once: about 1/3-1/2 of the recording time (LiDAR decode at 3x,
radar frames from the SD card, SVO2 replay).
Needs the conda env (pyzed, torch): ~/miniconda3/envs/rel_1_py_3_8/bin/python
"""
from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> int:
    ap = argparse.ArgumentParser(description="HTML report of a recorded session")
    ap.add_argument("--session", type=Path, nargs="+", required=True)
    ap.add_argument("--reuse", action="store_true", help="use <session>/sync/report_data.pkl if present")
    ap.add_argument("--out", type=Path, help="output file (default <session>/session_report.html)")
    ap.add_argument("--radar-cloud", action="store_true",
                    help="also pre-compute the filtered radar point cloud (team pipeline) for the replay "
                         "(radar_cloud.py; about 1.1x the recording time on 10 CPU cores)")
    a = ap.parse_args()
    from report import collect, render
    for S in a.session:
        S = S.resolve()
        print(f"\n=== {S}")
        if not (S / "sync" / "radar_timeline.csv").is_file():
            print("  aligning first (align_session.py) ...")
            subprocess.run([sys.executable, str(HERE / "align_session.py"), "--session", str(S)], check=True,
                           stdout=subprocess.DEVNULL)
        cache = S / "sync" / "report_data.pkl"
        t0 = time.monotonic()
        if a.reuse and cache.is_file():
            D = pickle.loads(cache.read_bytes())
        else:
            D = collect.collect(S)
            cache.write_bytes(pickle.dumps(D, protocol=4))
        page = render.build(D)
        out = a.out if (a.out and len(a.session) == 1) else S / "session_report.html"
        out.write_text(page, encoding="utf-8")
        print(f"  wrote {out}  ({out.stat().st_size / 1e6:.1f} MB, {time.monotonic() - t0:.0f} s)")
        if a.radar_cloud:
            print("  filtered radar point cloud (team pipeline) ...", flush=True)
            subprocess.run([sys.executable, str(HERE / "radar_cloud.py"), "--session", str(S)], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
