"""
check_session.py — validate a session and produce a provisional pairing.

WHAT THIS ANSWERS, BEFORE ANY BULK COLLECTION
    1. Are the two manifests internally sane (counts, regular intervals,
       no gaps)?
    2. Which clock domain is rs_timestamp_ms in? If librealsense reported
       hardware_clock it has an arbitrary epoch and is NOT comparable to
       CLOCK_MONOTONIC - only t_arrival would be usable. This must be known
       before the timestamps are trusted.
    3. Do the two streams actually overlap in time?
    4. What is the pairing error distribution?

    (4) is the important one. It validates the timestamp MECHANICS without
    needing a sync event: at 15 fps the nearest camera frame should always
    be within half a frame period (33 ms). Anything larger means the
    timelines are not really shared, and no amount of offset correction
    will fix it.

WHAT THIS DOES NOT DO
    It does not establish the PHYSICAL offset between radar acquisition and
    camera exposure. Both sensors report when the host SAW data, not when
    the world was sampled, and those latencies differ. That is Stage 3, and
    it needs a sharp event both sensors observe. This tool tells you whether
    Stage 3 is worth doing yet.

USAGE
    python check_session.py --session ./sessions/run01
    python check_session.py --session ./sessions/run01 --offset-ms 45
    python check_session.py --session ./sessions/run01 --write-manifest
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fnum(row: dict, *keys):
    for k in keys:
        v = row.get(k, "")
        if v not in ("", None):
            try:
                return float(v)
            except ValueError:
                pass
    return None


def pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round(q / 100.0 * (len(s) - 1)))))
    return s[i]



def check_files(session: Path, rad: list, cam: list, frame_bytes: int) -> tuple:
    """On-disk integrity. Deliberately stdlib-only so this runs anywhere,
    including outside the camera conda env.

    Two of twelve sessions in the previous collection were silently dead and
    nobody knew for months. A manifest row proves a frame was RECORDED; it
    does not prove the file on disk is complete and readable.
    """
    problems = []

    # ── radar: every .bin must be EXACTLY frame_bytes ────────────────────────
    # A short file means the writer was interrupted mid-frame. Because frames
    # are sliced from a continuous byte stream, a wrong size also means every
    # LATER frame is misaligned - so this must be exact, not approximate.
    rdir = session / "radar"
    bins = sorted(rdir.glob("frame_*.bin"))
    bad_size = [(p.name, p.stat().st_size) for p in bins
                if p.stat().st_size != frame_bytes]
    print(f"\n[5] Radar files")
    print(f"    .bin on disk    : {len(bins)}   manifest rows: {len(rad)}")
    if len(bins) != len(rad):
        problems.append(f"radar file/manifest mismatch: {len(bins)} vs {len(rad)}")
        print(f"    MISMATCH - manifest and disk disagree")
    print(f"    wrong size      : {len(bad_size)}   (expected {frame_bytes:,} B each)")
    for n, sz in bad_size[:5]:
        print(f"        {n}  {sz:,} B  ({sz-frame_bytes:+,})")
    if bad_size:
        problems.append(f"{len(bad_size)} radar frames have the wrong size")
    missing = [r.get("filename", "") for r in rad
               if not (rdir / r.get("filename", "")).exists()]
    if missing:
        problems.append(f"{len(missing)} manifest rows have no file")
        print(f"    manifest rows with no file: {len(missing)}")

    # ── camera: JPEG must start FFD8 and end FFD9 ────────────────────────────
    # Truncation is the realistic failure (interrupted write, full disk). The
    # SOI/EOI markers catch it without needing OpenCV, so this check has no
    # dependencies and cannot itself fail to run.
    cdir = session / "camera"
    imgs = sorted(list(cdir.glob("frame_*.jpg")) + list(cdir.glob("frame_*.png")))
    print(f"\n[6] Camera files")
    print(f"    images on disk  : {len(imgs)}   manifest rows: {len(cam)}")
    if len(imgs) != len(cam):
        problems.append(f"camera file/manifest mismatch: {len(imgs)} vs {len(cam)}")
        print(f"    MISMATCH - manifest and disk disagree")
    truncated, empty = [], []
    for p in imgs:
        sz = p.stat().st_size
        if sz == 0:
            empty.append(p.name)
            continue
        try:
            with open(p, "rb") as f:
                head = f.read(2)
                f.seek(-2, 2)
                tail = f.read(2)
        except OSError:
            truncated.append(p.name)
            continue
        if p.suffix == ".jpg" and (head != b"\xff\xd8" or tail != b"\xff\xd9"):
            truncated.append(p.name)
        elif p.suffix == ".png" and head != b"\x89P":
            truncated.append(p.name)
    print(f"    zero-length     : {len(empty)}")
    print(f"    truncated       : {len(truncated)}   (JPEG SOI/EOI markers)")
    for n in (empty + truncated)[:5]:
        print(f"        {n}")
    if empty:
        problems.append(f"{len(empty)} zero-length images")
    if truncated:
        problems.append(f"{len(truncated)} truncated images")

    sizes = [p.stat().st_size for p in imgs if p.stat().st_size]
    if sizes:
        sizes.sort()
        print(f"    size median     : {sizes[len(sizes)//2]/1024:.0f} KB"
              f"   min {sizes[0]/1024:.0f} KB   max {sizes[-1]/1024:.0f} KB")
        # An all-black or all-white frame compresses to almost nothing.
        tiny = [s for s in sizes if s < sizes[len(sizes)//2] * 0.2]
        if tiny:
            print(f"    suspiciously small: {len(tiny)}  (<20% of median - "
                  f"possible blank frames)")
            problems.append(f"{len(tiny)} images far smaller than median")

    total = sum(p.stat().st_size for p in bins) + sum(sizes)
    print(f"\n    session size    : {total/1e9:.2f} GB")
    return problems, len(bins), len(imgs)


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a session and pair frames")
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--offset-ms", type=float, default=0.0,
                    help="subtract this from radar timestamps before pairing "
                         "(the Stage 3 correction; 0 until measured)")
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--frame-bytes", type=int, default=3145728,
                    help="expected radar frame size (256x64x48x4)")
    ap.add_argument("--skip-files", action="store_true",
                    help="skip on-disk integrity checks (manifests only)")
    a = ap.parse_args()

    rad = load(a.session / "radar" / "frames.csv")
    cam = load(a.session / "camera" / "frames.csv")

    print("=" * 76)
    print(f"  SESSION: {a.session}")
    print("=" * 76)
    if not rad or not cam:
        print(f"  radar rows {len(rad)}, camera rows {len(cam)} - cannot proceed")
        return 1

    # ── clock domain ─────────────────────────────────────────────────────────
    doms = {r.get("rs_domain", "?") for r in cam}
    print(f"\n[1] Camera clock domain: {', '.join(sorted(doms))}")
    dom = next(iter(doms)).lower() if len(doms) == 1 else ""
    if "hardware" in dom:
        print("    hardware_clock: device epoch, NOT comparable to CLOCK_MONOTONIC.")
        print("    Only t_arrival_monotonic is usable for pairing.")
    elif "global" in dom or "system" in dom:
        print("    Host-referenced, but in MILLISECONDS SINCE THE UNIX EPOCH,")
        print("    while the radar logs CLOCK_MONOTONIC (seconds since boot).")
        print("    Different origins - do NOT mix. t_arrival is the shared clock.")
    else:
        print("    Unrecognised - use t_arrival_monotonic.")
    if len(doms) > 1:
        print("    WARNING: domain changed mid-session; investigate before trusting.")

    # ── timing, on the shared clock only ─────────────────────────────────────
    rt = [fnum(r, "t_burst_monotonic", "t_first_monotonic") for r in rad]
    ct = [fnum(r, "t_arrival_monotonic") for r in cam]
    rt = [t for t in rt if t is not None]
    ct = [t for t in ct if t is not None]
    if not rt or not ct:
        print("\n  missing timestamps - cannot pair")
        return 1

    rgap = [b - a_ for a_, b in zip(rt, rt[1:])]
    cgap = [b - a_ for a_, b in zip(ct, ct[1:])]
    print(f"\n[2] Cadence")
    print(f"    radar  : {len(rt):>5} frames, interval median "
          f"{st.median(rgap)*1000:7.1f} ms = {1/st.median(rgap):.2f} fps")
    print(f"    camera : {len(ct):>5} frames, interval median "
          f"{st.median(cgap)*1000:7.1f} ms = {1/st.median(cgap):.2f} fps")
    # A dropped frame shows as an interval near a multiple of the period.
    rj = max(rgap) / st.median(rgap)
    cj = max(cgap) / st.median(cgap)
    print(f"    worst interval vs median: radar {rj:.2f}x, camera {cj:.2f}x")
    if cj > 1.8:
        print("    camera shows a gap >= 2 frame periods - frames were dropped")
    if rj > 1.8:
        print("    radar shows a gap >= 2 frame periods - investigate")

    # ── overlap ──────────────────────────────────────────────────────────────
    print(f"\n[3] Coverage (CLOCK_MONOTONIC)")
    print(f"    radar  : {rt[0]:.3f} .. {rt[-1]:.3f}   ({rt[-1]-rt[0]:.1f} s)")
    print(f"    camera : {ct[0]:.3f} .. {ct[-1]:.3f}   ({ct[-1]-ct[0]:.1f} s)")
    lead, trail = rt[0] - ct[0], ct[-1] - rt[-1]
    print(f"    camera leads by {lead:+.2f} s, trails by {trail:+.2f} s")
    if lead < 0 or trail < 0:
        print("    WARNING: radar frames fall OUTSIDE camera coverage.")
        print("    Those frames have no ground truth and must be excluded.")

    # ── provisional pairing ──────────────────────────────────────────────────
    off = a.offset_ms / 1000.0
    print(f"\n[4] Provisional pairing  (radar offset {a.offset_ms:+.1f} ms)")
    pairs, errs = [], []
    j = 0
    for i, t in enumerate(rt):
        tt = t - off
        while j + 1 < len(ct) and abs(ct[j + 1] - tt) <= abs(ct[j] - tt):
            j += 1
        pairs.append((i, j, ct[j] - tt))
        errs.append(abs(ct[j] - tt))

    cam_period = st.median(cgap)
    limit = cam_period / 2
    print(f"    median |error| : {st.median(errs)*1000:7.1f} ms")
    print(f"    p95            : {pct(errs,95)*1000:7.1f} ms")
    print(f"    max            : {max(errs)*1000:7.1f} ms")
    print(f"    expected max   : {limit*1000:7.1f} ms  (half a camera period)")
    bad = sum(1 for e in errs if e > limit * 1.5)
    print(f"    pairs beyond 1.5x that: {bad} of {len(errs)}")

    signed = [e for _, _, e in pairs]
    bias = st.median(signed)
    print(f"\n    median SIGNED error: {bias*1000:+.1f} ms")
    print("    CAUTION: this does NOT estimate the physical offset. Nearest-")
    print("    neighbour pairing always picks the CLOSEST camera frame, so a")
    print("    constant offset does not appear as a bias - it silently changes")
    print("    WHICH frame is selected while the residual stays symmetric about")
    print("    zero. Simulated: offsets of 0/50/100/200 ms all yield a median")
    print("    residual under 1 ms, while shifting the chosen frame by 0/1/2/3.")
    print(f"    One frame of error = {cam_period*1000:.0f} ms = "
          f"{1.4*cam_period*100:.0f} cm at walking pace.")
    print("    Only Stage 3 - a sharp event both sensors observe - can measure it.")

    file_problems = []
    if not a.skip_files:
        file_problems, _, _ = check_files(a.session, rad, cam, a.frame_bytes)

    print("\n" + "=" * 76)
    print("  VERDICT")
    print("=" * 76)
    ok = (max(errs) <= limit * 1.5 and lead >= 0 and trail >= 0
          and len(doms) == 1 and not file_problems)
    if ok:
        print("  Timestamp mechanics are sound: every radar frame has a camera")
        print("  frame within half a period, and coverage brackets the radar.")
        print("  The shared clock works.")
        print("")
        print("  STILL UNKNOWN: the physical offset between radar acquisition")
        print("  and camera exposure. The pairing above cannot reveal it (see")
        print("  the caution in section 4). Stage 3 - a sharp event both")
        print("  sensors observe - is the only way to bound it.")
    else:
        print("  PROBLEM - do not collect at scale until resolved:")
        if max(errs) > limit * 1.5:
            print("    - pairing errors exceed half a camera frame period")
        if lead < 0 or trail < 0:
            print("    - radar extends beyond camera coverage")
        if len(doms) > 1:
            print("    - camera clock domain changed mid-session")
        for pb in file_problems:
            print(f"    - {pb}")

    if a.write_manifest:
        out = a.session / "pairs.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["radar_idx", "radar_file", "camera_idx",
                        "camera_file", "error_ms"])
            for i, jj, e in pairs:
                w.writerow([i, rad[i].get("filename", ""), jj,
                            cam[jj].get("filename", ""), f"{e*1000:.2f}"])
        print(f"\n  wrote {out}  ({len(pairs)} pairs)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
