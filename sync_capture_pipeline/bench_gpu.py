#!/usr/bin/env python3
"""
bench_gpu.py - the processing's speed and GPU use on a session, measured the same way every time,
so each optimisation can be shown to help.

Frames are loaded into RAM first (the session drive's speed does not count). Then:
  radar        RadarDetector.process per frame (moving + static paths), alone
  lidar        LidarDetector.detect per sweep (no per-point labels), alone
  render       the replay's bird's-eye render of a sweep (replay/sources.render_lidar), alone
  tracker      the LiDAR tracker update with the detector's objects, alone
  concurrent   radar and LiDAR (detect + track + render) in two threads at once, as the replay runs
               them: achieved rates against the sensors' 8 / 10 Hz
GPU load comes from tegrastats (GR3D_FREQ, sampled every 100 ms while each part runs); peak GPU
memory from torch.

    python bench_gpu.py --session /mnt/sdcard/sessions/<name> [--frames 200] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "replay")]


class GpuLoad:
    """GR3D_FREQ samples from tegrastats while running."""

    def __init__(self):
        self.samples, self.proc = [], None

    def __enter__(self):
        try:
            self.proc = subprocess.Popen(["tegrastats", "--interval", "100"], stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, text=True)
            threading.Thread(target=self._read, daemon=True).start()
        except OSError:
            self.proc = None
        return self

    def _read(self):
        for line in self.proc.stdout:
            m = re.search(r"GR3D_FREQ (\d+)%", line)
            if m:
                self.samples.append(int(m.group(1)))

    def __exit__(self, *exc):
        if self.proc is not None:
            self.proc.terminate()
            self.proc.wait(timeout=2)

    @property
    def mean(self):
        return float(np.mean(self.samples)) if self.samples else float("nan")


def timed(fn, items, warm=5):
    import torch
    for it in items[:warm]:
        fn(it)
    torch.cuda.synchronize()
    ms = []
    with GpuLoad() as g:
        for it in items:
            t0 = time.perf_counter()
            fn(it)
            torch.cuda.synchronize()
            ms.append(1e3 * (time.perf_counter() - t0))
    return {"ms_median": round(float(np.median(ms)), 2), "ms_p90": round(float(np.percentile(ms, 90)), 2),
            "gpu_load_pct": round(g.mean, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Processing speed and GPU use on a session")
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=200, help="radar frames and LiDAR sweeps to use")
    ap.add_argument("--start-s", type=float, default=100.0, help="start this far into the session")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    import torch
    from perception.core.lidar_io import decode_sweeps
    from perception.core.session import Session
    from perception.lidar import CLASSES, LIDAR_TRACKER, O, LidarDetector
    from perception.radar import RadarDetector, rx_phase_for
    from perception.tracking import Tracker
    from sources import render_lidar
    S = Session(a.session)
    k0 = int(np.searchsorted(S.radar.t, S.radar.t[0] + a.start_s))
    raws = [S.radar.raw(k) for k in range(k0, min(k0 + a.frames, len(S.radar)))]
    sweeps = []
    t_start = S.radar.t[k0]
    for t, p in decode_sweeps(S.path, 3.0, cfg_path=Path("/tmp") / f"bench_decode_{S.name}.yaml"):
        if t >= t_start:
            sweeps.append((t, p))
        if len(sweeps) >= a.frames:
            break
    print(f"{S.name}: {len(raws)} radar frames, {len(sweeps)} LiDAR sweeps in RAM", flush=True)
    det_r = RadarDetector(S.cfg_path, rx_phase_deg=rx_phase_for(S.variant))
    det_l = LidarDetector()
    view = dict(mode="top", range_m=20.0, color="height", near_m=0.5, aspect=1.6, px=700)
    out = {"session": S.name, "frames": len(raws), "torch": torch.__version__}
    torch.cuda.reset_peak_memory_stats()
    out["radar"] = timed(lambda raw: det_r.process(raw, static=True), raws)
    out["lidar"] = timed(lambda tp: det_l.detect(tp[1], labels=False), sweeps)
    out["render"] = timed(lambda tp: render_lidar(tp[1], view, -0.9), sweeps)
    objs = [det_l.detect(p, labels=False)["objects"] for _, p in sweeps]
    T = Tracker(**LIDAR_TRACKER)

    def trk(i):
        ob = objs[i]
        mov = ob[ob[:, O["cls"]] != CLASSES.index("structure")]
        T.update(sweeps[i][0], mov[:, [O["x"], O["y"]]], list(mov[:, O["cls"]]))
    out["tracker"] = timed(trk, list(range(len(sweeps))))
    # concurrent: both sensors as fast as they can for the same wall time, as in the replay
    stop = threading.Event()
    counts = {"radar": 0, "lidar": 0}

    def radar_loop():
        i = 0
        while not stop.is_set():
            det_r.process(raws[i % len(raws)], static=True)
            torch.cuda.synchronize()
            counts["radar"] += 1
            i += 1

    def lidar_loop():
        T2, i = Tracker(**LIDAR_TRACKER), 0
        while not stop.is_set():
            t, p = sweeps[i % len(sweeps)]
            ob = det_l.detect(p, labels=False)["objects"]
            mov = ob[ob[:, O["cls"]] != CLASSES.index("structure")]
            T2.update(t + 100.0 * (i // len(sweeps)), mov[:, [O["x"], O["y"]]], list(mov[:, O["cls"]]))
            render_lidar(p, view, -0.9)
            counts["lidar"] += 1
            i += 1
    dur = 15.0
    with GpuLoad() as g:
        th = [threading.Thread(target=radar_loop), threading.Thread(target=lidar_loop)]
        for x in th:
            x.start()
        time.sleep(dur)
        stop.set()
        for x in th:
            x.join()
    out["concurrent"] = {"radar_hz": round(counts["radar"] / dur, 1), "lidar_hz": round(counts["lidar"] / dur, 1),
                         "gpu_load_pct": round(g.mean, 1), "need_hz": {"radar": 8, "lidar": 10}}
    out["gpu_mem_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    for k in ("radar", "lidar", "render", "tracker"):
        print(f"  {k:8s} {out[k]['ms_median']:6.1f} ms median (p90 {out[k]['ms_p90']:5.1f})   GPU load {out[k]['gpu_load_pct']:4.0f}%")
    c = out["concurrent"]
    print(f"  concurrent: radar {c['radar_hz']} Hz, LiDAR {c['lidar_hz']} Hz (sensors: 8 / 10 Hz), GPU load {c['gpu_load_pct']:.0f}%")
    print(f"  peak GPU memory {out['gpu_mem_peak_mb']} MB")
    if a.json:
        a.json.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
