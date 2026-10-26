"""
The LiDAR recording: lidar.pcap (raw UDP packets stamped with host CLOCK_MONOTONIC) and its
decoding by our rs_driver wrapper (lidar_standalone_capture/rs_driver/python_wrapper_2).

    lidar_pcap_index   packet table, cached in sync/lidar_pcap_index.npz
    lidar_to_mono      LiDAR clock -> session CLOCK_MONOTONIC (align_session.py's envelope fit)
    sweep_time         the time a sweep's radar-boresight points were scanned
    decode_config      the session's driver config with split_angle 180 (sweeps split behind)
    decode_sweeps      the whole session decoded, sweep by sweep, in order
    decode_parallel    the same, several drivers on time chunks at once (processes), fn applied there

Decode at <= 3x real time: faster, the driver drops data. Points are N x 6: x, y, z (x forward,
y left, z up), range, intensity, time offset from the sweep timestamp. Directions with no return
come back as zero-range points.
"""
from __future__ import annotations

import contextlib
import io
import json
import queue
import struct
import sys
import time
from pathlib import Path

import numpy as np

PIPELINE = Path(__file__).resolve().parents[2]
ROOT = PIPELINE.parent
LIDAR_WRAPPER = ROOT / "lidar_standalone_capture" / "rs_driver" / "python_wrapper_2"
MSOP_ID = b"\x55\xaa\x05\x5a"


def lidar_pcap_index(session: Path, pcap=None) -> dict:
    """Packet table of lidar.pcap: byte offset, length, host CLOCK_MONOTONIC ns, port, and
    for MSOP packets the LiDAR timestamp (lts, s) with its host time (mts, ns).
    Cached in sync/lidar_pcap_index.npz - building it reads the whole PCAP once."""
    cache = session / "sync" / "lidar_pcap_index.npz"
    keys = ("off", "ln", "ht", "port", "lts", "mts")
    if cache.is_file():
        z = np.load(cache)
        return {k: z[k] for k in keys}
    buf = pcap if pcap is not None else np.memmap(session / "lidar" / "lidar.pcap", dtype=np.uint8, mode="r")
    off, ln, ht, port, lts, mts = [], [], [], [], [], []
    o, n = 24, len(buf)
    while o + 16 <= n:
        sec, nsec, incl, _ = struct.unpack_from("<IIII", buf, o)
        if o + 16 + incl > n:
            break
        t = sec * 1_000_000_000 + nsec
        p = struct.unpack_from("!H", buf, o + 16 + 36)[0]
        off.append(o); ln.append(16 + incl); ht.append(t); port.append(p)
        pl = o + 16 + 42
        if bytes(buf[pl:pl + 4]) == MSOP_ID:
            lts.append(int.from_bytes(bytes(buf[pl + 20:pl + 26]), "big")
                       + int.from_bytes(bytes(buf[pl + 26:pl + 30]), "big") * 1e-6)
            mts.append(t)
        o += 16 + incl
    idx = {"off": np.array(off), "ln": np.array(ln), "ht": np.array(ht, dtype=np.int64), "port": np.array(port),
           "lts": np.array(lts), "mts": np.array(mts, dtype=np.int64)}
    np.savez(cache, **idx)
    return idx


def lidar_to_mono(index: dict, offset_s: float = 0.0):
    """LiDAR clock (s) -> session CLOCK_MONOTONIC (s, LiDAR offset applied): the same
    lower-envelope fit as align_session.py."""
    if str(PIPELINE) not in sys.path:
        sys.path.insert(0, str(PIPELINE))
    import align_session as al
    a, b, l0, _, _ = al.envelope_fit(index["lts"], index["mts"] / 1e9)
    return lambda tl: a * (np.asarray(tl) - l0) + b + offset_s


def sweep_time(points: np.ndarray, ts: float) -> float:
    """LiDAR-clock time at which the sweep's radar-boresight points (within ~5 deg ahead) were
    scanned; points column 5 = per-point time offset from the sweep timestamp ts."""
    fwd = (points[:, 0] > 0) & (np.abs(points[:, 1]) <= 0.09 * points[:, 0])
    return ts + float(np.median(points[fwd, 5])) if fwd.any() else ts + 0.05


def decode_config(session: Path, out_path: Path, split_angle: float = 180.0) -> Path:
    """Write the session's driver config (lidar/config_used.yaml) with split_angle set."""
    import yaml
    cfg = yaml.safe_load((Path(session) / "lidar" / "config_used.yaml").read_text())
    cfg["lidar"]["split_angle"] = float(split_angle)
    Path(out_path).write_text(yaml.safe_dump(cfg, sort_keys=False))
    return Path(out_path)


def load_wrapper():
    """The wrapper's Lidar class."""
    if str(LIDAR_WRAPPER) not in sys.path:
        sys.path.insert(0, str(LIDAR_WRAPPER))
    from lidar import Lidar
    return Lidar


def lidar_offset_s(session: Path) -> float:
    """The LiDAR's Stage 3 offset (s) used by align_session.py (alignment.json), 0 if none."""
    al = Path(session) / "sync" / "alignment.json"
    return json.loads(al.read_text()).get("offsets", {}).get("lidar_ms", 0.0) / 1e3 if al.is_file() else 0.0


def decode_sweeps(session: Path, rate: float = 3.0, cfg_path: Path = None, pcap_path: Path = None):
    """Decode the whole lidar.pcap (or pcap_path: a part of it, see decode_parallel); yields
    (t_ref of the boresight points, points N x 6 float32) in time order. The driver runs at
    `rate` x real time (<= 3)."""
    session = Path(session)
    to_mono = lidar_to_mono(lidar_pcap_index(session), lidar_offset_s(session))
    cfg_path = decode_config(session, cfg_path or session / "sync" / "perception_lidar_decode.yaml")
    q: queue.Queue = queue.Queue(maxsize=200)

    def on_cloud(points, ts, seq):
        p = np.asarray(points)
        if ts > 0 and len(p):
            q.put((float(to_mono(sweep_time(p, ts))), p.astype(np.float32, copy=True)))

    lid = load_wrapper()(str(cfg_path))
    lid.on_pointcloud(on_cloud)
    with contextlib.redirect_stdout(io.StringIO()):
        lid.start(pcap_path=str(pcap_path or session / "lidar" / "lidar.pcap"), pcap_rate=rate)
    pending = []                      # sweeps arrive in order; keep a small reorder buffer anyway
    try:
        while not (lid.input_finished() and q.empty()):
            try:
                pending.append(q.get(timeout=0.5))
            except queue.Empty:
                continue
            if len(pending) >= 3:
                pending.sort(key=lambda w: w[0])
                yield pending.pop(0)
        time.sleep(1.0)
        while not q.empty():
            pending.append(q.get())
        for item in sorted(pending, key=lambda w: w[0]):
            yield item
    finally:
        with contextlib.redirect_stdout(io.StringIO()):
            lid.stop()


def _decode_chunk(session: Path, fn, lo: int, hi: int, first: bool, last: bool, rate: float, tmp_dir: Path):
    """One chunk of decode_parallel, in a worker process: packets lo..hi-1 of lidar.pcap copied
    into a temporary pcap and decoded; [(t, fn(points))] without the chunk's first and last sweep
    (possibly partial), unless the chunk starts or ends the session."""
    import os
    import tempfile
    idx = lidar_pcap_index(session)
    src = session / "lidar" / "lidar.pcap"
    a, b = int(idx["off"][lo]), int(idx["off"][hi - 1] + idx["ln"][hi - 1])
    fd, tmp = tempfile.mkstemp(suffix=".pcap", dir=str(tmp_dir))
    try:
        with open(src, "rb") as f, os.fdopen(fd, "wb") as g:
            g.write(f.read(24))                               # pcap global header
            f.seek(a)
            left = b - a
            while left > 0:
                buf = f.read(min(left, 1 << 24))
                if not buf:
                    break
                g.write(buf)
                left -= len(buf)
        cfg = Path(tmp).with_suffix(".yaml")
        out = [(t, fn(p)) for t, p in decode_sweeps(session, rate, cfg_path=cfg, pcap_path=Path(tmp))]
    finally:
        for q in (Path(tmp), Path(tmp).with_suffix(".yaml")):
            if q.exists():
                q.unlink()
    return out[(0 if first else 1):(len(out) if last else max(len(out) - 1, 0))]


def decode_parallel(session: Path, fn, workers: int = 4, chunk_s: float = None, rate: float = 3.0,
                    pad_s=(1.5, 0.3), tmp_dir: Path = Path("/dev/shm"), max_chunk_s: float = 300.0):
    """Like ((t, fn(points)) for t, p in decode_sweeps(session)), but several drivers decode time
    chunks of the pcap at once, in `workers` processes (one driver is limited to ~3x real time).
    fn runs in the workers (a module-level function, picklable; e.g. a detector), so only its
    results come back. Chunks overlap by pad_s (before: >= 1 DIFOP packet, which the driver needs
    first; after: the chunk's last sweep), each drops its possibly partial first and last sweep,
    and sweeps decoded twice are dropped by time. Yields in time order as the chunks finish.
    chunk_s None: the session split evenly over the workers (one round; a chunk costs ~2 s of driver
    start/stop), at most max_chunk_s each (a chunk's copy sits in tmp_dir, RAM, while it decodes)."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    session = Path(session)
    idx = lidar_pcap_index(session)
    ht = idx["ht"] / 1e9
    span = float(ht[-1] - ht[0])
    if chunk_s is None:
        n = max(int(workers), 1)
        while span / n > max_chunk_s:
            n += max(int(workers), 1)
        chunk_s = span / n + 1e-3
    edges = np.arange(ht[0], ht[-1], chunk_s)[1:]
    cuts = np.r_[0, np.searchsorted(ht, edges), len(ht)]
    jobs = []
    for i in range(len(cuts) - 1):
        lo = int(np.searchsorted(ht, ht[cuts[i]] - pad_s[0])) if i > 0 else 0
        hi = int(np.searchsorted(ht, ht[cuts[i + 1] - 1] + pad_s[1], side="right")) if i < len(cuts) - 2 else len(ht)
        jobs.append((lo, hi, i == 0, i == len(cuts) - 2))
    last_t = -np.inf
    with ProcessPoolExecutor(max(int(workers), 1), mp_context=multiprocessing.get_context("spawn")) as ex:
        futs = [ex.submit(_decode_chunk, session, fn, lo, hi, first, last, rate, tmp_dir)
                for lo, hi, first, last in jobs]
        for fu in futs:                                       # in chunk order
            for t, r in fu.result():
                if t > last_t + 1e-6:                         # a sweep the previous chunk already gave
                    last_t = t
                    yield t, r
