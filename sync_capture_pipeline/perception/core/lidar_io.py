"""
The LiDAR recording: lidar.pcap (raw UDP packets stamped with host CLOCK_MONOTONIC) and its
decoding by our rs_driver wrapper (lidar_standalone_capture/rs_driver/python_wrapper_2).

    lidar_pcap_index   packet table, cached in sync/lidar_pcap_index.npz
    lidar_to_mono      LiDAR clock -> session CLOCK_MONOTONIC (align_session.py's envelope fit)
    sweep_time         the time a sweep's radar-boresight points were scanned
    decode_config      the session's driver config with split_angle 180 (sweeps split behind)
    decode_sweeps      the whole session decoded, sweep by sweep, in order

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


def decode_sweeps(session: Path, rate: float = 3.0, cfg_path: Path = None):
    """Decode the whole lidar.pcap; yields (t_ref of the boresight points, points N x 6 float32)
    in time order. The driver runs at `rate` x real time (<= 3)."""
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
        lid.start(pcap_path=str(session / "lidar" / "lidar.pcap"), pcap_rate=rate)
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
