"""
An empty-scene background for the replay (--background): a recorded session (capture_session.py,
aligned) or a live-view calibration capture (the iPad page's "Save background":
<sessions-root>/calibration/<set>/background/). Read once, when the replay starts:

    radar_frames(path, n)    n raw radar frames from the middle of its radar recording, and its
                             firmware variant (the replayed session's must match: same channels)
    lidar_sweeps(path, n)    n LiDAR sweeps in the VEHICLE frame (each with its own capture's
                             LiDAR extrinsic, so a background of the same mounting lines up)

The radar subtracts the background coherently, channel by channel (RadarDetector.set_background);
the LiDAR keeps only the returns in voxels the background did not have
(perception/core/reflector.py). Both hold only while the rig has not moved since the background.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def is_live_capture(d: Path) -> bool:
    return (Path(d) / "capture.json").is_file()


def radar_frames(path: Path, n: int = 16) -> tuple:
    """(list of raw frames (int16), firmware variant)."""
    path = Path(path)
    if is_live_capture(path):
        f = path / "radar" / "frames.npy"
        if not f.is_file():
            raise FileNotFoundError(f"{f}: the background capture has no radar")
        variant = json.loads((path / "radar" / "meta.json").read_text()).get("variant")
        return list(np.load(f))[:n], variant
    from perception.core.session import Session
    sess = Session(path)
    tl = sess.radar
    mid = len(tl) // 2
    ks = range(max(0, mid - n // 2), min(len(tl), mid - n // 2 + n))
    return [tl.raw(k) for k in ks], sess.variant


def lidar_sweeps(path: Path, n: int = 20, skip_s: float = 1.0) -> list:
    """n sweeps (N x >= 5: x, y, z, range, intensity, ...) in the vehicle frame."""
    path = Path(path)
    from perception.core.extrinsic import IDENTITY, lidar_points_to_vehicle
    if is_live_capture(path):
        f = path / "lidar" / "sweeps.npz"
        if not f.is_file():
            raise FileNotFoundError(f"{f}: the background capture has no LiDAR")
        z = np.load(f)
        own = np.split(z["points"], np.cumsum(z["counts"])[:-1])          # the LiDAR's own frame
        meta = json.loads((path / "lidar" / "meta.json").read_text())
        ext = meta.get("extrinsic") or dict(IDENTITY)
        return [lidar_points_to_vehicle(np.array(p, np.float32, copy=True), ext) for p in own[:n]]
    from perception.core.lidar_io import decode_sweeps
    out, t0 = [], None
    for t, p in decode_sweeps(path):                       # vehicle frame, the session's extrinsic
        t0 = t if t0 is None else t0
        if t - t0 < skip_s:                                # the first sweep after the start is partial
            continue
        out.append(p[p[:, 3] > 0])
        if len(out) >= n:
            break
    return out


def name_of(path: Path) -> str:
    path = Path(path)
    return f"{path.parent.name}/{path.name}" if is_live_capture(path) else path.name
