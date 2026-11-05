"""
The LiDAR -> vehicle (radar) transform, per mounting: calibration/lidar_extrinsic.json.

The pipeline frame is the VEHICLE frame: x forward along the radar boresight, y left, z up
(perception/core/conventions.py). The radar and the camera face along x. The LiDAR may be mounted
turned: its points are rotated (yaw about z) and shifted into the vehicle frame right after
decoding, in every tool (lidar_points_to_vehicle), so sweep timing, detection, views and scores
all use one frame.

Which transform a session uses:
    1. the one recorded in <session>/sync/alignment.json ("lidar_extrinsic"), written by
       align_session.py: re-align to change it;
    2. otherwise the newest entry of the file whose valid_from is <= the session's start
       (session.json started_local), else identity (the mounting before any entry).
Live view (no session) uses the newest entry (current_extrinsic).

An identity transform returns the points unchanged (the same array): sessions before the first
entry are processed exactly as before.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EXTRINSIC_FILE = Path(__file__).resolve().parents[2] / "calibration" / "lidar_extrinsic.json"
IDENTITY = {"id": "identity", "yaw_deg": 0.0, "tx_m": 0.0, "ty_m": 0.0}


def _entries(path: Path = EXTRINSIC_FILE) -> list:
    if not Path(path).is_file():
        return []
    return sorted(json.loads(Path(path).read_text()).get("mountings", []), key=lambda e: e["valid_from"])


def _pick(when: str, path: Path = EXTRINSIC_FILE) -> dict:
    """Newest entry with valid_from <= when (ISO strings compare in time order), else identity."""
    best = dict(IDENTITY)
    for e in _entries(path):
        if e["valid_from"] <= when:
            best = e
    return {k: best[k] for k in ("id", "yaw_deg", "tx_m", "ty_m")}


def extrinsic_for_date(when: str, path: Path = EXTRINSIC_FILE) -> dict:
    """The mounting in use at `when` (ISO date or time, local)."""
    return _pick(str(when)[:19], path)


def current_extrinsic(path: Path = EXTRINSIC_FILE) -> dict:
    """The newest mounting (live view: the rig as it is now)."""
    return _pick("9999", path)


def extrinsic_for_session(session, path: Path = EXTRINSIC_FILE) -> dict:
    """The session's transform: alignment.json's record, else by its start time (see module doc)."""
    s = Path(session)
    al = s / "sync" / "alignment.json"
    if al.is_file():
        rec = json.loads(al.read_text()).get("lidar_extrinsic")
        if rec is not None:
            return rec
    return session_default(s, path)


def session_default(session, path: Path = EXTRINSIC_FILE) -> dict:
    """By the session's start time only (align_session.py records this one)."""
    sj = Path(session) / "session.json"
    when = json.loads(sj.read_text()).get("started_local", "") if sj.is_file() else ""
    return extrinsic_for_date(when, path) if when else dict(IDENTITY)


def is_identity(ext: dict) -> bool:
    return not (ext and (ext.get("yaw_deg") or ext.get("tx_m") or ext.get("ty_m")))


def lidar_points_to_vehicle(points: np.ndarray, ext: dict) -> np.ndarray:
    """N x >=4 LiDAR points [x, y, z, range, ...] -> the vehicle frame. Zero-range points (no
    return) stay zero; range is recomputed from the vehicle origin. Identity: the input itself."""
    if is_identity(ext):
        return points
    p = np.array(points, dtype=np.float32, copy=True)
    ok = p[:, 3] > 0
    c, s = np.cos(np.radians(ext["yaw_deg"])), np.sin(np.radians(ext["yaw_deg"]))
    x, y = p[ok, 0].copy(), p[ok, 1].copy()
    p[ok, 0] = c * x - s * y + ext["tx_m"]
    p[ok, 1] = s * x + c * y + ext["ty_m"]
    p[ok, 3] = np.sqrt(p[ok, 0] ** 2 + p[ok, 1] ** 2 + p[ok, 2] ** 2)
    return p


def describe(ext: dict) -> str:
    return "identity" if is_identity(ext) else (f"{ext['id']}: yaw {ext['yaw_deg']:+.1f} deg, "
                                                 f"t ({ext['tx_m']:+.2f}, {ext['ty_m']:+.2f}) m")
