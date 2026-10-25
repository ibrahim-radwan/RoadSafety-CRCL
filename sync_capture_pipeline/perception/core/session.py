"""
One recorded session, as the replay and the batch tools see it.

    s = Session("/mnt/sdcard/sessions/freewalk_ref_02")
    s.variant, s.cfg_path            radar firmware variant and its .cfg (radar_firmware.py)
    s.offsets                        Stage 3 offsets applied by align_session.py (ms)
    s.radar                          RadarTimeline: t_ref and file per radar frame
    s.camera_t, s.camera_index       camera frame times (t_ref) and SVO frame indices
    s.target                         the single LiDAR walker track (make_report.py) or None
    s.target_at(t)                   (range m, azimuth deg + left) at time t, or None
    s.pairs                          sync/pairs.csv rows by radar frame index

Everything comes from sync/ (align_session.py), so offsets applied there apply here.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PIPELINE = Path(__file__).resolve().parents[2]


def _csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Session:
    def __init__(self, path):
        self.path = Path(path)
        self.name = self.path.name
        self.sync = self.path / "sync"
        self.meta = json.loads((self.path / "session.json").read_text())
        self.sensors = self.meta.get("sensors", [])
        self._radar = self._target = self._pairs = self._camera = None

    def __repr__(self):
        return f"Session({self.path})"

    # ── radar firmware ──────────────────────────────────────────────────
    @property
    def variant(self) -> str:
        return self.firmware[0]

    @property
    def firmware(self) -> tuple:
        """(variant, how it was determined)."""
        if str(PIPELINE) not in sys.path:
            sys.path.insert(0, str(PIPELINE))
        import radar_firmware
        return radar_firmware.session_variant(self.path)

    @property
    def cfg_path(self) -> Path:
        import radar_firmware
        return radar_firmware.cfg_for_variant(self.firmware[0])

    # ── alignment ───────────────────────────────────────────────────────
    @property
    def alignment(self) -> dict:
        p = self.sync / "alignment.json"
        return json.loads(p.read_text()) if p.is_file() else {}

    @property
    def offsets(self) -> dict:
        return self.alignment.get("offsets", {})

    @property
    def radar(self):
        if self._radar is None:
            from .radar_io import RadarTimeline
            self._radar = RadarTimeline(self.path)
        return self._radar

    @property
    def camera_t(self) -> np.ndarray:
        return self._camera_rows()[0]

    @property
    def camera_index(self) -> np.ndarray:
        return self._camera_rows()[1]

    def _camera_rows(self):
        if self._camera is None:
            rows = _csv(self.sync / "camera_timeline.csv")
            self._camera = (np.array([float(r["t_ref"]) for r in rows]),
                            np.array([int(r["frame_index"]) for r in rows]))
        return self._camera

    @property
    def pairs(self) -> dict:
        if self._pairs is None:
            p = self.sync / "pairs.csv"
            self._pairs = {int(r["radar_idx"]): r for r in _csv(p)} if p.is_file() else {}
        return self._pairs

    # ── the LiDAR walker track (a single-target reference) ──────────────
    @property
    def target(self):
        """dict t, r, az (deg, + left) from sync/timing_lidar_track.npz, or None."""
        if self._target is None:
            p = self.sync / "timing_lidar_track.npz"
            if not p.is_file():
                return None
            z = np.load(p)
            self._target = {"t": z["t"], "r": z["r"], "az": z["az"]}
        return self._target

    def target_at(self, t: float, tol: float = 0.08):
        """(range, azimuth) of the LiDAR target at time t, or None (no target within tol s)."""
        return target_at(self.target, t, tol)


def target_at(track, t: float, tol: float = 0.08):
    """(range m, azimuth deg + left) of a target track {t, r, az} nearest to t, or None when there
    is none within tol s."""
    if track is None or not len(track["t"]):
        return None
    j = int(np.clip(np.searchsorted(track["t"], t), 1, len(track["t"]) - 1))
    j = j if abs(track["t"][j] - t) < abs(track["t"][j - 1] - t) else j - 1
    if abs(track["t"][j] - t) > tol or not np.isfinite(track["r"][j]):
        return None
    return float(track["r"][j]), float(track["az"][j])
