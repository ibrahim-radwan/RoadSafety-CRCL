"""Smoke test of the replay's data sources, headless (no Qt window): the radar and LiDAR sources start on
a reference session, follow the replay clock, and publish their products (maps, detections, tracks;
the rendered LiDAR view and its objects). This catches a source that cannot start, which the
detector tests alone do not (2026-10-25: a refactor dropped the radar read-ahead and only the
replay noticed).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

PIPELINE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PIPELINE), str(PIPELINE / "replay")]
pytest.importorskip("torch")

S = Path(os.environ.get("REPORT_TEST_SESSIONS", "/mnt/sdcard/sessions/freewalk_ref_02").split(",")[0])
pytestmark = pytest.mark.skipif(not (S / "sync" / "radar_timeline.csv").is_file(), reason="no reference session")


def _wait(src, timeout_s=60.0, accept=lambda item: item is not None):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if src.error:
            raise AssertionError(src.error)
        item = src.get()
        if accept(item):
            return item
        time.sleep(0.05)
    raise AssertionError(f"{src.name_}: no product within {timeout_s:.0f} s")


def test_radar_source_publishes():
    from sources import RadarSource, ReplayClock
    clock = ReplayClock(0.0, 1.0)
    bev_x = np.arange(0.1, 40, 0.2)
    bev_y = np.arange(-20, 20, 0.2)
    src = RadarSource(clock, S, bev_x, bev_y)
    clock.t0, clock.t1 = src.span()
    clock.seek(clock.t0 + 120.0)
    src.start()
    try:
        item = _wait(src)
    finally:
        src.stop()
    assert item["rd"].shape == (len(src.proc.dop_axis), src.proc.n_range)
    assert item["bev"].shape == (len(bev_x), len(bev_y))
    assert item["det"].ndim == 2 and isinstance(item["tracks"], list)


def test_lidar_source_publishes():
    from sources import LidarSource, ReplayClock
    clock = ReplayClock(0.0, 1.0)
    src = LidarSource(clock, S, 0.0, view=dict(mode="top", range_m=20.0, color="height", near_m=0.5))
    clock.t0, clock.t1 = src.span()
    clock.seek(clock.t0 + 120.0)
    clock.play()
    src.start()
    try:
        item = _wait(src, accept=lambda it: it is not None and it.get("det") is not None)
    finally:
        src.stop()
        clock.pause()
    assert item["rgb"].ndim == 3 and item["rgb"].shape[2] == 3
    assert item["det"]["objects"].ndim == 2
