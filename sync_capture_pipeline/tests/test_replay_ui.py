"""Replay window pieces that need no session (PyQt5, offscreen)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


def test_panel_info_does_not_widen_panel():
    """A panel's info line must not set its minimum width: the LiDAR's grew once a second (gap fill)
    and the splitter took the width from the camera panel, whose image shrank and grew (2026-11-07)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "replay")]
    import replay_session as rs
    p = rs.Panel("lidar", "LiDAR", QtWidgets.QLabel("body"), lambda k: None)
    p.set(info="52,346 pts @ 24.99 s")
    short = p.info.minimumSizeHint().width()
    p.set(info="52,346 pts @ 24.99 s  |  16.5 deg gap filled from the previous sweep  |  background: 3,318 changed")
    assert p.info.minimumSizeHint().width() == short == 0
    assert "gap filled" in p.info.toolTip()
