"""
Radar channel calibration: the .cfg antenna calibration (antennaCalibParams) combined with the
in-situ phase correction measured against the LiDAR (calibration/radar_rx_phase.json, written by
perception/estimate_rx_phase.py).

The in-situ correction is PROVISIONAL: the team owns the radar calibration, and their
recalibration with the v2 images replaces it (radar_correction(..., in_situ=False), or remove
the variant from the JSON file).
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .radar_config import RadarConfig, parse_calibration, parse_cfg_file

RX_PHASE_FILE = Path(__file__).resolve().parents[2] / "calibration" / "radar_rx_phase.json"


def rx_phase_for(variant: str):
    """The in-situ phase correction (deg) for a firmware variant: per RX (8) or per channel
    [tx][rx] (6 x 8); None when there is none."""
    if not RX_PHASE_FILE.is_file():
        return None
    entry = json.loads(RX_PHASE_FILE.read_text()).get("variants", {}).get(variant)
    return None if entry is None else entry["rx_phase_deg"]


def apply_phase(corr: np.ndarray, phase_deg) -> np.ndarray:
    """corr [tx, rx] complex times exp(-j phase): phase per RX (8) or per channel (tx x rx)."""
    if phase_deg is None:
        return corr
    ph = np.asarray(phase_deg, float)
    rot = np.exp(-1j * np.radians(ph))
    return corr * (rot[None, :] if ph.ndim == 1 else rot)


def load_radar_config(cfg_path) -> tuple:
    """(RadarConfig, .cfg calibration [tx, rx] complex) from the parser, silenced."""
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = parse_cfg_file(str(cfg_path))
        corr = parse_calibration(str(cfg_path), cfg.num_tx, cfg.num_rx)
    return cfg, np.asarray(corr)


def radar_correction(cfg_path, variant: str = None, in_situ: bool = True, phase_deg=None) -> tuple:
    """(cfg, combined correction [tx, rx] complex, info dict). The in-situ phase is phase_deg when
    given, else the stored one for `variant` when in_situ."""
    cfg, corr = load_radar_config(cfg_path)
    if phase_deg is None and in_situ and variant:
        phase_deg = rx_phase_for(variant)
    info = {"cfg": Path(cfg_path).name, "in_situ": phase_deg is not None,
            "rx_phase_file_md5": file_md5(RX_PHASE_FILE) if phase_deg is not None else None}
    return cfg, apply_phase(corr, phase_deg), info


def file_md5(p: Path) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()[:12] if Path(p).is_file() else "none"


__all__ = ["RX_PHASE_FILE", "RadarConfig", "apply_phase", "file_md5", "load_radar_config", "radar_correction",
           "rx_phase_for"]
