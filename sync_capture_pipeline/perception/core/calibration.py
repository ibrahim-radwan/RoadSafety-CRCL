"""
Radar array description and channel corrections.

MEASURED ARRAY (calibration/radar_array.json, perception/estimate_array_layout.py, 2026-11-07): the
virtual-array layout of the board as measured from the data (phase slope against the LiDAR or camera
azimuth, per channel), and per firmware variant the phase between the two AWR2243 chips (a constant
plus a ramp in range: the slave samples ~0.4 us late). perception/radar.py uses it by default
(layout="measured"), with neither the .cfg antennaCalibParams nor the in-situ phase: on the
corner-reflector and plate sessions they made the measured layout worse.

TEAM LAYOUT (layout="team"): the table in radar_config.py (_TX_AZ, _TX_EL, _RX_AZ), the .cfg
antenna calibration (antennaCalibParams) and the in-situ phase correction measured against the LiDAR
(calibration/radar_rx_phase.json, perception/estimate_rx_phase.py). The table does not match the
board (docs/knowledge/03_radar_firmware_and_dca.md, "Off-axis azimuth"); it stays for the old
chain's reproduction (radar_chain.py) and for comparisons. Both corrections are PROVISIONAL.
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
ARRAY_FILE = Path(__file__).resolve().parents[2] / "calibration" / "radar_array.json"


def array_doc() -> dict:
    """calibration/radar_array.json ({} when absent)."""
    return json.loads(ARRAY_FILE.read_text()) if ARRAY_FILE.is_file() else {}


def array_layout() -> dict:
    """The measured board layout: chirp_tx_pos (per chirp, lambda/2 within a chip's group; None =
    an elevated TX, not in the azimuth row), rx_chip, rx_pos (per RX), chip_offset (lambda/2 from
    the master's group to the slave's). None when the file has no layout."""
    return array_doc().get("layout")


def array_for(variant: str):
    """The inter-chip phase of a firmware variant: {"slave_phase_rad_per_m", "slave_phase_deg", ...};
    None when it was not measured (the detector then combines the chips incoherently only)."""
    return array_doc().get("variants", {}).get(variant) if variant else None


def variant_of_cfg(cfg_path) -> str:
    """The firmware variant of a processing .cfg (firmware/<variant>/cascade_<variant>.cfg), or None."""
    name = Path(cfg_path).name
    return name[len("cascade_"):-len(".cfg")] if name.startswith("cascade_") and name.endswith(".cfg") else None


def rx_phase_for(variant: str, layout: str = "measured"):
    """The in-situ phase correction (deg) for a firmware variant: per RX (8) or per channel
    [tx][rx] (6 x 8); None when there is none. It belongs to the TEAM layout: with the measured
    layout (the default, when calibration/radar_array.json has one) it is None."""
    if layout == "measured" and array_layout() is not None:
        return None
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


__all__ = ["ARRAY_FILE", "RX_PHASE_FILE", "RadarConfig", "apply_phase", "array_doc", "array_for", "array_layout",
           "file_md5", "load_radar_config", "radar_correction", "rx_phase_for", "variant_of_cfg"]
