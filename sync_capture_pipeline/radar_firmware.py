"""Which radar firmware produced a session, and which .cfg describes its data.

The .cfg is never sent to the device: it only tells processing what the
firmware does (frame period, I/Q order, range axis). Using the wrong one is
silent, so every tool resolves it here.

  - ~/.radar_flashed (key=value, written by flashing/boot/flash_variant.sh)
    records what is in QSPI now.
  - capture_session.py copies that record into session.json["radar_firmware"].
  - Sessions recorded before that field existed were all r075m.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIRMWARE_DIR = ROOT / "cascaded_radar_capture_pipeline" / "firmware"
FLASHED_RECORD = Path.home() / ".radar_flashed"
LEGACY_VARIANT = "r075m"          # every session before session.json had radar_firmware


def firmware_map() -> dict:
    return json.loads((FIRMWARE_DIR / "firmware_map.json").read_text())["variants"]


def read_flashed() -> dict | None:
    """Parse ~/.radar_flashed and add the map entry's period/validity fields."""
    if not FLASHED_RECORD.exists():
        return None
    rec = {}
    for line in FLASHED_RECORD.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            rec[k.strip()] = v.strip()
    v = firmware_map().get(rec.get("variant", ""), {})
    for k in ("frame_periodicity_ms", "compression", "dss_l3_meminit", "data_valid"):
        if k in v:
            rec[k] = v[k]
    return rec


def cfg_for_variant(variant: str) -> Path:
    return FIRMWARE_DIR / variant / firmware_map()[variant]["processing_cfg"]


def session_variant(session: Path) -> tuple[str, str]:
    """(variant, how it was determined) for a recorded session."""
    meta = Path(session) / "session.json"
    if meta.exists():
        fw = json.loads(meta.read_text()).get("radar_firmware")
        if fw and fw.get("variant"):
            return fw["variant"], "session.json"
    return LEGACY_VARIANT, "assumed (session predates the radar_firmware field)"


def cfg_for_session(session: Path) -> Path:
    return cfg_for_variant(session_variant(session)[0])
