"""Moved to sync_capture_pipeline/perception/core/radar_io.py (radar-chain merge, step 2, 2026-10-25).

This shim keeps the old imports working (`from processing.radar_reader import ...`, the old chain's
relative imports, and loading this file by path) until the old chain is retired. It re-exports
every name of the new module, private ones included: the objects are the same.
"""
import sys as _sys
from pathlib import Path as _Path

_PIPELINE = str(_Path(__file__).resolve().parents[2] / "sync_capture_pipeline")
if _PIPELINE not in _sys.path:
    _sys.path.insert(0, _PIPELINE)
from perception.core import radar_io as _core  # noqa: E402

globals().update({k: v for k, v in vars(_core).items() if not k.startswith("__")})
