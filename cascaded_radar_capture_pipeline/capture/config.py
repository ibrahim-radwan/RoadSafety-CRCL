"""
Project configuration for the fixed shortrange cascade capture app.

This module centralizes:
- project paths
- runtime capture settings
- cascade_r017m.cfg parsing
- frame geometry derived from the fixed cfg/header inputs
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAPTURE_DIR = PROJECT_ROOT / "capture"
CONFIG_DIR = PROJECT_ROOT / "config"
BUILD_DIR = CONFIG_DIR / "build_shortrange_inf"
DATA_DIR = CAPTURE_DIR / "data"
FRAMES_DIR = CAPTURE_DIR / "frames"

CASCADE_CFG_PATH = PROJECT_ROOT / "firmware" / "r017m" / "cascade_r017m.cfg"
DCA_CAPTURE_JSON = CONFIG_DIR / "AM273X_Capture_fixed.json"
CCXML_PATH = CONFIG_DIR / "AM273x_XDS110.ccxml"
DSS_SCRIPT_PATH = CONFIG_DIR / "reset_load_run_dualcore_reset.js"
MSS_BIN_PATH = BUILD_DIR / "am273x_cascade_mss.xer5f"
APP_HEADER_PATH = BUILD_DIR / "cascade_csirx.h"

DEFAULT_GEOMETRY_FALLBACK: dict[str, int] = {
    "TEST_NUM_ADC_SAMPLES": 256,
    "TEST_NUM_LOOPS": 64,
    "TEST_NUM_FRAMES": 0,
    "TEST_NUM_RX": 4,
    "TEST_NUM_TX": 3,
    "TEST_BYTES_PER_ADC_SAMPLE": 4,
    "TEST_NUM_CHIRPS": 6,
}

_TI_REF_CHIRP_TX_MAP_6 = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2))


@dataclass(frozen=True)
class CaptureSettings:
    """Fixed runtime settings for this capture project."""

    post_proc: Path
    ccs_dss: Path
    dss_js: Path
    ccxml: Path
    mss_bin: Path
    capture_json: Path
    dca_cli: Path
    data_dir: Path
    stop_file: Path
    app_header: Path
    geometry_fallback: dict[str, int]
    out_dir: Path
    num_devices: int = 2
    raw_bin_prefix: str = "am273x_capture"
    uart_ps1_name: str = "_uart_stop.ps1"
    uart_port: Optional[str] = "COM20"
    max_frames: int = 0
    frame_bytes_override: int = 0
    skip_load: bool = False
    sync_wait: float = 60.0
    diag: bool = False


@dataclass(frozen=True)
class CascadeFrameGeometry:
    """Geometry parsed from the fixed shortrange cfg."""

    num_adc_samples: int
    bytes_per_sample: int
    iq_swap: bool
    chan_interleaved: bool
    num_rx_per_device: int
    num_tx_per_device: int
    num_devices: int
    chirp_start_idx: int
    chirp_end_idx: int
    num_loops: int
    chirp_tx_map: tuple[tuple[int, int], ...]

    @property
    def num_chirps_per_loop(self) -> int:
        # Purpose: return chirps per loop from frame indices. Input: none. Output: integer chirp count.
        return self.chirp_end_idx - self.chirp_start_idx + 1

    @property
    def num_rx_total(self) -> int:
        # Purpose: return total RX channels across devices. Input: none. Output: integer RX count.
        return self.num_rx_per_device * self.num_devices

    @property
    def num_tx_total(self) -> int:
        # Purpose: return total TX channels across devices. Input: none. Output: integer TX count.
        return self.num_tx_per_device * self.num_devices

    @property
    def num_virtual_antennas(self) -> int:
        # Purpose: return total virtual antennas. Input: none. Output: integer virtual antenna count.
        return self.num_tx_total * self.num_rx_total

    @property
    def bytes_per_chirp(self) -> int:
        # Purpose: return bytes in one chirp block. Input: none. Output: integer byte count.
        return self.num_adc_samples * self.bytes_per_sample * self.num_rx_total

    @property
    def bytes_per_frame(self) -> int:
        # Purpose: return bytes in one frame. Input: none. Output: integer byte count.
        return self.bytes_per_chirp * self.num_chirps_per_loop * self.num_loops

    def describe(self) -> str:
        # Purpose: format geometry for console output. Input: none. Output: multi-line description string.
        lines = [
            "--- CascadeFrameGeometry ---",
            f"  ADC samples / chirp / RX : {self.num_adc_samples}",
            f"  Bytes / sample           : {self.bytes_per_sample}"
            f"  ({'complex' if self.bytes_per_sample == 4 else 'real'} int16)",
            f"  IQ swap (SampleSwap)     : {self.iq_swap}"
            f"  ({'Q,I' if self.iq_swap else 'I,Q'} raw order)",
            f"  Chan interleaved         : {self.chan_interleaved}",
            f"  RX per device            : {self.num_rx_per_device}",
            f"  TX per device            : {self.num_tx_per_device}",
            f"  Devices                  : {self.num_devices}",
            f"  Total RX                 : {self.num_rx_total}",
            f"  Total TX (TDM)           : {self.num_tx_total}",
            f"  Virtual antennas (TX*RX) : {self.num_virtual_antennas}",
            f"  Chirps / loop            : {self.num_chirps_per_loop}",
            f"  Loops / frame            : {self.num_loops}",
            f"  Bytes / chirp block      : {self.bytes_per_chirp:,}",
            f"  Bytes / frame            : {self.bytes_per_frame:,}",
            f"  Output shape [NTX,NL,NRX,N]: "
            f"({self.num_tx_total},{self.num_loops},{self.num_rx_total},{self.num_adc_samples})",
            "  Chirp -> TX map (loop-major time order):",
        ]
        for index, (device, tx) in enumerate(self.chirp_tx_map):
            global_tx = device * self.num_tx_per_device + tx + 1
            lines.append(
                f"    chirp {index} -> Dev{device} TX{tx + 1}"
                f"  (global TX{global_tx},  txEnable={1 << tx})"
            )
        lines.append(
            f"  Linear chirp index: loop*{self.num_chirps_per_loop} + chirp_in_loop"
            f"  (0 .. {self.num_loops * self.num_chirps_per_loop - 1})"
        )
        lines.append("-----------------------------------------------------")
        return "\n".join(lines)


def create_capture_settings(
    *,
    out_dir: Optional[Path] = None,
    uart_port: Optional[str] = "COM20",
    max_frames: int = 0,
    frame_bytes_override: int = 0,
    skip_load: bool = False,
    sync_wait: float = 60.0,
    diag: bool = False,
) -> CaptureSettings:
    # Purpose: build the fixed runtime settings object. Input: optional overrides. Output: CaptureSettings.
    local_dca_cli = PROJECT_ROOT / "DCA1000EVM_CLI_Control.exe"
    fixed_dca_cli = Path(r"C:\ti\mmwave_studio_03_00_00_14\mmWaveStudio\PostProc\DCA1000EVM_CLI_Control.exe")
    if fixed_dca_cli.is_file():
        dca_cli = fixed_dca_cli
    elif local_dca_cli.is_file():
        dca_cli = local_dca_cli
    else:
        dca_cli = PROJECT_ROOT.parent / "DCA1000EVM_CLI_Control.exe"
    return CaptureSettings(
        post_proc=PROJECT_ROOT,
        ccs_dss=Path(r"C:\ti\ccs2050\ccs\ccs_base\scripting\bin\dss.bat"),
        dss_js=DSS_SCRIPT_PATH.resolve(),
        ccxml=CCXML_PATH.resolve(),
        mss_bin=MSS_BIN_PATH.resolve(),
        capture_json=DCA_CAPTURE_JSON.resolve(),
        dca_cli=dca_cli.resolve(),
        data_dir=DATA_DIR,
        stop_file=PROJECT_ROOT / ".stop_capture",
        app_header=APP_HEADER_PATH.resolve(),
        geometry_fallback=dict(DEFAULT_GEOMETRY_FALLBACK),
        out_dir=(out_dir or FRAMES_DIR).resolve(),
        uart_port=uart_port,
        max_frames=max_frames,
        frame_bytes_override=frame_bytes_override,
        skip_load=skip_load,
        sync_wait=sync_wait,
        diag=diag,
    )


def replace_capture_settings(settings: CaptureSettings, **kwargs) -> CaptureSettings:
    # Purpose: safely clone settings with selected overrides. Input: base settings and keyword overrides. Output: CaptureSettings.
    declared = {field.name for field in settings.__dataclass_fields__.values()}
    return replace(settings, **{key: value for key, value in kwargs.items() if key in declared})


def _cfg_tokens(text: str, keyword: str) -> list[str] | None:
    # Purpose: read one cfg command line by keyword. Input: cfg text and command keyword. Output: token list or None.
    for raw in text.splitlines():
        line = raw.split("%")[0].strip()
        if not line:
            continue
        parts = line.split()
        if parts[0] == keyword:
            return parts[1:]
    return None


def load_cfg_text(cfg_path: Path = CASCADE_CFG_PATH) -> str:
    # Purpose: load the fixed cascade cfg text. Input: optional cfg path. Output: cfg file contents.
    return cfg_path.read_text(encoding="utf-8", errors="replace")


def get_profile_cfg_tokens(cfg_path: Path = CASCADE_CFG_PATH) -> list[str]:
    # Purpose: find the active profileCfg row for the frame chirps. Input: cfg path. Output: profileCfg tokens.
    text = load_cfg_text(cfg_path)
    frame_cfg = _cfg_tokens(text, "frameCfg")
    if frame_cfg is None:
        raise ValueError("frameCfg not found")
    chirp_low, chirp_high = int(frame_cfg[0]), int(frame_cfg[1])

    profile_ids: set[int] = set()
    for raw_line in text.splitlines():
        parts = raw_line.split("%")[0].strip().split()
        if len(parts) < 4 or parts[0] != "chirpCfg":
            continue
        chirp_start, chirp_end, profile_id = int(parts[1]), int(parts[2]), int(parts[3])
        if chirp_start <= chirp_high and chirp_end >= chirp_low:
            profile_ids.add(profile_id)

    if len(profile_ids) > 1:
        raise ValueError(
            f"frameCfg chirps {chirp_low}-{chirp_high} reference multiple profileIds {sorted(profile_ids)}"
        )
    if not profile_ids:
        raise ValueError(f"No chirpCfg overlaps frameCfg chirps {chirp_low}-{chirp_high}")

    profile_id = next(iter(profile_ids))
    for raw_line in text.splitlines():
        parts = raw_line.split("%")[0].strip().split()
        if len(parts) > 1 and parts[0] == "profileCfg" and int(parts[1]) == profile_id:
            return parts[1:]
    raise ValueError(f"profileCfg id={profile_id} not found")


def parse_cascade_cfg(cfg_path: Path = CASCADE_CFG_PATH) -> CascadeFrameGeometry:
    # Purpose: parse the fixed radar cfg into frame geometry. Input: cfg path. Output: CascadeFrameGeometry.
    text = load_cfg_text(cfg_path)

    channel_cfg = _cfg_tokens(text, "channelCfg")
    if channel_cfg is None:
        raise ValueError(f"'channelCfg' not found in {cfg_path}")
    rx_mask = int(channel_cfg[0])
    tx_mask = int(channel_cfg[1])
    cascading = int(channel_cfg[2])
    num_devices = 2 if cascading else 1
    num_rx_per_device = bin(rx_mask).count("1")
    num_tx_per_device = bin(tx_mask).count("1")

    adc_cfg = _cfg_tokens(text, "adcCfg")
    if adc_cfg is None:
        raise ValueError(f"'adcCfg' not found in {cfg_path}")
    bytes_per_sample = 4 if int(adc_cfg[1]) == 1 else 2

    adcbuf_cfg = _cfg_tokens(text, "adcbufCfg")
    if adcbuf_cfg is None:
        raise ValueError(f"'adcbufCfg' not found in {cfg_path}")
    iq_swap = False
    chan_interleaved = bool(int(adcbuf_cfg[3]))

    profile_cfg = get_profile_cfg_tokens(cfg_path)
    num_adc_samples = int(profile_cfg[9])

    frame_cfg = _cfg_tokens(text, "frameCfg")
    if frame_cfg is None:
        raise ValueError(f"'frameCfg' not found in {cfg_path}")
    chirp_start, chirp_end, num_loops = int(frame_cfg[0]), int(frame_cfg[1]), int(frame_cfg[2])

    chirp_cfgs: dict[int, tuple[int, int]] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("%")[0].strip()
        if not line:
            continue
        parts = line.split()
        if parts[0] != "chirpCfg":
            continue
        if len(parts) < 9:
            raise ValueError(f"{cfg_path}: malformed chirpCfg: {line!r}")
        cfg_start = int(parts[1])
        cfg_end = int(parts[2])
        tx_enable = int(parts[-1])
        if tx_enable == 0 or (tx_enable & (tx_enable - 1)) != 0:
            raise ValueError(f"{cfg_path}: chirpCfg txEnable={tx_enable} must select exactly one TX")
        tx_bit = (tx_enable - 1).bit_length()
        device = 0 if num_devices <= 1 else cfg_start // num_tx_per_device
        if device >= num_devices:
            raise ValueError(f"{cfg_path}: chirpCfg at chirp {cfg_start} maps to invalid device {device}")
        for chirp_index in range(cfg_start, cfg_end + 1):
            chirp_cfgs[chirp_index] = (device, tx_bit)

    missing = [index for index in range(chirp_start, chirp_end + 1) if index not in chirp_cfgs]
    if missing:
        raise ValueError(f"{cfg_path}: missing chirpCfg entries for chirps {missing}")

    chirp_tx_map = tuple(chirp_cfgs[index] for index in range(chirp_start, chirp_end + 1))
    if (
        cascading
        and num_devices == 2
        and len(chirp_tx_map) == 6
        and num_tx_per_device == 3
        and chirp_tx_map != _TI_REF_CHIRP_TX_MAP_6
    ):
        print(
            f"Note: {cfg_path.name}: TX order {chirp_tx_map} differs from reference {_TI_REF_CHIRP_TX_MAP_6}. "
            "Extraction follows the cfg."
        )

    return CascadeFrameGeometry(
        num_adc_samples=num_adc_samples,
        bytes_per_sample=bytes_per_sample,
        iq_swap=iq_swap,
        chan_interleaved=chan_interleaved,
        num_rx_per_device=num_rx_per_device,
        num_tx_per_device=num_tx_per_device,
        num_devices=num_devices,
        chirp_start_idx=chirp_start,
        chirp_end_idx=chirp_end,
        num_loops=num_loops,
        chirp_tx_map=chirp_tx_map,
    )


def apply_raw_iq_order_override(geom: CascadeFrameGeometry, mode: str) -> CascadeFrameGeometry:
    # Purpose: override IQ order handling for raw samples. Input: geometry and mode string. Output: updated geometry.
    if mode == "cfg":
        return geom
    if mode == "i-first":
        return replace(geom, iq_swap=False)
    if mode == "q-first":
        return replace(geom, iq_swap=True)
    raise ValueError(f"Unknown mode: {mode!r}")


def expected_frame_shape(geom: CascadeFrameGeometry) -> tuple[int, int, int, int]:
    # Purpose: report the extracted frame tensor shape. Input: geometry. Output: (tx, loops, rx, adc).
    return (geom.num_tx_total, geom.num_loops, geom.num_rx_total, geom.num_adc_samples)


def nyquist_beat_range_m(profile_tokens: list[str], c_light: float = 299_792_458.0) -> float:
    # Purpose: compute maximum unambiguous range. Input: active profileCfg tokens. Output: range in meters.
    slope_hz_per_s = float(profile_tokens[7]) * 1e12
    sample_rate_hz = float(profile_tokens[10]) * 1e3
    return c_light * sample_rate_hz / (4.0 * slope_hz_per_s)


def range_axis_m(n_range: int, cfg_path: Path = CASCADE_CFG_PATH, c_light: float = 299_792_458.0) -> np.ndarray:
    # Purpose: build the physical range axis. Input: fft size and cfg path. Output: float64 range axis in meters.
    profile_tokens = get_profile_cfg_tokens(cfg_path)
    slope_hz_per_s = float(profile_tokens[7]) * 1e12
    sample_rate_hz = float(profile_tokens[10]) * 1e3
    freq = np.fft.fftshift(np.fft.fftfreq(n_range, 1.0 / sample_rate_hz))
    return (c_light * freq / (2.0 * slope_hz_per_s)).astype(np.float64)


def velocity_axis_m_per_s(
    n_doppler: int,
    geom: CascadeFrameGeometry,
    cfg_path: Path = CASCADE_CFG_PATH,
    c_light: float = 299_792_458.0,
) -> np.ndarray:
    # Purpose: build the physical Doppler axis. Input: fft size, geometry, cfg path. Output: float64 velocity axis.
    profile_tokens = get_profile_cfg_tokens(cfg_path)
    lam = c_light / (float(profile_tokens[1]) * 1e9)
    chirp_time_s = (float(profile_tokens[2]) + float(profile_tokens[4])) * 1e-6
    loop_time_s = geom.num_chirps_per_loop * chirp_time_s
    velocity_resolution = lam / (2.0 * n_doppler * loop_time_s)
    return ((np.arange(n_doppler, dtype=np.float64) - n_doppler // 2) * velocity_resolution).astype(np.float64)


def unambiguous_velocity_halfspan_mps(
    geom: CascadeFrameGeometry,
    cfg_path: Path = CASCADE_CFG_PATH,
    c_light: float = 299_792_458.0,
) -> float:
    # Purpose: compute the TDM unambiguous velocity limit. Input: geometry and cfg path. Output: max abs velocity.
    profile_tokens = get_profile_cfg_tokens(cfg_path)
    lam = c_light / (float(profile_tokens[1]) * 1e9)
    loop_time_s = geom.num_chirps_per_loop * (float(profile_tokens[2]) + float(profile_tokens[4])) * 1e-6
    return lam / (4.0 * loop_time_s)


__all__ = [
    "APP_HEADER_PATH",
    "BUILD_DIR",
    "CAPTURE_DIR",
    "CASCADE_CFG_PATH",
    "CCXML_PATH",
    "CONFIG_DIR",
    "CaptureSettings",
    "CascadeFrameGeometry",
    "DCA_CAPTURE_JSON",
    "DEFAULT_GEOMETRY_FALLBACK",
    "DSS_SCRIPT_PATH",
    "FRAMES_DIR",
    "MSS_BIN_PATH",
    "PROJECT_ROOT",
    "apply_raw_iq_order_override",
    "create_capture_settings",
    "expected_frame_shape",
    "get_profile_cfg_tokens",
    "load_cfg_text",
    "nyquist_beat_range_m",
    "parse_cascade_cfg",
    "range_axis_m",
    "replace_capture_settings",
    "unambiguous_velocity_halfspan_mps",
    "velocity_axis_m_per_s",
]
