"""
DCA-side helpers for configuration, record control, frame splitting, and capture orchestration.

This module owns:
- DCA1000 CLI calls
- raw bin monitoring and frame splitting
- cleanup and validation
- end-to-end capture start/stop orchestration with the radar module
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .config import CaptureSettings
from . import radar


def _frame_stat_str(payload: bytes) -> str:
    # Purpose: compute a one-line per-frame ADC stat fingerprint. Input: raw frame bytes (int16 IQ). Output: short stat string.
    # Uses numpy if available; otherwise falls back to a manual loop on a downsampled view.
    try:
        import numpy as _np
        a = _np.frombuffer(payload, dtype=_np.int16)
        if a.size < 2 or (a.size % 2) != 0:
            return "stat=[malformed payload]"
        iq = a.reshape(-1, 2).astype(_np.float32)
        mag = _np.hypot(iq[:, 0], iq[:, 1])
        return (
            f"stat=[mean|x|={float(mag.mean()):6.1f} std={float(mag.std()):6.1f} "
            f"max={int(mag.max()):5d} I_dc={float(iq[:, 0].mean()):+5.2f} "
            f"Q_dc={float(iq[:, 1].mean()):+5.2f}]"
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"stat=[err: {exc}]"


@dataclass
class CaptureResult:
    """Summary of one capture run."""

    frame_idx: int
    total_bytes: int
    frame_bytes: int
    monitor_error: Optional[str]
    out_dir: Optional[Path]


def dca_json_ethernet_ips(settings: CaptureSettings) -> tuple[str, str]:
    # Purpose: read DCA and PC IPs from the capture JSON. Input: settings. Output: (dca_ip, pc_ip).
    try:
        data = json.loads(settings.capture_json.read_text(encoding="utf-8"))
        root = data.get("DCA1000Config") or {}
        ec = root.get("ethernetConfig") or {}
        eu = root.get("ethernetConfigUpdate") or {}
        return str(ec.get("DCA1000IPAddress", "?")), str(eu.get("systemIPAddress", "?"))
    except Exception:
        return "?", "?"


def build_cli_json_text(settings: CaptureSettings) -> str:
    # Purpose: build a TI-CLI-safe capture JSON with the current project data path. Input: settings. Output: JSON text.
    source_data = json.loads(settings.capture_json.read_text(encoding="utf-8"))
    root = source_data.setdefault("DCA1000Config", {})
    capture = root.setdefault("captureConfig", {})
    data_dir = settings.data_dir.resolve()
    capture["fileBasePath"] = str(data_dir) + "\\"
    capture["filePrefix"] = settings.raw_bin_prefix
    return json.dumps(source_data, indent=2)


def staged_json_matches_settings(path: Path, settings: CaptureSettings) -> bool:
    # Purpose: check whether an existing staged JSON already matches the project data path. Input: JSON path and settings. Output: True when usable.
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        capture = (data.get("DCA1000Config") or {}).get("captureConfig") or {}
        actual_base = str(capture.get("fileBasePath", "")).rstrip("\\/")
        expected_base = str(settings.data_dir.resolve()).rstrip("\\/")
        actual_prefix = str(capture.get("filePrefix", ""))
        return actual_base == expected_base and actual_prefix == settings.raw_bin_prefix
    except Exception:
        return False


def resolve_cli_json_path(settings: CaptureSettings) -> Path:
    # Purpose: stage a fresh TI-CLI JSON beside the executable. Input: settings. Output: CLI-safe JSON path.
    source = settings.capture_json.resolve()
    target = settings.dca_cli.parent / source.name
    if source == target:
        return target
    if target.exists() and staged_json_matches_settings(target, settings):
        return target
    staged_text = build_cli_json_text(settings)
    try:
        target.write_text(staged_text, encoding="utf-8")
        return target
    except PermissionError:
        print(
            f"  WARNING: Could not refresh staged DCA JSON at {target}; using existing file.",
            file=sys.stderr,
            flush=True,
        )
        return target


def cli_output_indicates_failure(text: str) -> bool:
    # Purpose: detect TI CLI failure text even when return code is zero. Input: combined stdout/stderr text. Output: True when it looks like a failure.
    lowered = (text or "").lower()
    return (
        "disconnected" in lowered
        or "timeout error" in lowered
        or "system is disconnected" in lowered
        or "command failed" in lowered
    )


def print_dca_connectivity_hint(settings: CaptureSettings) -> None:
    # Purpose: print quick troubleshooting details for DCA connectivity. Input: settings. Output: console text only.
    dca_ip, pc_ip = dca_json_ethernet_ips(settings)
    print(
        f"  DCA JSON: {settings.capture_json}\n"
        f"  Expect PC NIC static IP ~ {pc_ip}, DCA1000 at {dca_ip} (UDP 4096).\n"
        "  Check: Ethernet cable to DCA1000, firewall off for private net, ping DCA IP,\n"
        "  close mmWave Studio if it has the board open, power-cycle DCA1000.",
        file=sys.stderr,
        flush=True,
    )


def dca_cli(command: str, settings: CaptureSettings, *, timeout: float = 120) -> bool:
    # Purpose: run one DCA1000 CLI command. Input: command string, settings, timeout. Output: True on success.
    cli_json = resolve_cli_json_path(settings)
    if not settings.dca_cli.exists():
        print(f"  ERROR: DCA CLI executable not found: {settings.dca_cli}", flush=True)
        return False
    cmd = [str(settings.dca_cli), command, str(cli_json)]
    cwd = str(settings.dca_cli.parent)
    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)

    if command == "start_record":
        process = subprocess.Popen(cmd, cwd=cwd, creationflags=creationflags)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            print(f"  CLI {command}: TIMED OUT", flush=True)
            return False
        time.sleep(1)
        check = subprocess.run(
            ["tasklist", "/fi", "imagename eq DCA1000EVM_CLI_Record.exe"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        ok = "DCA1000EVM_CLI_Record" in (check.stdout or "")
        print(f"  CLI {command}: {'Success' if ok else 'FAILED'}", flush=True)
        return ok

    if command == "stop_record":
        try:
            result = subprocess.run(cmd, cwd=cwd, timeout=15, capture_output=True, text=True)
            out = ((result.stdout or "") + (result.stderr or "")).strip()
            print(f"  CLI {command}: {out}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"  CLI {command}: timed out", flush=True)
        return True

    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=max(5, int(timeout)))
    combined = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
    print(f"  CLI {command}: {' '.join(combined.split())}", flush=True)
    return result.returncode == 0 and not cli_output_indicates_failure(combined)


def dca_cli_retry(
    command: str,
    settings: CaptureSettings,
    *,
    attempts: int = 3,
    delay_s: float = 2.0,
    timeout: float = 120,
) -> bool:
    # Purpose: retry a DCA CLI command with delay. Input: command, settings, retry controls. Output: True on success.
    for attempt in range(attempts):
        if dca_cli(command, settings, timeout=timeout):
            return True
        if attempt + 1 < attempts:
            print(f"  Retrying {command} ({attempt + 2}/{attempts}) after {delay_s:.0f}s...", flush=True)
            time.sleep(delay_s)
    return False


def dca_configure_for_capture(settings: CaptureSettings) -> bool:
    # Purpose: bring DCA1000 into ready-to-record state. Input: settings. Output: True when all required steps succeed.
    if not dca_cli_retry("reset_fpga", settings, attempts=3, delay_s=2.0, timeout=90):
        return False
    time.sleep(1.5)
    if not dca_cli("fpga", settings, timeout=120):
        return False
    time.sleep(1.0)
    if not dca_cli("query_sys_status", settings, timeout=60):
        return False
    time.sleep(0.5)
    if not dca_cli("record", settings, timeout=120):
        return False
    time.sleep(0.5)
    return dca_cli("query_sys_status", settings, timeout=60)


def dca_start_recording(settings: CaptureSettings, *, timeout: float = 120) -> bool:
    # Purpose: start the DCA record process. Input: settings and timeout. Output: True on success.
    return dca_cli("start_record", settings, timeout=timeout)


def dca_stop_recording(settings: CaptureSettings) -> bool:
    # Purpose: stop the DCA record process. Input: settings. Output: True when stop command completes.
    return dca_cli("stop_record", settings)


def cleanup_leftover_dca_processes(settings: CaptureSettings) -> None:
    # Purpose: stop stale DCA helper processes before a new run. Input: settings. Output: none.
    try:
        cli_json = resolve_cli_json_path(settings)
        subprocess.run(
            [str(settings.dca_cli), "stop_record", str(cli_json)],
            cwd=str(settings.dca_cli.parent),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        pass
    for proc in ["DCA1000EVM_CLI_Record.exe", "DCA1000EVM_CLI_Control.exe"]:
        subprocess.run(["taskkill", "/f", "/t", "/im", proc], capture_output=True, timeout=10)
    time.sleep(2.5)


class FileMonitor(threading.Thread):
    """Watch DCA raw bin files and split them into one file per frame.
    
    If out_dir is None, frames are NOT written to disk; only the on_frame_saved
    callback is triggered with the frame bytes (for streaming mode).
    """

    def __init__(
        self,
        data_dir: Path,
        prefix: str,
        out_dir: Optional[Path],
        frame_bytes: int,
        max_frames: int,
        *,
        on_frame_saved: Optional[Callable[[Optional[Path], int, bytes], None]] = None,
        verbose: bool = True,
        warmup_frames: int = 0,
        frame_stats: bool = False,
    ):
        super().__init__(daemon=True)
        self.data_dir = data_dir
        self.prefix = prefix
        self.out_dir = out_dir
        self.frame_bytes = frame_bytes
        self.max_frames = max_frames
        self.frame_idx = 0
        self.total_bytes = 0
        self.stop_event = threading.Event()
        self.error: Optional[str] = None
        self._on_frame = on_frame_saved
        self._verbose = verbose
        self._warmup_remaining = max(int(warmup_frames), 0)
        self._warmup_total = self._warmup_remaining
        self._warmup_skipped = 0
        self._frame_stats = bool(frame_stats)

    def find_latest_bin(self) -> Optional[str]:
        # Purpose: find the newest raw capture bin matching the configured prefix. Input: none. Output: file path string or None.
        try:
            bins = list(self.data_dir.glob(f"{self.prefix}*.bin"))
        except OSError:
            return None
        if not bins:
            return None
        bins.sort(key=lambda path: path.name)
        return str(bins[-1])

    def run(self) -> None:
        # Purpose: continuously split incoming raw bytes into frame files. Input: thread state. Output: none.
        buffer = bytearray()
        buffer_pos = 0
        file_positions: dict[str, int] = {}
        last_file: Optional[str] = None
        try:
            while not self.stop_event.is_set():
                file_str = self.find_latest_bin()
                if file_str is None:
                    time.sleep(0.02)
                    continue

                if file_str != last_file:
                    last_file = file_str
                    file_positions.setdefault(file_str, 0)

                read_pos = file_positions[file_str]
                try:
                    size = os.path.getsize(file_str)
                except OSError:
                    time.sleep(0.005)
                    continue

                if size > read_pos:
                    try:
                        with open(file_str, "rb") as handle:
                            handle.seek(read_pos)
                            chunk = handle.read(size - read_pos)
                    except OSError:
                        time.sleep(0.005)
                        continue
                    file_positions[file_str] = read_pos + len(chunk)
                    buffer.extend(chunk)
                    self.total_bytes += len(chunk)

                while (len(buffer) - buffer_pos) >= self.frame_bytes:
                    frame_bytes_view = buffer[buffer_pos : buffer_pos + self.frame_bytes]
                    if self._warmup_remaining > 0:
                        # Discard this frame as firmware warm-up; do not write to disk,
                        # do not increment frame_idx, do not count toward max_frames.
                        self._warmup_remaining -= 1
                        self._warmup_skipped += 1
                        if self._verbose:
                            print(
                                f"  Discarded warm-up frame {self._warmup_skipped}/{self._warmup_total} "
                                f"({self.frame_bytes:,} B; not saved)",
                                flush=True,
                            )
                        buffer_pos += self.frame_bytes
                        continue
                    
                    # Optionally write frame to disk (only if out_dir is not None).
                    # For live streaming with disk writes disabled, out_dir is None and frames
                    # only trigger the callback (passed to FrameQueue via make_on_frame_saved).
                    out_path = None
                    if self.out_dir is not None:
                        out_path = self.out_dir / f"frame_{self.frame_idx:06d}.bin"
                        with open(out_path, "wb") as handle:
                            handle.write(frame_bytes_view)
                    
                    self.frame_idx += 1
                    if self._verbose and out_path is not None:
                        msg = f"  Saved {out_path.name}  ({self.frame_bytes:,} B)  [{self.frame_idx} frames]"
                        if self._frame_stats:
                            msg += "  " + _frame_stat_str(frame_bytes_view)
                        print(msg, flush=True)
                    elif self._verbose and out_path is None:
                        msg = f"  Frame {self.frame_idx:06d}  ({self.frame_bytes:,} B)  [streaming, no disk]"
                        if self._frame_stats:
                            msg += "  " + _frame_stat_str(frame_bytes_view)
                        print(msg, flush=True)
                    
                    # Call callback with frame bytes (for streaming pipeline).
                    # Pass bytes directly so callback doesn't need to read from disk.
                    if self._on_frame:
                        self._on_frame(out_path, self.frame_idx, bytes(frame_bytes_view))
                    
                    if self.max_frames and self.frame_idx >= self.max_frames:
                        self.stop_event.set()
                        return
                    buffer_pos += self.frame_bytes

                if buffer_pos > 4 * self.frame_bytes:
                    buffer = bytearray(buffer[buffer_pos:])
                    buffer_pos = 0
                time.sleep(0.001)
        except Exception as exc:
            self.error = str(exc)
            print(f"  [FileMonitor] ERROR: {exc}", file=sys.stderr)


def clean_capture_artifacts(settings: CaptureSettings, out_dir: Optional[Path], clear_frames: bool = True) -> None:
    # Purpose: remove stale raw capture files before a run. Input: settings, output dir, frame-clear flag. Output: none.
    for pattern in ["am273x_capture*.bin", "am273x__*.bin", "*_Raw_LogFile.csv"]:
        for old in settings.data_dir.glob(pattern):
            print(f"  Removing: {old.name}")
            old.unlink(missing_ok=True)
    if clear_frames and out_dir is not None:
        for old in out_dir.glob("frame_*.bin"):
            old.unlink(missing_ok=True)


def validate_saved_frames(out_dir: Path, frame_idx: int, frame_bytes: int, monitor_total_bytes: int) -> None:
    # Purpose: validate saved frame files after capture. Input: output dir, frame count, frame size, monitor bytes. Output: none.
    if frame_idx <= 0:
        print("\n=== Frame validation ===")
        print("  No frames saved (frame_idx=0).")
        return

    frames = sorted(out_dir.glob("frame_*.bin"))
    saved_count = len(frames)
    expected_total = frame_idx * frame_bytes
    verify_all = saved_count <= 5000
    mismatches: list[tuple[str, int]] = []
    saved_total = 0

    if verify_all:
        for path in frames:
            try:
                size = path.stat().st_size
            except OSError:
                continue
            saved_total += size
            if size != frame_bytes:
                mismatches.append((path.name, size))
    else:
        sample = frames[:3] + frames[-3:]
        for path in sample:
            try:
                size = path.stat().st_size
            except OSError:
                continue
            saved_total += size
            if size != frame_bytes:
                mismatches.append((path.name, size))
        saved_total = expected_total

    print("\n=== Frame validation (DCA -> per-frame split) ===")
    print(f"  frame_idx saved       : {frame_idx}")
    print(f"  Expected frame size  : {frame_bytes:,} B")
    print(f"  Expected saved bytes : {expected_total:,} B")
    if monitor_total_bytes >= 0:
        remainder = monitor_total_bytes - expected_total
        print(f"  Monitor read bytes   : {monitor_total_bytes:,} B")
        print("    (raw DCA record stream includes packet/FPGA framing, so it is larger than frame payload bytes)")
        print(f"  vs payload (N x frame): {remainder:+,} B  (overhead + any partial tail in buffer)")
    else:
        print("  Monitor read bytes   : (skipped; not available in --stop mode)")
    print(f"  Saved files found    : {saved_count if verify_all else '(sampled only)'}")
    print(f"  Saved total (sum)   : {saved_total:,} B")
    if mismatches:
        print(f"  MISMATCHES (size != frame_bytes): {len(mismatches)}")
        for name, size in mismatches[:10]:
            print(f"    - {name}: {size:,} B")
    else:
        print("  All checked frame sizes match expected size.")


def stop_capture(
    settings: CaptureSettings,
    *,
    out_dir: Optional[Path] = None,
    frame_bytes: Optional[int] = None,
    uart_port: Optional[str] = None,
) -> None:
    # Purpose: stop an active capture and validate the saved frames. Input: settings and optional overrides. Output: none.
    output_dir = Path(out_dir if out_dir is not None else settings.out_dir).resolve()
    if frame_bytes is None:
        _, effective_frame_bytes = radar.resolve_frame_bytes(settings, 0)
    else:
        effective_frame_bytes = frame_bytes
    uart = settings.uart_port if uart_port is None else uart_port
    settings.stop_file.write_text("stop", encoding="utf-8")
    print("=== Stopping capture ===")
    if uart:
        print(f"  Sending 's' to firmware on {uart}...")
        radar.uart_send_stop(uart, settings)
    dca_stop_recording(settings)
    validate_saved_frames(output_dir, len(sorted(output_dir.glob("frame_*.bin"))), effective_frame_bytes, -1)


def run_capture(
    settings: CaptureSettings,
    *,
    out_dir: Optional[Path] = None,
    max_frames: Optional[int] = None,
    skip_load: Optional[bool] = None,
    sync_wait: Optional[float] = None,
    uart_port: Optional[str] = None,
    frame_bytes_override: Optional[int] = None,
    diag: Optional[bool] = None,
    mss_bin: Optional[Path] = None,
    on_frame_saved: Optional[Callable[[Optional[Path], int, bytes], None]] = None,
    file_monitor_verbose: bool = True,
    warmup_frames: int = 0,
    frame_stats: bool = False,
) -> CaptureResult:
    # Purpose: run the full radar + DCA capture flow. Input: settings and optional overrides. Output: CaptureResult.
    # When out_dir is None (live streaming mode), frames are NOT written to disk.
    # When out_dir is set, frames are written and on_frame_saved callback is triggered with file path.
    output_dir = None if out_dir is None else Path(out_dir).resolve()
    effective_max_frames = settings.max_frames if max_frames is None else max_frames
    effective_skip_load = settings.skip_load if skip_load is None else skip_load
    effective_sync_wait = settings.sync_wait if sync_wait is None else sync_wait
    effective_uart = settings.uart_port if uart_port is None else uart_port
    effective_diag = settings.diag if diag is None else diag
    effective_frame_override = settings.frame_bytes_override if frame_bytes_override is None else frame_bytes_override

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.stop_file.unlink(missing_ok=True)

    geom, frame_bytes = radar.resolve_frame_bytes(settings, effective_frame_override)
    header_frame_bytes = int(geom["frame_bytes"])
    image = mss_bin or settings.mss_bin
    if not effective_skip_load and not image.exists():
        raise FileNotFoundError(f"Firmware not found: {image}")

    source = "header" if settings.app_header.is_file() else "fallback"
    print(f"Config   : {settings.app_header.name} ({source})")
    print(f"Firmware : {image.name}")
    print(f"Frame split: {frame_bytes:,} B  (each frame_*.bin payload; FileMonitor uses this)")
    print(
        f"  From firmware header: {geom['num_rx']} RX x {geom['adc_samples']} samp x "
        f"{geom['chirps_per_frame']} chirps x {geom['bytes_per_sample']} B => {header_frame_bytes:,} B"
    )
    if header_frame_bytes != frame_bytes:
        print(
            f"  WARNING: split size {frame_bytes:,} B != header {header_frame_bytes:,} B. "
            "If LVDS matches the header, frames are mis-aligned and RD/IQ will look wrong.",
            file=sys.stderr,
        )
    print(f"           {geom['num_chirps']} chirps/loop x {geom['num_loops']} loops x {settings.num_devices} device(s)")
    print(f"FW frames: {'infinite' if geom['num_frames'] == 0 else geom['num_frames']}")
    if output_dir is not None:
        print(f"Output   : {output_dir}  (frames written to disk)")
    else:
        print(f"Output   : None  (streaming mode, no disk writes)")
    print(f"Capture  : {'infinite (Ctrl-C or --stop to end)' if effective_max_frames == 0 else effective_max_frames}")
    if effective_uart:
        print(f"UART     : {effective_uart} (for stop command)")

    print("\nCleaning up...")
    cleanup_leftover_dca_processes(settings)
    clean_capture_artifacts(settings, output_dir, clear_frames=True)

    radar.start_and_load(settings, mss_bin=image, skip_load=effective_skip_load, diag=effective_diag)
    time.sleep(1.0)

    print("\n=== Configuring DCA1000 ===")
    if not dca_configure_for_capture(settings):
        print("\nFATAL: DCA1000 configuration failed (reset_fpga / fpga / record / link).", file=sys.stderr)
        print_dca_connectivity_hint(settings)
        raise RuntimeError("DCA1000 not reachable - fix Ethernet / JSON IPs, then retry.")

    print("\n=== Starting DCA1000 record ===")
    if not dca_start_recording(settings, timeout=120):
        print("\nFATAL: DCA1000 start_record failed.", file=sys.stderr)
        print_dca_connectivity_hint(settings)
        raise RuntimeError("DCA1000 start_record failed.")

    monitor = FileMonitor(
        settings.data_dir,
        settings.raw_bin_prefix,
        output_dir,
        frame_bytes,
        effective_max_frames or 0,
        on_frame_saved=on_frame_saved,
        verbose=file_monitor_verbose,
        warmup_frames=max(int(warmup_frames), 0),
        frame_stats=bool(frame_stats),
    )
    monitor.start()
    print(f"File monitor watching {settings.data_dir}")
    if warmup_frames > 0:
        print(
            f"  Warm-up: discarding first {int(warmup_frames)} frame(s) after sensorStart "
            "(firmware front-end calibration not yet steady-state).",
            flush=True,
        )
    if frame_stats:
        print(
            "  Per-frame ADC stats enabled (mean|x|, std, max, I_dc, Q_dc). "
            "Large frame-to-frame variation indicates non-stationary firmware state.",
            flush=True,
        )
    print(
        "  Split: each frame_*.bin = exactly one radar frame "
        f"({frame_bytes:,} B payload). Raw DCA .bin bytes include FPGA/packet headers.",
        flush=True,
    )

    radar.capture(
        settings,
        monitor,
        sync_wait=effective_sync_wait,
        skip_load=effective_skip_load,
        max_frames=effective_max_frames or 0,
    )

    monitor.stop_event.set()
    time.sleep(0.3)

    if effective_uart:
        radar.stop(settings, uart_port=effective_uart)

    payload_bytes = monitor.frame_idx * frame_bytes
    print(
        f"\n=== Done: {monitor.frame_idx} frames  raw_stream={monitor.total_bytes:,} B  "
        f"split_payload={payload_bytes:,} B ==="
    )
    if getattr(monitor, "_warmup_total", 0):
        print(
            f"  Warm-up frames discarded: {monitor._warmup_skipped}/{monitor._warmup_total}",
            flush=True,
        )
    if monitor.error:
        print(f"  Error: {monitor.error}", file=sys.stderr)

    dca_stop_recording(settings)
    if output_dir is not None:
        validate_saved_frames(output_dir, monitor.frame_idx, frame_bytes, monitor.total_bytes)
    settings.stop_file.unlink(missing_ok=True)
    return CaptureResult(
        frame_idx=monitor.frame_idx,
        total_bytes=monitor.total_bytes,
        frame_bytes=frame_bytes,
        monitor_error=monitor.error,
        out_dir=output_dir,
    )


__all__ = [
    "CaptureResult",
    "FileMonitor",
    "clean_capture_artifacts",
    "cleanup_leftover_dca_processes",
    "dca_cli",
    "dca_cli_retry",
    "dca_configure_for_capture",
    "dca_json_ethernet_ips",
    "dca_start_recording",
    "dca_stop_recording",
    "print_dca_connectivity_hint",
    "run_capture",
    "stop_capture",
    "validate_saved_frames",
]
