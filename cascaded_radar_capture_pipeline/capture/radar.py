"""
Radar-side helpers for firmware load, header parsing, UART stop, and capture wait logic.

This module owns:
- firmware header parsing
- frame-byte derivation from the firmware header
- CCS/DSS firmware load
- UART stop command
- radar-side wait loop while DCA capture is running
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from .config import CaptureSettings, DEFAULT_GEOMETRY_FALLBACK

if TYPE_CHECKING:
    from .dca import FileMonitor

_DEFINE_RE = re.compile(r"^\s*#define\s+(TEST_\w+)\s+\(?\s*(\d+)[Uu]?\s*\)?")


def parse_app_header(header_path: Path, fallback: Optional[dict[str, int]] = None) -> dict[str, int]:
    # Purpose: read frame constants from the firmware header. Input: header path and fallback values. Output: parsed dict.
    values = dict(fallback or DEFAULT_GEOMETRY_FALLBACK)
    if not header_path.is_file():
        return values
    for line in header_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = _DEFINE_RE.match(line)
        if match and match.group(1) in values:
            values[match.group(1)] = int(match.group(2))
    return values


def compute_frame_geometry(cfg: dict[str, int], num_devices: int = 2) -> dict[str, int]:
    # Purpose: derive frame sizes from parsed firmware constants. Input: TEST_* values and device count. Output: geometry dict.
    num_rx = cfg["TEST_NUM_RX"] * num_devices
    adc_samples = cfg["TEST_NUM_ADC_SAMPLES"]
    chirps_per_frame = cfg["TEST_NUM_CHIRPS"] * cfg["TEST_NUM_LOOPS"]
    bytes_per_sample = cfg["TEST_BYTES_PER_ADC_SAMPLE"]
    frame_bytes = num_rx * adc_samples * chirps_per_frame * bytes_per_sample
    return {
        "num_rx": num_rx,
        "adc_samples": adc_samples,
        "chirps_per_frame": chirps_per_frame,
        "bytes_per_sample": bytes_per_sample,
        "num_loops": cfg["TEST_NUM_LOOPS"],
        "num_chirps": cfg["TEST_NUM_CHIRPS"],
        "num_frames": cfg["TEST_NUM_FRAMES"],
        "frame_bytes": frame_bytes,
    }


def resolve_frame_bytes(settings: CaptureSettings, frame_bytes_override: int = 0) -> tuple[dict[str, int], int]:
    # Purpose: decide the effective frame size for capture splitting. Input: settings and optional override. Output: geometry dict and frame bytes.
    app_cfg = parse_app_header(settings.app_header, settings.geometry_fallback)
    geom = compute_frame_geometry(app_cfg, settings.num_devices)
    override = frame_bytes_override if frame_bytes_override > 0 else settings.frame_bytes_override
    frame_bytes = override if override > 0 else geom["frame_bytes"]
    return geom, frame_bytes


def load_firmware(mss_bin: Path, settings: CaptureSettings, *, diag: bool = False) -> bool:
    # Purpose: load the MSS image with CCS/DSS. Input: firmware path and settings. Output: True when DSS reports success.
    wait_ms = "30000" if diag else "1500"
    dss_cmd = [
        "cmd",
        "/c",
        str(settings.ccs_dss),
        str(settings.dss_js),
        str(settings.ccxml),
        str(mss_bin),
        "NONE",
        wait_ms,
    ]
    workdir = str(settings.dss_js.parent)

    if diag:
        proc = subprocess.Popen(
            dss_cmd,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            print(line, flush=True)
            lines.append(line)
        proc.wait()
        stdout = "\n".join(lines)
    else:
        result = subprocess.run(dss_cmd, cwd=workdir, capture_output=True, text=True, timeout=300)
        stdout = (result.stdout or "").strip()
        print(stdout)
        if (result.stderr or "").strip():
            print(result.stderr.strip(), file=sys.stderr)

    ok = "OK: reset/load/run completed" in stdout
    if not ok:
        print("WARNING: Firmware may not have loaded!", file=sys.stderr)
    return ok


def uart_send_stop(port: str, settings: CaptureSettings, *, baudrate: int = 115200) -> bool:
    # Purpose: send the firmware stop character over UART. Input: COM port, settings, baud rate. Output: True on success.
    ps_script = (
        f'$sp = New-Object System.IO.Ports.SerialPort("{port}", {baudrate}, "None", 8, "One")\n'
        "$sp.DtrEnable = $true\n"
        "$sp.RtsEnable = $true\n"
        "$sp.Open()\n"
        "Start-Sleep -Milliseconds 200\n"
        "$sp.Write('s')\n"
        "Start-Sleep -Milliseconds 500\n"
        "$sp.Close()\n"
        "$sp.Dispose()\n"
        "Write-Host 'Sent stop command (s) to firmware'\n"
    )
    ps_path = settings.post_proc / settings.uart_ps1_name
    ps_path.write_text(ps_script, encoding="utf-8")
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps_path)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        out = (result.stdout or "").strip()
        if out:
            print(f"  {out}")
        if (result.stderr or "").strip():
            print(f"  PS: {result.stderr.strip()}", file=sys.stderr)
        return result.returncode == 0
    except Exception as exc:
        print(f"  UART stop error: {exc}", file=sys.stderr)
        return False
    finally:
        try:
            ps_path.unlink(missing_ok=True)
        except OSError:
            pass


def start_and_load(
    settings: CaptureSettings,
    *,
    mss_bin: Optional[Path] = None,
    skip_load: bool = False,
    diag: Optional[bool] = None,
) -> bool:
    # Purpose: optionally load firmware before capture. Input: settings and optional overrides. Output: True when ready.
    if skip_load:
        return True
    image = mss_bin or settings.mss_bin
    diag_mode = settings.diag if diag is None else diag
    print(f"\n=== Loading firmware: {image.name} ===")
    return load_firmware(image, settings, diag=diag_mode)


def print_monitor_progress(monitor: "FileMonitor", elapsed_s: float) -> None:
    # Purpose: print a compact streaming progress line. Input: file monitor and elapsed seconds. Output: console text only.
    frame_bytes = monitor.frame_bytes
    frame_count = monitor.frame_idx
    raw_bytes = monitor.total_bytes
    payload_bytes = frame_count * frame_bytes
    extra_bytes = raw_bytes - payload_bytes
    print(
        f"  [{elapsed_s:.0f}s] frames={frame_count}  raw_stream={raw_bytes:,} B  "
        f"split_payload={payload_bytes:,} B  overhead+partial={extra_bytes:,} B",
        flush=True,
    )


def capture(
    settings: CaptureSettings,
    monitor: "FileMonitor",
    *,
    sync_wait: float,
    skip_load: bool,
    max_frames: int,
) -> None:
    # Purpose: wait for radar streaming and then hold until stop or frame limit. Input: settings, monitor, timing flags. Output: none.
    if not skip_load and sync_wait > 0:
        print(f"\n=== Waiting up to {sync_wait:.0f}s for firmware init + streaming ===")
        print("  (firmware: MMWave_open -> config -> calibrate -> start)")
        start_time = time.time()
        last_print = start_time
        data_seen_at = 0.0
        while time.time() - start_time < sync_wait:
            if settings.stop_file.exists():
                break
            now = time.time()
            if monitor.total_bytes > 0 and data_seen_at == 0.0:
                data_seen_at = now - start_time
                print(f"  [{data_seen_at:.0f}s] >> Data arriving! <<", flush=True)
            if now - last_print >= 5.0:
                print_monitor_progress(monitor, now - start_time)
                last_print = now
            time.sleep(0.2)
        elapsed = time.time() - start_time
        print(f"  Init done ({elapsed:.0f}s). Frames so far: {monitor.frame_idx}")
        if monitor.total_bytes == 0:
            print("  WARNING: No data received! Firmware may still be initializing.", file=sys.stderr)
            print("  Try sync_wait 60 or 90", file=sys.stderr)

    print("\n=== Capturing frames (Ctrl-C or --stop to end) ===", flush=True)
    try:
        capture_start = time.time()
        last_print = capture_start
        while True:
            if settings.stop_file.exists():
                print("\n  Stop signal received.")
                settings.stop_file.unlink(missing_ok=True)
                break
            if max_frames and monitor.frame_idx >= max_frames:
                print(f"\n  Reached {max_frames} frames.")
                break
            now = time.time()
            if now - last_print >= 5.0:
                print_monitor_progress(monitor, now - capture_start)
                last_print = now
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n  Stopped by user.")


def stop(settings: CaptureSettings, *, uart_port: Optional[str] = None) -> bool:
    # Purpose: stop radar streaming via UART if a port is configured. Input: settings and optional UART override. Output: True if stop succeeds or is skipped.
    port = settings.uart_port if uart_port is None else uart_port
    if not port:
        return True
    print("\n=== Sending stop to firmware ===")
    return uart_send_stop(port, settings)


__all__ = [
    "capture",
    "compute_frame_geometry",
    "load_firmware",
    "parse_app_header",
    "print_monitor_progress",
    "resolve_frame_bytes",
    "start_and_load",
    "stop",
    "uart_send_stop",
]
