"""
capture_session.py — run a synchronised radar + ZED 2i camera + RS-Fairy LiDAR session.

See SYNC_DESIGN_3_SENSORS.md for the timing model. The radar + D455 version
this replaces is kept in legacy/.

WHY THREE PROCESSES AND NOT THREADS
    The radar receiver handles ~17,300 packets/second (8 fps) on a socket
    with no flow control: if the reader stalls, the kernel drops packets
    SILENTLY. Each recorder runs in its own interpreter, so camera and LiDAR
    work cannot contend for the radar receiver's GIL.

    The ZED and LiDAR recorders run in the conda env (pyzed); the radar
    script imports only the stdlib and runs anywhere.

ORDERING
    LiDAR and camera start FIRST and stop LAST, so every radar frame has
    coverage on both sides. The launcher waits until both report that they
    are recording before it starts the radar (~5.5 s of DCA setup).

    Each recorder runs in its own process group. Ctrl+C therefore reaches
    only this launcher, which stops the radar first, lets the others trail,
    and then stops them - every stop is a SIGINT, so each one finalises.

WHAT THIS DOES NOT DO
    It does not align the streams. Every recorder logs CLOCK_MONOTONIC
    (the camera alongside its SDK capture time); conversion and pairing
    happen offline, after the fixed offsets have been MEASURED (Stage 3).

USAGE
    python capture_session.py --out ./sessions/run01                 # until Ctrl+C
    python capture_session.py --out ./sessions/run01 --secs 300 --delay 60 \\
        --note "outdoor, 2 pedestrians crossing"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import board_health
import radar_firmware

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SENSOR_PYTHON = "/home/jetson/miniconda3/envs/rel_1_py_3_8/bin/python"
ZED_DIR = ROOT / "camera_standalone_capture" / "ZED_2i_Sensor"
LIDAR_DIR = ROOT / "lidar_standalone_capture"

# Lines each recorder prints once it is actually recording.
READY = {"camera": "Recording 1920x1080", "lidar": "Recording RS-Fairy"}
# Storage per hour at 8 fps (SYNC_DESIGN_3_SENSORS.md section 9).
SD_MOUNT = Path("/mnt/sdcard")
RADAR_RATE_RE = re.compile(r"t\+\s*\d+s\s+([\d,]+) pkt/s")
# Radar watchdog: the DCA1000 occasionally comes up throttled (seen 2026-10-11:
# ~2,970 pkt/s instead of ~17,300; it buffered for 102 s, then stopped
# capturing, and arrival times stopped meaning anything). Stop early instead
# of recording minutes of unusable radar data.
RADAR_LOW_FRACTION, RADAR_LOW_SECONDS = 0.5, 5
GB_PER_HOUR = {"radar": 90, "camera": 5, "lidar": 18}   # measured 2026-10-10 at 8 fps: 90.6 / 3.9 / 17.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synchronised radar + camera + LiDAR session")
    p.add_argument("--out", type=Path, required=True, help="new session directory")
    p.add_argument("--secs", type=float, default=0.0,
                   help="radar recording length; 0 = until Ctrl+C (default)")
    p.add_argument("--lead", type=float, default=3.0,
                   help="seconds camera/LiDAR record before and after the radar")
    p.add_argument("--delay", type=float, default=0.0,
                   help="countdown BEFORE anything starts, so the operator can "
                        "disconnect peripherals and clear the scene")
    p.add_argument("--note", default="", help="free-text session note")
    p.add_argument("--log", type=Path, default=None,
                   help="log file (default <out>/session.log). Essential for "
                        "unattended runs: a sensor that dies mid-ride is "
                        "otherwise invisible until the vehicle returns.")
    p.add_argument("--sensor-python", default=SENSOR_PYTHON,
                   help="interpreter with pyzed (the conda env), for camera and LiDAR")
    p.add_argument("--radar-python", default=sys.executable,
                   help="interpreter for the radar script (stdlib only)")
    p.add_argument("--camera-stage", type=Path, default=None,
                   help="record the camera to this directory (e.g. on the eMMC) and move it into "
                        "<out>/camera after the session. Keeps the ZED off the disk that takes the "
                        "radar's 25 MB/s, which otherwise stalls its writes and drops frames")
    p.add_argument("--no-radar", action="store_true", help="bench test without the radar")
    p.add_argument("--no-camera", action="store_true")
    p.add_argument("--no-lidar", action="store_true")
    p.add_argument("--ready-timeout", type=float, default=45.0,
                   help="seconds to wait for camera/LiDAR to start recording")
    p.add_argument("--min-free-gb", type=float, default=5.0,
                   help="stop the session if free space falls below this")
    return p.parse_args()


def clock_pair() -> dict:
    m1 = time.monotonic_ns()
    r = time.time_ns()
    m2 = time.monotonic_ns()
    return {"monotonic_ns": (m1 + m2) // 2, "realtime_ns": r}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


class Recorder:
    """One child process; its output is prefixed, teed to the log, and
    watched for the line that says it is recording."""

    def __init__(self, name, cmd, cwd, emit):
        self.name, self.cmd, self.cwd, self.emit = name, cmd, cwd, emit
        self.ready = threading.Event()
        self.proc = None
        self.reader = None
        self.rates = []             # radar: per-second packet counts, as the receiver reports them

    def start(self):
        self.proc = subprocess.Popen(
            self.cmd, cwd=self.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True)
        self.reader = threading.Thread(target=self._read, name=f"{self.name}-out", daemon=True)
        self.reader.start()

    def _read(self):
        for line in self.proc.stdout:
            line = line.rstrip()
            self.emit(f"  [{self.name}] {line}")
            if self.name in READY and READY[self.name] in line:
                self.ready.set()
            if self.name == "radar":
                m = RADAR_RATE_RE.search(line)
                if m:
                    self.rates.append(int(m.group(1).replace(",", "")))

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def interrupt(self):
        if self.alive():
            self.proc.send_signal(signal.SIGINT)

    def wait(self, timeout) -> int:
        if self.proc is None:
            return None
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.emit(f"  {self.name}: no exit after {timeout:.0f} s - killing")
            self.proc.kill()
            self.proc.wait()
        if self.reader:
            self.reader.join(timeout=5)
        return self.proc.returncode


def main() -> int:
    a = parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        print(f"{a.out} already exists and is not empty - use a new session name")
        return 1
    # Sessions on the SD card must land ON the card: if it is missing, the
    # bare /mnt/sdcard directory on the eMMC would silently take the data.
    resolved = a.out.resolve()
    if resolved == SD_MOUNT or SD_MOUNT in resolved.parents:
        try:
            os.listdir(SD_MOUNT)                 # triggers the systemd automount
        except OSError:
            pass
        if not os.path.ismount(SD_MOUNT):
            print(f"ERROR: {SD_MOUNT} is not mounted - insert the SD card (or check /etc/fstab). Not recording.")
            return 2
    a.out.mkdir(parents=True, exist_ok=True)
    log_path = a.log or a.out / "session.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w", encoding="utf-8")
    lock = threading.Lock()

    def emit(line: str = "") -> None:
        with lock:
            print(line, flush=True)
            log_fh.write(line + "\n")
            log_fh.flush()      # flush per line: a crash must not lose the tail

    sensors = [n for n, off in (("lidar", a.no_lidar), ("camera", a.no_camera), ("radar", a.no_radar))
               if not off]
    if not sensors:
        print("all sensors disabled - nothing to record")
        return 1

    emit("=" * 70)
    emit("  CAPTURE SESSION")
    emit("=" * 70)
    emit(f"  output   : {a.out}")
    emit(f"  sensors  : {', '.join(sensors)}")
    emit(f"  radar    : {'until Ctrl+C' if a.secs <= 0 else f'{a.secs:.0f} s'}")
    emit(f"  note     : {a.note}")

    # ── preflight ───────────────────────────────────────────────────────────
    problems = []
    thermal = {"samples": 0, "tj_max_c": None, "fan_max_pct": None, "start": None, "end": None,
               "warn_c": board_health.WARN_C, "hot_c": board_health.HOT_C}
    free_gb = shutil.disk_usage(a.out).free / 1e9
    rate = sum(GB_PER_HOUR[s] for s in sensors)
    emit(f"  storage  : {free_gb:.1f} GB free, ~{rate} GB/hour "
         f"-> ~{max(0.0, free_gb - a.min_free_gb) / rate * 60:.0f} min of recording")
    if a.secs > 0 and free_gb - a.min_free_gb < rate * a.secs / 3600:
        problems.append(f"not enough space for {a.secs:.0f} s")
    board = board_health.snapshot()
    thermal["start"] = board
    emit(f"  board    : {board_health.log_line(board)}")
    if board["level"] == "hot":
        problems.append(f"board too hot to start (tj {board['tj_c']:.0f} C >= {board_health.HOT_C:.0f} C) "
                        f"- cool it down / shade it first")
    elif board["level"] == "warm":
        emit(f"  WARNING: board is warm (tj {board['tj_c']:.0f} C) - check airflow and shade")
    cam_dir = a.out / "camera"
    if a.camera_stage and "camera" in sensors:
        cam_dir = a.camera_stage / a.out.name / "camera"
        if cam_dir.exists():
            problems.append(f"camera staging dir {cam_dir} already exists")
        else:
            cam_dir.parent.mkdir(parents=True, exist_ok=True)
            stage_gb = shutil.disk_usage(cam_dir.parent).free / 1e9
            emit(f"  camera   : staged on {cam_dir.parent} ({stage_gb:.1f} GB free, "
                 f"~{stage_gb / GB_PER_HOUR['camera']:.1f} h), moved to {a.out / 'camera'} afterwards")
            if a.secs > 0 and stage_gb < GB_PER_HOUR["camera"] * a.secs / 3600 + 2:
                problems.append(f"not enough space for the camera on {cam_dir.parent}")
    radar_fw = radar_firmware.read_flashed() if "radar" in sensors else None
    if "radar" in sensors:
        if radar_fw is None:
            emit(f"  WARNING: no {radar_firmware.FLASHED_RECORD} - the radar firmware in flash is unknown, "
                 f"so processing will assume {radar_firmware.LEGACY_VARIANT}")
        else:
            emit(f"  radar fw : {radar_fw.get('variant')}  "
                 f"({radar_fw.get('frame_periodicity_ms')} ms frames, near floor {radar_fw.get('hpf_floor_m')} m, "
                 f"md5 {radar_fw.get('appimage_md5', '?')[:8]})")
            if str(radar_fw.get("data_valid", "yes")).lower() != "yes":
                emit(f"  WARNING: firmware_map.json says this image's data is not valid: {radar_fw['data_valid']}")
    rmem = int(Path("/proc/sys/net/core/rmem_max").read_text())
    if rmem < 32 * 1024 * 1024 and ("radar" in sensors or "lidar" in sensors):
        emit(f"  WARNING: net.core.rmem_max is {rmem:,} B; run "
             f"'sudo sysctl -w net.core.rmem_max=33554432'")
    if "camera" in sensors:
        r = subprocess.run([a.sensor_python, "-c",
                            "import pyzed.sl as sl; print(len(sl.Camera.get_device_list()))"],
                           capture_output=True, text=True)
        n_zed = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "?"
        if n_zed != "1":
            problems.append(f"ZED SDK sees {n_zed} cameras (expected 1) - unplug and "
                            f"replug the ZED, then retry")
    for pb in problems:
        emit(f"  PREFLIGHT FAILED: {pb}")
    if problems:
        return 1

    if a.delay > 0:
        emit(f"\n  Starting in {a.delay:.0f} s - disconnect peripherals and clear the scene now.")
        for rem in range(int(a.delay), 0, -1):
            print(f"\r    {rem:>3} s ...", end="", flush=True)
            time.sleep(1.0)
        print("\r" + " " * 20 + "\r", end="")

    # Ctrl+C reaches only this process (children have their own sessions).
    stop_requested = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_requested.set())
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.set())

    clocks = {"start": clock_pair()}
    started_local = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    recs: dict = {}
    if "lidar" in sensors:
        recs["lidar"] = Recorder("lidar", [a.sensor_python, "-u", "record_lidar_raw.py",
                                           "--session-dir", str((a.out / "lidar").resolve())],
                                 LIDAR_DIR, emit)
    if "camera" in sensors:
        recs["camera"] = Recorder("camera", [a.sensor_python, "-u", "recorder.py",
                                             "--session-dir", str(cam_dir.resolve())],
                                  ZED_DIR, emit)
    if "radar" in sensors:
        recs["radar"] = Recorder("radar", [a.radar_python, "-u", str(HERE / "dca_capture_radar_sync.py"),
                                           "--secs", str(a.secs), "--out", str((a.out / "radar").resolve())],
                                 HERE, emit)

    stop_reason = "radar finished"
    t_radar_start = None
    try:
        # ── camera + LiDAR first, wait until both are recording ─────────────
        for name in ("lidar", "camera"):
            if name in recs:
                emit(f"\n  starting {name} ...")
                recs[name].start()
        deadline = time.monotonic() + a.ready_timeout
        for name in ("lidar", "camera"):
            if name not in recs:
                continue
            while not recs[name].ready.wait(0.2):
                if not recs[name].alive() or time.monotonic() > deadline or stop_requested.is_set():
                    raise RuntimeError(f"{name} did not start recording")
            emit(f"  {name} is recording")

        # ── radar ───────────────────────────────────────────────────────────
        if "radar" in recs:
            time.sleep(a.lead)
            emit("\n  starting radar ...")
            t_radar_start = clock_pair()
            recs["radar"].start()

        # ── run: until radar ends, Ctrl+C, a sensor dies, or the disk fills ─
        next_disk = next_temp = time.monotonic()
        period_ms = float((radar_fw or {}).get("frame_periodicity_ms") or 125)
        expected_pps = 3_145_728 / 1456 / (period_ms / 1000)
        while True:
            rates = recs["radar"].rates[2:] if "radar" in recs else []   # skip start-up seconds
            if len(rates) >= RADAR_LOW_SECONDS and \
                    all(v < RADAR_LOW_FRACTION * expected_pps for v in rates[-RADAR_LOW_SECONDS:]):
                stop_reason = (f"radar rate low ({rates[-1]:,} pkt/s, expected ~{expected_pps:,.0f}) - "
                               f"DCA1000 throttled; power-cycle the radar + DCA1000")
                emit(f"\n  ERROR: {stop_reason} - stopping the session")
                break
            if time.monotonic() >= next_temp:
                # Board temperature every 5 s: summer rides in an enclosure. A
                # warning only - stopping would not protect data already saved.
                next_temp += 5.0
                b = board_health.snapshot()
                thermal["samples"] += 1
                thermal["tj_max_c"] = max(thermal["tj_max_c"] or 0.0, b["tj_c"] or 0.0)
                thermal["fan_max_pct"] = max(thermal["fan_max_pct"] or 0, b["fan_pct"] or 0)
                thermal["end"] = b
                emit("  " + board_health.log_line(b))
                if b["level"] == "hot":
                    emit(f"  WARNING: board is HOT (tj {b['tj_c']:.0f} C) - near throttling; "
                         f"consider stopping and cooling it")
            if "radar" in recs and not recs["radar"].alive():
                rc = recs["radar"].proc.returncode
                if rc != 0:
                    stop_reason = f"radar exited with code {rc}"
                    emit(f"\n  ERROR: {stop_reason} - stopping the session")
                break
            if "radar" not in recs and a.secs > 0 and \
                    time.monotonic_ns() - clocks["start"]["monotonic_ns"] > a.secs * 1e9:
                stop_reason = "duration reached"
                break
            if stop_requested.is_set():
                stop_reason = "operator stop (Ctrl+C)"
                break
            dead = [n for n in ("lidar", "camera") if n in recs and not recs[n].alive()]
            if dead:
                stop_reason = f"{', '.join(dead)} exited early"
                emit(f"\n  ERROR: {stop_reason} - stopping the session")
                break
            if time.monotonic() >= next_disk:
                next_disk += 10.0
                if shutil.disk_usage(a.out).free / 1e9 < a.min_free_gb:
                    stop_reason = f"free space below {a.min_free_gb} GB"
                    emit(f"\n  ERROR: {stop_reason} - stopping the session")
                    break
            time.sleep(0.2)
    except RuntimeError as exc:
        stop_reason = f"startup failed: {exc}"
        emit(f"\n  ERROR: {stop_reason}")

    # ── stop: radar first, then let camera/LiDAR trail, then stop them ──────
    codes = {}
    if "radar" in recs and recs["radar"].proc is not None:
        if recs["radar"].alive():
            emit("\n  stopping radar ...")
            recs["radar"].interrupt()
        codes["radar"] = recs["radar"].wait(timeout=180)
        if not stop_reason.startswith(("startup", "lidar", "camera")):
            emit(f"  radar stopped - camera/LiDAR trailing {a.lead:.0f} s ...")
            time.sleep(a.lead)
    clocks["stop"] = clock_pair()
    for name in ("lidar", "camera"):
        if name in recs and recs[name].proc is not None:
            emit(f"  stopping {name} ...")
            recs[name].interrupt()
    if "lidar" in recs and recs["lidar"].proc is not None:
        codes["lidar"] = recs["lidar"].wait(timeout=60)
    if "camera" in recs and recs["camera"].proc is not None:
        # The ZED verifies the finalized SVO2 after stopping (~0.55x the
        # recording time), so allow for it.
        rec_s = (clocks["stop"]["monotonic_ns"] - clocks["start"]["monotonic_ns"]) / 1e9
        emit(f"  camera is verifying its SVO2 (up to ~{rec_s * 0.6:.0f} s) ...")
        codes["camera"] = recs["camera"].wait(timeout=120 + rec_s)

    # ── staged camera -> session directory ──────────────────────────────────
    camera_staged = None
    if cam_dir != a.out / "camera" and cam_dir.exists():
        t0 = time.monotonic()
        size_gb = sum(f.stat().st_size for f in cam_dir.rglob("*") if f.is_file()) / 1e9
        emit(f"  moving camera data ({size_gb:.2f} GB) from {cam_dir} to {a.out / 'camera'} ...")
        try:
            shutil.move(str(cam_dir), str(a.out / "camera"))
            try:
                cam_dir.parent.rmdir()
            except OSError:
                pass
            camera_staged = {"staged_in": str(cam_dir), "moved_s": round(time.monotonic() - t0, 1),
                             "gb": round(size_gb, 3)}
            emit(f"  camera data moved in {camera_staged['moved_s']} s")
        except Exception as exc:
            camera_staged = {"staged_in": str(cam_dir), "error": str(exc)}
            emit(f"  ERROR moving camera data: {exc} - it is still in {cam_dir}")

    # ── manifest ────────────────────────────────────────────────────────────
    files = {
        "capture_session.py": HERE / "capture_session.py",
        "dca_capture_radar_sync.py": HERE / "dca_capture_radar_sync.py",
        "zed/recorder.py": ZED_DIR / "recorder.py",
        "zed/config.yaml": ZED_DIR / "config.yaml",
        "lidar/record_lidar_raw.py": LIDAR_DIR / "record_lidar_raw.py",
        "lidar/config.yaml": LIDAR_DIR / "rs_driver" / "config" / "config.yaml",
    }
    meta = {
        "started_local": started_local,
        "host": socket.gethostname(),
        "note": a.note,
        "sensors": sensors,
        "stop_reason": stop_reason,
        "exit_codes": codes,
        "clocks": dict(clocks, radar_start=t_radar_start),
        "clock": "CLOCK_MONOTONIC is the session reference; each pair is "
                 "(monotonic, realtime) sampled back-to-back",
        "args": {k: str(v) for k, v in vars(a).items()},
        "commands": {n: r.cmd for n, r in recs.items()},
        "dirs": {n: n for n in recs},
        "net_core_rmem_max": rmem,
        "thermal": thermal,
        "radar_firmware": radar_fw,
        "camera_staged": camera_staged,
        "tool_sha256": {k: sha256(p) for k, p in files.items()},
        "pairing": "NOT applied. Offsets must be measured (Stage 3) before association.",
    }
    (a.out / "session.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ── summary ─────────────────────────────────────────────────────────────
    emit("\n" + "=" * 70)
    emit("  SESSION SUMMARY")
    emit("=" * 70)
    emit(f"  stop reason : {stop_reason}")
    if thermal["samples"]:
        emit(f"  board       : tj max {thermal['tj_max_c']:.1f} C, fan max {thermal['fan_max_pct']}% "
             f"({thermal['samples']} samples){'  - WARNING: got hot' if thermal['tj_max_c'] >= board_health.WARN_C else ''}")
    ok = not stop_reason.startswith(("startup", "lidar", "camera", "free"))
    for name in sensors:
        code = codes.get(name)
        detail = ""
        if name == "radar":
            detail = f"{len(list((a.out / 'radar').glob('frame_*.bin')))} frames"
        elif name == "camera" and (a.out / "camera" / "capture_status.json").is_file():
            st = json.loads((a.out / "camera" / "capture_status.json").read_text())
            detail = (f"complete={st.get('complete')} frames={st.get('recorded_frames')} "
                      f"fps={st.get('observed_fps') or 0:.2f} dropped={st.get('estimated_missing_camera_frames')}")
            ok = ok and bool(st.get("complete"))
        elif name == "lidar" and (a.out / "lidar" / "metadata.json").is_file():
            md = json.loads((a.out / "lidar" / "metadata.json").read_text())
            detail = (f"complete={md.get('complete')} packets={md.get('packets')} "
                      f"lost={md.get('msop_lost')}")
            ok = ok and bool(md.get("complete"))
        ok = ok and code == 0
        emit(f"  {name:7s}: exit {code}  {detail}")
    emit(f"  manifest    : {a.out / 'session.json'}")
    emit(f"  log         : {log_path}")
    if not ok:
        emit("\n  SESSION INCOMPLETE - check the log above before relying on this data.")
    log_fh.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
