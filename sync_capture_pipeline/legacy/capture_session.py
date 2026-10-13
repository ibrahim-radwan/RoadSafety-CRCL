"""
capture_session.py — run a synchronised radar + camera collection session.

WHY TWO PROCESSES AND NOT TWO THREADS
    At 8 fps the radar receiver handles ~17,300 packets/second on a socket
    with no flow control: if the reader stalls, the kernel drops packets
    SILENTLY. JPEG encoding in the same interpreter would contend for the
    GIL with that socket loop. Separate processes get separate GILs, so
    image work cannot starve the receiver. This is the single most
    important structural decision here.

    It also lets the camera run in its conda env (pyrealsense2, opencv,
    Python 3.8) while the radar script runs anywhere - it imports only
    stdlib and has no third-party dependencies at all.

ORDERING
    Camera starts FIRST and stops LAST, so every radar frame has camera
    coverage on both sides. The radar spends ~5.5 s on DCA setup before
    START_RECORD, so the camera naturally leads it regardless.

WHAT THIS DOES NOT DO
    It does not align the two streams. Both write CLOCK_MONOTONIC stamps to
    their own frames.csv; pairing happens offline (Stage 4), after the
    fixed radar-vs-camera offset has been MEASURED (Stage 3). Do not assume
    that offset is zero - a radar frame takes ~54 ms to transmit, plus LVDS
    and network latency, and that bias shifts every pair the same way.

USAGE
    python capture_session.py --secs 30 --out ./sessions/run01
    python capture_session.py --secs 30 --out ./sessions/run01 \\
        --camera-python /home/jetson/miniconda3/envs/rel_1_py_3_8/bin/python
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synchronised radar + camera session")
    p.add_argument("--secs", type=float, required=True,
                   help="radar recording duration; the camera runs longer")
    p.add_argument("--out", type=Path, required=True, help="session directory")
    p.add_argument("--camera-python", default=sys.executable,
                   help="interpreter with pyrealsense2 (the conda env)")
    p.add_argument("--radar-python", default=sys.executable,
                   help="interpreter for the radar script (stdlib only)")
    p.add_argument("--camera-script", default=str(HERE / "capture_d455_radar_sync.py"))
    p.add_argument("--radar-script", default=str(HERE / "dca_capture_radar_sync.py"))
    p.add_argument("--fps", type=int, default=15, help="camera fps")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--lead", type=float, default=2.0,
                   help="seconds the camera leads and trails the radar")
    p.add_argument("--note", default="", help="free-text session note")
    p.add_argument("--delay", type=float, default=0.0,
                   help="countdown BEFORE anything starts, so the operator can "
                        "disconnect peripherals and clear the scene. Without "
                        "this, the first ~30-60 s of every session records the "
                        "operator leaving and a stationary vehicle.")
    p.add_argument("--log", type=Path, default=None,
                   help="also write all output here. Essential for unattended "
                        "runs: a radar that dies mid-ride is otherwise invisible "
                        "until the vehicle returns.")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    cam_dir = a.out / "camera"
    rad_dir = a.out / "radar"
    for d in (cam_dir, rad_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Radar setup is ~5.5 s (3.0 FPGA re-init + 1.0 + 0.5 + 0.5 + 0.5) before
    # START_RECORD, so the camera must cover that plus lead and trail.
    RADAR_SETUP = 6.0
    cam_secs = a.secs + RADAR_SETUP + 2 * a.lead

    print("=" * 70)
    print("  CAPTURE SESSION")
    print("=" * 70)
    print(f"  output        : {a.out}")
    print(f"  radar         : {a.secs:.0f} s recording")
    print(f"  camera        : {cam_secs:.0f} s at {a.fps} fps "
          f"(covers radar setup + {a.lead:.0f} s either side)")
    print(f"  camera python : {a.camera_python}")
    print(f"  radar python  : {a.radar_python}")
    print()

    # Tee everything to a log file when asked. On an unattended ride nobody
    # is watching the terminal, so the per-second packet counts - the only
    # live indication the radar is alive - would otherwise be lost.
    log_fh = None
    if a.log:
        a.log.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(a.log, "w", encoding="utf-8")

    def emit(line: str = "") -> None:
        print(line)
        if log_fh:
            log_fh.write(line + "\n")
            log_fh.flush()      # flush per line: a crash must not lose the tail

    if a.delay > 0:
        emit(f"\n  Starting in {a.delay:.0f} s - disconnect peripherals and "
             f"clear the scene now.")
        for rem in range(int(a.delay), 0, -1):
            print(f"\r    {rem:>3} s ...", end="", flush=True)
            time.sleep(1.0)
        print("\r" + " " * 20 + "\r", end="")
        emit("  starting.")

    t_session0 = time.monotonic()
    started = time.time()

    cam_cmd = [a.camera_python, a.camera_script,
               "--output-dir", str(cam_dir),
               "--duration", str(cam_secs),
               "--fps", str(a.fps),
               "--width", str(a.width), "--height", str(a.height)]
    rad_cmd = [a.radar_python, a.radar_script,
               "--secs", str(a.secs), "--out", str(rad_dir)]

    print("  starting camera ...")
    cam = subprocess.Popen(cam_cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)

    # Let the camera open the device and start streaming before the radar
    # begins, so its first frames are not part of RealSense start-up.
    time.sleep(a.lead)
    if cam.poll() is not None:
        print("  CAMERA EXITED EARLY - aborting before touching the radar:")
        print(cam.stdout.read() if cam.stdout else "")
        return 1

    print("  starting radar ...\n")
    t_radar_launch = time.monotonic()
    rad = subprocess.Popen(rad_cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)

    rad_out = []
    try:
        if rad.stdout:
            for line in rad.stdout:
                emit("  [radar] " + line.rstrip())
                rad_out.append(line)
        rad.wait()
    except KeyboardInterrupt:
        print("\n  interrupted - stopping radar cleanly ...")
        # SIGINT, not kill: the radar script must still send STOP_RECORD,
        # or the DCA stays armed and the next session starts in an unknown
        # state.
        rad.send_signal(signal.SIGINT)
        try:
            rad.wait(timeout=15)
        except subprocess.TimeoutExpired:
            rad.kill()

    # Let the camera TRAIL the radar. camera_secs budgeted a trailing window
    # but the wrapper previously SIGINTed the camera the instant the radar
    # exited, so the last radar frame had no camera frame after it and was
    # the only one without two-sided support.
    print(f"\n  radar finished - camera trailing {a.lead:.0f} s ...")
    time.sleep(a.lead)
    print("  stopping camera ...")
    # SIGINT so the camera drains its writer queue and closes frames.csv.
    cam.send_signal(signal.SIGINT)
    try:
        cam_out = cam.communicate(timeout=30)[0]
    except subprocess.TimeoutExpired:
        cam.kill()
        cam_out = cam.communicate()[0]
    for line in (cam_out or "").splitlines():
        emit("  [camera] " + line)

    meta = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "t_session0_monotonic": t_session0,
        "t_radar_launch_monotonic": t_radar_launch,
        "radar_secs": a.secs,
        "camera_secs": cam_secs,
        "camera_fps": a.fps,
        "camera_resolution": [a.width, a.height],
        "radar_exit_code": rad.returncode,
        "camera_exit_code": cam.returncode,
        "note": a.note,
        "clock": "CLOCK_MONOTONIC (time.monotonic), shared across processes",
        "pairing": "NOT applied. Offsets must be measured (Stage 3) before "
                   "association (Stage 4).",
    }
    meta["delay_s"] = a.delay
    (a.out / "session.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("  SESSION COMPLETE")
    print("=" * 70)
    n_bin = len(list(rad_dir.glob("frame_*.bin")))
    n_img = len(list(cam_dir.glob("frame_*.jpg"))) + len(list(cam_dir.glob("frame_*.png")))
    print(f"  radar frames  : {n_bin}   ({rad_dir})")
    print(f"  camera frames : {n_img}   ({cam_dir})")
    print(f"  metadata      : {a.out / 'session.json'}")
    ok = rad.returncode == 0 and n_bin > 0 and n_img > 0
    if not ok:
        print("\n  SESSION INCOMPLETE - check the logs above before relying on this data.")
    if n_bin and n_img:
        print(f"\n  ratio         : {n_img/n_bin:.1f} camera frames per radar frame")
        print("  Next: Stage 3 - record a sharp event both sensors observe")
        print("  (clap in view / LED flash) and measure the fixed offset.")
    if log_fh:
        log_fh.close()
        print(f"  log           : {a.log}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
