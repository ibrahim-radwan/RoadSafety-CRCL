"""
capture_d455_radar_sync.py — D455 colour capture for radar/camera ground-truth pairing.

CHANGES FROM capture_d455_standalone.py, and why each matters for pairing
--------------------------------------------------------------
1. TIMESTAMPS ARE NOW ABSOLUTE, NOT RELATIVE.
       old:  elapsed = time.monotonic() - start_time
   That is seconds since THIS PROCESS started. The radar process has a
   different start_time, so the two timelines shared no origin and the
   streams could not be aligned at all.

   On Linux time.monotonic() IS CLOCK_MONOTONIC - counted since boot and
   shared by every process on the machine. The common epoch was always
   there; subtracting start_time threw it away. We now log the raw value.

   CLOCK_MONOTONIC is preferred over time.time(): wall clock can step
   under NTP correction mid-session, silently corrupting alignment.

2. THREE TIMESTAMPS PER FRAME, not one.
       t_arrival  host CLOCK_MONOTONIC when wait_for_frames() returned
       rs_ts      librealsense frame timestamp (ms)
       rs_domain  which clock rs_ts is in
   librealsense may report hardware, system, or global time depending on
   kernel metadata support. rs_ts accounts for USB transfer latency and is
   usually the better estimate - but only if the domain is known. Record
   both and let Stage 3 decide from measurement rather than assumption.

3. JPEG INSTEAD OF PNG.
   Annotation needs visually faithful images, not bit-exact ones. At
   30 fps this is 12.0 -> 2.7 MB/s. The CPU saving matters as much as the
   storage: on the Jetson, image encoding competes with a radar receiver
   that drops packets silently when starved.

4. WRITING IS OFF THE CAPTURE THREAD.
   The old loop encoded and wrote inline, so any disk or encode stall
   directly caused a dropped camera frame. Frames now go to a queue and a
   writer thread handles them.

5. DROPPED FRAMES ARE DETECTED AND REPORTED.
   RealSense frame numbers are sequential; gaps mean drops. Previously
   invisible. A silently truncated ground-truth stream is worse than none.

6. SIDECAR CSV.
   Filenames carry the index only; all timing goes to frames.csv, which is
   what the Stage 4 association tool consumes.

USAGE
    python capture_d455_radar_sync.py --output-dir ./cam --duration 30 --fps 15
"""
from __future__ import annotations

import argparse
import csv
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D455 colour capture with shared-clock timestamps")
    p.add_argument("--output-dir", type=Path,
                   default=Path("recordings") / time.strftime("%Y_%m_%d_%H_%M_%S"))
    p.add_argument("--duration", type=float, help="capture duration in seconds")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--serial", help="D455 serial number")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    # 15 fps is the sensible floor. Worst-case pairing error is half a frame
    # period: at 1.4 m/s that is 4.7 cm at 15 fps (0.7 of a 6.6 cm range bin)
    # but 17.5 cm at 4 fps - larger than the radar can resolve. Do NOT match
    # the camera rate to the radar rate.
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--png", action="store_true",
                   help="write lossless PNG instead of JPEG (4x storage and CPU)")
    return p.parse_args()


class Writer(threading.Thread):
    """Encode and write off the capture thread, so a disk stall drops no frames."""

    def __init__(self, outdir: Path, png: bool, quality: int):
        super().__init__(daemon=True)
        self.q: queue.Queue = queue.Queue(maxsize=240)   # ~8 s at 30 fps
        self.outdir, self.png, self.quality = outdir, png, quality
        self.written = 0
        self.dropped_full = 0        # frames discarded because the queue filled
        self.errors = 0
        # NOT self._stop: threading.Thread already defines a private _stop()
        # method, and Thread.join() calls it via _wait_for_tstate_lock().
        # Shadowing it with an Event gives
        #     TypeError: 'Event' object is not callable
        # at join() - which fired in the finally block and prevented
        # fh.close() and the session summary from ever running.
        self._stop_evt = threading.Event()

    def submit(self, idx: int, img: np.ndarray) -> bool:
        try:
            self.q.put_nowait((idx, img))
            return True
        except queue.Full:
            # Better to lose a frame than to block the capture loop and lose
            # several - but this must be COUNTED, not silent.
            self.dropped_full += 1
            return False

    def run(self) -> None:
        ext = ".png" if self.png else ".jpg"
        params = [] if self.png else [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        while not (self._stop_evt.is_set() and self.q.empty()):
            try:
                idx, img = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            path = self.outdir / f"frame_{idx:06d}{ext}"
            try:
                if cv2.imwrite(str(path), img, params):
                    self.written += 1
                else:
                    self.errors += 1
            except Exception:
                self.errors += 1

    def finish(self) -> None:
        self._stop_evt.set()


def main() -> None:
    a = parse_args()
    if a.duration is not None and a.duration <= 0:
        raise ValueError("--duration must be > 0")
    if a.max_frames is not None and a.max_frames <= 0:
        raise ValueError("--max-frames must be > 0")

    a.output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    if a.serial:
        config.enable_device(a.serial)
    config.enable_stream(rs.stream.color, a.width, a.height, rs.format.bgr8, a.fps)

    profile = pipeline.start(config)
    device = profile.get_device()

    # Ask librealsense for host-synchronised timestamps where supported. This
    # is what makes rs_ts comparable to the radar's CLOCK_MONOTONIC stamps.
    try:
        for s in device.query_sensors():
            if s.supports(rs.option.global_time_enabled):
                s.set_option(rs.option.global_time_enabled, 1)
    except Exception as exc:                                  # noqa: BLE001
        print(f"  note: could not enable global time ({exc})")

    print(f"Camera : {device.get_info(rs.camera_info.name)}")
    print(f"Output : {a.output_dir}")
    print(f"Format : {a.width}x{a.height} @ {a.fps} fps, "
          f"{'PNG' if a.png else f'JPEG q{a.jpeg_quality}'}")
    print("Ctrl+C to stop.")

    writer = Writer(a.output_dir, a.png, a.jpeg_quality)
    writer.start()

    csv_path = a.output_dir / "frames.csv"
    fh = open(csv_path, "w", newline="", encoding="utf-8")
    cw = csv.writer(fh)
    cw.writerow(["idx", "filename", "t_arrival_monotonic", "rs_timestamp_ms",
                 "rs_domain", "rs_frame_number", "queued"])

    t_session0 = time.monotonic()
    idx = 0
    prev_rs_num = None
    gaps = 0
    missing = 0
    ext = ".png" if a.png else ".jpg"

    try:
        while True:
            if a.duration is not None and time.monotonic() - t_session0 >= a.duration:
                break
            if a.max_frames is not None and idx >= a.max_frames:
                break

            frames = pipeline.wait_for_frames()
            # Stamp IMMEDIATELY on return, before any processing, so the value
            # is as close to acquisition as the host can observe.
            t_arrival = time.monotonic()

            cf = frames.get_color_frame()
            if not cf:
                continue

            rs_num = int(cf.get_frame_number())
            if prev_rs_num is not None and rs_num != prev_rs_num + 1:
                # Sequential frame numbers; a gap means the device or USB
                # dropped frames. Previously invisible.
                gaps += 1
                missing += max(0, rs_num - prev_rs_num - 1)
            prev_rs_num = rs_num

            try:
                rs_ts = float(cf.get_timestamp())
                rs_dom = str(cf.get_frame_timestamp_domain())
            except Exception:                                 # noqa: BLE001
                rs_ts, rs_dom = float("nan"), "unknown"

            img = np.asanyarray(cf.get_data())
            queued = writer.submit(idx, img.copy())

            cw.writerow([idx, f"frame_{idx:06d}{ext}", f"{t_arrival:.6f}",
                         f"{rs_ts:.3f}", rs_dom, rs_num, int(queued)])
            idx += 1
    except KeyboardInterrupt:
        print()
    finally:
        # Each step guarded: a failure in one must not prevent the others.
        # Previously an exception in join() skipped fh.close(), leaving the
        # manifest unflushed and the summary unprinted - losing the very
        # diagnostics that reveal a bad session.
        for label, fn in (("pipeline.stop", pipeline.stop),
                          ("writer.finish", writer.finish),
                          ("writer.join", lambda: writer.join(timeout=15.0)),
                          ("csv.close", fh.close)):
            try:
                fn()
            except Exception as exc:                          # noqa: BLE001
                print(f"  teardown warning ({label}): {type(exc).__name__}: {exc}")

    elapsed = time.monotonic() - t_session0
    print("\n" + "=" * 60)
    print("  CAMERA SESSION")
    print("=" * 60)
    print(f"  frames captured : {idx}")
    print(f"  frames written  : {writer.written}")
    print(f"  elapsed         : {elapsed:.1f} s   ({idx/elapsed if elapsed else 0:.1f} fps)")
    print(f"  device drops    : {gaps} gaps, {missing} frames "
          f"(from RealSense frame-number gaps)")
    print(f"  queue overflows : {writer.dropped_full}")
    print(f"  write errors    : {writer.errors}")
    print(f"  manifest        : {csv_path}")
    if missing or writer.dropped_full or writer.errors:
        print("\n  WARNING: frames were lost. Ground truth is incomplete for the")
        print("  affected interval - check before annotating.")


if __name__ == "__main__":
    main()
