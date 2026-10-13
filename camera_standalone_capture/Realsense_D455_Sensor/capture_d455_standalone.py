import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture color frames from an Intel RealSense D455")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("recordings") / time.strftime("%Y_%m_%d_%H_%M_%S"),
        help="Directory for PNG frames (default: recordings/<timestamp>)",
    )
    parser.add_argument("--duration", type=float, help="Capture duration in seconds")
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames")
    parser.add_argument("--serial", help="D455 serial number to use")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration is not None and args.duration <= 0:
        raise ValueError("--duration must be greater than zero")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be greater than zero")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(
        rs.stream.color,
        args.width,
        args.height,
        rs.format.bgr8,
        args.fps,
    )

    profile = pipeline.start(config)
    device = profile.get_device()
    print(
        f"Capturing {device.get_info(rs.camera_info.name)} "
        f"to {args.output_dir}"
    )
    print("Press Ctrl+C to stop.")

    start_time = time.monotonic()
    frame_count = 0
    try:
        while True:
            if args.duration is not None and time.monotonic() - start_time >= args.duration:
                break
            if args.max_frames is not None and frame_count >= args.max_frames:
                break

            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            image = np.asanyarray(color_frame.get_data())
            elapsed = time.monotonic() - start_time
            seconds = int(elapsed)
            fraction = int((elapsed - seconds) * 100000)
            frame_number = int(color_frame.get_frame_number())
            filename = f"{seconds:010d}_{fraction:05d}_{frame_number:012d}.png"
            filepath = args.output_dir / filename
            if not cv2.imwrite(str(filepath), image):
                raise RuntimeError(f"Could not write frame to {filepath}")
            frame_count += 1
    except KeyboardInterrupt:
        print()
    finally:
        pipeline.stop()

    print(f"Saved {frame_count} frames to {args.output_dir}")


if __name__ == "__main__":
    main()