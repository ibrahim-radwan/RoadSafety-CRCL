#!/usr/bin/env python3
"""Example: read live LiDAR and IMU messages from the rs_driver Python bridge.

Usage:
    python3 python_wrapper/read_lidar_stream.py --duration 10 --save-csv imu.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from demo_online_wrapper import OnlineLidarDemo


def print_sample(label: str, payload: dict | None) -> None:
    if payload is None:
        print(f"{label}: waiting...")
        return

    if label == "pointcloud":
        points = payload.get("points", [])
        print(f"{label}: seq={payload.get('seq')} points={len(points)} timestamp={payload.get('timestamp')}")
        if points:
            x, y, z, intensity = points[0]
            print(f"  first point: x={x:.3f}, y={y:.3f}, z={z:.3f}, intensity={intensity}")
    else:
        ang = payload.get("angular_velocity", [0.0, 0.0, 0.0])
        lin = payload.get("linear_acceleration", [0.0, 0.0, 0.0])
        print(
            f"{label}: ts={payload.get('timestamp')} "
            f"gyro=({ang[0]:.4f}, {ang[1]:.4f}, {ang[2]:.4f}) "
            f"acc=({lin[0]:.4f}, {lin[1]:.4f}, {lin[2]:.4f}) "
            f"roll={payload.get('roll_deg', 0.0):.2f}deg "
            f"pitch={payload.get('pitch_deg', 0.0):.2f}deg "
            f"yaw={payload.get('yaw_deg', 0.0):.2f}deg"
        )


def save_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    out = Path(path)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Display and save rs_driver point cloud + IMU data in Python.")
    parser.add_argument("--print-every", type=float, default=1.0, help="Print every N seconds.")
    parser.add_argument("--save-csv", type=str, default="", help="Optional CSV output path for IMU samples.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the LiDAR YAML config file used by the example. Default: config/config.yaml",
    )
    parser.add_argument(
        "--gyro-calibration",
        action="store_true",
        help="Average several IMU samples to estimate the gyroscope bias before using yaw.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=5000,
        help="Number of IMU samples to collect for gyroscope bias estimation.",
    )
    args = parser.parse_args()

    demo = OnlineLidarDemo(
        config_path=args.config,
        stream_log=False,
        gyro_calibration=args.gyro_calibration,
        calibration_samples=args.calibration_samples,
    )
    print("Loaded config parameters\n", demo.get_config_parameters())
    demo.start()

    print("Waiting for first LiDAR and IMU packets...")
    cloud = demo.wait_for_pointcloud(timeout=5.0)
    imu = demo.wait_for_imu(timeout=5.0)

    if cloud is not None:
        print_sample("pointcloud", cloud)
    if imu is not None:
        print_sample("imu", imu)

    rows: list[dict] = []
    last_print = time.monotonic() - args.print_every

    while True:
        sample = demo.sample_once()
        cloud = sample["pointcloud"]
        imu = sample["imu"]
        filtered_imu = sample["filtered_imu"]
        velocity = sample["velocity"]
        speed = sample["speed_mps"]

        now = time.monotonic()
        # if cloud is not None and now - last_print >= args.print_every:
        #     print_sample("pointcloud", cloud)
        #     last_print = now

        #print IMU sample every print_every seconds
        if filtered_imu is not None and now - last_print >= args.print_every:
            # print_sample("imu", imu)
            # print("speed", {"speed_mps": speed})
            last_print = now

        # cloud_hz = demo.get_cloud_rate()
        # imu_hz = demo.get_imu_rate()

        # print("cloud:", cloud_hz, "Hz")
        # print("imu:", imu_hz, "Hz")

        time.sleep(0.01)

    if args.save_csv:
        save_csv(args.save_csv, rows)

    demo.stop()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
