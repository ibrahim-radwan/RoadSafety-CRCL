#!/usr/bin/env python3
"""Python wrapper for the C++ online LiDAR demo.

This wrapper starts a bridge executable that streams both point cloud and IMU
messages as JSON. Python code can read the latest values directly from the
running process without manually parsing the C++ SDK internals.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "demo"
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
GRAVITY = 9.80665
ACCEL_FILTER_ALPHA = 0.98
GYRO_BIAS = {
    "x": 0.0147152,
    "y": -0.0116165,
    "z": -0.00163704,
}
DEFAULT_CALIBRATION_SAMPLES = 5000

previous_cloud_timestamp = None
previous_imu_timestamp = None


def load_yaml_config(config_path: Optional[os.PathLike[str] | str] = None) -> Dict[str, Any]:
    """Load the YAML config used by the C++ example."""
    path = Path(config_path) if config_path else CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to parse the lidar config. Install it with 'pip install pyyaml'."
        )

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    if not isinstance(data, dict):
        raise TypeError(f"Config file must contain a YAML mapping: {path}")

    return data


def _candidate_binaries() -> list[Path]:
    """Return likely build paths for the compiled bridge or demo binaries."""
    candidates = [
        REPO_ROOT / "build" / "demo" / "demo_online_bridge",
        REPO_ROOT / "build" / "demo_online_bridge",
        REPO_ROOT / "demo_online_bridge",
        REPO_ROOT / "build" / "demo" / "demo_online",
        REPO_ROOT / "build" / "demo_online",
        REPO_ROOT / "demo_online",
    ]
    return [p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve() for p in candidates]


def resolve_demo_binary(binary_path: Optional[os.PathLike[str] | str] = None) -> Path:
    """Find the demo_online_bridge or fallback demo_online executable."""
    if binary_path:
        path = Path(binary_path).expanduser()
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"Binary not found: {path}")

    for candidate in _candidate_binaries():
        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not find the compiled bridge or demo binary. Build it first with:\n"
        "  cmake -S . -B build -DCOMPILE_DEMOS=ON && cmake --build build --target demo_online_bridge"
    )


class OnlineLidarDemo:
    """Python API for reading LiDAR point clouds and IMU data from the rs_driver example."""

    @staticmethod
    def _count_rate_by_timestamp(samples: Deque[float], window_seconds: float = 1.0) -> float:
        """Estimate the message rate in Hz from a deque of timestamps.

        The timestamps may be in nanoseconds or seconds. The estimate is based on the
        most recent samples within the requested window and returns messages/sec.
        """
        if window_seconds <= 0 or not samples:
            return 0.0

        values = []
        for ts in samples:
            value = float(ts)
            if value >= 1e9:
                value /= 1e9
            values.append(value)

        if not values:
            return 0.0

        latest = values[-1]
        recent = [v for v in values if latest - v <= window_seconds]
        if not recent:
            recent = values[-1:]

        if len(recent) <= 1:
            return 1.0 / window_seconds

        elapsed = recent[-1] - recent[0]
        if elapsed <= 0:
            return 0.0
        return (len(recent) - 1) / elapsed

    def __init__(
        self,
        binary_path: Optional[os.PathLike[str] | str] = None,
        config_path: Optional[os.PathLike[str] | str] = None,
        stream_log: bool = False,
        gyro_calibration: bool = False,
        calibration_samples: int = DEFAULT_CALIBRATION_SAMPLES,
    ) -> None:
        self.config_path = Path(config_path).resolve() if config_path else CONFIG_PATH
        self.binary_path = resolve_demo_binary(binary_path)
        self.config = load_yaml_config(self.config_path)
        self.process: Optional[subprocess.Popen[str]] = None
        self._reader_thread: Optional[threading.Thread] = None
        self.latest_pointcloud: Optional[Dict[str, Any]] = None
        self.latest_imu: Optional[Dict[str, Any]] = None
        self.latest_pointcloud_seq = -1
        self.latest_pointcloud_timestamp = 0.0
        self.last_pointcloud_update = 0.0
        self.cloud_timestamps: Deque[float] = deque(maxlen=10000)
        self.imu_timestamps: Deque[float] = deque(maxlen=10000)

        calibration_cfg = self.config.get("calibration", {})
        default_gyro_bias = {
            "x": float(calibration_cfg.get("x", GYRO_BIAS["x"])),
            "y": float(calibration_cfg.get("y", GYRO_BIAS["y"])),
            "z": float(calibration_cfg.get("z", GYRO_BIAS["z"])),
        }

        self.gyro_calibration = bool(gyro_calibration)
        self.calibrating_gyro = bool(gyro_calibration)
        self.calibration_samples = max(1, calibration_samples)
        self.gyro_bias = default_gyro_bias
        self.gyro_sum_x = 0.0
        self.gyro_sum_y = 0.0
        self.gyro_sum_z = 0.0
        self.calibration_count = 0
        self.imu_state: Dict[str, float] = {
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "velocity_x": 0.0,
            "velocity_y": 0.0,
            "velocity_z": 0.0,
            "previous_timestamp": 0.0,
            "initialized": 0.0,
        }
        self.stream_log = stream_log
        self._lock = threading.Lock()

    def update_imu_state(
        self,
        timestamp: float,
        ax: float,
        ay: float,
        az: float,
        gx: float,
        gy: float,
        gz: float,
    ) -> Dict[str, float]:
        """Compute filtered roll, pitch, and yaw similar to demo_online.cpp."""
        if not self.imu_state["initialized"]:
            self.imu_state["previous_timestamp"] = timestamp
            self.imu_state["roll"] = math.atan2(ay, az)
            self.imu_state["pitch"] = math.atan2(-ax, math.sqrt(ay * ay + az * az))
            self.imu_state["yaw"] = 0.0
            self.imu_state["velocity_x"] = 0.0
            self.imu_state["velocity_y"] = 0.0
            self.imu_state["velocity_z"] = 0.0
            self.imu_state["initialized"] = 1.0
            return {
                "roll": self.imu_state["roll"],
                "pitch": self.imu_state["pitch"],
                "yaw": self.imu_state["yaw"],
                "velocity_x": self.imu_state["velocity_x"],
                "velocity_y": self.imu_state["velocity_y"],
                "velocity_z": self.imu_state["velocity_z"],
            }

        dt = timestamp - self.imu_state["previous_timestamp"]
        self.imu_state["previous_timestamp"] = timestamp
        if dt <= 0.0 or dt > 1.0:
            return {
                "roll": self.imu_state["roll"],
                "pitch": self.imu_state["pitch"],
                "yaw": self.imu_state["yaw"],
                "velocity_x": self.imu_state["velocity_x"],
                "velocity_y": self.imu_state["velocity_y"],
                "velocity_z": self.imu_state["velocity_z"],
            }

        roll_gyro = self.imu_state["roll"] + gx * dt
        pitch_gyro = self.imu_state["pitch"] + gy * dt
        yaw_gyro = self.imu_state["yaw"] + gz * dt

        roll_acc = math.atan2(ay, az)
        pitch_acc = math.atan2(-ax, math.sqrt(ay * ay + az * az))

        self.imu_state["roll"] = ACCEL_FILTER_ALPHA * roll_gyro + (1.0 - ACCEL_FILTER_ALPHA) * roll_acc
        self.imu_state["pitch"] = ACCEL_FILTER_ALPHA * pitch_gyro + (1.0 - ACCEL_FILTER_ALPHA) * pitch_acc
        self.imu_state["yaw"] = yaw_gyro

        cr = math.cos(self.imu_state["roll"])
        sr = math.sin(self.imu_state["roll"])
        cp = math.cos(self.imu_state["pitch"])
        sp = math.sin(self.imu_state["pitch"])
        cy = math.cos(self.imu_state["yaw"])
        sy = math.sin(self.imu_state["yaw"])

        r11 = cy * cp
        r12 = cy * sp * sr - sy * cr
        r13 = cy * sp * cr + sy * sr
        r21 = sy * cp
        r22 = sy * sp * sr + cy * cr
        r23 = sy * sp * cr - cy * sr
        r31 = -sp
        r32 = cp * sr
        r33 = cp * cr

        world_ax = r11 * ax + r12 * ay + r13 * az
        world_ay = r21 * ax + r22 * ay + r23 * az
        world_az = r31 * ax + r32 * ay + r33 * az
        world_az -= GRAVITY

        self.imu_state["velocity_x"] += world_ax * dt
        self.imu_state["velocity_y"] += world_ay * dt
        self.imu_state["velocity_z"] += world_az * dt

        return {
            "roll": self.imu_state["roll"],
            "pitch": self.imu_state["pitch"],
            "yaw": self.imu_state["yaw"],
            "velocity_x": self.imu_state["velocity_x"],
            "velocity_y": self.imu_state["velocity_y"],
            "velocity_z": self.imu_state["velocity_z"],
        }

    def calibrate_gyro_bias(self, gx: float, gy: float, gz: float) -> bool:
        """Collect gyroscope samples until enough data are gathered, then compute the bias."""
        if not self.gyro_calibration:
            return False

        self.gyro_sum_x += gx
        self.gyro_sum_y += gy
        self.gyro_sum_z += gz
        self.calibration_count += 1

        if self.calibration_count >= self.calibration_samples:
            self.gyro_bias["x"] = self.gyro_sum_x / self.calibration_samples
            self.gyro_bias["y"] = self.gyro_sum_y / self.calibration_samples
            self.gyro_bias["z"] = self.gyro_sum_z / self.calibration_samples
            self.gyro_calibration = False
            self.calibrating_gyro = False

            print(
                f"Gyroscope bias calibrated: {self.gyro_bias} "
                f"from {self.calibration_count} samples.")
            return True

        return False

    def _consume_stream(self) -> None:
        """Read JSON messages from stdout and cache the latest LiDAR/IMU payload."""
        if self.process is None or self.process.stdout is None:
            return

        for raw_line in self.process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                if self.stream_log:
                    print(f"[rs_driver_bridge] ignored: {line}", file=sys.stderr)
                continue

            kind = msg.get("type")
            if kind == "pointcloud":
                ts = float(msg.get("timestamp", 0.0))
                with self._lock:
                    self.latest_pointcloud = msg
                    self.latest_pointcloud_seq = int(msg.get("seq", self.latest_pointcloud_seq))
                    self.latest_pointcloud_timestamp = ts
                    self.last_pointcloud_update = time.monotonic()
                    self.cloud_timestamps.append(ts)
                if self.stream_log:
                    print("pointcloud", msg.get("seq"), len(msg.get("points", [])))
            elif kind == "imu":
                ts = float(msg.get("timestamp", 0.0))
                ang = msg.get("angular_velocity", [0.0, 0.0, 0.0])
                lin = msg.get("linear_acceleration", [0.0, 0.0, 0.0])
                gx = float(ang[0])
                gy = float(ang[1])
                gz = float(ang[2])

                with self._lock:
                    if self.gyro_calibration:
                        calibrated = self.calibrate_gyro_bias(gx, gy, gz)
                        if calibrated:
                            if self.stream_log:
                                print(
                                    "gyro bias calibrated:",
                                    self.gyro_bias,
                                    file=sys.stderr,
                                )
                        else:
                            # keep waiting for the required number of IMU samples
                            self.latest_imu = dict(msg)
                            self.latest_imu["gyro_calibration_in_progress"] = True
                            self.latest_imu["calibration_count"] = self.calibration_count
                            self.latest_imu["calibration_samples"] = self.calibration_samples
                            self.imu_timestamps.append(ts)
                            continue

                    gx -= self.gyro_bias["x"]
                    gy -= self.gyro_bias["y"]
                    gz -= self.gyro_bias["z"]

                    attitude = self.update_imu_state(
                        ts,
                        float(lin[0]),
                        float(lin[1]),
                        float(lin[2]),
                        gx,
                        gy,
                        gz,
                    )
                    msg = dict(msg)
                    msg["roll_rad"] = attitude["roll"]
                    msg["pitch_rad"] = attitude["pitch"]
                    msg["yaw_rad"] = attitude["yaw"]
                    msg["roll_deg"] = math.degrees(attitude["roll"])
                    msg["pitch_deg"] = math.degrees(attitude["pitch"])
                    msg["yaw_deg"] = math.degrees(attitude["yaw"])
                    msg["velocity_x"] = attitude["velocity_x"]
                    msg["velocity_y"] = attitude["velocity_y"]
                    msg["velocity_z"] = attitude["velocity_z"]
                    msg["gyro_bias"] = dict(self.gyro_bias)
                    self.latest_imu = msg
                    self.imu_timestamps.append(ts)
                if self.stream_log:
                    print(
                        "imu",
                        msg.get("timestamp"),
                        "roll_deg=",
                        msg.get("roll_deg"),
                        "pitch_deg=",
                        msg.get("pitch_deg"),
                        "yaw_deg=",
                        msg.get("yaw_deg"),
                    )

    def summary(self) -> Dict[str, Any]:
        lidar_cfg = self.config.get("lidar", {})
        net_cfg = self.config.get("network", {})
        return {
            "lidar_type": lidar_cfg.get("lidar_type"),
            "lidar_ip": net_cfg.get("lidar_ip"),
            "host_ip": net_cfg.get("host_ip"),
            "msop_port": lidar_cfg.get("msop_port"),
            "difop_port": lidar_cfg.get("difop_port"),
            "imu_port": lidar_cfg.get("imu_port"),
            "binary": str(self.binary_path),
            "config": str(self.config_path),
        }

    def start(self) -> subprocess.Popen[str]:
        """Launch the bridge process and begin reading the JSON stream."""
        if self.process is not None and self.process.poll() is None:
            return self.process

        self.process = subprocess.Popen(
            [str(self.binary_path)],
            cwd=str(DEMO_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

        self._reader_thread = threading.Thread(target=self._consume_stream, daemon=True)
        self._reader_thread.start()
        return self.process

    def get_latest_pointcloud(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.latest_pointcloud

    def get_pointcloud_freshness(self) -> Dict[str, float | int]:
        """Return the current cloud sequence, timestamp, and age in seconds."""
        with self._lock:
            age = 0.0 if self.last_pointcloud_update == 0.0 else max(0.0, time.monotonic() - self.last_pointcloud_update)
            return {
                "seq": self.latest_pointcloud_seq,
                "timestamp": self.latest_pointcloud_timestamp,
                "age_sec": age,
            }

    def get_latest_imu(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.latest_imu

    def get_filtered_orientation(self) -> Dict[str, float]:
        """Return the latest filtered roll, pitch, and yaw in radians and degrees."""
        imu = self.get_latest_imu()
        if imu is None:
            return {"roll_rad": 0.0, "pitch_rad": 0.0, "yaw_rad": 0.0, "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0}
        return {
            "roll_rad": float(imu.get("roll_rad", 0.0)),
            "pitch_rad": float(imu.get("pitch_rad", 0.0)),
            "yaw_rad": float(imu.get("yaw_rad", 0.0)),
            "roll_deg": float(imu.get("roll_deg", 0.0)),
            "pitch_deg": float(imu.get("pitch_deg", 0.0)),
            "yaw_deg": float(imu.get("yaw_deg", 0.0)),
        }

    def get_velocity(self) -> Dict[str, float]:
        """Return the latest integrated IMU velocity in m/s."""
        imu = self.get_latest_imu()
        if imu is None:
            return {"velocity_x": 0.0, "velocity_y": 0.0, "velocity_z": 0.0}
        return {
            "velocity_x": float(imu.get("velocity_x", 0.0)),
            "velocity_y": float(imu.get("velocity_y", 0.0)),
            "velocity_z": float(imu.get("velocity_z", 0.0)),
        }

    def get_speed(self) -> float:
        """Return the current IMU speed magnitude in m/s."""
        velocity = self.get_velocity()
        return math.hypot(
            velocity["velocity_x"],
            velocity["velocity_y"],
            velocity["velocity_z"],
        )

    def wait_for_pointcloud(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            cloud = self.get_latest_pointcloud()
            if cloud is not None:
                return cloud
            time.sleep(min(0.01, max(deadline - time.monotonic(), 0.0)))
        return None

    def wait_for_imu(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            imu = self.get_latest_imu()
            if imu is not None:
                return imu
            time.sleep(min(0.01, max(deadline - time.monotonic(), 0.0)))
        return None

    def get_config_parameters(self) -> Dict[str, Any]:
        """Return the loaded YAML config parameters."""
        return self.config

    def get_cloud_rate(self, window_seconds: float = 1.0) -> float:
        """Return the recent cloud message rate in Hz."""
        with self._lock:
            return self._count_rate_by_timestamp(self.cloud_timestamps, window_seconds)

    def get_imu_rate(self, window_seconds: float = 1.0) -> float:
        """Return the recent IMU message rate in Hz."""
        with self._lock:
            return self._count_rate_by_timestamp(self.imu_timestamps, window_seconds)

    def sample_once(self) -> Dict[str, Optional[Dict[str, Any]]]:
        """Return the latest point cloud and IMU packet available in Python."""
        return {
            "pointcloud": self.get_latest_pointcloud(),
            "imu": self.get_latest_imu(),
            "filtered_imu": self.get_filtered_orientation(),
            "velocity": self.get_velocity(),
            "speed_mps": self.get_speed(),
            "cloud_rate_hz": self.get_cloud_rate(),
            "imu_rate_hz": self.get_imu_rate(),
        }

    def stop(self, timeout: float = 5.0) -> None:
        """Terminate the child process gracefully."""
        if self.process is None or self.process.poll() is not None:
            return

        try:
            self.process.send_signal(signal.SIGINT)
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=timeout)

    def run(self, *, timeout: Optional[float] = None) -> int:
        """Start the demo and wait until it exits."""
        self.start()
        if timeout is None:
            return self.process.wait()
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream rs_driver point cloud and IMU data into Python.")
    parser.add_argument(
        "--binary",
        type=str,
        default=None,
        help="Path to the compiled bridge executable. Defaults to the repo build output.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="Path to the YAML lidar config used by the demo. Defaults to config/config.yaml.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the loaded YAML config and exit.",
    )
    parser.add_argument(
        "--gyro-calibration",
        action="store_true",
        help="Run gyroscope bias calibration before using the IMU yaw estimate.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=DEFAULT_CALIBRATION_SAMPLES,
        help="Number of IMU samples to average to estimate the gyro bias.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Launch the stream and return immediately without waiting for it to exit.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        demo = OnlineLidarDemo(
            binary_path=args.binary,
            config_path=args.config,
            stream_log=True,
            gyro_calibration=args.gyro_calibration,
            calibration_samples=args.calibration_samples,
        )
    except (FileNotFoundError, RuntimeError, TypeError) as exc:
        print(f"Wrapper error: {exc}", file=sys.stderr)
        return 2

    if args.print_config:
        print(demo.config)
        return 0

    print("Starting rs_driver Python bridge with:")
    print(f"  binary: {demo.binary_path}")
    print(f"  config: {demo.config_path}")
    print(f"  lidar: {demo.summary()['lidar_type']} @ {demo.summary()['lidar_ip']}")

    if args.no_wait:
        demo.start()
        print(f"Bridge started with PID: {demo.process.pid}")
        return 0

    return demo.run()


if __name__ == "__main__":
    raise SystemExit(main())
