#!/usr/bin/env python3
"""Display the latest LiDAR point cloud in 3D using Python."""

from __future__ import annotations

import argparse
import math
import os
import threading
import time
from typing import Any

import numpy as np

try:  # pragma: no cover - direct script execution
    from demo_online_wrapper import OnlineLidarDemo
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from python_wrapper.demo_online_wrapper import OnlineLidarDemo

try:  # pragma: no cover - optional dependency
    import open3d as o3d
except Exception:  # pragma: no cover - fallback when Open3D is unavailable
    o3d = None

def _select_matplotlib_backend() -> bool:
    """Select the best available Matplotlib backend for this environment."""
    try:
        import matplotlib
    except Exception:
        return False

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    candidates = ["TkAgg", "Qt5Agg", "QtAgg", "GTK3Agg", "macosx", "Agg"]
    if not has_display:
        candidates = ["Agg", *[b for b in candidates if b != "Agg"]]

    for backend in candidates:
        try:
            matplotlib.use(backend, force=True)
            return True
        except Exception:
            continue
    return False


try:  # pragma: no cover - fallback backend
    if not _select_matplotlib_backend():
        raise RuntimeError("No usable matplotlib backend found")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    plt = None
    Axes3D = None


def extract_points(cloud: dict[str, Any] | None) -> np.ndarray:
    """Return an Nx3 array from a LiDAR point-cloud payload."""
    if cloud is None:
        return np.empty((0, 3), dtype=np.float64)

    points = cloud.get("points", [])
    if not points:
        return np.empty((0, 3), dtype=np.float64)

    xyz = np.empty((len(points), 3), dtype=np.float64)
    valid = 0
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 3:
            continue
        xyz[valid] = (float(point[0]), float(point[1]), float(point[2]))
        valid += 1

    if valid == 0:
        return np.empty((0, 3), dtype=np.float64)
    return xyz[:valid]


def _make_colors(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    z_values = points[:, 2]
    z_min = float(np.min(z_values)) if np.size(z_values) else 0.0
    z_max = float(np.max(z_values)) if np.size(z_values) else 1.0
    span = z_max - z_min
    if span <= 0:
        span = 1.0

    colors = np.zeros((len(points), 3), dtype=np.float64)
    intensity = (z_values - z_min) / span
    colors[:, 0] = 0.2 + 0.8 * intensity
    colors[:, 1] = 0.6 - 0.3 * intensity
    colors[:, 2] = 0.9 - 0.2 * intensity
    return colors


class CloudState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.points: np.ndarray | None = None
        self.frame_hash: int | None = None

    def update(self, points: np.ndarray | None) -> None:
        with self._lock:
            self.points = points.copy() if points is not None else None
            self.frame_hash = None if points is None else hash(
                float(points.size)
                + float(points.shape[0])
                + float(np.sum(points[:, 0]))
                + float(np.sum(points[:, 1]))
                + float(np.sum(points[:, 2]))
            )

    def snapshot(self) -> np.ndarray | None:
        with self._lock:
            if self.points is None:
                return None
            return self.points.copy()


def _cloud_update_worker(demo: OnlineLidarDemo, state: CloudState, stop_event: threading.Event, max_points: int = 5000, poll_hz: float = 30.0) -> None:
    """Background thread that periodically refreshes the latest LiDAR cloud."""
    while not stop_event.is_set():
        cloud = demo.get_latest_pointcloud()
        points = extract_points(cloud)
        if len(points) > max_points:
            idx = np.linspace(0, points.shape[0] - 1, max_points, dtype=int)
            points = points[idx]
        state.update(points if len(points) > 0 else None)
        time.sleep(1.0 / max(poll_hz, 1.0))


def visualize_open3d(demo: OnlineLidarDemo, update_hz: float = 10.0) -> None:
    """Open an interactive 3D point cloud viewer using Open3D."""
    if o3d is None:
        raise RuntimeError("Open3D is not installed. Install it with: pip install open3d")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LiDAR 3D View", width=1280, height=720)
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    render_opt = vis.get_render_option()
    render_opt.background_color = np.asarray([0.0, 0.0, 0.0])
    render_opt.point_size = 1.5

    state = CloudState()
    stop_event = threading.Event()
    worker = threading.Thread(target=_cloud_update_worker, args=(demo, state, stop_event), daemon=True)
    worker.start()

    last_drawn_hash = None
    print("Streaming cloud to Open3D (main thread rendering only)...")
    try:
        while True:
            points = state.snapshot()
            if points is not None:
                frame_hash = state.frame_hash
                if frame_hash != last_drawn_hash:
                    pcd.points = o3d.utility.Vector3dVector(points)
                    pcd.colors = o3d.utility.Vector3dVector(_make_colors(points))
                    vis.update_geometry(pcd)
                    vis.poll_events()
                    vis.update_renderer()
                    last_drawn_hash = frame_hash
            else:
                time.sleep(0.02)
            time.sleep(1.0 / max(update_hz, 1.0))
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        vis.destroy_window()


def visualize_matplotlib(demo: OnlineLidarDemo, update_hz: float = 10.0) -> None:
    """Fallback 3D viewer using Matplotlib."""
    if plt is None:
        raise RuntimeError("Matplotlib is not installed. Install it with: pip install matplotlib")
    if Axes3D is None:
        raise RuntimeError(
            "Matplotlib 3D support is unavailable in this environment. "
            "Install the 3D toolkit or use Open3D instead."
        )

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("LiDAR Point Cloud")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    scatter = None
    last_points = None
    last_hash = None
    last_update = 0.0
    headless = not bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    snapshot_path = "/tmp/rs_lidar_pointcloud.png"
    print("Streaming cloud to Matplotlib" + (" (headless snapshot mode)" if headless else " (press Ctrl+C to stop)..."))

    try:
        while True:
            now = time.monotonic()
            if now - last_update < 1.0 / max(update_hz, 1e-6):
                time.sleep(0.01)
                continue

            cloud = demo.get_latest_pointcloud()
            points = extract_points(cloud)
            if len(points) > 0:
                max_points = 5000
                if points.shape[0] > max_points:
                    idx = np.linspace(0, points.shape[0] - 1, max_points, dtype=int)
                    points = points[idx]
                frame_hash = hash(float(points.size) + float(points.shape[0]) + float(np.sum(points[:, 0])) + float(np.sum(points[:, 1])) + float(np.sum(points[:, 2])))
                if last_hash != frame_hash:
                    if scatter is None:
                        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.0, c=points[:, 2], cmap="viridis")
                    else:
                        scatter.remove()
                        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.0, c=points[:, 2], cmap="viridis")
                    ax.set_xlim(np.min(points[:, 0]) - 1.0, np.max(points[:, 0]) + 1.0)
                    ax.set_ylim(np.min(points[:, 1]) - 1.0, np.max(points[:, 1]) + 1.0)
                    ax.set_zlim(np.min(points[:, 2]) - 1.0, np.max(points[:, 2]) + 1.0)
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                    if headless:
                        fig.savefig(snapshot_path, dpi=150)
                        print(f"Saved latest cloud snapshot to {snapshot_path}")
                    else:
                        plt.pause(0.001)
                    last_points = points.copy()
                    last_hash = frame_hash
                    last_update = now
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize the live LiDAR point cloud in 3D.")
    parser.add_argument("--binary", type=str, default=None, help="Path to the compiled bridge executable.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the YAML lidar config.")
    parser.add_argument("--update-hz", type=float, default=10.0, help="Refresh rate for the viewer in Hz.")
    parser.add_argument(
        "--matplotlib",
        action="store_true",
        help="Use Matplotlib instead of Open3D for fallback visualization.",
    )
    args = parser.parse_args()

    demo = OnlineLidarDemo(
        binary_path=args.binary,
        config_path=args.config,
        stream_log=False,
    )
    demo.start()

    try:
        cloud = demo.wait_for_pointcloud(timeout=5.0)
        if cloud is None:
            print("No point cloud packet was received yet.")
            return 1

        if o3d is not None and not args.matplotlib:
            visualize_open3d(demo, update_hz=args.update_hz)
        elif plt is not None:
            visualize_matplotlib(demo, update_hz=args.update_hz)
        else:
            raise RuntimeError("Open3D is not installed and Matplotlib is unavailable. Install Open3D with: pip install open3d")
    finally:
        demo.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
