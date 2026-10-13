from lidar import Lidar
import open3d as o3d
import numpy as np
import threading
import time


lidar = Lidar("../config/config.yaml")


# Latest point cloud
latest_points = None
points_lock = threading.Lock()

# Statistics
pointcloud_count = 0
last_sequence = -1


def pointcloud_callback(points, timestamp, sequence):
    global latest_points
    global pointcloud_count
    global last_sequence

    # Convert to NumPy
    points = np.asarray(points)

    # Remove NaN / Inf
    points = points[np.isfinite(points).all(axis=1)]

    # Store only the newest cloud
    with points_lock:
        latest_points = points.copy()
        pointcloud_count += 1
        last_sequence = sequence


def imu_callback(imu):
    pass


lidar.on_pointcloud(pointcloud_callback)
lidar.on_imu(imu_callback)

lidar.start()

print("LiDAR started")


# ---------------------------------------------------------
# Open3D visualization
# ---------------------------------------------------------

vis = o3d.visualization.Visualizer()

vis.create_window(
    window_name="RoboSense RS-Fairy Point Cloud",
    width=1280,
    height=720
)

pcd = o3d.geometry.PointCloud()

# Add an empty cloud initially
pcd.points = o3d.utility.Vector3dVector(
    np.zeros((1, 3))
)

vis.add_geometry(pcd)


# Coordinate frame
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=1.0,
    origin=[0, 0, 0]
)

vis.add_geometry(axis)


# Camera/view settings
view_control = vis.get_view_control()

view_control.set_front(
    [0.0, 0.0, -1.0]
)

view_control.set_up(
    [0.0, -1.0, 0.0]
)


try:

    last_count = 0
    last_time = time.monotonic()

    while True:

        # Get latest point cloud
        with points_lock:
            if latest_points is not None:
                current_points = latest_points.copy()
            else:
                current_points = None

        # Update Open3D
        if current_points is not None and len(current_points) > 0:

            # Make sure only XYZ are used
            xyz = current_points[:, :3]

            pcd.points = o3d.utility.Vector3dVector(xyz)

            vis.update_geometry(pcd)

        # Process Open3D events
        if not vis.poll_events():
            break

        vis.update_renderer()

        # Print callback frequency once per second
        now = time.monotonic()

        if now - last_time >= 1.0:

            frequency = pointcloud_count - last_count

            print(
                f"Pointcloud: "
                f"{frequency:.2f} Hz | "
                f"Points: "
                f"{len(current_points) if current_points is not None else 0} | "
                f"Sequence: "
                f"{last_sequence}"
            )

            last_count = pointcloud_count
            last_time = now

        # Don't run the visualization loop unnecessarily fast
        time.sleep(0.01)


except KeyboardInterrupt:

    print("Stopping...")


finally:

    lidar.stop()
    vis.destroy_window()