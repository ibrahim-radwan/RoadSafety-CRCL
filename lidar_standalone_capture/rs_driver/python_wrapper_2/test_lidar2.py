from lidar import Lidar
import time
import threading

lidar = Lidar("../config/config.yaml")

pointcloud_count = 0
imu_count = 0

counter_lock = threading.Lock()


def pointcloud_callback(points, timestamp, sequence):
    global pointcloud_count

    with counter_lock:
        pointcloud_count += 1


def imu_callback(imu):
    global imu_count

    with counter_lock:
        imu_count += 1


lidar.on_pointcloud(pointcloud_callback)
lidar.on_imu(imu_callback)

lidar.start()

print("LiDAR started...")

last_time = time.monotonic()

try:

    while True:

        time.sleep(1)

        now = time.monotonic()
        elapsed = now - last_time
        last_time = now

        with counter_lock:
            pc_count = pointcloud_count
            imu_count_local = imu_count

            pointcloud_count = 0
            imu_count = 0

        pc_hz = pc_count / elapsed
        imu_hz = imu_count_local / elapsed

        print(
            f"Pointcloud: {pc_hz:.2f} Hz | "
            f"IMU: {imu_hz:.2f} Hz"
        )

except KeyboardInterrupt:

    print("Stopping LiDAR...")
    lidar.stop()