#!/usr/bin/env python3

import time

from lidar import Lidar


def main():

    lidar = Lidar(
        config="../config/config.yaml",
        remove_nan=True,
    )

    try:

        lidar.start()

        while True:

            # -----------------------------------------
            # Point cloud
            # -----------------------------------------

            cloud = lidar.get_pointcloud()

            if cloud is not None:

                points = cloud["points"]

                print(
                    "\nPOINT CLOUD"
                )

                print(
                    "Sequence:",
                    cloud["seq"]
                )

                print(
                    "Timestamp:",
                    cloud["timestamp"]
                )

                print(
                    "Number of points:",
                    len(points)
                )

                print(
                    "Array shape:",
                    points.shape
                )

                if len(points) > 0:

                    print(
                        "First point:",
                        points[0]
                    )


            # -----------------------------------------
            # IMU
            # -----------------------------------------

            imu = lidar.get_imu()

            if imu is not None:

                print(
                    "\nIMU"
                )

                print(
                    "Timestamp:",
                    imu["timestamp"]
                )

                print(
                    "Acceleration:",
                    imu["linear_acceleration"]
                )

                print(
                    "Gyroscope:",
                    imu["angular_velocity"]
                )

                print(
                    "Roll:",
                    imu["roll_deg"]
                )

                print(
                    "Pitch:",
                    imu["pitch_deg"]
                )

                print(
                    "Yaw:",
                    imu["yaw_deg"]
                )

                print(
                    "Velocity:",
                    imu["velocity"]
                )

                print(
                    "Speed:",
                    imu["speed_mps"],
                    "m/s"
                )


            time.sleep(
                0.1
            )


    except KeyboardInterrupt:

        print(
            "\nStopping..."
        )

    finally:

        lidar.stop()


if __name__ == "__main__":

    main()