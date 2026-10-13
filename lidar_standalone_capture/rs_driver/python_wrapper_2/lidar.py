#!/usr/bin/env python3

import time
import threading
import numpy as np

import robosense_lidar


class Lidar:
    """
    Python interface for RoboSense RSFAIRY.

    Point cloud format:

        points[:, 0] = X
        points[:, 1] = Y
        points[:, 2] = Z
        points[:, 3] = range
        points[:, 4] = intensity / reflectivity
        points[:, 5] = time of this point minus the cloud timestamp (s)

    Point-cloud timestamps are float seconds (LiDAR clock when
    use_lidar_clock is true, host clock otherwise). A point's own time is
    timestamp + points[:, 5].

    IMU contains:

        timestamp
        angular velocity
        linear acceleration
        roll
        pitch
        yaw
        velocity
        speed
    """

    def __init__(
        self,
        config="../config/config.yaml",
        remove_nan=False,
        gyro_bias=None,
    ):

        self._lidar = robosense_lidar.Lidar()

        self._lidar.set_remove_nan(remove_nan)

        if gyro_bias is not None:

            self._lidar.set_gyro_bias(
                float(gyro_bias[0]),
                float(gyro_bias[1]),
                float(gyro_bias[2]),
            )

        self._config = config

        self._cloud_lock = threading.Lock()
        self._imu_lock = threading.Lock()

        self._latest_cloud = None
        self._latest_imu = None

        self._running = False

    # ==========================================================
    # START
    # ==========================================================

    def start(self, pcap_path=None, pcap_rate=1.0):

        """
        Start receiving live packets, or decode a recorded PCAP file
        once when pcap_path is given (see input_finished()).
        """

        if self._running:
            return

        print("Starting RoboSense LiDAR...")

        self._lidar.start(
            self._config,
            str(pcap_path) if pcap_path else "",
            float(pcap_rate),
        )

        self._running = True

        print("RoboSense LiDAR started.")

    # ==========================================================
    # STOP
    # ==========================================================

    def stop(self):

        if not self._running:
            return

        print("Stopping RoboSense LiDAR...")

        self._lidar.stop()

        self._running = False

        print("LiDAR stopped.")

    # ==========================================================
    # POINT CLOUD
    # ==========================================================

    def get_pointcloud(self):

        cloud = self._lidar.get_pointcloud()

        if cloud is None:
            return None

        points = cloud["points"]

        if points.size == 0:
            return None

        return {
            "timestamp": float(
                cloud["timestamp"]
            ),

            "seq": int(
                cloud["seq"]
            ),

            "points": np.asarray(
                points,
                dtype=np.float32
            ),
        }

    # ==========================================================
    # INDIVIDUAL POINT CLOUD ARRAYS
    # ==========================================================

    def get_xyz(self):

        cloud = self.get_pointcloud()

        if cloud is None:
            return None

        return cloud["points"][:, 0:3]

    def get_range(self):

        cloud = self.get_pointcloud()

        if cloud is None:
            return None

        return cloud["points"][:, 3]

    def get_intensity(self):

        cloud = self.get_pointcloud()

        if cloud is None:
            return None

        return cloud["points"][:, 4]

    # ==========================================================
    # IMU
    # ==========================================================

    def get_imu(self):

        imu = self._lidar.get_imu()

        if imu is None:
            return None

        return imu

    # ==========================================================
    # CALLBACKS
    # ==========================================================

    def on_pointcloud(self, callback):

        """
        Register Python callback.

        callback(points, timestamp, sequence)
        """

        self._lidar.set_pointcloud_callback(
            callback
        )

    def on_imu(self, callback):

        """
        Register Python callback.

        callback(imu_dictionary)
        """

        self._lidar.set_imu_callback(
            callback
        )

    # ==========================================================
    # RESET IMU
    # ==========================================================

    def reset_imu(self):

        self._lidar.reset_imu()

    # ==========================================================
    # RUNNING
    # ==========================================================

    def is_running(self):

        return self._lidar.is_running()

    def input_finished(self):

        """True once a PCAP file passed to start() has been fully read."""

        return self._lidar.input_finished()