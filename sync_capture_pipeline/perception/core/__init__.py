"""Shared core of the perception library: configuration, calibration, readers, sessions and
conventions. numpy only (no torch), so the report and the capture-side tools can import it.

    radar_config   the radar .cfg parser, array geometry and axes (RadarConfig)
    radar_io       raw radar frames: reading, the session's frame timeline, read-ahead
    calibration    the .cfg antenna calibration combined with the in-situ correction
    lidar_io       the LiDAR pcap: packet index, clock mapping, sweep decoding
    session        one recorded session: metadata, firmware, offsets, timelines, reference track
    conventions    the pipeline frame and sign conventions, in one place
"""
