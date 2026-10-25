"""
The pipeline's frame and sign conventions, in one place. Everything that converts between the
radar processing's own angles and the pipeline frame goes through here.

Pipeline frame (the LiDAR's): x forward, y LEFT, z up, metres. Azimuth = atan2(y, x), positive to
the LEFT. Times: t_ref (CLOCK_MONOTONIC seconds, Stage 3 offsets applied). The radar is assumed
co-located with the LiDAR until the team's extrinsic calibration.

Radar processing angles (radar_config.az_axis_deg, the azimuth FFT bins): positive to the RIGHT.
That is the original processing's convention ("x = lateral, positive = right of boresight"), not
an error. Verified against the LiDAR walker on freewalk_ref_02: 90% sign agreement beyond 8 deg.

Radar radial velocity (radar_config.doppler_axis): positive = RECEDING. Verified: 99% sign
agreement with the LiDAR range rate on freewalk_ref_02.

The original processing's xyz_cloud (process_frame) is [x lateral (+ right), y forward, z, power,
doppler, range, az]; team_xyz_to_pipeline converts it.
"""
from __future__ import annotations

import numpy as np

AZIMUTH_SIGN = -1.0          # world azimuth (+ left) = AZIMUTH_SIGN * processing azimuth (+ right)
RADIAL_VELOCITY_RECEDING_POSITIVE = True


def world_azimuth(proc_az):
    """Processing azimuth (deg or sin, + right) -> pipeline azimuth (same unit, + left)."""
    return AZIMUTH_SIGN * np.asarray(proc_az)


def processing_azimuth(world_az):
    """Pipeline azimuth (+ left) -> processing azimuth (+ right); its own inverse."""
    return AZIMUTH_SIGN * np.asarray(world_az)


def polar_to_xyz(r, az_deg, el_deg=0.0):
    """Range, pipeline azimuth (deg, + left) and elevation (deg, + up) -> x forward, y left, z up."""
    r, az, el = np.asarray(r, float), np.radians(az_deg), np.radians(el_deg)
    ce = np.cos(el)
    return np.stack(np.broadcast_arrays(r * ce * np.cos(az), r * ce * np.sin(az), r * np.sin(el)), axis=-1)


def team_xyz_to_pipeline(team_xyz: np.ndarray) -> np.ndarray:
    """process_frame's xyz_cloud [N, >=3] (x lateral + right, y forward, z up) -> [N, 3] pipeline xyz."""
    t = np.asarray(team_xyz)
    return np.c_[t[:, 1], -t[:, 0], t[:, 2]].astype(np.float32)
