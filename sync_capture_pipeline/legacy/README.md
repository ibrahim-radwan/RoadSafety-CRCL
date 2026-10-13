# Legacy radar + D455 pipeline (pre three-sensor)

Unmodified copies of the scripts as they were before the three-sensor sync
work (2026-10-09). `capture_session.py` here launches the D455 camera
(`../capture_d455_radar_sync.py`) and the radar; `dca_capture_radar_sync.py`
is the radar receiver that writes frames inside its receive loop; `check_session.py`
validates radar + D455 sessions and writes a provisional `pairs.csv`;
`measure_sync_offset.py` is the two-sensor Stage 3 (motion-energy correlation).

Note: sessions from these scripts have `t_burst_monotonic` one radar frame
late (see `../SYNC_DESIGN_3_SENSORS.md` section 3.1).

See `../SYNC_DESIGN_3_SENSORS.md` for what replaced them and why.
