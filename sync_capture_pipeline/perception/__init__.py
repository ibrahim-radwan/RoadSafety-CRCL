"""Our own GPU detection pipeline (radar, LiDAR) and a shared multi-target tracker.

Independent of the research team's chain (cascaded_radar_capture_pipeline/
processing), which is left unchanged. Everything here outputs the pipeline
frame: x forward, y LEFT, z up, metres; times are t_ref (CLOCK_MONOTONIC,
Stage 3 offsets applied).
"""
