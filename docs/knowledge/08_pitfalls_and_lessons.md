# Pitfalls and lessons learned

Each item cost real debugging time. Read before changing the related code.

## Hardware / system
- **Radar on a weak supply:** it accepts every command and sends **no data**,
  with no error anywhere. Use the 12 V mains adapter.
- **Interface names swap.** `eth0`/`eth1` depend on whether the USB adapter is
  present at boot, so NetworkManager profiles must be bound by **MAC**.
- **`setup_jetson_capture.sh` owns `/etc/sysctl.d/90-sensor-capture.conf`.**
  Keep all three settings in it: rmem and the two `vm.dirty_*` values.
- **`uart_uniflash.py` can exit 0 on failure.** Read its output after flashing.
- **The DCA1000 can come up throttled.** Power-cycle the radar + DCA1000; the
  watchdog stops such sessions.
- **The LiDAR can be knocked out of alignment.** Check it faces forward.
- **A round LiDAR shows no "forward".** The RS-Fairy's 0° is its encoder's,
  and the cylinder sits in a round collar. The team mounted it "facing
  forward" and it was 37° off; once it was 57° off. Mark the housing and
  its collar, and check in the data.

## Radar
- **Subtract a background coherently, not in dB.** Differencing dB static
  maps lands on the walls' sidelobes (three positions read exactly +22°).
  Subtracting the zero-Doppler channels (complex) before beamforming
  leaves the reflector alone, even across sessions.
- **The `.cfg` is documentation, not device configuration.** Resolve it per
  session through `radar_firmware.py`.
- **The "mirrored azimuth" was a convention mix-up, not a bug.** The team's
  azimuth is right-positive, and our frame is left-positive. Read a
  codebase's axis convention (their README states it) before calling its
  output mirrored.
- **Phase-residual estimates need the snapshot's common phase removed.**
  Every snapshot carries an arbitrary absolute phase. Averaging without
  removing it gave a coherence of 0.04 and a wrong pattern; with it, 0.33 to
  0.46 and a stable per-channel pattern.
- **The radar–LiDAR yaw leaks into a phase correction.** A phase linear in
  aperture position only steers the beam. Remove it, or the correction fits
  one session's mounting and biases the next (+11° on run_20261014_1326).
- **A walking person is a poor calibration target:** extended, with
  micro-Doppler and multipath. The plate sessions (a near-point reflector)
  gave a correction that generalised; the walker's did not.
- **Two azimuth peaks per cell doubled the false tracks.** With the current
  calibration the beam has a split lobe ~13° away and only ~4 dB down.
- **"Needs antenna calibration" was concluded too quickly (2026-11-07).**
  The split beam was blamed on the calibration until a test showed the
  error depends on the target's angle, which a per-channel calibration
  cannot cause or fix. Before blaming a calibration, check that one
  correction holds at every angle (estimate it on the left and right
  targets separately, against a same-side noise control). Then suspect the
  array model in the processing.
- **The radar's array table was never checked against the data** (found
  2026-11-07). Three TX positions and the elevation TX were wrong. Weeks of
  "calibration" patches (the in-situ phase) only bent it into shape near
  boresight. A layout can be measured from any session with a single
  target and an angle reference: each channel's phase against sin(azimuth)
  gives its position (`perception/estimate_array_layout.py`). Do it for any
  new board or firmware chirp order.
- **The DCA packet delay limits the frame rate:** 50 µs caps the stream at
  7.44 fps. Use 25 µs.
- **I/Q order is I first.**
- **The azimuth sign is mirrored** in the processing.
- **Frame stamping:** use the burst that carried the frame's **first** byte.

## LiDAR
- **Zero-range points:** filter `range > 0`.
- **The RS-Fairy skips 16.5° once a second** (its 0–16.5°, 4.6 ms, no packet
  loss). A wall there blinks at 1 Hz. Before blaming the decoder or the
  network, look at the block azimuths: they jump from 359.6° to 16.1°.
- **Fitting an extrinsic to nominal target positions absorbs the rig's own
  turn.** On 2026-11-03 the floor marks gave 39.8° for the LiDAR, but the
  camera, which sees the "centre" reflectors 1.6–1.9° to the right, gives
  37.2°. Reference the yaw to a sensor on the rig (the camera), not to
  the floor.
- **One ground plane fails** on slopes and under roofs. The carport scene of
  freewalk_ref_02 made a single 13 × 8 m "object". Use local ground and drop
  overhead structures (see [05](05_lidar.md)).
- **`cv2.minAreaRect`'s angle is along its first side**, not the long side.
  Swap the sides, and the box turns by 90°.
- **One stray point per cell can bridge two objects** in grid clustering.
  Mixed pixels at depth edges lie between a person and a wall. Cluster
  through dense cells only (`perception/lidar.py`, `cluster="density"`);
  closing the grid brings the bridges back.
- **More objects make the tracker slow if it loops in Python.** With ~45
  LiDAR tracks, the per-track gating and the all-pairs duplicate check took
  7.5 ms per sweep in the replay's LiDAR thread, which holds the GIL; the
  LiDAR rate fell from ~10 to ~7 updates/s. Vectorised: 2.5 ms, identical
  tracks.
- **Decode pcaps at ≤ 3×** real time per driver. Run several drivers on time
  chunks for more (`decode_parallel`); each chunk needs a DIFOP packet
  (once a second) before its first sweep, so chunks start 1.5 s early.
- **`split_angle: 180`**, or the forward field of view splits across two
  sweeps.

## Camera
- **Share a disk with the radar and the camera drops frames.** Stage the camera
  on its own disk.
- **`pyzed.grab()` holds the GIL.** Run the camera in its own process.
- **Lossless H.264 decodes at only ~40–50 fps.**

## Software
- **Never install a systemd unit with a fixed user and path.** The web
  service's file named user `jetson` and
  `/home/jetson/dev/CascadedRadarRoadSafetyOnTwoWheelers`, and
  `setup_jetson_capture.sh` copied it as is. On another board (217/USER)
  or a checkout under the GitHub name `RoadSafety-CRCL` (200/CHDIR), the
  service failed every 3 s and the page stayed blank, which looked like a
  network problem. The script now writes the unit for the board it runs on
  (2026-11-04).
- **Never let the launcher's stdout be a pipe owned by the web server.** If the
  server dies, the recording dies with a BrokenPipeError. The launcher writes
  to a file and the server follows `session.log`. The service needs
  `KillMode=process`.
- **`pgrep`/`kill` patterns:** anchor them, or they match the shell running
  them.
- **Python 3.8 in the capture environment:** no `Path.is_relative_to`, and
  `dict | None` type hints need `from __future__ import annotations`.
- **The camera recorder and pyzed need the conda environment.** The web server
  runs the launcher with `/usr/bin/python3`, which is also 3.8.
- **pyqtgraph 0.13:** `ImageItem.setRect` needs an image set first.
- **`conda activate rel_1_py_3_8` sets `QT_XCB_GL_INTEGRATION=none`**, through
  the `qt-main` package's activation script. That disables Qt's OpenGL on X11,
  so the 3D LiDAR window fails with "Cannot create platform OpenGL context,
  neither GLX nor EGL are enabled". `replay_session.py` clears that value for
  its own process (fixed 2026-10-14). Any other Qt OpenGL tool needs the same,
  or `unset QT_XCB_GL_INTEGRATION`.
  - **Test GUI/OpenGL changes the way users run them:** on the real display,
    in a terminal with the environment *activated*. Calling the environment's
    Python directly, or Qt's off-screen platform, skips that script and hides
    the problem.

- **scikit-learn in a per-frame loop starves a Qt GUI.** On every call,
  DBSCAN's `threadpool_limits` walks all loaded libraries (`realpath` on each)
  while holding the GIL. The replay froze, and even a screenshot timer could
  not fire. Use scipy's `csgraph` (C, releases quickly) for small clustering
  jobs.
- **A replay source must join its thread in `stop()`.** The LiDAR source
  wraps rs_driver, whose threads call back into Python. If the process
  exits while that thread still runs, pybind11 aborts ("dec_ref() is being
  called while the GIL is either not held or invalid") and the process
  dumps core. Before 2026-10-25 a 0.3 s sleep in the replay hid this; the
  new headless source test exposed it. `Source.stop()` now joins.
- **Cutting a block of code by text range can take a neighbour with it.**
  Removing the "filtered" cloud from `RadarSource.__init__` also removed the
  read-ahead line just below it. Unit and score tests passed; only the
  replay failed. `tests/test_replay_sources.py` now starts the replay's
  sources headless.
- **Profile GPU code with CUDA synchronised.** Without it, the time lands on
  the first sync point (`torch.nonzero`, `.cpu()`) and looks like CPU time.
- **The first replay after a reboot is slow on the external drive.** The file
  cache is empty, so every read is cold: camera 0, radar ~2 updates/s at
  first. A second run of the same part is normal (camera ~29, radar ~8).
- **Sessions on NTFS through FUSE are slow.** On the external drive, a 3 MB
  radar frame takes 78 ms median to read (up to 390 ms), and the replay got
  only 3 radar updates/s. The replay now reads ahead. For full speed, use
  ext4.
  - The drive is a 2.5" USB hard disk (WD10JMVW). Frame files come at
    ~31 MB/s with 1 to 8 reading threads alike (seeks), so batch radar runs
    from it are disk-bound (~70 ms/frame).
- **A read from the external drive once returned different bytes**
  (2026-10-26): `freewalk_ref_02/radar/frame_003308.bin` gave a slightly
  different frame in one cold run (noise level 75.74 instead of 75.83 dB, 32
  static detections moved) and stayed wrong in the page cache until evicted.
  Six later cold reads all agree. Re-run a result that looks odd before
  trusting it, and check copies against the SD card originals (checksums)
  before relying on this drive.
- **numpy on tiny arrays is slow.** `np.mean` of a 3-element list costs
  ~25 µs; the tracker's moving/static check did that 3× per track per sweep
  (60% of the tracker). Plain float arithmetic: identical, 3× faster.
- **Boolean-mask indexing over a whole image is slow.** The LiDAR render
  spent 21 of its 30 ms in `img[mask] = ...` over ~300 k pixels; working on
  the ~40 k points and `cv2.copyTo` with a mask: 7 ms, identical.
- **CUDA graph capture in torch 2.1 is global.** Any CUDA call from another
  thread during a capture breaks it (there is no `capture_error_mode` here).
  Capture under `perception/core/gpu.py` `GPU_LOCK`, and hold it in every
  other GPU user that can run in a parallel thread.
- **A CUDA graph only helps launch-bound work.** The radar front end saved
  ~1 ms of 6; the big gain was the ~100 tiny ops per frame after `nonzero`,
  moved into per-cell kernels inside the graph. Time graph and eager parts
  separately before deciding.
- **Several processes on the GPU time-slice it.** Four LiDAR worker
  processes and the radar together were slower (242 s) than two workers and
  the radar (169 s). Measure before adding GPU processes.
- **A binary protocol on stdout needs fd 1 moved first.** rs_driver
  ("Starting RoboSense LiDAR...") and the ZED SDK print to stdout from C++,
  where reassigning `sys.stdout` does not reach. The live-view workers
  `dup` fd 1 for their messages and point fd 1 at stderr
  (`live/common.py`).
- **Evict files from the page cache without root** to time cold reads
  repeatably: `os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)` per file;
  `fincore` shows what is cached.

## Environment (any new board)
- **`pip install ultralytics` replaces the CUDA torch with a CPU one** (via
  torchvision). Install NVIDIA's torch first and pin it with a constraints
  file. Nothing in the repo imports ultralytics.
- **`fpga-udp==1.3.0` (in the environment file) is not on PyPI.** Nothing in
  the repo imports it.
- **CuPy 12.3 (the last series for Python 3.8)** needs `fastrlock==0.8.2`.
  The newest fastrlock sdist needs Python ≥ 3.9 to build.
- **A ROS install can put its packages on `PYTHONPATH`** (`ibrahim-desktop`
  sources ROS Foxy). pytest then loads ROS plugins and fails on `lark`. Run
  the tools with `PYTHONPATH` unset.
- **User site-packages leak into conda envs.** On `ibrahim-desktop`,
  `~/.local/lib/python3.8` supplies numpy to `rel_1_py_3_8`. The version
  matches today; set `PYTHONNOUSERSITE=1` if it ever does not.

## Analysis
- **A clean PASS from `check_session` doesn't prove timing.** Use Stage 3 or
  the velocity check.
- **When there is no ground truth, score one sensor against another.** The
  radar false-alarm proxy is the share of confirmed radar tracks with no LiDAR
  object nearby; that is how the multipath ghosts were found. But a
  single-target LiDAR track says nothing about frames where it has no target:
  the "empty" stage3 frames still had the walker in view.
- **A file several tools write must be written the same way by all of
  them.** `sync/timing_lidar_track.npz` (the LiDAR target) is written by the
  report and by `timing_consistency.py`; when the LiDAR extrinsic came in,
  only the report applied it (found 2026-11-07, before it bit). When a
  convention changes, grep for every writer, not only the readers.
- **Validate tools against ground truth before trusting them.** Each of these
  bugs was caught this way:
  - the radar range extraction absorbed a walker who stayed in one range span;
  - the camera tracker followed trees;
  - the LiDAR statistics counted empty returns.
