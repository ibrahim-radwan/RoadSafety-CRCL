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

## Radar
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
- **The DCA packet delay limits the frame rate:** 50 µs caps the stream at
  7.44 fps. Use 25 µs.
- **I/Q order is I first.**
- **The azimuth sign is mirrored** in the processing.
- **Frame stamping:** use the burst that carried the frame's **first** byte.

## LiDAR
- **Zero-range points:** filter `range > 0`.
- **One ground plane fails** on slopes and under roofs. The carport scene of
  freewalk_ref_02 made a single 13 × 8 m "object". Use local ground and drop
  overhead structures (see [05](05_lidar.md)).
- **`cv2.minAreaRect`'s angle is along its first side**, not the long side.
  Swap the sides, and the box turns by 90°.
- **Decode pcaps at ≤ 3×** real time.
- **`split_angle: 180`**, or the forward field of view splits across two
  sweeps.

## Camera
- **Share a disk with the radar and the camera drops frames.** Stage the camera
  on its own disk.
- **`pyzed.grab()` holds the GIL.** Run the camera in its own process.
- **Lossless H.264 decodes at only ~40–50 fps.**

## Software
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
- **Sessions on NTFS through FUSE are slow.** On the external drive, a 3 MB
  radar frame takes 78 ms median to read (up to 390 ms), and the replay got
  only 3 radar updates/s. The replay now reads ahead. For full speed, use
  ext4.

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
- **Validate tools against ground truth before trusting them.** Each of these
  bugs was caught this way:
  - the radar range extraction absorbed a walker who stayed in one range span;
  - the camera tracker followed trees;
  - the LiDAR statistics counted empty returns.
