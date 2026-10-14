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
- **The DCA packet delay limits the frame rate:** 50 µs caps the stream at
  7.44 fps. Use 25 µs.
- **I/Q order is I first.**
- **The azimuth sign is mirrored** in the processing.
- **Frame stamping:** use the burst that carried the frame's **first** byte.

## LiDAR
- **Zero-range points:** filter `range > 0`.
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

## Analysis
- **A clean PASS from `check_session` doesn't prove timing.** Use Stage 3 or
  the velocity check.
- **Validate tools against ground truth before trusting them.** Each of these
  bugs was caught this way:
  - the radar range extraction absorbed a walker who stayed in one range span;
  - the camera tracker followed trees;
  - the LiDAR statistics counted empty returns.
