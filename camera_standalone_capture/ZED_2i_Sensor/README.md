# ZED 2i Python dataset recorder

A standalone Python recorder using the ZED SDK, with no ROS requirement.
One editable configuration, `config.yaml`, controls acquisition and export:
**HD1080, 1920×1080, 30 FPS requested, SYSTEM_CLOCK**.

## Application folder

All files needed to run the standalone application are together:

```text
zed2i_python/
  config.yaml        single editable recorder/export configuration
  recorder.py        run this to record
  sensors.py         IMU and physical-sensor acquisition
  images.py          optional direct-image writers
  dataset.py         shared configuration, calibration and file helpers
  export.py          automatic or manual PNG export
  requirements.txt   Python application dependencies
  README.md          installation and usage
  tests/             regression tests
  dataset/           generated sessions, each with its own config snapshot
```

If the SDK and packages are already installed, start from the project root:

```bash
cd zed2i_python
python3 recorder.py --check-config
python3 recorder.py
```

Activate `.venv` first if you created it during installation. Edit `config.yaml`
to change settings; the recorder finds it beside `recorder.py` automatically.

## Recording workflow

> **In this project** `config.yaml` sets `recording.auto_export: false`: sessions keep only the lossless
> `camera.svo2` (plus IMU and calibration), and images are extracted offline. See
> `docs/knowledge/06_camera.md`. The text below describes the recorder's general behaviour.

**By default, PNG export runs after recording stops, not during recording.**
You only start `recorder.py`; it calls the exporter automatically when
`recording.auto_export: true` and capture has finalized successfully.

1. **During recording:** save `camera.svo2`, timestamped IMU/sensor CSVs and calibration.
2. **After Ctrl+C or the configured `--duration`:** stop acquisition, drain pending writes and finalize/verify the archive.
3. **After successful finalization:** `export.py` replays the archive and creates the PNG streams, sensor copies and calibration under `export/`.

| How you run it | When are PNG files written? |
|---|---|
| `python3 recorder.py` | Automatically after recording stops, with the default config. |
| `python3 recorder.py --no-export` | Later, when you run `export.py` manually. |
| `python3 recorder.py --direct-images --no-export` | During recording, through `images.py`, under `capture/images/`; no automatic offline export. |

Direct image writing is an optional capture mode. It still retains the SVO2
archive and can reduce recording FPS because PNG encoding and file writes compete
with acquisition. The default keeps that work after capture to reduce this load.

The default workflow records lossless stereo to SVO2 and saves timestamped physical
sensor CSV files during acquisition. After Ctrl+C, it automatically produces
**RGB PNG images** for rectified and unrectified left/right views. Depth and
positional tracking are disabled during capture, and the default export does not
compute depth, disparity, confidence or point clouds. Keeping PNG encoding after
capture avoids compression and large image-file writes in the acquisition loop.

PNG is a common lossless stereo dataset format; for example,
[KITTI-360](https://www.cvlibs.net/datasets/kitti-360/documentation.php) distributes
rectified stereo pairs as 8-bit PNGs. JPEG is also supported explicitly for direct
saving, but is lossy. The program never changes image format, resolution or quality
silently. RGB PNGs preserve the three color channels and omit the SDK's unused
opaque alpha byte. Unrectified means camera-ISP color, not Bayer sensor data.

## Requirements

**Install ZED SDK 4.1.4 to reproduce the hardware-tested setup.** The code requires
SDK **4.1 or newer** because it records SVO2; support for later releases has not
been verified by this project's hardware test. SDK 4.1.4 includes fixes for SVO2
recording and lossless H26X compression; see the
[official 4.1 release notes](https://docs.stereolabs.com/docs/development/zed-sdk/release-notes/4-x/4-1).
Use the same SDK version for recording and export: the exporter checks the saved
version and rejects a mismatch by default.

| Requirement | What to install or provide |
|---|---|
| Camera | ZED 2i connected through a USB 3 port and suitable cable. |
| GPU | An NVIDIA GPU supported by your chosen ZED SDK, or a supported NVIDIA Jetson. GPU access is needed for recording and SDK replay, including images-only export. |
| CUDA and driver | Versions compatible with the selected SDK installer. On Jetson, use the CUDA and driver supplied by the matching JetPack installation. |
| Python | 64-bit Python 3.8 or newer, with a matching `pyzed` wheel available for your SDK and platform. Python 3.8.10 was used in the hardware test. |
| ZED Python API | `pyzed` matching the installed SDK and Python interpreter; install it using the SDK's `get_python_api.py` helper. |
| Storage | A writable dataset directory with room for both the SVO2 archive and four PNG streams. An SSD is preferable for sustained recording and export. |

The tested machine is a **Jetson AGX Orin**, running **Ubuntu 20.04.6,
L4T R35.4.1, CUDA 11.4.19 and SDK 4.1.4**. These identify the verified setup;
other machines need their own matching SDK installer. The default
`H264_LOSSLESS` codec requires encoder support. If your platform does not support
it, explicitly set `recording.compression: LOSSLESS` in `config.yaml` and check
the resulting capture rate. The recorder does not silently change codecs.

### Python packages

`requirements.txt` defines the application dependencies:

| Package | Version range | Purpose |
|---|---|---|
| NumPy | `>=1.20,<2` | SDK image arrays and sensor calculations; keep the NumPy 1.x ABI for this SDK profile. |
| PyYAML | `>=5.3,<7` | Configuration parsing. |
| Pillow | `>=8,<12` | PNG, BMP and optional JPEG writing. |
| pytest | Optional, install separately | Regression tests only. |

`pyzed` is installed separately after the native SDK. The SDK's API installation
helper also uses `requests` and may install its own build dependencies, such as
Cython. OpenCV, PyOpenGL, CuPy, ROS and FFmpeg are not application dependencies.

## Installation

### 1. Install the native SDK

Download **SDK 4.1.4** from the
[Stereolabs SDK download page](https://www.stereolabs.com/developers/release/).
Choose the installer for your operating system, architecture and CUDA or JetPack
version. Follow the official installation instructions for
[Ubuntu/Linux](https://docs.stereolabs.com/docs/development/zed-sdk/linux) or
[NVIDIA Jetson](https://www.stereolabs.com/docs/development/zed-sdk/linux/work-with-nvidia-jetson).
On Jetson, check `cat /etc/nv_tegra_release` before choosing the installer;
the installer must support that L4T release. CUDA comes with JetPack.

The usual Linux SDK installation directory is `/usr/local/zed`. To confirm camera
detection on a machine with a desktop, open `/usr/local/zed/tools/ZED_Explorer`,
check that the camera streams, then close it before starting this recorder.

### 2. Prepare Python and install packages

On Ubuntu, install the Python setup tools if they are missing:

```bash
sudo apt update
sudo apt install python3-pip python3-venv python3-dev
```

Use a virtual environment so the recorder's dependencies are separate from other
projects. From the project folder:

```bash
cd '/media/ibrahim/94DD-7E88/ZED 2i-Camera/zed2i_python'
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install requests
python3 /usr/local/zed/get_python_api.py
python3 -m pip install -r requirements.txt
```

The last command reapplies the project's package constraints after the SDK helper
runs. Run the helper from this writable folder and with the environment activated,
so it downloads the wheel here and installs it for the same interpreter that will
run the recorder. Internet access is needed to download packages. See the
[official Python API installation guide](https://www.stereolabs.com/docs/development/api-languages/python).
If your existing Python environment already imports the correct `pyzed` and has
the required packages, you can keep using it instead of creating a new environment.

### 3. Verify the installation

```bash
python3 --version
python3 -m pip check
python3 -c "import pyzed.sl as sl; import numpy, yaml; from PIL import Image; print('ZED SDK:', sl.Camera.get_sdk_version())"
python3 recorder.py --check-config
```

For the verified setup, the SDK check should print **4.1.4**. `--check-config`
validates the YAML without opening the camera; it does not prove camera detection
or GPU availability. The following short capture checks the complete workflow:

```bash
python3 recorder.py --duration 3
```

## Record and export

In a new terminal, enter the project folder and activate `.venv` if you created it:

```bash
cd '/media/ibrahim/94DD-7E88/ZED 2i-Camera/zed2i_python'
source .venv/bin/activate
python3 recorder.py
```

Press Ctrl+C once to stop capture, drain sensor writes, finalize/verify SVO2 and
start the configured PNG export. Export can take longer than recording. Ctrl+C
during export cancels the generated output safely and preserves the original
recording. GPU access is required for SDK capture and replay; restricted containers
may not expose the necessary devices. No ROS environment needs to be sourced.

To postpone PNG generation:

```bash
python3 recorder.py --no-export
python3 export.py --session dataset/SESSION_FOLDER --images-only
```

To compute optional depth/disparity/confidence later:

```bash
python3 export.py --session dataset/SESSION_FOLDER --with-depth --overwrite
```

`--overwrite` replaces only a recognized generated export, after its replacement
has passed validation. Each session preserves `config_used.yaml` for reproducibility;
these are historical snapshots, not additional editable configurations.

## Troubleshooting

| Symptom | Action |
|---|---|
| `No module named pyzed` | Activate the intended environment and rerun `/usr/local/zed/get_python_api.py` with its `python3`. Installing `requirements.txt` alone does not install the SDK API. |
| NumPy binary incompatibility on import | Restore the NumPy range in `requirements.txt`, reinstall the API matching your SDK, and repeat the import check. |
| `CAMERA NOT DETECTED` | Check the USB 3 connection, close other camera applications, and verify detection with ZED Explorer. |
| CUDA initialization or shared-library error | Check the SDK/driver/CUDA pairing and GPU access. On Jetson, use the installer matching the installed L4T release. |
| Recording codec is unsupported | Select `LOSSLESS` explicitly in `config.yaml`; verify performance before a long recording. |
| Export reports an SDK version mismatch | Use the SDK version saved in that session's `metadata.json`. |
| Export takes longer than recording | PNG encoding and disk writes run after capture by design. Keep `auto_export: true` to generate PNGs automatically, or use `--no-export` and run the exporter later. |

## Left/right image meanings

| Directory | SDK view | Image geometry |
|---|---|---|
| `left/` | `VIEW.LEFT` | Rectified left camera |
| `right/` | `VIEW.RIGHT` | Rectified right camera |
| `left_raw/` | `VIEW.LEFT_UNRECTIFIED` | Unrectified left camera |
| `right_raw/` | `VIEW.RIGHT_UNRECTIFIED` | Unrectified right camera |

All four are 1920×1080 and share the original frame timestamp. Their pixel
coordinates and intrinsics differ between raw and rectified geometry. Both sets
of calibration are saved. One shared SDK view mapping is used for direct capture
and offline export; tests give each view different pixels to detect swapped eyes,
duplicated images or incorrect raw/rectified selection.

## Dataset files

```text
dataset/zed2i_sdk_session_.../
  camera.svo2                    original stereo + native SDK sensor archive
  config_used.yaml
  metadata.json                  camera identity, SDK, clock checks and codec
  dataset_schema.json            units, timestamp and orientation conventions
  capture_status.json            actual rates, counts, gaps and finalization status
  calibration/
    SN....conf                   preserved SDK factory calibration with hash
    camera.json                  raw/rectified intrinsics, distortion and transforms
  capture/
    frames.csv                   original image timestamps and frame intervals
    timing/camera_gaps.csv        acquisition gaps, only when observed
    imu/imu.csv                  quaternion, roll/pitch/yaw, calibrated acceleration/gyro
    imu/imu_raw.csv               uncalibrated acceleration/gyro
    imu/magnetometer.csv          calibrated/raw field and heading quality
    sensors/barometer.csv         available pressure/relative altitude
    sensors/temperature.csv       temperatures with explicit host-poll times
  export/
    left/ right/                 rectified RGB PNGs
    left_raw/ right_raw/          unrectified RGB PNGs
    frames.csv                   PNG names, original timestamps, dimensions and sources
    imu/ sensors/ timing/         copied original CSVs, where available
    calibration/                 recorded and replay calibration
    tf/static.json               stereo, camera/IMU and magnetometer transforms
    export_summary.json          validated stream counts and processing settings
```

File names contain the original nanosecond SDK timestamp and frame index. Host
receipt times are separate; file-writing times are not acquisition timestamps.
The SDK 4.1 SVO decoder can round timestamps to microseconds. Export matches each
archived frame by index and timestamp at that precision, then preserves the exact
original nanosecond ledger timestamp in PNG names and the image index. Decoder
skips are retried by seeking; irrecoverable mismatches fail validation.

SVO2 is retained because it contains original stereo and native-rate integrated
sensor data. It enables later re-export and optional SDK processing. The SDK does
not embed the factory calibration file, so the recorder copies it and checks its
hash on replay. Self-calibration is disabled by default for consistent geometry.
Rectified images and optional depth maps are computed from the recorded stereo;
SDK-version and calibration checks protect reproducibility.

## IMU angles and sensor conventions

`imu.csv` saves quaternion components in **x,y,z,w** order. Roll, pitch and yaw
are in radians, with the intrinsic ZYX convention and the original SDK orientation
reference. Yaw is not a guaranteed absolute heading. At Euler gimbal lock the
representation sets yaw to zero; use the quaternion as the authoritative rotation.
Gyroscope readings are angular rates about sensor axes, not Euler angle derivatives.

Calibrated and raw gyroscope readings are converted from degrees/s to rad/s;
gyroscope covariance uses rad²/s². Acceleration is m/s² and includes gravity,
magnetic field is µT, and pressure is hPa. Covariance names ending in `_sdk`
retain SDK conventions. Raw channels have no invented orientation/covariance.
Temperature timestamps are labeled host-poll times because the API supplies no
measurement timestamp for them.

Independent `TIME_REFERENCE.CURRENT` sensor polling deduplicates timestamps. The
CSV contains latest samples observed by Python, not every native 400 Hz IMU sample;
SVO2 is the native-rate archive. Observed rates are recorded. SYSTEM_CLOCK epoch
checks run at startup and periodically, and backward image/IMU timestamps stop
capture. `tracking.enabled: true` optionally enables live SDK tracking/depth and
pose CSVs; the default does not invent world position or odometry.

## Efficiency and verification

Default capture has no PNG encoding, CPU image copies, depth maps or tracking in
the recording loop. CSV queues are bounded and flush every second, including idle
streams. Capture protects a disk reserve plus pending-write headroom. Export owns
SDK buffer copies, bounds pending frames, stages its output and installs it only
when all expected streams have complete counts and matching timestamps.

For direct PNG files during capture, explicitly opt in:

```bash
python3 recorder.py --direct-images --no-export
```

**Note:** With the default PNG configuration, `--direct-images` saves all four
1920×1080 streams (`left`, `right`, `left_raw`, `right_raw`) during recording under
`capture/images/<stream>/`. IMU, orientation angles and other enabled sensor CSVs
are also saved during recording under `capture/`; calibration is saved under
`calibration/`. The SVO2 archive is still retained.

`--no-export` skips the automatic post-recording export only. It does not disable
direct images, sensor files or SVO2 recording. Press Ctrl+C once and wait for
pending writes and archive finalization. This command does not create an
`export/` directory; you can run `export.py` manually later if needed.

Direct PNG compression can substantially reduce camera FPS. For less processing
during capture, use `python3 recorder.py` and let PNG export run afterward.
`images.workers`, `queue_frames`
and `writer_backend` control bounded background work. Compressed direct formats
use spawned processes with reusable shared buffers; workers receive no camera
handle or CUDA state. Completed file indexes are published in frame order, and
buffers are released on failure. BMP and explicit JPEG overrides remain available,
but PNG is the default dataset format. Extra queued memory cannot fix insufficient
CPU or sustained disk throughput.

Ctrl+C drains accepted data. Cleanup continues even if camera closure fails, and
errors mark the session incomplete. A complete session means saved frames passed
integrity checks; inspect gap counts separately for acquisition continuity.
Flushes and atomic renames do not guarantee power-loss durability. Actual capture
rates are workload-dependent; hardware-test sessions retain `verification.json`
with their measurements rather than promising a sustained 30 FPS on every system.

## Modules and tests

**You only run `recorder.py` and edit `config.yaml` for normal recording.**
The other Python files are supporting modules; they do not need separate terminals
or manual startup. `export.py` also has a command-line interface for manual export.

| File | Responsibility |
|---|---|
| `recorder.py` | Starts the camera, manages the recording session and shutdown, then calls automatic export when enabled. |
| `sensors.py` | Collects calibrated/raw IMU, orientation and available magnetometer, pressure and temperature measurements. |
| `images.py` | Saves image files during recording only when direct image writing is enabled; manages bounded background writers. |
| `dataset.py` | Shared configuration validation, timestamps, calibration, schemas and file handling used by capture and export. |
| `export.py` | Replays a finalized recording to produce PNGs, copied sensor data and calibration; optionally computes derived measurements. |
| `config.yaml` | The single editable configuration for recording and export. |

Separating these responsibilities makes the code easier to maintain and test.
The file split does not meaningfully slow recording: camera processing, image
encoding and storage throughput determine performance. Combining the modules
would mainly change code organization.

Regression tests are retained because they cover timestamps, buffer ownership,
stream selection, queues, error cleanup, atomic export and calibration checks:

```bash
python3 -m pip install pytest
python3 -m pytest -q -p no:cacheprovider tests
```

## Official SDK references

- [Stereolabs tutorials](https://github.com/stereolabs/zed-sdk/tree/master/tutorials)
- [Python multi-camera SVO recording](https://github.com/stereolabs/zed-sdk/blob/master/recording/recording/multi%20camera/python/recording_multi.py)
- [SVO contents and replay](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/recording)
- [IMU orientation and measurements](https://docs.stereolabs.com/docs/development/zed-sdk/modules/sensors/imu)
