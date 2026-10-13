# Storage and recording

## Setup (decided 2026-10-10)

- **SD card:** 1 TB SanDisk (U3/V30/A2), ext4, label `sensor_data`.
- **`/etc/fstab`:** mounted at **`/mnt/sdcard`** by UUID, with
  `defaults,noatime,data=writeback,nofail,x-systemd.automount,x-systemd.device-timeout=10`.
  - Reboot and re-insertion need no commands.
  - Never remove it while recording.
  - The UUID belongs to this card. A reformatted or new card needs a new
    fstab line.
- **Sessions:** `/mnt/sdcard/sessions/<name>/`.
- **The camera is staged on the eMMC** (`/home/jetson/camera_stage`) and moved
  into the session when it ends (`capture_session.py --camera-stage`; the web
  server passes it).
  - **Why:** any disk shared with the radar's 25 MB/s stream stalls the ZED's
    writes. Measured drops:
    - all on SD: 5.44%;
    - with writeback tuning: 2.36%;
    - camera on eMMC: **0.18%**.
  - Radar and LiDAR were lossless in every test.
- **Safety check:** the launcher refuses to record under `/mnt/sdcard` unless
  it is mounted, so the bare directory on the eMMC never silently takes the
  data.

## Data rates at 8 fps

| Sensor | Rate |
|---|---|
| Radar | 90.6 GB/h |
| LiDAR | 17.5 GB/h |
| Camera | 3.9 GB/h |
| **Total** | **~112 GB/h (31 MB/s)** |

The card holds about 8 h; the eMMC holds about 4 h of camera.

## History (why not other options)

- **eMMC only:** too small, and its write stalls caused camera drops.
- **NVMe:** recommended at first; dropped once the SD card plus staging passed.
- **USB stick:** tested once; not used.

## Recording rules

- Raw data only. **No PNG export at capture** (the ZED writes lossless SVO2).
  The LiDAR records raw UDP to pcap with monotonic receipt stamps, not decoded
  points. Images and points are produced offline.
- **Sessions are valuable:** never delete recordings without the owner's
  approval. The safety check in the tooling blocks deleting from inside a
  session directory, and it must not be worked around.
