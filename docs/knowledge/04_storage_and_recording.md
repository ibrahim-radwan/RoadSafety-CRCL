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
- **The eMMC limits session length** (2026-11-23). The camera's lossless
  video grows with the scene: 2–4 GB/h indoors, **16.6 GB/h on the road**
  (`run_20261123_1619`). That session started with 8.3 GB free on the eMMC,
  the ZED stopped itself at its 2 GB minimum after 18 min, and the launcher
  ended the session abruptly. Now:
  - the launcher budgets **20 GB/h** for the camera and prints the minutes of
    road recording the eMMC holds (a warning below 30 min);
  - it watches the eMMC every 10 s, warns at 10 and 3 min left, and below
    `--stage-min-free-gb` (3 GB) **stops all sensors in order**: stop reason
    `storage limit: ...`, a complete session, a warning in the check.
  - **The iPad page shows the camera space** (minutes of road recording,
    counting down while recording) and refuses Start below 1 min or for a
    duration longer than the space (2026-11-25). The launcher refuses the
    same below 1 min.
  - **Leftovers:** a staging folder that never reached the SD card (a failed
    move, a power loss) is listed on the page and in the launcher's
    preflight; **Move to SD card** moves it into its session. Until then it
    shortens every session.
  - The move after a session is copy-then-delete (two disks): the eMMC is
    freed only once the copy is complete.
  - For sessions longer than the eMMC allows: free space on it, or add an
    NVMe drive (recommended), or (open) segment the camera and move segments
    to the SD card while recording.

## Data rates at 8 fps

| Sensor | Rate |
|---|---|
| Radar | 90.6 GB/h |
| LiDAR | 17.5 GB/h |
| Camera | 2–4 GB/h indoors, 16.6 GB/h on the road (launcher budgets 20) |
| Altos V4 | ~0.3 GB/h (a 3–4 m room, 2026-11-15; the launcher budgets 1 GB/h; at most ~13 GB/h at its 12,000-point maximum) |
| **Total** | **~110–130 GB/h (31–36 MB/s)** |

The card holds about 7–8 h. The eMMC holds (free GB − 3) / 20 h of road
camera: about 1 h per 23 GB free.

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
