# Project status and roadmap (as of 2026-10-24)

## Where the project stands

A three-sensor data-collection rig for two-wheeler road safety, running on an
NVIDIA Jetson AGX Orin. All of the following are built and verified on
hardware:

| Area | State |
|---|---|
| Radar firmware | `r075m_v2_8fps` flashed (8 fps, 2.62–75 m). All 20 variants rebuilt with the DSS-L3 memory-init fix. See [03](03_radar_firmware_and_dca.md). |
| Capture | Three-sensor launcher with preflight checks, board-temperature monitoring and a radar-rate watchdog; iPad web control. See `sync_capture_pipeline/DATA_COLLECTION_GUIDE.md`. |
| Storage | 1 TB SD card (ext4), camera staged on the eMMC; 0.2% camera drops. See [04](04_storage_and_recording.md). |
| Synchronisation | Offline alignment on CLOCK_MONOTONIC. **Stage 3 offsets measured**: radar +12.65 ± 1.45 ms, camera +23.34 ± 3.28 ms (LiDAR reference). Applied automatically. See [07](07_synchronisation_and_stage3.md). |
| Validation | `check_session.py` (completeness, pass/fail) runs after every session. `make_report.py` produces a descriptive HTML report per session, with no verdict. |
| Replay | One window, six panels (camera, LiDAR, LiDAR 3D, radar BEV, range-Doppler, objects), each can be maximised; bird's-eye views fill their panels; live GPU detection and tracking on radar and LiDAR; event strip; radar point cloud (detections, raw). See `sync_capture_pipeline/replay/README.md`. |
| Processing and detection | One PyTorch library, `sync_capture_pipeline/perception/`, plus `run_perception.py`: radar signal-processing stages (`radar_dsp.py`), GPU radar CFAR (TDM Doppler compensation, in-situ phase correction, multipath-ghost flags), GPU LiDAR objects (local ground, overhead structures removed), Doppler-aware tracking. Scored against the LiDAR walker: see `perception/README.md`. The old radar chain was merged in and deleted on 2026-10-25 (below). |
| Tests | `sync_capture_pipeline/tests/`: report, perception, baseline (golden frames: ours, and the old chain's frozen outputs) and radar-stage tests. Session tests need the reference sessions and skip if absent. |

## Reference recordings (not in git)

Kept on the 1 TB SD card (`/mnt/sdcard/sessions/`) and to be copied to an
external drive by the project owner.

| Session | Content |
|---|---|
| `freewalk_ref_02` | **The main reference.** A 10-minute scripted walk with these segments: empty scene, left/right check, free walk, close range (~2 m), far range (~20 m), standing, two people, plate check, empty scene. |
| `freewalk_ref_01` | DCA1000 throttle fault (only the first 103 s of radar is valid; nominal frame times). The fault test case. |
| `stage3_01..03` | Daylight Stage 3 calibration sessions (90 s each). |

Older eMMC sessions (`sync_capture_pipeline/sessions/`) predate the radar
firmware fix; their radar data is about 50% corrupted. They are kept only as
history.

## Ownership

- **End-user research team:** all sensor calibration (radar antenna/intrinsic
  and the three-sensor extrinsics), data collection, and the use of the data
  (annotation, ML). They pull this repository and read
  `sync_capture_pipeline/NOTES_FOR_TEAM.md`.
- **This repository:** capture, synchronisation, validation, replay,
  firmware packaging, and **all processing, filtering and detection** (since
  2026-10-25, at the team's request). That includes the radar processing
  chain that came from the team's initial Windows work
  (`cascaded_radar_capture_pipeline/processing`), which is to be merged into
  `sync_capture_pipeline/perception/`. The team no longer runs the tools in
  that folder.
- **The Windows firmware-build machine** (TI SDK, firmware source) has its own
  repository. Briefs to it go through a USB handover folder (see
  [09](09_working_practices.md)).

## Second machine: `ibrahim-desktop` (2026-10-24)

A second Orin (JetPack R35.4.1, same as the rig) set up for replay and
processing work, user `ibrahim`.

- `rel_1_py_3_8` was created per `docs/NEW_BOARD_SETUP.md`, with these
  adjustments, which are now in that file:
  - no `fpga-udp`;
  - no `ultralytics`, which would replace the CUDA torch;
  - `fastrlock` 0.8.2 and CuPy 12.3 added.
- Pyzed 4.1.4 is installed and the LiDAR wrapper is built.
- The reference sessions are on the external drive at
  `/media/ibrahim/Elements/HASSAN_JETSON_SYNC_CAPTURE_SESSIONS/`. It is NTFS
  through FUSE, which is slow (see [08](08_pitfalls_and_lessons.md)).
- ROS Foxy sets `PYTHONPATH` in this user's shell, so run the tools with it
  unset.

## Merging the radar chain into `perception/` (since 2026-10-25)

The plan, agreed with the owner:

- **One PyTorch library** (`sync_capture_pipeline/perception/`) holds every
  processing, filtering and detection stage, for both GPU and CPU.
- It replaces the old chain (`cascaded_radar_capture_pipeline/processing`)
  and its separate CuPy copy, plus the report's `RadarProcessor`.
- It is migrated stage by stage. Each port is tested for equivalence against
  the golden frames and scored against the baseline before the old code goes.

| Step | State |
|---|---|
| 1. Baseline | **Done 2026-10-25.** `perception/baseline.py`. Scores of all 7 reference sessions in `perception/baseline/scores.json` (git), for ours and the old chain's clouds on both paths. Golden per-frame outputs of 20 frames on `freewalk_ref_02`, `stage3_02` and `run_20261014_1326` (`<session>/sync/baseline/`, not in git). Our detector's regression test: `tests/test_baseline.py`. |
| 2. Shared core | **Done 2026-10-25.** `perception/core/`: `radar_config`, `radar_io` (both moved from the old chain, with history), `calibration`, `lidar_io`, `session`, `conventions`. The replay, `run_perception.py`, `radar_cloud.py`, `baseline.py`, `estimate_rx_phase.py` and the report's `RadarProcessor` use it. The old chain reads its config and frames through shims. Verified: no output changes (the old chain reproduces its golden frames exactly on both paths; our golden frames and scores are unchanged). |
| 3. Port the old chain's stages | **Done 2026-10-25.**
- `perception/radar_dsp.py`: every general stage of `process_frame` in PyTorch.
- `perception/radar_chain.py`: those stages composed as the old front end. It reproduces the golden frames of both old paths at ~21 ms/frame.
- Our `RadarDetector` (`perception/radar.py`) runs on the same stages; its golden frames and scores are unchanged.
- Tests: `tests/test_baseline.py`, `tests/test_radar_dsp.py`.
- **Retired** (owner's decision, 2026-10-25): stage 8's single-person gates, the old xyz cloud, and the single-subject accumulation and micro-Doppler. The pipeline's radar point cloud is our multi-object detections. Accumulation and micro-Doppler are to be rebuilt per tracked object in the techniques phase. The old behaviour stays reproducible from git history and the golden files. |
| 4. Remove duplicates | **Done 2026-10-25.**
- Deleted: the old chain (`cascaded_radar_capture_pipeline/processing/`, including `gpu_accel.py`, its GUI `streaming.py`, `main.py`, `accumulate.py`, `microdoppler.py`); its tools (`live_capture.py`, `sync_replay.py`, `capture/`, the Windows workspace); `radar_cloud.py`, `make_report.py --radar-cloud`, the replay's "filtered" option; the old-chain scripts in `utils/benchmarking_and_frames_analysis/`. All are in git history.
- The report's `RadarProcessor` is now a thin wrapper over `RadarDetector` (equal within 0.0002 dB).
- The old chain's outputs stay as frozen references: `golden_team_{cpu,gpu}.npz` (test_baseline.py) and `golden_stages.npz` (test_radar_dsp.py).
- `cascaded_radar_capture_pipeline/` now holds only the firmware.
- Verified:
  - 140 tests pass, including a new headless test of the replay sources;
  - the report, rebuilt from raw data on `stage3_02`, passes;
  - the radar scores are unchanged;
  - the replay runs and exits cleanly.
- The new test exposed a shutdown crash in the LiDAR source, now fixed: `Source.stop()` waits for the thread. |

What porting the old chain showed (2026-10-25):

- **Its Bartlett beamformer could never run:** `_az_bartlett`'s `einsum`
  reads the [azimuth, channel] steering matrix as [channel, azimuth] and
  raises `ValueError` for any input. The port computes the intended aᴴRa.
- **It processes only to 16 m** (`max_range_m`), though the firmware reaches
  75 m. It also requires 10 dB above the CFAR threshold plus a 12 dB power
  floor. These are parameters of `RadarChain`, not fixed.
- **Its CPU and GPU paths differ in more than the CFAR:**
  - In range bins without detections, the range-azimuth focus mask takes the
    top 3 of nearly equal noise cells, and float differences pick different
    ones.
  - MUSIC is ill-conditioned where a range bin has no clear signal.

  Neither matters for detection. The port agrees with the GPU path.
- **Stage 8 is built for one person.** It keeps azimuth peaks only within
  0.3 m of the peak nearest boresight, drops a second cluster at the same
  range more than 0.3 m to the side, and gates everything to within 0.9–3 m
  of the tracked subject. On a road that removes a second road user.

What the baseline showed:

- **The old chain on the GPU path (CASO) matches the CPU path (OS)** on
  `freewalk_ref_02`: 49% / 50% detected, 5.0° / 4.9° spread. Its cloud there
  is now the GPU one. The CPU one is kept as
  `sync/radar_cloud_filtered_cpu.npz`.
- **Our LiDAR detection drops to 47% / 27% on `run_20261014_1325` / `1326`**
  (92–97% elsewhere). There the walker stays within 2.7–6.6 m, close to other
  structures, and our grid clustering merges them into one 6–7 m object.
  **Fixed 2026-10-26** with density-based clustering (84% / 94% now; 99–99.9%
  on the other sessions). See [05](05_lidar.md).
- **The radar azimuth spread on `freewalk_ref_01`** (11.7°) reflects its
  nominal radar times after the DCA throttle fault. The radar–LiDAR yaw of
  the Oct 14 mounting is about +2.4° to +7.7°.
- **`stage3_01` scores like `stage3_02`/`03`.** Its radar data is clean: it
  was recorded with `r075m_v2_8fps`. The replay README's note that it is
  degraded seems to refer to an older recording of the same name.

## Open items and ideas

- **On hold (owner's decision):** recommending a 90 s Stage 3 check on the
  first field day after mounting. Do not add it to the guide unless asked.
- **The azimuth "mirror" is closed:** it is a frame convention (the team's
  azimuth is right-positive), not an error in `process_frame`. See
  [03](03_radar_firmware_and_dca.md).
- **Radar calibration findings for the team** (in `NOTES_FOR_TEAM.md`):
  - an inter-device TX phase error (TX3/TX4 ~100–125° off);
  - beam sidelobes only 5–10 dB down;
  - the radar–LiDAR yaw differs per mounting;
  - the elevation bias.
  
  Our in-situ correction (`calibration/radar_rx_phase.json`) is provisional
  until their recalibration.
- **The old chain's CuPy and CPU paths differed** (CASO vs OS CFAR on
  range-Doppler). Reported to the team. The chain is retired;
  `RadarChain(cfar=...)` reproduces either.
- **Techniques phase (from 2026-10-26).** Done so far:
  - LiDAR density-based clustering;
  - the tracker vectorised (3× faster with many tracks);
  - per-track radar accumulation and micro-Doppler (`perception/track_features.py`;
    cadence experimental);
  - **a moving / static state per track** (stationary rig): on `freewalk_ref_02`
    the walker's tracks are moving 79% of the time (the rest is the standing
    segment and turns), and the other tracks are static 85% of the time.
    **Ego-motion compensation is still needed** for riding sessions; it needs
    riding data to validate;
  - **radar stationary reflectors** (zero Doppler; experimental, opt-in in
    the replay). LiDAR support is 82–100% within 6–10 m, and 55% overall
    outdoors: beyond that, sidelobe and multipath images dominate until the
    antenna recalibration.
- Next for detection:
  - a learned LiDAR classifier (walls now break into more pieces, ~37 objects
    per sweep; the moving/static flag helps tell road users from structure);
  - camera detection (needs a model and the team's camera extrinsics for
    overlays);
  - a per-session radar yaw estimate, if the team wants one before their
    extrinsics.
- The DCA1000 throttle fault has an unknown cause (power cycle clears it).
- The Windows capture config `AM273X_Capture_fixed.json` still uses a 50 µs
  packet delay, which loses data at 8 fps.
- Only `r075m_v2` and `r075m_v2_8fps` are hardware-proven. Run the acceptance
  check before using any other variant.
- CuPy (12.3) was installed in `rel_1_py_3_8` on `ibrahim-desktop`
  (2026-10-24) for the old chain's GPU path. Nothing uses it since the
  merge.
- Possible later work: a CI subset that runs without hardware or data. The
  synthetic perception tests already do.
- Possible later work: GPU LiDAR rendering in the replay. The numpy render
  takes ~10 ms per sweep and holds the GIL.
