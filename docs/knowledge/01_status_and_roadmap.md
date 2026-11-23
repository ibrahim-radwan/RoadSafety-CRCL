# Project status and roadmap (updated 2026-11-23)

## Where the project stands

A three-sensor data-collection rig for two-wheeler road safety, running on an
NVIDIA Jetson AGX Orin. All of the following are built and verified on
hardware:

| Area | State |
|---|---|
| Radar firmware | `r075m_v2_8fps` flashed (8 fps, 2.62–75 m). All 20 variants rebuilt with the DSS-L3 memory-init fix. See [03](03_radar_firmware_and_dca.md). |
| Capture | Three-sensor launcher with preflight checks, board-temperature monitoring and a radar-rate watchdog; iPad web control. See `sync_capture_pipeline/DATA_COLLECTION_GUIDE.md`. |
| Live view | iPad page card for the extrinsic alignment (2026-10-31): camera, LiDAR top view, radar map with detections, strongest reflector per sensor; nothing saved; never together with recording; off after 10 min. **Working on the rig (team, 2026-11-05):** hotspot set up, live view runs, recording works. See `sync_capture_pipeline/live/README.md`. |
| Storage | 1 TB SD card (ext4), camera staged on the eMMC; 0.1–0.5% camera drops. The eMMC limits session length (~1 h of road camera per 23 GB free); the launcher stops a session cleanly before it fills. See [04](04_storage_and_recording.md). |
| Synchronisation | Offline alignment on CLOCK_MONOTONIC. **Stage 3 offsets measured**: radar +12.65 ± 1.45 ms, camera +23.34 ± 3.28 ms (LiDAR reference). Applied automatically. See [07](07_synchronisation_and_stage3.md). |
| Validation | Session notes after every session (`check_session.py --quick`, seconds); the full check (`verify_svo.py`, then `check_session.py`) offline. `make_report.py` produces a descriptive HTML report per session, with no verdict. |
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

## Altos V4 radar: the fourth sensor (since 2026-11-15)

An Altos V4 imaging radar. It outputs a processed point cloud, not raw ADC
data. **Decided 2026-11-17: an additional sensor**, recorded alongside the TI
radar, LiDAR and camera. It connects through a second USB-Ethernet adapter
and keeps its factory addresses (02).

**Done:**
- First contact on `ibrahim-desktop`.
- A stdlib Python recorder and a checker: `altos_standalone_capture/`.
- A 3D viewer, live or replay (`view_altos.py`).
- A settings switch (`altos_settings.py`). Every change restarts the radar.
- A room analysis against tape measurements (`analyse_altos.py`).
- First numbers: threshold 0 gives ~55% more points than the default;
  150m_1 gives more points than 150m_0; the threshold is not a plain SNR cut.
- 30 s recorded at 15 fps with no losses.
- **Pipeline integration (2026-11-17):**
  - `capture_session.py`: the Altos starts and stops with the LiDAR and
    camera, after a preflight; `--no-altos` leaves it out.
  - Web page: an Altos checkbox for recording and live view, a status card,
    a live panel and calibration captures (`live/altos_live.py`).
  - `check_session.py`: an Altos section.
  - `align_session.py`: capture times (07), `sync/altos_timeline.csv`,
    `sync/altos_points.npz`, and the Altos columns of `pairs.csv`.
  - Replay: an Altos panel (trail, Doppler colours, optional 3D overlay) and an
    "Altos gaps" row; tested on Altos-only sessions.
  - Report: an Altos section, with Altos rows, panels and columns in the
    timeline, intervals, pairing, velocity check, range overlay and
    snapshots.
- **Tested here:**
  - the launcher, with the Altos replayed on this machine;
  - check and alignment on Altos-only and synthetic radar + Altos sessions;
  - the live worker and calibration capture from a recording;
  - the report's Altos figures on stand-in data;
  - `tests/test_altos.py`.
- **Not yet run:** a real 4-sensor session on the rig, and the full report
  on one (the reference drive wasn't attached).

**Open:**
- The small-room tests (README): no reflector or LiDAR available,
  so it's a foil corner reflector or a plate, and the tape measure.
- Check axis signs.
- On the rig:
  - create the `altos-usb-eth` network profile (NEW_BOARD_SETUP section 6;
    needs the owner's go-ahead);
  - record a 4-sensor session with someone walking;
  - run the report and set `altos_ms` from its Altos velocity check.
- Calibrate the Altos's mounting relative to the vehicle frame; its points
  are in its own frame. That is the team's calibration.

- PTP, which needs a decision from the owner.
- The team's decision on replacement. It depends on whether they need raw
  radar data.

See `altos_standalone_capture/README.md`.

## Long sessions: storage, session notes (2026-11-23)

The team's first sessions with all four sensors (11 sessions, 18–23 Nov,
logs only) raised four points: the eMMC filling on long sessions, the check
holding up the next session, checkpoints during a session, camera drops.

- **Found in their logs:**
  - the one long road session (18 min) stopped when the eMMC filled: the
    camera writes 16.6 GB/h on the road (2–4 indoors; the budget was 5), and
    only 8.3 GB were free;
  - the end-of-session wait was mostly the ZED decoding every frame (~0.6×
    the recording time);
  - camera drops were 0.1–0.5% in every session, all under the 1% limit;
  - two sessions were damaged in the copy to a Windows drive;
  - one session (`check`) had its radar capture start mid-frame
    ([03](03_radar_firmware_and_dca.md)).
- **Done (step 2 of the plan):** the camera budget is 20 GB/h; an eMMC
  watchdog stops all sensors cleanly before it fills (warnings at 10 and
  3 min); local time on every log line; session notes
  (`session_notes.txt`, `check_session.py --quick`) written by the launcher
  in seconds; the SVO2 decode moved to `verify_svo.py` (offline, required by
  the full check); the launcher stops a session whose radar capture starts
  mid-frame. Tested on synthetic sessions and with a stand-in ZED (the
  watchdog); `test_session_notes.py`.
- **Not yet verified on the rig:** a real ZED recording with
  `verify_after_stop: false` (how long the frame-count check takes on a
  large SVO2) and `verify_svo.py` on a real SVO2; the misalignment stop on a
  real start.
- **Next:**
  - salvage `run_20261123_1619`: `verify_svo.py` on it (radar, LiDAR and
    Altos are complete; the camera's SVO2 was closed but not verified);
  - **step 3: camera segments + a background mover to the SD card +
    checkpoints** (`checkpoints.jsonl`, flushed files at each), for sessions
    longer than the eMMC holds; measure the frames lost at each switch and
    the drops while the mover runs;
  - an NVMe drive (the owner's decision: fstab);
  - a file list with sizes at stop, so copies can be checked;
  - re-cutting a mid-frame radar session offline (no bytes are lost).

## Open items and ideas

- **Now (2026-11-07): waiting for the team's pilot from the wheeler.** The
  reply (`docs/team_messages/2026-11-07_reply.md`) asks them to: mark the
  LiDAR housing and collar; record a calibration set on the wheeler as
  mounted (empty scene first, as a session named `background`, then the
  reflector positions in the same folder); record a 10–15 min pilot
  (calibration set, a minute of walking in front of the stopped wheeler,
  a ride at mixed speeds, a few stops); share it; then collect, with our
  feedback after each batch. **When it arrives:**
  1. `check_session.py`, `align_session.py` (the extrinsic: is there an
     entry valid for the new mounting? the 2026-11-03 one is for the rig on
     its stand), `make_report.py`, `run_perception.py`; board health,
     storage and gaps in the report.
  2. Calibration set: `replay_session.py --calibration` per position and
     `calib_reflector.py --background <bg> --positions ...` → LiDAR yaw from
     the camera, radar − camera/LiDAR. A new mounting = a new entry in
     `calibration/lidar_extrinsic.json` (later `valid_from`).
  3. Timing on the walking minute: the report's velocity check; radar range
     vs the LiDAR target (lag within ±0.03 s on the October sessions).
  4. The ride: how often targets exceed ±4.7 m/s radial (they wrap; no
     de-aliasing; a firmware decision for the owner); how the processing
     behaves on a moving platform (MTI and the stationary detector assume
     a still rig; LiDAR moving/static flag needs ego-motion; vibration);
     ground and size classes on roads; radar false alarms.
  5. Answer in `NOTES_FOR_TEAM.md` as the current state (09: no history or
     corrections there); the owner writes the message.
- **Still open (ours):** radar elevation (the elevated TX positions); the
  inter-chip phase of the other firmware variants (unmeasured: they combine
  the chips incoherently only); velocity de-aliasing; a walker merging
  sideways with tall office furniture in the LiDAR; the team's mounting-photo
  question was not answered in the reply (owner's choice).

- **Team's corner-reflector sessions of 2026-11-03 (analysed 2026-11-05).**
  They are on the team's drive at `/media/ibrahim/94DD-7E88/`:
  `calibration_dataset/` (16 sessions) and `run_20261103_1600-Lidar-AR1`.
  All are aligned; AR1 also has its report and perception. Findings are in
  `NOTES_FOR_TEAM.md` before its 2026-11-07 rewrite (git history, "Your
  corner-reflector sessions"):
  - The scene names are given from the reflector's side: `*_left` is the
    vehicle's right.
  - On this mounting the LiDAR is yawed **+37.2°** (to the left) from the
    camera (camera-referenced, 2026-11-07). The first fit, 39.8° to the 13
    nominal positions, absorbed the rig's ~2.5° turn against the floor
    marks. The team's manual fit gives 38.6°.
  - Radar range is accurate (+0.1 m). Radar azimuth is within ±3° on
    boresight but 5–13° off at 0.5–1.5 m to the side. Cause: our
    virtual-array table, not the antenna calibration; fixed with the
    measured layout (2026-11-07, see below).
  - AR1 is not usable for calibration: the radar barely sees the reflector
    there.
  - Replay changes made for this (2026-11-05):
    - any view range;
    - a range filter;
    - stationary radar reflectors in the objects table, with x / y and
      absolute dB;
    - raw threshold up to 80 dB;
    - `--radar-all`, `--range-filter`.
  - The report no longer calls the radar azimuth "mirrored".
  - **LiDAR extrinsic applied (2026-11-06, owner's decision on this
    data).**
    - Our fit is in `calibration/lidar_extrinsic.json`, per mounting
      (`valid_from`); `perception/core/extrinsic.py` reads it.
    - The points are turned into the vehicle (radar) frame right after
      decoding, in align (boresight timing), report, replay, live view,
      `run_perception` and Stage 3.
    - It is recorded per session in `sync/alignment.json`. Sessions before
      2026-11-03 use identity and are bit-identical.
    - The Nov-3 sessions were re-aligned, and AR1's report and perception
      rebuilt.
    - Provisional until the team's extrinsic.
  - **Live view "Take background" (2026-11-06).**
    - The reflector is picked as the largest change from an empty-scene
      reference, held in the workers' memory only: nothing is saved, and a
      new live view starts without it.
    - Verified on `calibration_dataset`: LiDAR within 0.1 m at 3–5 m;
      radar right on boresight.
    - `tests/test_live_view.py` takes and clears a background and checks
      that the session's files are unchanged.

- **Team's follow-up questions of 2026-11-06 (answered 2026-11-07,**
  the reply sent: `docs/team_messages/2026-11-07_reply.md`; current state in
  `NOTES_FOR_TEAM.md`):
  - **Yaw revised to 37.2°, referenced to the camera.** All 17 sessions on
    the team's drive were re-aligned; AR1's report and perception were
    rebuilt.
  - **Radar off-axis azimuth: our array table was wrong, not the antenna
    calibration (found after the owner's challenge; fixed 2026-11-07).**
    - The `.cfg` calibration barely matters (0.85 vs 0.82 with none) and the
      error depended on the target's angle (left vs right 0.57–0.60, same
      side 0.99): no per-channel calibration can fix that.
    - **Measured layout** (`perception/estimate_array_layout.py`,
      `calibration/radar_array.json`): per chip, chirps 0, 1, 2, 5 at 4, 2,
      0, 7 λ/2 (the table: 0, 4, 7, 11 and chirp 5 elevated); chirps 3, 4
      elevated; RX as in the table; the slave's phase +0.356 rad/m, −11°.
    - **Processing** (`perception/radar.py` `layout="measured"`, default):
      per-chip beams added incoherently (sidelobes ~20 dB down), the angle
      refined coherently across chips; no `.cfg` calibration, no in-situ
      phase; elevation 0 for now. `layout="team"` keeps the old one
      (`--radar-layout team` in the replay).
    - **Result:** 11-03 corner reflector − camera −2.4° ± 0.37° at all 9
      positions at 2–5 m (before: outliers to 16°); walker spread vs LiDAR
      1.8° instead of 2.7°, > 5° off 12% instead of 36% (freewalk_ref_02).
      Details in 03, "Off-axis azimuth".
    - Tests: `tests/test_array_layout.py` (synthetic targets, −40..+40°).
      Baseline re-made 2026-11-07: `golden_ours.npz` on the three golden
      sessions and `perception/baseline/scores.json`. Radar detected
      +0.08…+0.32 on every reference session except `run_20261014_1325`,
      where the LiDAR was turned ~20° against the radar and the camera
      (set-up session; checked on the camera images).
  - **Done before the reply (2026-11-07):**
    - Processing checks. Radar chain (03, "Processing checks"): axes,
      range, velocity, TDM compensation, I/Q and timing all correct; one bug
      fixed (`timing_consistency.py` wrote the LiDAR target cache without
      the extrinsic). Detector settings (03, "Detector settings"): CFAR
      threshold 14 → 16 dB (−25…40% unsupported radar tracks, walker
      −0.013 at most), the rest kept. LiDAR module (05): positions, heights
      and ground right; fixed: a roof or ceiling over a person counted as
      part of them (vertical-gap cut; walker classed pedestrian 0.80 →
      0.87–0.99 under the carport). Baseline re-made after each change.
    - Replay for calibration sessions: `--background` (radar subtracted
      coherently per channel, LiDAR new returns only) and `--calibration`
      (sibling `background` found by itself; the reflector read out in both
      sensors, radar − LiDAR −1.9° at 5 and 8 m on 2026-11-03); the camera
      panel no longer shrinks when the LiDAR's info line grows.
    - `NOTES_FOR_TEAM.md` rewritten as the current state; the owner's reply
      is in `docs/team_messages/2026-11-07_reply.md`.
  - **Their manual calibration file was never used** by any code.
  - **The blinking wall** is the RS-Fairy skipping 0–16.5° once a second.
    The replay fills it for display only.
  - **Live view calibration capture:**
    - *Save background* / *Save position* write raw radar, LiDAR (own
      frame) and camera to `<sessions-root>/calibration/<set>/<name>/`,
      with the reflector found as a coherent / voxel change, and a
      *reflector only* view.
    - `calib_reflector.py` processes captures and recorded sessions, and
      gives the camera-referenced yaw.
    - Tested by playing the team's sessions through live view; **not yet
      with the live sensors.**
  - **Replay:**
    - the `S1..S3` rows use a 2 s mean of the static map, interpolated
      between cells (they blinked: one cell of one frame);
    - a radar *map* tick for points only (FusionApp-like);
    - stationary points in 3D.
  - **Still to do:**
    - ask RoboSense about the skipped sector;
    - Capon for the static map (FusionApp's sharper points), possibly;
    - run the reference-session regression (the Elements drive was
      absent again). The `_front` refactor (`_static_stats`) must stay
      bit-identical: `test_speed_paths` covers it on a session.

- **Live view (2026-10-31): working on the rig (team's report,
  2026-11-05).** Over the Jetson's hotspot, the team set up the page, ran
  live view, and recorded.
  - Not reported in numbers yet: the radar pkt/s during live view, and a
    `check_session` PASS for a recording made straight after a live view.
  - The 2026-11-02 "stuck" report was never explained by a log. It came
    before the hotspot, while the page was reached over the university
    network, and it has not come back.
  - Built and tested on `ibrahim-desktop`, which has no sensors, by playing
    `freewalk_ref_02` (`--live-from-session`).
  - **First rig report (2026-11-02):** with camera + LiDAR only, the page
    "got stuck".
    - The screenshot showed the server answering POSTs ("live view is
      already on"), while `/api/status` and `/api/live/state` never
      returned usable JSON.
    - Not reproduced on `ibrahim-desktop`: camera + LiDAR works on recorded
      data, and the live LiDAR socket path works with a localhost UDP
      replay.
    - Cause still unknown. The page now shows the real error (HTTP status,
      the server's message), the server answers every failure in JSON with
      the traceback in its log, and its JSON never contains NaN. Waiting for
      the team's retest and `journalctl -u capture-web`.
  - **Page blocked on the university network (2026-11-03).** The team
    could not open the page by name or IP. Fix: the Jetson's own Wi-Fi
    hotspot (`setup_jetson_capture.sh --hotspot`, page at
    `http://192.168.50.1:8080`; [02](02_hardware_and_network.md)).
    - **Tested on `ibrahim-desktop` (2026-11-04)** by following the docs:
      the hotspot came up and a phone joined, but the page was blank. The
      cause was the web service: its file fixed user `jetson` and the old
      checkout path, so it failed with 217/USER. This is probably also the
      team's failure (on the rig, 200/CHDIR if the checkout is
      `RoadSafety-CRCL`).
    - Fixed: the script writes the unit for the user, checkout and conda
      Python of the board it runs on, waits up to 20 s for the page, and
      prints the cause if it fails. The docs have a "page stays blank"
      section.
    - Verified without sudo: the generated unit's command serves the page
      in systemd's clean environment.
    - **Resolved (2026-11-05).** The full documented run passed on
      `ibrahim-desktop`: a phone joined `JetsonHotSpot` and the page loaded.
      The rig had run a script older than 7450f46, which ignored
      `--hotspot`. After pulling f164682, the team configured the hotspot
      on the rig.
    - `ibrahim-desktop` has the end user's hand-made profile with the same
      settings.
  - Still to check on the rig: live DCA1000 control and STOP_RECORD, the
    LiDAR sockets, the live ZED, the page over the rig's Wi-Fi, the radar
    pkt/s during live view, and a recording straight after a live view
    (must PASS).
  - The server's lock does not cover `capture_session.py` started from a
    terminal during live view. A lock file checked by the launcher's
    preflight would; that is not done, because it changes the recording
    path.
  - Possible later: LiDAR over the camera image, once the team's camera
    calibration exists.

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
  
  Our in-situ correction (`calibration/radar_rx_phase.json`) was a patch.
  2026-11-07: the split beam came from our array table; the measured layout
  replaces both (03, "Off-axis azimuth").
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
    outdoors with the team layout: beyond that, sidelobe and multipath
    images dominated. To be re-scored with the measured layout.
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
- **Speed and the Orin's resources (from 2026-10-26, owner's top priority).**
  Done (all outputs identical; `perception/README.md` "Speed",
  `bench_gpu.py`):
  - LiDAR render 29.7 → 6.8 ms, tracker 6.3 → 3.0 ms, radar 18.5 → 16.9 ms;
    the replay's LiDAR loop 15 → 24 Hz (sensor: 10);
  - `run_perception.py` on 10 min of data: ~324 s → 169 s cached, ~700 s →
    353 s from the USB disk (radar frames read ahead; LiDAR decoded by
    several drivers on time chunks, alongside the radar).
  
  - On `ibrahim-desktop` (owner's approval, 2026-10-26): `whoopsie` disabled
    (it held one core at ~94% when idle) and `jetson_clocks` run (not
    persistent across reboots; see [02](02_hardware_and_network.md)). The radar
    detector went 16.9 → 13.3 ms at ~50% GPU load: launch-bound.

  - The radar as a CUDA graph with per-cell detection rows: 13.3 → 8.3 ms per
    frame, identical outputs; `run_perception` radar pass 92 → 59 s cached.

  Next candidates: the radar front end's GPU work (the 64 × 256 × 256
  azimuth cube; not bit-exact rewrites, so they need the baseline/score
  check); batched frames for `run_perception`; LiDAR detector (~17 ms). From
  the USB hard disk, the radar is disk-bound: sessions on a fast disk gain
  most.
