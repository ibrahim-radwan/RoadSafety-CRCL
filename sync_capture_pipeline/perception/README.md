# Perception: our GPU detection and tracking (radar, LiDAR)

The project's detection pipeline. It runs live in the replay and offline over
a whole session. Frame: x forward, y **left**, z up. Times: `t_ref`.

The early radar chain from the team's Windows work
(`cascaded_radar_capture_pipeline/processing`) is still separate and
unchanged. At the team's request (2026-10-25) it is being merged into this
library, stage by stage, each port checked against the old output and the
reference-session scores. See `docs/knowledge/09_working_practices.md`.

| File | What |
|---|---|
| `core/` | **The shared core** (merge step 2; numpy only, no torch, so the report and capture-side tools can import it). See the table below. |
| `radar_dsp.py` | **Radar signal-processing stages in PyTorch** (merge step 3), ported from the old chain and tested against it:
- range and Doppler FFT, MTI, DC notch;
- non-coherent power;
- Doppler-stripe suppression;
- CFAR (CA, CASO, CAGO, OS);
- range-Doppler sidelobe pruning, the cross-channel consistency check;
- the range-azimuth focus mask, the focused range-azimuth map;
- Capon, MUSIC, Bartlett;
- the elevation map. |
| `radar_chain.py` | `RadarChain`: the old chain's front end (`process_frame` stages 1–7 and 9) composed from those stages. It reproduces the old chain's golden frames on both paths (`cfar="caso"` or `"os"`), at ~21 ms/frame against ~100 ms (old GPU path) and ~1 s (old CPU path). |
| `radar.py` | `RadarDetector`: one raw frame → range-Doppler and range-azimuth maps plus a detection list. It runs on the GPU (PyTorch) in ~16 ms, built on the `radar_dsp.py` stages. Also `flag_multipath`, `cluster_detections`, `RADAR_TRACKER`. |
| `lidar.py` | `LidarDetector`: one sweep → objects. Local ground, overhead structures removed, oriented boxes, size class. ~20 ms. |
| `tracking.py` | `Tracker`: a constant-velocity Kalman filter with Hungarian association and M-of-N confirmation. For radar, Doppler is an EKF measurement plus a consistency check. |
| `estimate_rx_phase.py` | The in-situ radar phase correction against the LiDAR, written to `../calibration/radar_rx_phase.json`. |
| `baseline.py` | The frozen reference for the merge: golden per-frame outputs (old chain on both paths, ours) and the scores of every reference session (`baseline/scores.json`). |
| `../run_perception.py` | A whole session → `sync/perception_{radar,lidar}.npz` and `sync/perception.json` (scores). |

### `core/`: one place for configuration, data access and conventions

| Module | What |
|---|---|
| `radar_config.py` | The radar `.cfg` parser, array geometry, axes and the processing defaults (`RadarConfig`). Moved from the old chain. |
| `radar_io.py` | Raw radar frames: the old `read_frame` (moved here), plus `RadarTimeline` (t_ref and file per frame) and `FramePrefetcher` (reads ahead on slow disks). |
| `calibration.py` | The `.cfg` antenna calibration combined with the in-situ correction: `load_radar_config`, `radar_correction`, `rx_phase_for`. |
| `lidar_io.py` | `lidar.pcap`: packet index, LiDAR-to-monotonic clock, sweep timing, `decode_sweeps` (a whole session, in order). |
| `session.py` | `Session`: metadata, firmware variant and `.cfg`, offsets, radar/camera timelines, pairs, and the LiDAR walker track (`target_at`). |
| `conventions.py` | The frame and sign conventions: world azimuth = − processing azimuth, v_r > 0 receding, the old chain's xyz to the pipeline frame. |

The old chain's `processing/radar_config.py` and `processing/radar_reader.py`
are shims that re-export these modules, so `process_frame` runs unchanged
until it is ported. `tests/test_baseline.py` checks that it still reproduces
its golden frames on both paths.

```bash
conda activate rel_1_py_3_8
python run_perception.py --session /mnt/sdcard/sessions/<name>          # ~4 min for 10 min of data
python perception/estimate_rx_phase.py --session <s1> [<s2> ...] [--write]
python -m pytest -q tests/test_perception.py
```

## Baseline and regression check (merge step 1, 2026-10-25)

Every change of an algorithm is measured before it replaces the old one:

```bash
python run_perception.py --session <s> ...            # current outputs
python perception/baseline.py compare --session <s> ...   # scores vs perception/baseline/scores.json
python -m pytest -q tests/test_baseline.py            # our detector still reproduces the golden frames
```

- **`perception/baseline/scores.json`** (in git) holds the scores of every
  reference session for three outputs, against the LiDAR walker:
  - our radar and LiDAR;
  - the old chain's filtered clouds, on both its paths: `team_cpu` (OS-CFAR)
    and `team_gpu` (CASO).

  The radar scores of `stage3_02`/`03` are in-sample, because the phase
  correction was measured on them.
- **`<session>/sync/baseline/golden_*.npz`** (on the data drive, not in git)
  holds the per-frame intermediate outputs on 20 fixed frames:
  - for the old chain, on both paths: RD, RA, RE, CFAR masks and thresholds,
    and xyz;
  - for our detector: RD, RA and detections.

  Ported stages are tested against them for equivalence. If a deliberate
  change makes them obsolete, re-create them with
  `baseline.py golden --session <s>`.

## Radar, per frame

1. Calibration (the `.cfg`, times the in-situ correction), then the range FFT.
2. MTI, the Doppler FFT, and **TDM-MIMO Doppler compensation**: the 6 TX
   slots of a loop see a moving target at different phases.
3. The uniform 38-position virtual array, then the azimuth FFT of every cell.
   The detection map is each cell's angle-maximum power.
4. CA-CFAR in dB (circular in Doppler), zero-Doppler band excluded, 3×3
   local maxima.
5. Per detection:
   - one azimuth peak with sub-bin interpolation;
   - range interpolation;
   - elevation from the TX5 pair phase (**unverified**).
6. **Multipath ghosts flagged:**
   - images at n× range with n× radial velocity (n = 2, 3; radar → target →
     rig → target → radar);
   - images behind a stronger target at its radial velocity (via a wall).
7. Objects: single-linkage clusters of the non-ghost detections. Doppler is
   weighted 1 m per 4 m/s, so a walker's limbs stay one object.
8. Tracks: Doppler-aware. A track whose Doppler contradicts its motion is not
   reported.

Conventions, verified against the LiDAR walker:

- World azimuth = − processing azimuth. The processing is right-positive; the
  pipeline frame is left-positive.
- v_r > 0 means receding.

## LiDAR, per sweep

1. Valid returns only: range > 0, beyond `near_m` (default 0.5 m).
2. **Local ground:**
   - the lowest return per 1 m cell;
   - a cell more than 0.35 m above its 3×3 minimum is "lifted" (for example
     under a car body) and takes that minimum instead;
   - empty cells are filled from wider windows.
3. Obstacles: points 0.2–3 m above the ground, in 0.2 m columns that reach
   down to within 1 m of the ground. Overhead roofs, canopies and branches
   drop out; posts, walls, people and vehicles stay.
4. An occupancy grid, closed by one cell, then 8-connected components give
   the objects.
5. Per object: a minimum-area box (from the grid cells), height above ground,
   point count, and a **size class**:

   | Class | Length | Width | Height |
   |---|---|---|---|
   | pedestrian | ≤ 1.3 m | ≤ 1.0 m | 0.9–2.3 m |
   | two-wheeler | 1.3–2.5 m | ≤ 1.2 m | 0.9–2.3 m |
   | vehicle | 2.5–7 m | 1.3–3 m | — |
   | structure | > 8 m or taller than 4.5 m | — | — |
   | other | anything else | — | — |

   These are heuristics: a wall segment with a bike against it can be boxed
   as a "vehicle".
6. Tracks: every class except structure.

## Measured (2026-10-24)

All scores are against the LiDAR walker track (`sync/timing_lidar_track.npz`),
a single-target reference.

On `freewalk_ref_02` (10 min), which was not used to measure the radar phase
correction:

| | Ours | Team filtered (`process_frame`, CPU path) |
|---|---|---|
| Radar: walker detected (within 0.6 m, 10°) | **71.8%** | 49.5% |
| Radar: azimuth spread (median \|err − bias\|) | **2.41°** | 4.93° |
| Radar: within 3° of the bias | **54%** | 31% |
| Radar: walker has a reported track | 73.5% (within 2 m) | — |
| Radar: tracks not supported by a LiDAR object | 0.66 per frame, 58% supported (at the start of tuning: 1.75 here, 4.55 on stage3_02) | — |
| LiDAR: walker detected / track | **95.0% / 94.9%** | — |
| LiDAR: walker classed pedestrian | 83.5% (18.9 ms per sweep) | — |

- **The radar azimuth bias** (−5.0° on this session) is the radar–LiDAR yaw
  of that mounting (extrinsic).
- **Radar–LiDAR support** is our false-alarm proxy, since there is no ground
  truth.
- On the plate sessions (`stage3_02`/`03`), the phase correction cuts the
  held-out azimuth spread from 5.2° to 1.3°.

## Limitations

- The radar beam still has sidelobes only 5–10 dB down. That is calibration:
  the team's recalibration with a reflector over a wide angle span supersedes
  `radar_rx_phase.json`.
- Radar elevation is not verified (z = 0 in the displays by default).
- Moving-target radar loses a person who stops or walks across the line of
  sight (zero Doppler). Radar tracks break at such moments, about 9 times a
  minute on freewalk_ref_02.
- The LiDAR size classes are heuristics; there is no learned classifier yet.
- There are no camera detections yet: no model is installed, and there is no
  camera extrinsic calibration to overlay them.
