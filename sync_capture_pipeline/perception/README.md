# Perception: the processing, detection and tracking library (radar, LiDAR)

The project's one processing pipeline, in PyTorch on the GPU. It runs live in
the replay and offline over a whole session. Frame: x forward, y **left**,
z up. Times: `t_ref`.

At the team's request (2026-10-25), the early radar chain from their Windows
work was merged into this library and deleted:

- each general stage was ported and checked against the old output;
- the single-person stages were retired;
- the old chain's outputs stay as frozen references (see *Baseline* below).

See `docs/knowledge/01_status_and_roadmap.md`.

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
| `radar_chain.py` | `RadarChain`: the old chain's front end (`process_frame` stages 1–7 and 9) composed from those stages. It reproduces the old chain's frozen golden frames on both of its paths (`cfar="caso"` or `"os"`), at ~21 ms/frame; the old chain took ~100 ms (GPU path) or ~1 s (CPU path). |
| `radar.py` | `RadarDetector`: one raw frame → range-Doppler and range-azimuth maps plus a detection list. It runs on the GPU (PyTorch) in ~16 ms, built on the `radar_dsp.py` stages. Also `flag_multipath`, `cluster_detections`, `RADAR_TRACKER`. |
| `lidar.py` | `LidarDetector`: one sweep → objects. Local ground, overhead structures removed, density-based clustering, oriented boxes, size class. ~23 ms. |
| `tracking.py` | `Tracker`: a constant-velocity Kalman filter with Hungarian association and M-of-N confirmation; a **moving / static** state per track (displacement over 1.5 s, hysteresis 0.25–0.5 m/s; sensor frame: needs ego-motion compensation on a moving platform); prediction, gating and duplicate merging vectorised over all tracks (~2.5 ms with 45 LiDAR tracks). For radar, Doppler is an EKF measurement plus a consistency check. |
| `track_features.py` | `TrackFeatures`: per radar track, motion-compensated **accumulation** of its detections (last 1 s) and a rolling **micro-Doppler** spectrogram at its cells → torso velocity, Doppler spread (limbs), cadence (experimental). Replaces the old single-subject `accumulate.py`/`microdoppler.py`. |
| `estimate_rx_phase.py` | The in-situ radar phase correction against the LiDAR, written to `../calibration/radar_rx_phase.json`. |
| `baseline.py` | The reference: golden per-frame outputs (the old chain's, frozen; ours) and the scores of every reference session (`baseline/scores.json`); `compare` is the regression check. |
| `../run_perception.py` | A whole session → `sync/perception_{radar,lidar}.npz` and `sync/perception.json` (scores). |

### `core/`: one place for configuration, data access and conventions

| Module | What |
|---|---|
| `radar_config.py` | The radar `.cfg` parser, array geometry, axes and the processing defaults (`RadarConfig`). Moved from the old chain. |
| `radar_io.py` | Raw radar frames: the old `read_frame` (moved here), plus `RadarTimeline` (t_ref and file per frame), `FramePrefetcher` (the replay's random-access read-ahead on slow disks) and `iter_frames` (in-order read-ahead for batch runs). |
| `calibration.py` | The `.cfg` antenna calibration combined with the in-situ correction: `load_radar_config`, `radar_correction`, `rx_phase_for`. |
| `lidar_io.py` | `lidar.pcap`: packet index, LiDAR-to-monotonic clock, sweep timing, `decode_sweeps` (a whole session, in order, one driver) and `decode_parallel` (the same, several drivers on time chunks in worker processes; identical sweeps). |
| `session.py` | `Session`: metadata, firmware variant and `.cfg`, offsets, radar/camera timelines, pairs, and the LiDAR walker track (`target_at`). |
| `conventions.py` | The frame and sign conventions: world azimuth = − processing azimuth, v_r > 0 receding, the old chain's xyz to the pipeline frame. |

`radar_config.py` and `radar_io.py` came from the old chain (git history
follows them).

```bash
conda activate rel_1_py_3_8
python run_perception.py --session /mnt/sdcard/sessions/<name>          # ~3 min for 10 min of data (cached; ~6 min from a USB hard disk)
python bench_gpu.py --session <s> [--json out.json]                    # per-part speed and GPU load (see Speed below)
python perception/estimate_rx_phase.py --session <s1> [<s2> ...] [--write]
python -m pytest -q tests/test_perception.py
```

## Baseline and regression check (merge step 1, 2026-10-25)

Every change of an algorithm is measured before it replaces the old one:

```bash
python run_perception.py --session <s> ...            # current outputs
python perception/baseline.py compare --session <s> ...   # scores vs perception/baseline/scores.json
python -m pytest -q tests/test_baseline.py tests/test_radar_dsp.py   # golden frames: ours and the old chain's
```

- **`perception/baseline/scores.json`** (in git) holds the scores of every
  reference session for three outputs, against the LiDAR walker:
  - our radar and LiDAR;
  - the old chain's filtered clouds, on both its paths: `team_cpu` (OS-CFAR)
    and `team_gpu` (CASO), from the clouds made before it was retired.

  The radar scores of `stage3_02`/`03` are in-sample, because the phase
  correction was measured on them.
- **`<session>/sync/baseline/golden_*.npz`** (on the data drive, not in git)
  holds per-frame intermediate outputs on 20 fixed frames:
  - **`golden_team_{cpu,gpu}.npz`:** the old chain on both its paths (RD,
    RA, RE, CFAR masks and thresholds, xyz). These are frozen: they were made
    before the old chain was deleted. `RadarChain` must keep reproducing
    them.
  - **`golden_stages.npz`** (`freewalk_ref_02`): the old CFAR variants and
    beamformers, with their inputs.
  - **`golden_ours.npz`:** our detector (RD, RA and detections). After a
    deliberate change, re-create it with `baseline.py golden --session <s>`.

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
   Steps 1–5 run as a **CUDA graph** (`RadarDetector(cuda_graph=True)`, the
   default on CUDA): one graph per (moving_only, detect, static), captured on
   first use. Step 5 is computed for every cell inside the graph (for one
   peak, the FOV maximum is the peak, so it costs two gathers); after the
   graph only the thresholds, `nonzero` and a gather of the hit cells run.
   Identical to the eager path (tests/test_speed_paths.py).
7. Objects: single-linkage clusters of the non-ghost detections. Doppler is
   weighted 1 m per 4 m/s, so a walker's limbs stay one object.
8. Tracks: Doppler-aware. A track whose Doppler contradicts its motion is not
   reported.
9. **Stationary reflectors** (`process(..., static=True)`, experimental,
   2026-10-26). The moving path removes everything static, so parked cars,
   poles, stopped people and the whole scene at a red light are missed.
   - The static path: the zero-Doppler band without clutter removal; an
     azimuth FFT; a 2D CFAR over range × azimuth; local peaks; a 10 dB
     sidelobe prune per range row; up to 60 m; multipath images flagged.
   - LiDAR support is 82–100% within 6–10 m. Overall it is 80% indoors and
     55% outdoors (`freewalk_ref_02`). Beyond ~10 m many detections are
     sidelobe images of strong reflectors, since this array's sidelobes are
     only 5–10 dB down, until the antenna recalibration.
   - A higher threshold makes it worse: real nearby objects have ~17 dB SNR,
     the artifacts more.
10. **Per-track features** (`track_features.py`, 2026-10-26):
   - **Accumulation:** about 21 points per walker over 1 s, against a few per
     frame (`freewalk_ref_02`).
   - **Micro-Doppler** on the walker: the torso trace and the limb flashes
     (to about ±2.5 m/s) are clear.
   - **Cadence is experimental.** On the walker, 4 s windows give about
     2.0 Hz (the step) in some windows and about 1.0 Hz (the stride) in
     others, and often none. The waveform limits it: ~13 ms coherent dwell
     per frame, 0.15 m/s Doppler bins, 8 frames/s.

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
4. **Density-based clustering on a bird's-eye grid** (0.2 m; since
   2026-10-26). Objects grow through *core* cells only: cells with at least
   `round(200 / r²)` points, clamped to 1–8, which is 8 at 5 m, 2 at 10 m
   and 1 beyond 14 m, since density falls with range squared. A sparse cell
   joins a neighbouring object but cannot connect two.
   - The old method (`cluster="grid"`: any occupied cell, closed by one cell,
     8-connected) let single stray points bridge objects. In a cluttered
     office (`run_20261014_1325`/`1326`), a chain of one-point cells at the
     depth edge merged the walker into a 7 m object with the walls and desks.
     Such cells are mixed pixels, which average a near and a far surface.
   - Closing the core cells brings the merging back, so it is not used.
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

## Measured (2026-10-24; LiDAR 2026-10-26)

All scores are against the LiDAR walker track (`sync/timing_lidar_track.npz`),
a single-target reference.

On `freewalk_ref_02` (10 min), which was not used to measure the radar phase
correction:

| | Ours | The old chain (`process_frame`, CPU path) |
|---|---|---|
| Radar: walker detected (within 0.6 m, 10°) | **71.8%** | 49.5% |
| Radar: azimuth spread (median \|err − bias\|) | **2.41°** | 4.93° |
| Radar: within 3° of the bias | **54%** | 31% |
| Radar: walker has a reported track | 73.5% (within 2 m) | — |
| Radar: tracks not supported by a LiDAR object | 0.66 per frame, 58% supported (at the start of tuning: 1.75 here, 4.55 on stage3_02) | — |
| LiDAR: walker detected / track | **99.0% / 99.1%** (grid clustering: 95.0% / 94.9%) | — |
| LiDAR: walker classed pedestrian | 85.5% (~23 ms per sweep) | — |

- **The radar azimuth bias** (−5.0° on this session) is the radar–LiDAR yaw
  of that mounting (extrinsic).
- **Radar–LiDAR support** is our false-alarm proxy, since there is no ground
  truth.
- On the plate sessions (`stage3_02`/`03`), the phase correction cuts the
  held-out azimuth spread from 5.2° to 1.3°.
- **LiDAR density clustering, all sessions**, walker detected / confirmed
  track, grid → density:
  - `run_20261014_1326`: 27% → 94% / 24% → 95%;
  - `run_20261014_1325`: 47% → 84% / 43% → 83%;
  - `freewalk_ref_01`: 92% → 99.8%;
  - `stage3_01..03`: 97% → 99.7–99.9%.

  The walker is split into two objects in 0–8% of sweeps (the tracker's
  duplicate merge absorbs most of that). There are ~37–42 objects per sweep
  instead of ~27, as walls break into pieces. Radar tracks without LiDAR
  support fell as a side effect (`1326`: 1.24 → 0.72 per frame).

## Speed (2026-10-26, `ibrahim-desktop`, MAXN)

`bench_gpu.py` times each part on frames held in RAM, and both sensors
together in two threads as the replay runs them. On `freewalk_ref_02`:

| Part | Before | Now | How |
|---|---|---|---|
| Radar detector (moving + static) | 18.5 ms | 16.9 ms | the static path computes its 3 zero-Doppler bins as a DFT matmul, not a second full Doppler FFT; aperture maps cached |
| LiDAR detector | 17–19 ms | same | |
| LiDAR bird's-eye render (replay) | 29.7 ms | 6.8 ms | numpy on the ~40 k points, not boolean masks over the ~300 k-pixel image; sort the uint8 colour index (radix), not the values |
| LiDAR tracker update | 6.3 ms | 3.0 ms | the moving/static check in plain floats (numpy on 3-element lists was 60% of it) |
| Both sensors in two threads | radar 37 Hz, LiDAR 15 Hz | radar 36 Hz, LiDAR 24 Hz | (sensors: 8 / 10 Hz) |

All of these give identical outputs (tests/test_speed_paths.py, and
`run_perception.py` outputs compared array by array on `stage3_02` and the
full `freewalk_ref_02`).

`run_perception.py` on `freewalk_ref_02` (10 min of data):

| | Before | Now |
|---|---|---|
| Files cached in RAM | ~324 s (radar 103 s, then LiDAR 221 s) | **169 s** (both at once) |
| Cold, from the USB hard disk | ~700 s (radar 480 s, then LiDAR 221 s) | **353 s** (radar 335 s, disk-bound; LiDAR 107 s alongside) |

With `jetson_clocks` (clocks locked at their maximum) the benchmark gives:
radar 13.3 ms, LiDAR 16.9 ms, render 6.1 ms, tracker 2.9 ms; both threads
43–45 Hz radar and 28–29 Hz LiDAR. The radar gains most (16.9 → 13.3 ms) at
~50% GPU load.

**Radar as a CUDA graph (2026-10-26): 13.3 → 8.3 ms per frame** (GPU load
~78%), identical outputs (all 4805 frames of `freewalk_ref_02`). Measured
with graphs on and off:

| | Eager | Graph |
|---|---|---|
| Front end (range FFT .. CFAR noise maps) | 6.3 ms | 5.1 ms replay + 0.8 ms upload |
| Per-detection steps after `nonzero` | 5.0 ms (~100 small ops) | 1.0 ms (rows precomputed per cell in the graph) |
| Static path after `nonzero` | 1.5 ms | 0.9 ms |

- The front end itself is GPU-bound (the 64 × 256 × 256 azimuth cube: FFT,
  power, shifts), so the graph saves ~1 ms there; most of the gain is moving
  the per-detection work into large per-cell kernels.
- Computing the full azimuth spectrum in dB for every cell (the general
  multi-peak path) costs ~6 ms in the graph; with `max_peaks=1` (the
  default) the peak is the FOV maximum, which the detection map already has.
- `run_perception.py`'s radar pass on `freewalk_ref_02`, cached: 92 → 59 s.
- In the benchmark's unpaced two-thread test the radar runs at ~99 Hz and
  takes GPU time from the LiDAR thread (16.5 Hz); the replay paces the radar
  at 8 Hz.
- **Capture and threads:** this torch (2.1) only has the global capture mode,
  in which any other thread's CUDA work breaks a capture. The radar captures
  under `perception/core/gpu.py` `GPU_LOCK`, which the LiDAR detector holds
  while it works. Tested with configurations switching mid-run beside a
  LiDAR thread.

- The radar pass reads frames ahead in a thread (`iter_frames`). From the
  USB hard disk it is disk-bound: ~31 MB/s for 3 MB frame files (seeks),
  however many threads read.
- One rs_driver decodes at most ~3× real time (faster, it drops packets), so
  the LiDAR pass runs several drivers on time chunks in worker processes
  (`decode_parallel`, detection in the workers), then tracks in order.
  Chunks overlap by 1.5 s before (the driver needs a DIFOP packet first) and
  0.3 s after; each drops its possibly partial first and last sweep.
- **The GPU is shared by time-slicing between processes**, so more is not
  faster: with the radar running, 2 LiDAR workers (169 s) beat 4 (242 s),
  and 4 workers alone take 101 s for the LiDAR. Hence the default: 2 with
  the radar, 4 for the LiDAR alone (`--lidar-workers`).
- LiDAR detection on the CPU (torch, 2 threads, 23 ms) differs from the GPU
  in ~1 sweep in 100, so the workers stay on the GPU.

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
