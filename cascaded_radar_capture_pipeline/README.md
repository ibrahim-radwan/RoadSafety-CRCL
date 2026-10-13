# Cascaded Radar Project — AWR2243-2X Body-Shape Perception

> **Context in this repository (RoadSafety-CRCL).** This folder is the research team's radar code,
> copied from their Windows development repository. The README below is theirs and describes their
> Windows workflow. On the Jetson rig:
> - the firmware actually in use and its status: [firmware/FIRMWARE_STATUS.md](firmware/FIRMWARE_STATUS.md);
> - capture and sync: `../sync_capture_pipeline/`;
> - what to know before processing the data: `../sync_capture_pipeline/NOTES_FOR_TEAM.md`;
> - the overview: `../AGENTS.md`.
>
> `processing/` is used unchanged by the Jetson tools (`radar_cloud.py`, the report and the replay).

This project is a Windows-based capture, processing, and live-streaming
workflow for a fixed shortrange cascade radar (AWR2243-2X, 2-chip cascade,
6 TX TDM MIMO, 8 RX) aimed at human body-shape perception.

> **Documentation status: verified against code 2026-06-20.**
> Phases 0–6 of `docs/HUMAN_BODY_SHAPE_PLAN.md` are complete.
> The pipeline now includes temporal accumulation, DBSCAN clustering,
> subject-centric normalisation, body envelope fitting, micro-Doppler
> spectrogram, and gait feature extraction. The live viewer shows five
> panels. See [Pipeline State](#pipeline-state) for the full picture.

---

## Project Layout

### Capture

| File | Purpose |
|---|---|
| `capture/test.py` | Main entry point: capture, stop, geometry print, cfg validation, offline extraction |
| `capture/config.py` | Fixed paths and `cascade_r017m.cfg` parsing |
| `capture/dca.py` | DCA1000 control, raw file monitoring, frame splitting |
| `capture/radar.py` | DSS firmware load and UART stop helper |

### Processing

| File | Purpose |
|---|---|
| `processing/radar_config.py` | `RadarConfig` dataclass — all waveform parameters, CFAR settings, and pipeline flags |
| `processing/radar_processing.py` | Full DSP pipeline: range FFT → Doppler FFT → MTI → RD-CFAR → multi-peak azimuth → elevation → xyz cloud |
| `processing/radar_plots.py` | Offline viewer: RD/RE/RA heatmaps with CFAR markers. Label `az-pts` (was `RA-CFAR`) |
| `processing/streaming.py` | Live pipeline: `FrameQueue`, `StreamingProcessor`, `LiveViewer` (5-panel body-shape and radar-only software-rendered `volume` viewers) |
| `processing/gpu_accel.py` | CuPy GPU acceleration for all 8 DSP stages; persistent GPU buffers; CASO/OS CFAR |
| `processing/accumulate.py` | **Phase 2–6:** `TemporalAccumulator`, `subject_frame()`, `fit_body_ellipse()`, `gait_features()` |
| `processing/microdoppler.py` | **Phase 6:** `MicroDopplerBuffer` — Doppler-time spectrogram and cadence estimation |

### Diagnostics

| File | Purpose |
|---|---|
| `diag_pipeline_audit.py` | Batch 400-frame audit → `frames.csv` + `points.csv` with all phase columns |
| `diag_xyz.py` | Per-frame body-shape geometry: x/y/z spread, az/RD ratio, verdict |
| `benchmark_pipeline.py` | End-to-end per-stage latency benchmark (CPU vs GPU) |

### Entry Points

| File | Purpose |
|---|---|
| `live_capture.py` | Primary live hardware entry point; also drives replay mode |
| `sync_replay.py` | Replay paired camera/radar sessions from `pairs.csv` |

### Config

| File | Purpose |
|---|---|
| `firmware/r017m/cascade_r017m.cfg` | Fixed radar waveform configuration |
| `config/build_shortrange_inf/am273x_cascade_mss.xer5f` | DSP firmware |
| `config/AM273X_Capture_fixed.json` | DCA1000 capture config |
| `config/AM273x_XDS110.ccxml` | XDS110 debugger config |

---

## Pipeline State

Phases 0–6 of `HUMAN_BODY_SHAPE_PLAN.md` are complete. All acceptance
criteria pass. The system is ready for multi-session outdoor data collection.

| Phase | Description | Status | Key metric |
|---|---|---|---|
| 0 | Observability — per-frame CSV audit | ✓ Complete | 400-row `frames.csv`, stage attribution visible |
| 1 | Per-frame cloud correctness | ✓ Complete | grating/FoV/cc filters; prom=6 dB; cc_rej −71% |
| 2 | Temporal accumulation + motion comp | ✓ Complete | M4 median accum_pts = 32 ≥ 30 |
| 3 | DBSCAN single-subject clustering | ✓ Complete | 0 centroid jumps (was 61); noise = 2 pts median |
| 4 | Subject-centric coordinate normalisation | ✓ Complete | M5 dx_std = 0.141 m, range-independent |
| 5 | Body envelope visualisation | ✓ Complete | PCA ellipse tracks subject within ±0.50 m bounds |
| 6 | Micro-Doppler / gait features | ✓ Complete | Spectrogram panel; cadence via autocorrelation |
| 7 | Pose descriptors | Pending | Requires ≥ 20 multi-subject sequences |

### What the live viewer shows (5 panels)

| Panel | Content |
|---|---|
| LEFT (3D) | Full motion-compensated accumulated cloud, coloured by Doppler |
| CENTRE (X-Y) | Top-down body footprint; oval tracks DBSCAN cluster centroid |
| RIGHT (X-Doppler) | Lateral vs Doppler — trunk near zero, limbs extending outward |
| 4th (body-frame) | Subject-centric ±0.5 m; cyan cluster pts; amber PCA ellipse |
| 5th (spectrogram) | Doppler-time waterfall; amber torso line; cadence annotation |

Status bar: `vc=±X.XX m/s  σ=X.XX  limb=XX%  cadence=X.X Hz`

### Active pipeline flags (all True by default)

```python
emit_all_cluster_bins = True   # Phase 1: depth bins emitted
temporal_accum_en     = True   # Phase 2: motion-compensated accumulation
subject_frame_en      = True   # Phase 4: body-frame panel active
body_envelope_en      = True   # Phase 5: PCA ellipse overlay active
gait_features_en      = True   # Phase 6: gait descriptors in status bar
microdoppler_en       = True   # Phase 6: spectrogram panel active
```

---

## New Modules (Phases 2–6)

### `processing/accumulate.py`

Contains the full temporal pipeline from Phase 2 through Phase 6.

**`TemporalAccumulator`** — stateful ring buffer with motion compensation,
DBSCAN clustering, subject-frame normalisation, and gait feature extraction.

```python
acc = TemporalAccumulator(
    window=16,              # frames in ring buffer (4s at 4fps)
    max_jump_m=0.6,         # centroid jump gate (2.4 m/s max)
    alpha=0.3,              # EMA smoothing for centroid
    dbscan_eps=0.80,        # DBSCAN neighbourhood radius (m)
    dbscan_min_samples=3,   # minimum pts per cluster
    max_cluster_radius=0.65,# reject implausibly wide clusters (m)
    subject_frame_en=True,  # populate AccumResult.subject_cloud
    gait_features_en=True,  # populate AccumResult.gait
    dop_limb_thr=0.30,      # torso/limb Doppler boundary (m/s)
)

result = acc.update(xyz_cloud, frame_idx=fi)
# result.cloud          [M,7]   motion-compensated accumulated cloud
# result.cluster        ClusterResult — DBSCAN body cluster
# result.subject_cloud  [K,7]   body-local (dx,dy,dz,...) coords
# result.gait           GaitFeatures — dop_centroid, limb_frac, cadence
```

**`subject_frame(accum_result)`** — translates cluster cloud to body-local
coordinates centred on the cluster centroid. Used by Phase 4/5 viewer.

**`fit_body_ellipse(sf_cloud)`** — PCA ellipse fit to (dx, dy) body-frame
cloud. Returns cx, cy, rx, ry, angle_deg, and outline points. Clamped to
±0.50 m per plan requirement.

**`gait_features(cluster_cloud, dop_limb_thr=0.30)`** — per-frame
micro-Doppler descriptor. Returns `GaitFeatures`:
- `dop_centroid` — power-weighted trunk velocity (m/s)
- `dop_spread` — std of cluster Doppler values
- `dop_max` — fastest-moving scatterer (m/s)
- `n_torso_pts` / `n_limb_pts` — pts below/above `dop_limb_thr`
- `limb_frac` — gait activity index (0 = stance, 1 = all limbs)

### `processing/microdoppler.py`

**`MicroDopplerBuffer`** — Doppler-time spectrogram ring buffer.

```python
mdb = MicroDopplerBuffer(
    time_frames=64,     # columns (16s of history at 4fps)
    v_max=1.5,          # ±1.5 m/s Doppler axis
    n_dop_bins=40,      # rows in spectrogram
    frame_rate_hz=4.0,
)
mdb.update(cluster_dop_mps, limb_frac=lf, dop_centroid=dc)

mdb.spectrogram     # [40, 64] float32 — column-normalised heatmap
mdb.torso_line      # [64] float32 — dop_centroid history
mdb.cadence_hz      # float — gait cadence via autocorrelation
```

Cadence is estimated from the autocorrelation of `limb_frac` history.
Valid range: 0.3–3.0 Hz. Returns `nan` if no clear peak above 0.20.

---

## Diagnostic Scripts

### `diag_pipeline_audit.py` — Phase 0–6 batch audit

Runs `process_frame()` over every `.bin` file and writes two CSVs with
all phase columns:

```powershell
# Full Phase 0–6 audit (recommended for each new recording session)
python diag_pipeline_audit.py `
    --folder  capture\frames `
    --cfg     firmware\r017m\cascade_r017m.cfg `
    --temporal-accum --accum-window 16 --subject-frame --gait-features `
    --out-dir debug_context\session_name

# Phase 0 only (fast, ~50s for 400 frames)
python diag_pipeline_audit.py `
    --folder capture\frames --cfg ... --out-dir debug_context\p0
```

**`frames.csv` columns (26 total):**

| Group | Columns |
|---|---|
| Stage attribution | `rd_cfar`, `after_consistency`, `n_clusters`, `az_peaks_raw`, `fov_rej`, `grating_rej`, `cc_rej`, `ra_det` |
| Cloud | `cloud_pts`, `accum_pts`, `cluster_pts`, `cluster_noise` |
| Cluster geometry | `cluster_radius_x`, `cluster_radius_y`, `cluster_cx`, `cluster_cy` |
| Phase 4 | `subject_spread_x`, `subject_spread_y` |
| Phase 6 gait | `dop_centroid`, `dop_spread`, `dop_max`, `n_torso_pts`, `n_limb_pts`, `limb_frac` |
| Centroid | `centroid_x`, `centroid_y`, `centroid_z`, `dop_zero_pts`, `dop_max`, `proc_ms` |

### `diag_xyz.py` — per-frame geometry diagnostic

```powershell
python diag_xyz.py                          # 15-frame diagnostic set
python diag_xyz.py --all                    # all detected frames
python diag_xyz.py --radar-height 0.33      # apply z offset (metres)
python diag_xyz.py --frames 9 24 301 --verbose 9 24 301
```

---

## Software Requirements

Windows 10 or 11, Python 3.10+.

```powershell
pip install numpy scipy matplotlib pyqtgraph cupy-cuda12x PyOpenGL scikit-learn colorcet
```

| Package | Purpose |
|---|---|
| `numpy` | Core array operations |
| `scipy` | CPU CFAR fallback; autocorrelation |
| `matplotlib` | Offline plots |
| `pyqtgraph` | Live display — all five viewer panels |
| `cupy-cuda12x` | GPU acceleration (CUDA 12; falls back to CPU if absent) |
| `PyOpenGL` | OpenGL for 3D scatter panel |
| `scikit-learn` | DBSCAN clustering (Phase 3) |
| `colorcet` | Perceptually uniform spectrogram colourmap (Phase 6) |

> **PyOpenGL:** install `PyOpenGL` only — do **not** install
> `PyOpenGL_accelerate`. If already present: `pip uninstall PyOpenGL_accelerate`

> **scikit-learn** is required for Phase 3 DBSCAN. Without it the accumulator
> falls back to raw median centroid (Phase 2 behaviour) with a warning.

---

## Hardware Requirements

- TI AWR2243-2X cascaded radar (2-chip, 6 TX / 8 RX, 384 chirps/frame)
- DCA1000 capture board
- XDS110 debugger connection
- Ethernet: PC static IP `192.168.33.30`, DCA1000 IP `192.168.33.180`
- UART connection (default COM20)
- NVIDIA GPU (RTX series recommended)

### Radar Geometry

| Parameter | Value |
|---|---|
| TX / RX | 6 TX TDM MIMO / 8 RX (4 per device × 2 devices) |
| Virtual aperture | 48 channels |
| Range resolution | 6.60 cm |
| Max range | 16.8 m |
| Doppler max | ±2.498 m/s |
| Doppler resolution | 0.078 m/s |
| Az resolution at 3 m | 2.3 cm/bin |
| Az resolution at 8 m | 6.2 cm/bin |
| Frame rate | 4 fps (256 ms period) |
| Frame size | 3,145,728 bytes |

---

## Live Capture — `live_capture.py`

### Typical live commands

```powershell
# Full body-shape pipeline (all phases active, 5-panel viewer)
python live_capture.py --display pointcloud --quiet

# Replay recorded frames at 0.5 fps for detailed inspection
python live_capture.py --replay capture\frames --display pointcloud --replay-fps 0.5 --quiet

# Skip firmware reload (radar already running)
python live_capture.py --skip-load --display pointcloud --quiet

# Record new session while streaming
python live_capture.py --record --out-dir capture\frames --display pointcloud --quiet
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--cfg` | `cascade_r017m.cfg` | Radar configuration file |
| `--display` | `heatmap` | `heatmap` or `pointcloud` (5-panel body-shape) |
| `--no-cfar` | off | Skip CFAR — heatmap display only |
| `--quiet` | off | Suppress per-frame diagnostic logs |
| `--record` | off | Write `frame_*.bin` to disk while streaming |
| `--out-dir` | `capture\frames` | Output directory for `--record` |
| `--frames` | infinite | Stop after N frames |
| `--replay FOLDER` | off | Replay recorded frames (no hardware) |
| `--replay-fps` | 0 (auto) | Replay feed rate; 0 = pace to processing speed |
| `--loop` | off | Loop replay until window closes |
| `--skip-load` | off | Skip firmware load |
| `--uart` | `COM20` | UART port for firmware stop |
| `--warmup` | 0 | Frames to discard before display |

---

## DSP Pipeline (per-frame, stateless)

```
Raw IQ bytes (3,145,728 B)
  │
  ├─ 1.  IQ reorder (i-first)
  ├─ 2.  Antenna calibration (complex correction per virtual channel)
  ├─ 3.  Range FFT [6TX × 64loops × 8RX × 256samples] → [64, 256, 48 VA]
  ├─ 4.  Doppler FFT + MTI clutter removal + zero-Doppler notch
  ├─ 5.  Non-coherent integration (Σ|x|² over 48 VA → dB)
  │      + Doppler-stripe suppression
  ├─ 6.  RD-domain CFAR (CASO/OS, GPU) → rd_det [64×256 bool]
  │      + cfar_margin_db gate + cfar_min_power_db gate + NMS
  │      + RD-sidelobe prune + range gate
  ├─ 6b. Cross-channel consistency fusion (consistency_k)
  ├─ 7.  RA map for heatmap display
  ├─ 8.  RD-confirmed multi-peak azimuth → ra_det
  │      - cluster adjacent RD-confirmed range bins (rd_range_cluster_bins=2)
  │      - prominence-filtered local az peaks (az_peak_min_prom_db=6.0 dB)
  │      - FoV gate (|sin(az)| ≤ max_az_sin=0.866, ±60°)
  │      - grating-lobe gate (body_az_half_width_m=0.30 m, boresight-nearest)
  │      - cross-cluster coherence gate (cross_cluster_range_m=0.35 m)
  │      - CSV audit row appended when audit_csv_path is set
  ├─ 9.  Elevation FFT → RE map
  └─ 11. XYZ point-cloud extraction → xyz_cloud [N × 7]
         x(m), y(m), z(m), pwr_db, dop_mps, rng_m, az_deg
```

**`process_frame()` is strictly per-frame and stateless.** All temporal
reasoning (accumulation, clustering, tracking, gait) lives in
`processing/accumulate.py` and `processing/microdoppler.py`, which are
called by the live viewer and the audit script, not by the processing pipeline.

---

## Temporal Pipeline (stateful, in viewer and audit)

```
xyz_cloud (per frame)
  │
  ├─ TemporalAccumulator.update()
  │    ├─ Phase 2: Y+X EMA motion compensation → 16-frame merged cloud
  │    ├─ Phase 3: DBSCAN(eps=0.80, min_samples=3) → body cluster
  │    ├─ Phase 4: subject_frame() → body-local (dx, dy, dz) cloud
  │    └─ Phase 6: gait_features() → GaitFeatures descriptor
  │
  └─ MicroDopplerBuffer.update()
       └─ Phase 6: Doppler histogram → spectrogram column
                   limb_frac autocorrelation → cadence_hz
```

---

## Accepted Metrics (Phase 3 audit — 400 frames, single outdoor clip)

| Metric | Value | Target |
|---|---|---|
| Detected frames | 320 / 400 (80%) | — |
| avg cloud pts / det frame | 2.22 | — |
| avg accum_pts / frame | 25.1 | — |
| **M4** median cluster_pts (active) | 26 | ≥ 30 |
| **M4** median accum_pts (active) | 32 | ≥ 30 ✓ |
| Centroid jumps > 0.6 m | 0 | < 15 ✓ |
| Median noise pts | 2.0 | ≤ 3 ✓ |
| Cluster radius_x | 0.441 m | ≤ 0.55 m ✓ |
| **M5** dx_std (full window) | 0.141 m | 0.10–0.25 m ✓ |
| **M5** range independence | identical at 4.19–5.17 m | ✓ |
| grating_rej total | 1 275 | visible ✓ |
| cc_rej total | 119 | < 423 baseline ✓ |

---

## Key RadarConfig Fields (body-shape pipeline)

### Phase 1 — per-frame cloud

| Field | Default | Description |
|---|---|---|
| `az_multi_peak_en` | `True` | Enable multi-peak azimuth search |
| `az_peak_min_prom_db` | `6.0` | Minimum az peak prominence (dB) |
| `az_peak_min_gap_bins` | `6` | Minimum bin separation between peaks |
| `az_peak_max_n` | `6` | Maximum peaks per cluster |
| `emit_all_cluster_bins` | `True` | Emit az peaks for all depth bins |
| `body_az_half_width_m` | `0.30` | Grating-lobe gate: body half-width (m) |
| `max_az_sin` | `0.866` | FoV gate: ±60° |
| `cross_cluster_range_m` | `0.35` | Cross-cluster coherence window (m) |

### Phase 2–3 — accumulation and clustering

| Field | Default | Description |
|---|---|---|
| `temporal_accum_en` | `True` | Master switch for accumulator |
| `accum_window` | `16` | Ring buffer size (frames) |
| `accum_max_centroid_jump_m` | `0.6` | Ghost gate (m/frame) |
| `accum_alpha` | `0.3` | EMA smoothing weight |
| `dbscan_eps` | `0.80` | DBSCAN neighbourhood radius (m) |
| `dbscan_min_samples` | `3` | Minimum pts per cluster |
| `max_cluster_radius` | `0.65` | Reject implausibly wide clusters (m) |

### Phase 4–6 — body shape and gait

| Field | Default | Description |
|---|---|---|
| `subject_frame_en` | `True` | Enable body-frame panel |
| `body_envelope_en` | `True` | Enable PCA ellipse overlay |
| `gait_features_en` | `True` | Enable gait descriptors in status bar |
| `dop_limb_thr` | `0.30` | Torso/limb Doppler boundary (m/s) |
| `microdoppler_en` | `True` | Enable spectrogram panel |
| `spectrogram_frames` | `64` | Spectrogram time columns (16s at 4fps) |
| `spectrogram_v_max` | `1.5` | Spectrogram Doppler axis limit (m/s) |

### Phase 0 — audit

| Field | Default | Description |
|---|---|---|
| `audit_csv_path` | `None` | When set, appends one CSV row per frame |

---

## GPU Acceleration

All major DSP stages run on GPU via CuPy:

| Stage | Typical time |
|---|---|
| Range FFT | 4 ms |
| Doppler FFT × 4 | 14 ms |
| Power reduction | 3 ms |
| Azimuth FFT × 2 | 20 ms |
| RA collapse | 8 ms |
| RD CFAR (CASO) | 2 ms |
| Elevation FFT | 4 ms |

**Typical live performance (RTX 3080 Ti, Windows):**

| Mode | Avg latency |
|---|---|
| Full pipeline (CFAR on) | ~270 ms |
| `--no-cfar` | ~200 ms |

The temporal pipeline (accumulator, DBSCAN, spectrogram) adds ~40–60 ms
per frame on CPU. Total live latency with all phases active: ~310–330 ms.

---

## Current Limitations and Open Items

| Item | Status | Details |
|---|---|---|
| Config hygiene | Open | 5 unwired fields: `strict_object_filtering`, `require_rd_confirmation`, `ra_sidelobe_db`, `filter_high_moving`, `moving_objects_only` — plan says delete or wire |
| GPU/CPU parity test | Open | Test script not created; GPU path stable in practice |
| Kalman tracker | Not implemented | DBSCAN+EMA achieves equivalent metrics (0 jumps); Kalman would complete Phase 3 structurally |
| Phase 7 gated | Blocked on data | Requires ≥ 20 multi-subject sequences; single 400-frame clip is insufficient |
| Windows WDDM latency | Known | Trimodal ~185/280/380 ms cycle; driver-level, not fixable in Python |
| Sparse cloud (~2 pts/frame) | Expected | Body shape emerges only after 16-frame accumulation |
| Elevation (z) unreliable | Expected | 16 el VA → ~19 cm/bin at 3 m; z not used for body shape |
| Single-subject assumption | Known | DBSCAN selects one body cluster; multi-subject requires tracker |

---

## Next Steps (Data Collection — Phase 7 Prerequisite)

The pipeline is complete. The bottleneck for Phase 7 is data volume.

### Minimum viable dataset

20–30 sequences across multiple subjects before beginning Phase 7.

| Scenario | Pattern | Priority |
|---|---|---|
| A — Radial walk | Out-and-back 3–9 m, multiple speeds and subjects | 1st |
| D — Multi-leg | 6–8 direction changes per run, 3 speeds mixed | 2nd |
| B — Lateral walk | Perpendicular to boresight, 10 m corridor | 3rd |
| C — Stand then walk | 8 s stationary → walk → return → stationary | 4th |

### Per-session checklist

Before each session:
- Record radar mounting height (cm)
- Mark ground positions with tape/cones (3 m, 5 m, 7 m, 9 m from radar)
- Run 5-frame test capture, confirm viewer shows detections

After each session:
```powershell
python diag_pipeline_audit.py `
    --folder capture\frames `
    --cfg firmware\r017m\cascade_r017m.cfg `
    --temporal-accum --accum-window 16 --subject-frame --gait-features `
    --out-dir debug_context\session_name
```

Review `summary.txt` before the next session to catch issues immediately.

---

## Troubleshooting

### No detections (`cloud:0`) despite target present

1. Check `[PROC] C7 RD-CFAR margin` lines — margin consistently `BELOW by 15+ dB` means target SNR is too low.
2. Try `--cfar-margin 5` for broader detection.
3. Match firmware CFAR: `--cfar caso --cfar-guard 0 0 --cfar-train 16 16`
4. ~2–4 pts/frame is **expected** for a single frame. Body shape only emerges after accumulation.

### DBSCAN cluster_pts = 0 in audit

- Confirm `scikit-learn` is installed: `pip install scikit-learn`
- Without sklearn, accumulator falls back to Phase 2 (no clustering)

### Spectrogram panel blank / no colourmap

- Confirm `colorcet` is installed: `pip install colorcet`
- Without colorcet, `ImageItem` uses a greyscale fallback — heatmap still renders

### Cadence shows `nan` in status bar

- Normal until 16+ frames with valid gait data accumulate
- Check `limb_frac` column in audit CSV — if always 0.0, `dop_limb_thr` may be too high

### 3D point cloud window crashes (OpenGL error)

```powershell
pip uninstall PyOpenGL_accelerate
pip install --upgrade PyOpenGL
```

### DCA1000 configuration fails

- PC NIC static IP: `192.168.33.30`
- DCA1000 IP: `192.168.33.180`
- Close mmWave Studio if holding the DCA board
- Power-cycle the DCA1000

### GPU not detected / CuPy import fails

```powershell
nvcc --version
python -c "import cupy; print(cupy.__version__)"
```

Pipeline falls back to CPU automatically (~3500 ms/frame vs ~270 ms/frame).

---

## File Checksums (current deployed versions)

| File | MD5 |
|---|---|
| `processing/accumulate.py` | `392b1610a37c05d0a3a3a32b573f20dc` |
| `processing/microdoppler.py` | `dce61d6fdece3ae5bc75e4e2c5aa866e` |
| `processing/radar_config.py` | `551a5911d29a697b627e9d9d7d8df3ac` |
| `processing/radar_processing.py` | `e74723d56ad4f6d6d94493190c2918de` |
| `processing/streaming.py` | `8dde27434c6b0305ec086b3b761d62dd` |
| `processing/radar_plots.py` | `dc8def38048f07c5cfb102202ab92f2b` |
| `diag_pipeline_audit.py` | `be7f9162613094067a6889fa35d87438` |
| `diag_xyz.py` | `982eaf9e9069c6489d59b6fc879fde83` |

---

*For the detailed step-by-step algorithm roadmap see
[`docs/HUMAN_BODY_SHAPE_PLAN.md`](docs/HUMAN_BODY_SHAPE_PLAN.md).*

# AWR2243-2X Live Viewer — Five-Panel Reference

**Audience:** radar engineers and computer vision researchers working
with the AWR2243-2X cascaded body-shape perception pipeline.

This document describes each of the five live viewer panels in depth:
what signal processing stage each one represents, what is rendered and
how, what the correct appearance is for each scenario, what failure modes
look like, and how the panels relate to each other diagnostically.

---

## System context

The viewer sits at the end of a two-stage pipeline:

```
Raw IQ (3,145,728 B/frame)
  └─ process_frame()      — stateless DSP per frame
       range FFT → Doppler FFT → MTI → RD-CFAR → az multi-peak → xyz cloud
  └─ TemporalAccumulator  — stateful, 16-frame ring buffer
       motion compensation → DBSCAN → subject_frame() → gait_features()
  └─ MicroDopplerBuffer   — stateful, 64-frame ring buffer
       Doppler histogram → spectrogram → cadence (autocorrelation)
  └─ LiveViewer           — 5 panels updated at 4 fps
```

All five panels consume the output of the stateful layer, not the
per-frame output directly. The 16-frame accumulation window is the
fundamental temporal unit: features extracted from fewer than 16 frames
(the warm-up period at the start of each recording) are statistically
incomplete.

### Hardware reference values

| Parameter | Value |
|---|---|
| Frame rate | 4 fps (256 ms per frame) |
| Range resolution | 6.6 cm/bin |
| Doppler resolution | 0.078 m/s per bin |
| Doppler range | ±2.498 m/s |
| Az resolution at 3 m | 2.34 cm/bin → shoulder (0.45 m) spans 19 bins |
| Az resolution at 8 m | 6.25 cm/bin → shoulder spans 7 bins |
| Accumulation window | 16 frames = 4 s |
| EMA centroid lag | 2.3 frames = 0.6 s (α = 0.3) |
| Spectrogram window | 64 frames = 16 s |

---

## Panel 1 — 3D accumulated cloud

**Processing stage:** Phase 2 (temporal accumulation with motion compensation)

**Position in viewer:** leftmost, largest, OpenGL 3D scatter

### What is rendered

All points from the current 16-frame accumulator buffer after Y+X EMA
motion compensation, displayed in world-frame coordinates:

```
x  = lateral (azimuth), positive = right of boresight
y  = range (forward), positive = away from radar
z  = elevation + radar_height_offset
```

Each point is coloured by its radial Doppler value using a three-way
diverging colormap:

- Red — receding (positive radial velocity, walking away)
- White — stationary (|dop| near zero, tangential motion)
- Blue — approaching (negative radial velocity, walking toward radar)

Point size scales linearly with SNR (power_dB), so strong returns are
larger and visually dominant.

### Why this panel exists

Panel 1 provides the unfiltered ground truth of what the radar sees
across the current 4-second accumulation window. Every upstream decision
in the pipeline — grating lobe rejection, FoV gate, cross-cluster filter,
DBSCAN noise labelling — is visible here as the presence or absence of
points. It is the diagnostic entry point: if the downstream panels show
no tracking, the first question is whether Panel 1 shows any points at
all.

The Doppler colouring serves two specific purposes:

1. **Sign convention verification.** During radial walking, all points
should shift from white through red (walking away) or blue (walking
toward) as the subject moves. A subject walking away who produces blue
points indicates a sign flip somewhere in the Doppler FFT output.

2. **Clutter discrimination.** Stationary infrastructure (walls, poles,
vehicles not yet rejected by MTI) appears white regardless of frame.
Body returns appear coloured. This separation is immediate and requires
no numerical analysis.

### Motion compensation behaviour

The EMA centroid (α = 0.3, lag ≈ 2.3 frames = 0.6 s) tracks the
subject's range position and translates all historical frames to align
with the current centroid estimate. X compensation uses the most recent
frame's raw centroid (not the EMA) to avoid ghost-position lock-on.

During a direction reversal, the accumulated cloud shows a temporal
smear: points from the previous direction (old colour) persist for up
to 16 frames while new-direction points accumulate. The smear length in
the range axis equals the EMA lag × walking speed:

```
smear_length ≈ lag_frames × v_walk × (1/fps)
             ≈ 2.3 × 1.2 × 0.25 ≈ 0.69 m
```

This is not a pipeline failure. It is the expected transient before the
ring buffer flushes. The smear disappears after one full 16-frame window
(4 seconds).

### What correct output looks like

During radial walking: a compact elongated cloud aligned with the range
axis, shifting between red (outward leg) and blue (return leg) at each
direction reversal. The cloud should span approximately 0.3–0.5 m in x
(lateral) and 0.3–0.6 m in y (range depth of body). The elevation extent
(z) is typically 0.3–0.6 m and is less reliable due to the coarser
elevation resolution (48 virtual aperture channels vs 256 azimuth).

During lateral walking: the cloud moves horizontally (in x) while y
and z remain roughly constant. All points appear near-white (near-zero
Doppler) regardless of lateral speed.

### Failure modes

| Symptom | Cause | Check |
|---|---|---|
| No points visible | No detections passing CFAR | `cloud_pts` in audit CSV; try `--cfar-margin 5` |
| Points visible but wrong colour | Doppler sign flip | `dop_centroid` sign in gait features vs walk direction |
| Persistent white cloud at fixed position | MTI clutter not rejected | Is the target actually moving? Check MTI filter threshold |
| Cloud explodes to large spatial extent | Ghost return survived all filters | `cc_rej` count in audit — if zero, cross-cluster gate may need tightening |
| z values far from zero | Radar height offset not applied | Adjust `h_radar` slider in viewer or measure mounting height |

---

## Panel 2 — X-Y top-down footprint

**Processing stage:** Phase 3 (DBSCAN single-subject clustering and
centroid tracking)

**Position in viewer:** centre, 2D scatter, boresight looking down

### What is rendered

The projection of the accumulated cloud onto the horizontal plane:
x (lateral azimuth) on the horizontal axis, y (range) on the vertical
axis, with the radar at the top and range increasing downward.

Over this scatter an oval is drawn at the DBSCAN cluster centroid
(cluster_cx, cluster_cy) with fixed semi-axes:

```
semi-axis x:  ±0.225 m  (nominal human shoulder half-width)
semi-axis y:  ±0.150 m  (nominal human body half-depth, front-to-back)
```

The oval semi-axes are a fixed reference standard, not fitted to the
data. They show where the body should be relative to the centroid, not
where the cluster points actually are (that is the role of Panel 4).

### Why this panel exists

The top-down view is the canonical radar surveillance plane and the
primary track continuity diagnostic. It answers two questions that no
other panel makes as immediately visible:

**Is the centroid tracking the correct target?** If the oval locks onto
a wall reflection or grating lobe instead of the walking subject, the
oval will sit at a fixed position in y while the subject moves — visible
immediately as a stationary oval during an active recording.

**Is the track stable?** Each frame, the centroid is updated via the
EMA. Any centroid jump greater than 0.6 m between consecutive frames
(the ghost gate, corresponding to a speed of 2.4 m/s — faster than any
realistic walking speed) causes the accumulator to freeze the centroid
for that frame. A jump visible in the top-down view that exceeds the
oval diameter in one frame is a track failure event.

### DBSCAN parameters

```
eps            = 0.80 m    neighbourhood radius
min_samples    = 3         minimum points to form a cluster
max_cluster_radius = 0.65 m  reject implausibly wide clusters
jump_gate      = 0.60 m/frame = 2.4 m/s max allowed speed
```

The eps value of 0.80 m was calibrated to span the full body depth
(0.389 m measured radius, 0.778 m diameter) within a single DBSCAN
neighbourhood, so the body does not split into two depth sub-clusters
at far range.

### What correct output looks like

During radial walking: the oval moves smoothly along the y axis (range),
advancing during the outward leg and retreating during the return leg.
Lateral position (x) stays near boresight (±0.3 m) unless the subject
walks at an angle. The transition from outward to return should be a
smooth curve, not a step change.

During lateral walking (Scenario B): the oval moves horizontally (in x)
at fixed y = 3 m. The lateral drift should be smooth and continuous as
the subject crosses the ±3 m corridor.

### Failure modes

| Symptom | Cause | Check |
|---|---|---|
| Oval locks to fixed position while subject walks | Ghost return (wall/vehicle) captured by DBSCAN | `cc_rej` in audit; tighten `cross_cluster_range_m` |
| Oval jumps > 0.5 m between frames | Ghost point shifted cluster centroid | `cluster_noise` column; check if a second target is present |
| Oval absent (no oval drawn) | DBSCAN found no cluster this frame | `cluster_pts = 0` in audit; subject may be outside P1–P4 corridor |
| Oval oscillates at wrong range | EMA centroid lag during fast walking | Expected at high replay fps; occurs in production if actual speed > 1.8 m/s |

---

## Panel 3 — X-Doppler lateral velocity profile

**Processing stage:** Phase 1 (azimuth multi-peak detection — the
per-frame cloud structure diagnostic)

**Position in viewer:** right side, 2D scatter, x axis = lateral
azimuth, y axis = radial velocity

### What is rendered

Each accumulated point plotted as (x, dop_mps):

```
x axis:   lateral azimuth position (metres from boresight)
y axis:   radial Doppler velocity (m/s), positive = receding
colour:   same Doppler diverging map as Panel 1
```

No fitting or smoothing is applied. This is the raw (azimuth,
velocity) scatter of every point in the 16-frame accumulator buffer,
in world-frame coordinates.

### Why this panel exists

Panel 3 is the azimuth multi-peak diagnostic and the per-point
micro-Doppler cross-section. It answers a question that no other panel
addresses directly: **is the pipeline resolving lateral body width
within a single frame, or is every frame producing only a single azimuth
peak?**

When az multi-peak fires correctly (prominence threshold 6.0 dB met),
multiple peaks are detected across the shoulder width in a single frame.
These peaks appear at different x positions in Panel 3. When multi-peak
fails (prominence threshold too high, or subject too far for the az
resolution to separate shoulder returns), all points from a frame
collapse to a single x value — Panel 3 shows a vertical stripe at
x = centroid_x with no lateral spread.

The (x, v_r) structure also makes limb-torso separation visible in a
way that per-frame gait scalars cannot. During radial walking:

- Torso returns: x near centroid_x, |dop| near walking speed (~0.4 m/s)
- Limb returns: x at centroid_x ± limb_offset, |dop| > 0.30 m/s
  (the dop_limb_thr boundary)

The two dashed horizontal lines at ±0.30 m/s mark the boundary between
the torso and limb Doppler regimes. Points above/below these lines count
as limb returns (n_limb_pts) and contribute to limb_frac.

### Azimuth resolution and multi-peak detectability

```
az bin size at range R:  Δx = R × (2 / az_fft_size)
                              = R × 0.0078 m/bin

At 3 m:  Δx = 0.023 m/bin → shoulder (0.45 m) spans 19 bins
At 8 m:  Δx = 0.063 m/bin → shoulder spans 7 bins
```

At 8 m range with 7 bins across the shoulder, the peak prominence
between adjacent shoulder returns drops significantly. This is why
multi-peak is more likely to fire at close range and the lateral scatter
in Panel 3 is wider at 3 m than at 8 m for the same subject. This is
not a pipeline bug — it is a fundamental resolution constraint.

### What correct output looks like

During radial walking at 3–5 m range: a compact cluster centred near
(centroid_x, walking_speed), with some points displaced ±0.05–0.15 m
laterally at matching Doppler (shoulder returns from the same range bin).
Occasionally points appear at higher |dop| (limb returns). The cluster
should not span more than ±0.3 m in x in a single frame.

During lateral walking: all points collapse toward dop ≈ 0 across the
full x range of the subject's corridor. This produces a flat horizontal
band near v_r = 0, which is the correct signature for tangential motion.
The absence of Doppler spread in this scenario is expected and correct.

### The az/RD ratio diagnostic

```
az/RD ratio = cloud_pts / rd_cfar_survivors
```

This ratio is the per-frame measure of multi-peak activity. A ratio of
1.0 means every RD detection produced exactly one azimuth peak — no
multi-peak fired. A ratio of 2.5 means 2.5 az peaks per RD detection on
average — multi-peak is resolving lateral structure. Values consistently
above 1.5 indicate the pipeline is capturing body width. This ratio is
computed in `diag_xyz.py` and is the numerical complement of the lateral
scatter visible in Panel 3.

### Failure modes

| Symptom | Cause | Check |
|---|---|---|
| Single vertical stripe, no lateral spread | Multi-peak not firing | `az_peak_min_prom_db` — reduce from 10 to 6 dB if still at old value |
| All points near v_r = 0 during radial walk | Doppler sign not being detected | MTI filter too aggressive; check Doppler FFT stage |
| Wide lateral spread > ±0.5 m | Grating lobe not rejected | `grating_rej` count in audit; check `body_az_half_width_m` |
| Points at extreme Doppler (> 1.5 m/s) | Vehicle or other fast-moving object | `max_az_sin` gate may not be rejecting it; check FoV |

---

## Panel 4 — Body-frame panel

**Processing stages:** Phase 4 (subject-centric coordinate
normalisation) and Phase 5 (PCA body envelope fitting)

**Position in viewer:** 4th from left, 2D scatter, fixed ±0.5 m extent
in both axes

### What is rendered

The full 16-frame accumulated cloud translated to body-local
coordinates by subtracting the DBSCAN cluster centroid:

```
dx = x - cluster_cx    (lateral offset from body centre, m)
dy = y - cluster_cy    (range offset from body centre, m)
dz = z - cluster_cz    (elevation offset from body centre, m)
```

After translation, the display is fixed at ±0.5 m in both dx and dy.
A crosshair at (0, 0) marks the cluster centroid (the body centre).

**Point colouring:**
- Cyan (RGBA 80, 180, 255, 160) — DBSCAN body-cluster inlier points
  (within 1.5 × eps = 1.2 m of centroid)
- Dim grey (RGBA 80, 80, 80, 100) — noise-labelled points (outliers
  beyond the cluster boundary, visible but subdued)

Semi-transparent rendering (α = 160/255 ≈ 63%) allows density to be
inferred from point overlap in the tight body-width region.

**Reference overlays:**
- Dashed blue box at ±0.225 m × ±0.15 m — nominal human body footprint
  (shoulder width × body depth), fixed reference scale
- Crosshair at (0, 0) — body centroid
- Amber PCA ellipse — 2σ boundary fitted to body-cluster points in
  (dx, dy), clamped to ±0.50 m semi-axes

**Status annotation (top left of panel):**

```
accum=64 pts  cluster=58  noise=6  env 0.28×0.43m  θ=12°
```

### Why this panel exists

Panel 4 is the primary body-shape output panel and the deliverable of
Phases 4 and 5. By translating to subject-centric coordinates, the body
footprint becomes range-independent: the same physical body produces the
same (dx, dy) distribution at 3 m and at 9 m. This is the M5 criterion
(dx_std = 0.141 m, confirmed range-independent across 4.19–5.17 m in the
Phase 4 audit).

The PCA ellipse encodes two body-shape descriptors per frame that are
the primary Phase 7 feature candidates:

- `rx` — lateral semi-axis (lateral half-width of body footprint)
- `ry` — range semi-axis (range half-depth of body footprint)
- `θ` — orientation of the major axis (aspect angle relative to boresight)

### PCA ellipse fitting

The ellipse is fitted to the body-cluster points only (not noise points)
using PCA on the 2×2 covariance matrix of (dx, dy):

```
cov = [[var(dx), cov(dx,dy)],
       [cov(dx,dy), var(dy)]]

eigenvalues  → 2σ radii:    rx = 2√λ_major,  ry = 2√λ_minor
eigenvectors → orientation: θ = atan2(v_major)
```

The 2σ boundary encloses approximately 86% of a 2D Gaussian
distribution. Outliers (points beyond 2.5σ of the median distance from
the centroid) are excluded from the fit before computing the covariance,
to prevent a single ghost point from stretching the ellipse.

Both semi-axes are clamped to a maximum of 0.50 m before drawing, per
the plan's requirement that the envelope stays within human body bounds.

### The dx_std metric (M5 acceptance criterion)

```
dx_std = std(dx across all cluster points in current buffer)
       = 0.141 m  (measured, full 16-frame window)
```

This value is the normalised lateral spread — the radar's effective
measurement of body half-width after accumulation. It is range-independent
because the subject-centric translation removes the range-proportional az
bin size effect. The theoretical expectation for a shoulder width of
0.45 m sampled uniformly is 0.45 / (2√3) = 0.130 m; the observed 0.141 m
is consistent with body slightly wider than pure shoulder span (arms,
clothing) and az FFT bin-level quantisation.

### What correct output looks like

During radial walking with a full 16-frame buffer: a compact cyan cluster
centred near (0, 0), spanning approximately ±0.14 m in dx and ±0.22 m
in dy. The amber ellipse should be stable frame-to-frame and orient with
its major axis aligned with dy (range direction) for frontal radial
walking — the body is deeper front-to-back than it is wide in this
aspect. The grey noise points (if any) should appear at larger |dx| or
|dy|, clearly separated from the cyan cluster.

During lateral walking (Scenario B): the cluster may appear narrower in
dx (body is seen side-on, presenting its front-to-back depth into the
range axis rather than its shoulder width). The ellipse should rotate:
θ shifts from ~0° (radial walk, major axis in dy) toward ~90° (lateral
walk, major axis in dx).

### Why overlapping points look like few points

The body footprint in subject-centric coordinates spans approximately
±0.14 m in dx and ±0.22 m in dy — a total area of 0.28 m × 0.44 m.
At a ±0.5 m panel extent with a typical display width of ~400 px, this
region occupies approximately 112 px × 176 px. With 58 cyan points at
8 px each, the overlap is heavy. The semi-transparent rendering (α = 160)
means overlapping points accumulate brightness, giving a density
impression rather than individual dot counting. This is not missing
data — every point is rendered.

### Failure modes

| Symptom | Cause | Check |
|---|---|---|
| Ellipse rx > 0.50 m (clamped at boundary) | Ghost point survived cluster gate, or rapid lateral motion during EMA lag | `cluster_noise` count; check if `max_cluster_radius` is being respected |
| Cluster not centred at (0, 0) | Centroid offset — EMA not yet converged | Expected in first 4–8 frames; check if persists beyond warm-up |
| Ellipse orientation does not change between radial and lateral | Subject-frame translation using wrong centroid | Verify `subject_frame_en = True` in config |
| Only grey points, no cyan | DBSCAN found no cluster (cluster_pts = 0) | Subject may be outside range, or DBSCAN eps too small |
| Panel blank | `subject_frame_en = False` | Enable in radar_config.py |

---

## Panel 5 — Doppler-time spectrogram

**Processing stage:** Phase 6 (micro-Doppler extraction and cadence
estimation via MicroDopplerBuffer)

**Position in viewer:** rightmost (5th panel), 2D image + overlay curves

### What is rendered

A Doppler-time spectrogram rendered as a `pyqtgraph.ImageItem`:

```
Horizontal axis:  time, 64 frames = 16 s, oldest left → newest right
Vertical axis:    Doppler velocity, −1.5 to +1.5 m/s
Cell value:       count of cluster points in that (Doppler bin, frame),
                  normalised to [0, 1] per column
Bin resolution:   75 mm/s per bin  (40 bins across ±1.5 m/s)
```

Each column is the Doppler histogram of the DBSCAN body-cluster for
that frame — how many cluster points fell into each 75 mm/s Doppler
bin. Column-wise normalisation (each column divided by its maximum) means
the display shows relative Doppler distribution per frame rather than
absolute point counts. This prevents a high-point-count frame from
visually dominating.

**Overlays:**
- Amber curve — `torso_line`: the power-weighted Doppler centroid
  (`dop_centroid`) plotted across all 64 time columns. This is the
  first moment of the column's Doppler distribution, equivalent to the
  spectral centroid of the marginal Doppler density per frame.
- Two dashed blue lines at ±0.30 m/s — the `dop_limb_thr` boundary
  separating torso returns (|dop| < 0.30 m/s) from limb returns
  (|dop| ≥ 0.30 m/s).
- Zero line at 0 m/s — stationary reference.

**Cadence annotation (top left):**
```
cadence ≈ 1.0 Hz  (1.0s/stride)
```
or
```
filling… (24/64 frames)
```
or
```
cadence: no clear peak
```

### Why this panel exists

The Doppler-time spectrogram is the canonical micro-Doppler feature in
radar-based gait analysis, introduced by Chen and Ling (2002). It
externalises the temporal Doppler structure that is invisible in
per-frame scalar statistics. Three specific properties are readable
directly from the waterfall that no single-frame metric captures:

**1. Direction reversal.** At each turnaround the amber torso line
crosses zero — from positive (walking away) to negative (returning) or
vice versa. The time between successive zero-crossings is half the
out-and-back period. This is the most direct visual confirmation that
the Doppler sign convention is correct end-to-end.

**2. Speed change.** When the subject changes pace (Scenario D), the
torso line shifts to a new Doppler level and the spectrogram column
distribution changes width. The transition is visible as a gradual
shift because the 16-frame EMA centroid takes ~2.3 frames (0.6 s) to
track a step change in speed.

**3. Limb-swing periodicity.** During steady-state walking, the
spectrogram shows a periodic brightening above the 0.30 m/s limb
boundary as the swinging arm or leg produces a high-Doppler return every
stride. This periodicity is what the cadence estimator quantifies.

### Cadence estimation

The cadence is estimated from the normalised autocorrelation of the
`limb_frac` time series — the fraction of cluster points with
|dop| ≥ 0.30 m/s per frame, sampled at 4 fps across the 64-frame buffer:

```
limb_frac[t] = n_limb_pts[t] / cluster_pts[t]

AC[lag] = corr(limb_frac, shifted limb_frac by lag frames) / AC[0]

peak_lag = argmax(AC[lag_min : lag_max])
         where lag_min = fps / cadence_max = 4/3.0 ≈ 1 frame
               lag_max = fps / cadence_min = 4/0.3 ≈ 13 frames

cadence_hz = fps / peak_lag
```

The limb_frac signal has a period equal to the stride cycle because limb
Doppler peaks twice per stride (once per swinging limb, for a biped with
two limbs) — but the practical detection is of the dominant periodicity
in the autocorrelation, which at 4 fps with a 64-frame window can
resolve cadences from 0.3 Hz (one stride every 3.3 s) to 3.0 Hz (three
strides per second, above any human walking rate).

The estimate updates every 8 frames (2 s). A minimum autocorrelation
peak height of 0.20 is required — below this the signal is treated as
aperiodic and NaN is returned.

### Minimum frames required for cadence

```
n_cycles_needed  = 4  (minimum for reliable periodicity detection)
At 1.0 Hz cadence:  4 cycles × 4 frames/cycle = 16 frames = 4 s
At 0.5 Hz cadence:  4 cycles × 8 frames/cycle = 32 frames = 8 s
```

The cadence annotation shows `filling…` until 16 frames are available.
After that, if no clear peak is found, it shows `no clear peak` rather
than a potentially spurious value.

### What correct output looks like

**Scenario A (radial walk, forward and back):** The waterfall shows a
bright band centred on the walking speed (~0.4–0.5 m/s) during the
outward leg, and the amber torso line traces through that band. At each
turnaround the torso line crosses zero and the bright band flips sign.
After ~16 s (64 frames), the cadence annotation stabilises at
approximately 0.8–1.1 Hz depending on the subject's natural pace.

**Scenario B (lateral walk):** The waterfall is nearly blank — nearly all
cluster points have Doppler near zero. The torso line sits near 0 m/s
throughout. The cadence annotation returns NaN. This is the correct
output for tangential motion and confirms the Doppler sign convention.

**Scenario D (multi-leg mixed speed):** The torso line oscillates between
positive and negative values at the walking speed, with the amplitude
shifting when the subject changes speed (normal → slow → fast). The
bright band in the waterfall changes its Doppler level with each speed
change. The cadence estimate changes with speed but lags the actual
change by approximately one autocorrelation update period (8 frames = 2s).

### Failure modes

| Symptom | Cause | Check |
|---|---|---|
| Waterfall blank, torso line at zero | No cluster detections or lateral walk | Panel 2 for cluster oval; is subject in radial or lateral scenario? |
| Cadence = NaN after 30+ active frames | `limb_frac` consistently low (< 0.1) | Subject walking tangentially; cadence unreliable for non-radial motion |
| Torso line does not cross zero at turnarounds | EMA lag smearing direction reversal | Expected — smear duration is ~2.3 frames (0.6 s) around each reversal |
| Cadence estimate unstable (changing each update) | Insufficient gait cycles in 64-frame window | Occurs at slow speeds (<0.5 Hz cadence, needs 32+ frames of steady walk) |
| Panel blank / not visible | `microdoppler_en = False` | Enable in radar_config.py |

---

## Cross-panel interpretation guide

The five panels are designed to be read together. A failure or anomaly
that appears in one panel usually has a complementary signature in one
or more others.

| Observation | Panel(s) | Diagnosis |
|---|---|---|
| Panel 1 has points, Panel 2 oval absent | 1, 2 | DBSCAN found no cluster — cluster_pts = 0. Subject may be outside corridor or eps too small for the cloud density |
| Panel 2 oval stationary while subject walks | 1, 2 | Centroid locked to a ghost return. Check cc_rej in audit; verify cross_cluster_range_m = 0.35m |
| Panel 3 single vertical stripe, no lateral spread | 3 | Az multi-peak not firing. Verify az_peak_min_prom_db = 6.0 dB (not 10.0 dB) |
| Panel 4 ellipse rx > 0.50 m (clamped) | 2, 4 | Ghost point survived DBSCAN or max_cluster_radius gate. Check cluster_noise in audit |
| Panel 4 cluster not at (0, 0) | 4 | EMA not yet converged — expected in first 8 frames of each run |
| Panel 5 blank waterfall during Scenario A | 1, 5 | No cluster detections — no Doppler data to histogram. Check cloud_pts > 0 first |
| Panel 5 cadence = NaN, Scenario A | 5 | limb_frac too low (subject walking tangentially or dop_limb_thr too high). For Scenario B this is expected |
| All panels populated but Panel 4 ellipse wrong shape | 4, 5 | Subject not walking radially — check orientation from Panel 3 and gait from Panel 5 |
| Temporal smear in Panel 1 at every turnaround | 1, 5 | Normal — direction reversal flush takes 16 frames. Torso line zero-crossing in Panel 5 is the precise marker |
| z values far from zero in Panel 1 | 1 | Radar height not applied. Adjust h_radar slider; use diag_xyz.py --radar-height to find correct value |

---

## Panel-to-pipeline-stage mapping

| Panel | Phase | Module | Key config fields |
|---|---|---|---|
| 1 — 3D cloud | Phase 2 | `accumulate.py` → `TemporalAccumulator` | `accum_window=16`, `accum_alpha=0.3` |
| 2 — X-Y footprint | Phase 3 | `accumulate.py` → `TemporalAccumulator._cluster()` | `dbscan_eps=0.80`, `dbscan_min_samples=3`, `max_cluster_radius=0.65` |
| 3 — X-Doppler | Phase 1 | `radar_processing.py` → `process_frame()` az multi-peak | `az_peak_min_prom_db=6.0`, `emit_all_cluster_bins=True` |
| 4 — Body-frame | Phase 4+5 | `accumulate.py` → `subject_frame()` + `fit_body_ellipse()` | `subject_frame_en=True`, `body_envelope_en=True` |
| 5 — Spectrogram | Phase 6 | `microdoppler.py` → `MicroDopplerBuffer` | `microdoppler_en=True`, `spectrogram_frames=64`, `spectrogram_v_max=1.5`, `dop_limb_thr=0.30` |

---

## Status bar

The status bar below the panels shows per-frame gait scalars from
`GaitFeatures` when `gait_features_en = True`:

```
Frame 142  |  pts (accum×16(motion-comp)): 64  vy=+0.42m/s
           |  vc=+0.42m/s  σ=0.18  limb=65%
           |  radar h: 0.33m  |  dropped: 0
```

| Field | Source | Meaning |
|---|---|---|
| `pts` | `len(accum_result.cloud)` | Total accumulated cloud size (all 16 frames) |
| `vy` | `accum_result.velocity_y` | EMA-derived radial velocity estimate (m/s) |
| `vc` | `gait.dop_centroid` | Power-weighted Doppler centroid of body cluster (trunk velocity, m/s) |
| `σ` | `gait.dop_spread` | Std of cluster Doppler values (velocity diversity, m/s) |
| `limb%` | `gait.limb_frac` | Fraction of cluster pts with \|dop\| ≥ 0.30 m/s |

`vc` and `vy` should agree in sign during steady radial walking and
both cross zero at the same direction reversal. A discrepancy between
`vc` and `vy` indicates the EMA centroid is lagging the actual speed —
the power-weighted Doppler centroid (`vc`) responds instantaneously,
while the EMA velocity (`vy`) lags by ~2.3 frames.

---

*For the processing pipeline implementation see `processing/accumulate.py`,
`processing/microdoppler.py`, and `processing/streaming.py`.
For the full phase development roadmap see `docs/HUMAN_BODY_SHAPE_PLAN.md`.*
