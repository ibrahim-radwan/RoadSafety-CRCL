"""
radar_config.py - AWR2243-CAS-2X-EVM cascade geometry and processing defaults.

The virtual array follows the 38+8 layout used by the cascaded EVM:
  - bottom row: 40 azimuth channels with 38 independent positions
  - top row:    8 elevation channels
  - azimuth spacing: 0.5 lambda
  - elevation spacing: 0.8 lambda

The bottom-row overlaps occur at virtual azimuth positions 7 and 26, so angle
estimation should treat the aperture as 38 independent azimuth channels plus a
matched 8-channel elevation row.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

C = 3e8  # speed of light (m/s)

_TX_AZ  = [0,  4,  7, 11, 15,  2]   # az position per chirp (λ/2)
_TX_EL  = [0,  0,  0,  0,  0,  1]   # chirp 5 = elevation TX
_RX_AZ  = [0,  1,  2,  3, 19, 20, 21, 22]
_RX_EL  = [0,  0,  0,  0,  0,  0,  0,  0]
_D_EL   = 0.8    # elevation spacing in λ units
_EL_TX  = 5      # chirp index of elevation TX


@dataclass
class RadarConfig:
    # ── Hardware ──────────────────────────────────────────────────────────────
    num_devices:      int   = 2
    num_rx_per_dev:   int   = 4
    num_tx:           int   = 6

    # ── Waveform (shortrange defaults matching cascade_r017m.cfg) ─────────────
    start_freq_ghz:   float = 77.0
    slope_mhz_us:     float = 44.41
    sample_rate_ksps: float = 5000.0
    num_samples:      int   = 256
    idle_time_us:     float = 5.0
    ramp_end_us:      float = 60.0
    num_loops:        int   = 64
    iq_swap:          bool  = False  # Default raw capture is [I,Q]; SampleSwap=1 / q-first sets this True

    # ── Virtual array ─────────────────────────────────────────────────────────
    tx_az:       List[int] = field(default_factory=lambda: list(_TX_AZ))
    tx_el:       List[int] = field(default_factory=lambda: list(_TX_EL))
    rx_az:       List[int] = field(default_factory=lambda: list(_RX_AZ))
    rx_el:       List[int] = field(default_factory=lambda: list(_RX_EL))
    d_el_lambda: float     = _D_EL
    elev_tx_idx: int       = _EL_TX

    # ── FFT sizes ─────────────────────────────────────────────────────────────
    range_fft_size:   int = 256
    doppler_fft_size: int = 64
    az_fft_size:      int = 256
    el_fft_size:      int = 64

    # ── Windowing ─────────────────────────────────────────────────────────────
    range_window:       str   = "hann"
    doppler_window:     str   = "hann"
    az_window:          str   = "chebyshev"
    el_window:          str   = "hann"
    chebyshev_atten_db: float = 60.0

    # ── Calibration ───────────────────────────────────────────────────────────
    apply_calibration:  bool          = True
    calib_cfg_path:     Optional[str] = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "firmware", "r017m", "cascade_r017m.cfg")

    # ── Clutter / static removal ──────────────────────────────────────────────
    clutter_removal:  bool  = True   # MTI: subtract slow-time mean
    zero_dc_bins:     int   = 2      # ±N Doppler bins zeroed around DC

    # ── DSP mode ─────────────────────────────────────────────────────────────
    dsp_mode: str = "cascade"

    # ── AoA beamforming ───────────────────────────────────────────────────────
    aoa_method:        str = "fft"   # fft | capon | music | bartlett
    music_num_sources: int = 3
    music_diag_load:   float = 0.01
    music_forward_backward_avg: bool = True
    music_use_unique_aperture: bool = True

    # ── CFAR (K-Radar style) ──────────────────────────────────────────────────
    cfar_method:   str        = "os"
    cfar_guard:    List[int]  = field(default_factory=lambda: [4, 6])
    cfar_train:    List[int]  = field(default_factory=lambda: [8, 16])
    cfar_fa_rate:  float      = 3e-5
    cfar_os_rank:  float      = 0.75
    cfar_margin_db:float      = 10.0
    cfar_min_power_db: float  = 12.0
    # ── C4: firmware reference ────────────────────────────────────────────────
    # C4: firmware fixed CFAR threshold from cfarCfg — READ-ONLY REFERENCE.
    # Not used by the pipeline algorithm (which uses FA-rate + margin_db).
    # Populated by parse_cfg_file; used only for startup comparison/logging.
    cfar_threshold_db: float  = 0.0

    # ── D4: diagnostic verbosity ──────────────────────────────────────────────
    # When True, per-frame always-on diagnostic prints (C7 margin, C3 fallback,
    # C2 suppression) are suppressed.  Frame summary lines still appear.
    # Set via --quiet on the CLI.
    quiet_mode: bool = False
    skip_cfar:  bool = False   # skip CFAR stages 6+8 for max display throughput
    consistency_k: int = 2     # cross-channel consistency: min RX channels a
                               # candidate (range, Doppler) cell must exceed the
                               # per-channel noise floor to survive.
                               # 0 = disabled (bypass mask entirely).
                               # 2 = recommended outdoor default — rejects single-
                               # channel noise spikes without affecting real targets
                               # which appear coherently across all RX channels.
                               # Implements the cross-channel fusion step described
                               # in the AE-CFAR cascaded radar literature.

    # ── NMS / range gate ──────────────────────────────────────────────────────
    nms_size:    List[int] = field(default_factory=lambda: [3, 3])
    min_range_m: float     = 0.1
    max_range_m: float     = 16.0   # 0 = full axis
    roi_x_m: Optional[tuple[float, float]] = None
    roi_y_m: Optional[tuple[float, float]] = None
    roi_z_m: Optional[tuple[float, float]] = None

    # ── Gaussian smoothing ────────────────────────────────────────────────────
    gaussian_sigma: float = 0.0   # 0 = disabled

    # ── Moving-objects gate ───────────────────────────────────────────────────
    moving_objects_only:    bool  = False  # show only moving returns
    min_abs_doppler_mps:    float = 0.0    # 0 = auto (0.5 × dop_res)
    require_rd_confirmation:bool  = True

    # ── High-reflection moving-object filter ──────────────────────────────────
    # When enabled, the CFAR detection mask is additionally filtered to keep
    # only cells that are BOTH moving (|Doppler| > min) AND high-power
    # (power > local noise floor + high_moving_db_above).
    # Use this to isolate strong, moving reflectors (vehicles, metal objects).
    filter_high_moving:      bool  = False  # master enable
    high_moving_min_dop:     float = 0.0    # min |Doppler| (m/s); 0 = auto
    high_moving_db_above:    float = 8.0    # dB above local noise floor

    # Additional post-CFAR peak cleanup to suppress sidelobes around the
    # strongest object return in each range bin / azimuth row.
    apply_sidelobe_filters: bool  = True
    rd_sidelobe_db:         float = 8.0
    ra_sidelobe_db:         float = 8.0
    doppler_stripe_removal: bool  = True
    strict_object_filtering: bool = True
    max_az_detections_per_range: int = 4
    ra_peak_prominence_db: float = 3.0
    el_peak_prominence_db: float = 2.0

    # ── Multi-peak azimuth extraction (body-shape point cloud) ────────────────
    # These three fields control Stage 8 multi-peak az extraction.
    # The goal is to resolve multiple scattering centres across a human body
    # (shoulders, torso, hips) within a single RD-confirmed range bin.
    #
    # az_multi_peak_en:
    #   True  = find all local peaks in the az spectrum (body-shape mode)
    #   False = argmax only, one point per range bin (legacy behaviour)
    #
    # az_peak_max_n:
    #   Hard cap on peaks per range bin.  6 is sufficient to cover
    #   shoulders + torso + hips at close range.
    #
    # az_peak_min_prom_db:
    #   Minimum prominence above surrounding valley in dB.
    #   Hann-windowed FFT sidelobes are ~57 dB below the main lobe so any
    #   value above ~3 dB is safe against pure spectral sidelobes.
    #
    #   Phase 1 Step 2 (2026-06-15): reduced from 10.0 → 6.0.
    #   Rationale: Phase 1 audit showed avg 2.20 pts/frame at prom=10dB.
    #   The grating filter (boresight-nearest + 0.30m body gate) is robust
    #   enough to suppress grating lobes at any prom ≥ 3dB.  Shoulder
    #   returns sit at ~22.5cm from the torso centre (within the 0.30m gate)
    #   and have measured FFT prominence >30dB — well above 6dB — so they
    #   will be recovered.  At 6dB, a single-target scene produces up to 9
    #   raw peaks which the grating filter collapses to 2–3 (all within the
    #   body window).  Net effect: weaker shoulder/arm scatterers now qualify
    #   as secondary peaks without admitting grating lobes.
    #
    # az_peak_min_gap_bins:
    #   Minimum bin separation between accepted peaks.
    #   6 = physical Hann beamwidth (safe minimum — prevents same reflector
    #   being counted twice due to spectral leakage).
    #   Previously 12, which blocked shoulder resolution beyond 3m because
    #   shoulder-to-shoulder spans only 7–12 bins at 3–6m range.
    #
    # rd_range_cluster_bins:
    #   Adjacent RD-confirmed range bins within this many bins of each other
    #   are grouped into one cluster.  The cluster is represented by its
    #   strongest range bin for az/el estimation, eliminating depth-scatter
    #   from multiple range bins covering the same body part.
    #   0 = disabled (all range bins processed independently).
    #   2 = recommended: merges bins ≤13.2cm apart (2 × 6.6cm range res).
    #
    # el_shared_across_range_bins:
    #   When True, all range bins within the same RD frame share one
    #   elevation estimate (median across confirmed bins).  This eliminates
    #   per-bin elevation noise (±19cm at 3m) that causes vertical scatter.
    #   The body parts on a walking person are all at similar elevation —
    #   averaging is physically justified.
    az_multi_peak_en:           bool  = True
    az_peak_max_n:              int   = 6
    az_peak_min_prom_db:        float = 6.0   # Phase 1 Step 2: was 10.0
    az_peak_min_gap_bins:       int   = 6
    rd_range_cluster_bins:      int   = 2
    el_shared_across_range_bins: bool = True

    # ── Grating-lobe / multipath rejection ───────────────────────────────────
    # After finding all az peaks for a cluster, discard any peak whose lateral
    # position x = range × sin(az) differs from the STRONGEST peak in that
    # cluster by more than body_az_half_width_m.
    #
    # Physical basis:
    #   A human body cannot span more than ~±30 cm laterally at any range.
    #   TDM MIMO virtual arrays with non-uniform element spacing produce
    #   grating lobes at ±25–35° from the main lobe (observed at ±29.5°
    #   in the AWR2243-2X cascade recordings).  At 2.4 m range a ±29.5°
    #   grating lobe appears at ±1.2 m lateral — 2.8× the shoulder width.
    #   This single constraint removes all grating-lobe false peaks while
    #   leaving all real body scatterers (torso, shoulders, hips) intact.
    #
    # Default 0.30 m = slightly wider than shoulder half-width (0.225 m)
    # to allow for arm swing and measurement noise.
    # Set to 0.0 to disable the filter.
    body_az_half_width_m: float = 0.30

    # ── Az field-of-view gate ─────────────────────────────────────────────────
    # Discard any az peak whose |sin(az)| exceeds this value.
    # The az FFT axis uses sin_v = linspace(-1, 1, N_az); bins near 0 and N_az-1
    # have |sin_v| → 1.0 (az → ±90°).  Spectral leakage at these extreme bins
    # produces false peaks that translate to physically impossible x values
    # (|x| can exceed the target range).  Default 0.866 = ±60° FoV, which
    # covers the full usable illumination cone of the AWR2243-2X antenna.
    # Set to 1.0 to disable.
    max_az_sin: float = 0.866

    # ── Cross-cluster coherence gate ─────────────────────────────────────────
    # After per-cluster az estimation, compare all cluster representatives.
    # If two clusters are within cross_cluster_range_m in range (y) but more
    # than body_az_half_width_m apart laterally (x), only the highest-power
    # cluster survives.  This eliminates wall/multipath reflections that form
    # separate clusters at nearly the same range as the real target.
    # Default 1.0 m: clusters within 1m of each other in range are considered
    # co-located.  Set to 0.0 to disable.
    cross_cluster_range_m: float = 0.35

    # ── P0-6: subject-proximity gate ─────────────────────────────────────────
    # The cross-cluster filter above rejects cluster j only when
    #     |dy| < cross_cluster_range_m  AND  |dx| > body_az_half_width_m
    # i.e. ONLY same-range azimuth ambiguities (grating lobes). A return at a
    # DIFFERENT range is never rejected however far away it is, because
    # |dy| >= cross_cluster_range_m short-circuits the test. Measured effect:
    # comp_residual_x median 0.73-3.98 m against a 0.45 m body, and a 32%
    # ghost-gate trip rate driven by centroids jumping 2-8 m between frames.
    #
    # This gate rejects clusters further than subject_gate_radius_m from the
    # CURRENT TRACK POSITION (Euclidean), supplied by the Phase 3 accumulator
    # via process_frame(track_hint=...). It is the anchor_mode feedback path
    # that radar_config has always anticipated but never wired.
    #
    # subject_gate_radius_m: body half-width + one frame of motion.
    #   0.30 m (body) + 0.60 m (max_jump at 4 fps = 2.4 m/s) = 0.90 m
    # SAFETY: if the gate would remove every cluster the frame is passed
    # through UNGATED (counted as a bypass). Without that, a track that
    # drifts onto a ghost would filter the real subject out permanently —
    # positive feedback with no escape.
    # ── Stage 2.2: focused azimuth FFT ───────────────────────────────────────
    # True  = gather only the RD-focus-masked (doppler, range) cells and run one
    #         batched FFT. Mathematically identical to the full-cube path — the
    #         collapse only ever read those cells — but ~5.6x less work.
    # False = original full [n_dop, n_r, n_az] cube. Kept as a kill switch so
    #         the two paths can be A/B compared from the SAME binary, and so a
    #         rollback needs no file swap.
    # The full cube is also used automatically whenever cfg.save_drea_path is
    # set, because arrDREA needs it.
    az_fft_focused_en:      bool  = True

    subject_gate_en:        bool  = True
    subject_gate_radius_m:  float = 0.90

    # ── Stage 3.6: two-tier gate ─────────────────────────────────────────────
    # subject_gate_radius_m (0.90 m = body + one frame of motion) is sized for
    # TRACKING a subject already being followed. It is the wrong criterion for
    # ADMITTING a new one: a second person walking 2.86 m away can never get
    # in, so they never form a cluster, never become a track, and the gate
    # keeps excluding them. Measured on outdoor_long: disabling the gate raised
    # multi-subject frames 5.2x (22 -> 115 of 1447), confirming the lock-in.
    #
    # But the gate cannot simply be widened - with it off, every contamination
    # metric regressed to its pre-P0-6 value (wide_cluster_rej 22 -> 169,
    # rx REJECTED max 0.832 -> 2.087 m).
    #
    # The data separates the two cases by distance:
    #     second PERSON   : separation 1.1 - 3.0 m from the tracked subject
    #     CONTAMINATION   : comp_residual reaches 6.6 m, 'separation' 11.5 m
    #
    # So: a TIGHT radius holds an existing track, and a WIDER admit radius
    # lets a plausible new subject bootstrap while still rejecting distant
    # walls, posts and vehicles. A cluster admitted this way must also carry
    # enough points to be a plausible body, so isolated specks do not seed
    # spurious tracks.
    subject_admit_radius_m: float = 3.00
    # Default 1: outdoor_long averages 0.70 az peaks per range cluster,
    # so requiring 2 would reject most legitimate second-subject clusters and
    # make the admit tier inert. Raise to 2 only if admission proves too
    # permissive on real data.
    subject_admit_min_pts:  int   = 1

    # ── Phase 1 — per-frame cloud correctness ────────────────────────────────
    # ── Phase 1 config flags ──────────────────────────────────────────────────
    # Phase 0 evidence (2026-06-15):
    #   - cc_rej=423 was the dominant rejection stage (29.4% of RD survivors)
    #     because cross_cluster_range_m=1.0m grouped real body-depth clusters
    #     as multipath duplicates. Fixed: cross_cluster_range_m=0.35m (body depth).
    #   - 78% of detected frames had only 1-2 pts because emit_all_cluster_bins
    #     was False — only the representative bin's az was estimated, discarding
    #     body-depth coverage from other bins in the cluster.
    #   - These two fixes are now the default (emit_all=True, cc_range=0.35m).
    #
    # emit_all_cluster_bins:
    #   When True, each confirmed range bin in a cluster gets its own az FFT,
    #   producing multiple points spread in y (body depth) as well as x (width).
    #   When False (legacy), only the representative (strongest) bin is emitted.
    #   Default True (Phase 1 validated by Phase 0 evidence).
    #
    # cluster_max_points:
    #   Hard cap on total peaks emitted per cluster across all bins.
    #   Prevents noisy multi-bin clusters from flooding the cloud.
    #   8 = enough for 4 bins × 2 az peaks each (full body cross-section).
    #
    # anchor_mode:
    #   "boresight"         — keep boresight-nearest cluster (current default)
    #   "power_continuity"  — keep highest-power cluster (Phase 3 will upgrade
    #                         this to use the track centroid as reference)
    emit_all_cluster_bins: bool  = True
    cluster_max_points:    int   = 8
    anchor_mode:           str   = "boresight"
    # ── Phase 0 audit CSV ────────────────────────────────────────────────────
    # When set to a file path, process_frame() appends one CSV row per call
    # with stage-attribution counts. Default None = off, zero behaviour change.
    # Header is written once if the file does not already exist.
    # Columns: frame_idx,rd_cfar,after_consistency,n_clusters,az_peaks_raw,
    #          fov_rej,grating_rej,cc_rej,ra_det,cloud_pts,
    #          centroid_x,centroid_y,centroid_z,dop_zero_pts
    audit_csv_path: Optional[str] = None

    # ── Phase 2: temporal accumulation ───────────────────────────────────────
    # Default True: motion-compensated accumulation is always active in the
    # live viewer. Set False to revert to naive 5-frame concat.
    temporal_accum_en:          bool  = True
    accum_window:               int   = 16
    accum_max_centroid_jump_m:  float = 0.6
    accum_alpha:                float = 0.3

    # ── Phase 3: DBSCAN single-subject clustering ─────────────────────────────
    # Applied to the motion-compensated accumulated cloud from Phase 2.
    # Finds the dominant body cluster, eliminating ghost outliers that cause
    # centroid jumps.  Requires scikit-learn (pip install scikit-learn).
    #
    # dbscan_eps:
    #   DBSCAN neighbourhood radius (m).  0.50 m connects all points within
    #   half a shoulder-width of each other.  A wall ghost at ±3 m lateral
    #   is never within 0.50 m of the body cluster → classified as noise.
    #   0.0 = disable DBSCAN clustering (Phase 2 centroid only).
    #
    # dbscan_min_samples:
    #   Minimum points to form a core point.  3 is safe — body cluster
    #   always has >> 3 pts; a single ghost point = noise (label -1).
    dbscan_eps:                 float = 0.80   # covers max body Y-depth at far range
    dbscan_min_samples:        int   = 3
    max_cluster_radius:        float = 0.65   # reject clusters wider than this in X (m)

    # ── Phase 4: subject-centric coordinate normalisation ─────────────────────
    # Default True: body-frame panel is active when the viewer is running.
    # Requires temporal_accum_en=True and a valid DBSCAN cluster.
    subject_frame_en: bool = True

    # ── Phase 5: body envelope visualisation ─────────────────────────────────
    # Default True: PCA ellipse overlay active in body-frame panel.
    # Requires subject_frame_en=True.
    body_envelope_en: bool = True

    # ── Phase 6: micro-Doppler / gait features ───────────────────────────────
    # Default True: gait descriptor (vc, σ, limb%) shown in status bar.
    # Requires temporal_accum_en=True and a valid DBSCAN cluster.
    gait_features_en: bool  = True
    dop_limb_thr:     float = 0.30   # legacy 'absolute' mode only

    # ── Stage 4.2 / P0-3: adaptive limb classification ───────────────────────
    # 'absolute' (legacy) : is_limb = |dop| >= dop_limb_thr
    #   Saturated limb_frac at ~1.0 during radial walking because the MTI floor
    #   (0.234 m/s) sits below the 0.30 m/s threshold, collapsing the variance
    #   the cadence autocorrelation needs. Cadence returned NaN on the cleanest
    #   sessions (A_S01 limb_frac 100%, D_S01 96.2%).
    # 'relative' (default): is_limb = |dop - dop_centroid| >= dop_limb_offset
    #   Measures motion relative to the TORSO, which is the physically
    #   meaningful question, and restores the stance/swing oscillation.
    # 'adaptive' (default): offset = dop_limb_k x EMA(median |dop - centroid|).
    #   A FIXED threshold cannot work - measured on the A_S01 condition,
    #   absolute 0.30 gave limb_frac 1.0000 (cadence NaN) and a fixed relative
    #   0.35 gave 0.0109 (cadence 0.33 Hz vs 2.0 expected). Both saturated, in
    #   opposite directions. The scale must follow the subject.
    dop_limb_mode:    str   = 'adaptive'
    dop_limb_offset:  float = 0.35   # 'relative' mode only
    # k = 1.0 sets the threshold AT the median deviation, so ~50% of points
    # lie above it BY DEFINITION, for any distribution. Measured on all 12
    # sessions with k=1.0: limb% was 46.3-57.1% with scenario means A 53.1%,
    # B 55.0%, C 50.6%, D 50.0% - the statistic was measuring nothing. That is
    # a third saturation, at 0.5 rather than 0 or 1.
    # k = 1.75 places the threshold in the TAIL, where the surviving fraction
    # depends on the SHAPE of the Doppler distribution - bimodal for radial
    # walking (torso cluster + limbs at velocity extremes), near-Gaussian for
    # lateral. Simulated radial-vs-lateral separation: k=1.0 -> 0.0%,
    # k=1.5 -> 12.8%, k=1.75 -> 16.5%, k=2.0 -> 20.3%. 1.75 keeps limb% near
    # 24-41%, clear of both the 0 and 1 rails, with room to move either way.
    dop_limb_k:       float = 1.75   # multiplier on the smoothed scale
    dop_limb_scale_alpha: float = 0.15  # EMA rate; slow enough not to track gait

    # ── Phase 6: micro-Doppler spectrogram ───────────────────────────────────
    # microdoppler_en:
    #   When True, a Doppler-time spectrogram panel is shown in the viewer.
    #   Requires gait_features_en=True and temporal_accum_en=True.
    # spectrogram_frames:
    #   Number of time columns in the spectrogram (default 64 = 16s at 4fps).
    # spectrogram_v_max:
    #   Maximum Doppler velocity shown on the spectrogram axis (m/s).
    microdoppler_en:       bool  = True
    volume_en:             bool  = False

    # Which 3D volume to build when volume_en is set. Only ONE is produced per
    # frame - computing both would waste ~4 ms and 17 MB for a view that can
    # only show one at a time.
    #   "RDE"  Range x Doppler x Elevation - MEASURED. Elevation is a single
    #          interferometric angle per range-Doppler cell (see el_axis_deg).
    #   "RAE"  Range x Azimuth x Elevation - FUSED, not measured. Built as
    #          RA[:,:,None] + RE[:,None,:] (dB addition == linear outer
    #          product). RA knows range+azimuth, RE knows range+elevation;
    #          neither knows which azimuth went with which elevation, so N
    #          targets sharing ONE range bin produce N^2 blobs of which N are
    #          real. Ghosting is per-range-bin only, so targets at different
    #          ranges never ghost each other.
    volume_type:           str   = "RDE"    # "RDE" | "RAE"
    volume_percentile:     float = 97.5
    volume_max_voxels:     int   = 4000
    spectrogram_frames:    int   = 64
    spectrogram_v_max:     float = 1.5

    # ── Display / colour scale ────────────────────────────────────────────────
    # use_fixed_db = False  → adaptive per-frame (RECOMMENDED — always shows
    #   the strongest moving return in red, noise floor in dark blue)
    # use_fixed_db = True   → fixed limits (good for multi-frame comparison)
    use_fixed_db:  bool  = False
    db_vmin:       float = 55.0   # fixed-mode minimum (dB)
    db_vmax:       float = 90.0   # fixed-mode maximum (dB)
    db_span:       float = 24.0   # adaptive window width (dB)
    cmap_pct_high: float = 99.7   # percentile for adaptive vmax
    ra_display_snr_db: float = 4.0
    ra_display_floor_db: float = 6.0
    pc2d_dynamic_db: bool = True
    pc2d_db_vmin: Optional[float] = None
    pc2d_db_vmax: Optional[float] = None
    pc2d_db_span: float = 18.0
    pc2d_cmap_pct_low: float = 12.0
    pc2d_cmap_pct_high: float = 99.5
    point_cloud_max_points: int = 50000
    pc2d_ranges_to_show: int = 40
    pc2d_points_per_range: int = 8
    pc2d_min_points: int = 320

    # ── Colormap ──────────────────────────────────────────────────────────────
    cmap_pct_low:  float = 8.0    # adaptive floor percentile to suppress noise

    # ── Save outputs ──────────────────────────────────────────────────────────
    # Defaults are None — auto-save is DISABLED unless the user explicitly
    # passes --save-drea or --save-xyz on the command line.
    # The previous hardcoded TI PostProc paths caused an attempted filesystem
    # write on every process_frame() call on every machine, regardless of
    # whether the user requested it.  Set via CLI: --save-drea FILE --save-xyz FILE
    save_drea_path: Optional[str] = None
    save_xyz_path:  Optional[str] = None
    # =========================================================================
    # Derived properties
    # =========================================================================

    @property
    def num_rx(self) -> int:
        return self.num_rx_per_dev * self.num_devices

    @property
    def chirps_per_frame(self) -> int:
        return self.num_tx * self.num_loops

    @property
    def bytes_per_frame(self) -> int:
        return self.chirps_per_frame * self.num_rx * self.num_samples * 2 * 2

    @property
    def lambda_m(self) -> float:
        return C / (self.start_freq_ghz * 1e9)

    @property
    def range_res_m(self) -> float:
        bw = self.slope_mhz_us * 1e12 * self.num_samples / (self.sample_rate_ksps * 1e3)
        return C / (2.0 * bw)

    @property
    def n_range(self) -> int:
        return self.range_fft_size or self.num_samples

    @property
    def range_axis(self) -> np.ndarray:
        return np.arange(self.n_range) * self.range_res_m

    @property
    def tdm_period_s(self) -> float:
        return self.num_tx * (self.idle_time_us + self.ramp_end_us) * 1e-6

    @property
    def dop_max(self) -> float:
        """Max unambiguous Doppler speed (m/s) — calculated from waveform."""
        return self.lambda_m / (4.0 * self.tdm_period_s)

    @property
    def dop_res(self) -> float:
        """Doppler resolution (m/s) — calculated from waveform."""
        return self.lambda_m / (2.0 * self.num_loops * self.tdm_period_s)

    @property
    def doppler_axis(self) -> np.ndarray:
        """
        Doppler axis in m/s, zero-centred.
        Limits ±dop_max are physically derived from the waveform config:
          dop_max = λ / (4 × N_TX × T_chirp)
        """
        n = self.doppler_fft_size or self.num_loops
        return np.linspace(-self.dop_max, self.dop_max, n, endpoint=False)

    @property
    def az_axis_deg(self) -> np.ndarray:
        sin_v = np.linspace(-1.0, 1.0, self.az_fft_size, endpoint=False)
        return np.degrees(np.arcsin(np.clip(sin_v, -1, 1)))

    @property
    def el_axis_deg(self) -> np.ndarray:
        lim = min(1.0 / (2.0 * self.d_el_lambda), 1.0)
        sin_v = np.linspace(-lim, lim, self.el_fft_size, endpoint=False)
        return np.degrees(np.arcsin(np.clip(sin_v, -1, 1)))

    def build_channel_table(self):
        from collections import Counter
        ch_az, ch_el = [], []
        for t in range(self.num_tx):
            for r in range(self.num_rx):
                ch = dict(az=self.tx_az[t]+self.rx_az[r], tx=t, rx=r,
                          cal_idx=t*self.num_rx+r)
                (ch_el if self.tx_el[t] else ch_az).append(ch)
        ch_az.sort(key=lambda c: c["az"])
        ch_el.sort(key=lambda c: c["az"])
        pos_to_idx: dict = {}
        for i, c in enumerate(ch_az):
            pos_to_idx.setdefault(c["az"], []).append(i)
        overlaps = [tuple(v[:2]) for v in pos_to_idx.values() if len(v) > 1]
        return ch_az, ch_el, overlaps

    def print_summary(self):
        fw_thr_str = (f"  fw_threshold={self.cfar_threshold_db:.1f} dB"
                      if self.cfar_threshold_db > 0 else "")
        print(f"  CFAR cfg    : guard={self.cfar_guard} train={self.cfar_train}  "
              f"fa={self.cfar_fa_rate:g} margin={self.cfar_margin_db:g} dB  "
              f"nms={self.nms_size}{fw_thr_str}")
        print(f"  Filters     : clutter={'ON' if self.clutter_removal else 'OFF'}  "
              f"zeroDC=±{self.zero_dc_bins}  "
              f"sidelobes={'ON' if self.apply_sidelobe_filters else 'OFF'}")
        print("=" * 62)
        print(f"  AWR2243-2X  |  {self.num_tx}TX / {self.num_rx}RX  |  "
              f"{self.chirps_per_frame} chirps/frame")
        print(f"  Range res   : {self.range_res_m*100:.2f} cm   "
              f"Max range : {self.range_axis[-1]:.1f} m")
        print(f"  Doppler max : ±{self.dop_max:.3f} m/s  "
              f"(calc. from waveform)")
        print(f"  Doppler res : {self.dop_res:.4f} m/s")
        print(f"  dB scale    : {'fixed' if self.use_fixed_db else 'adaptive'}  "
              f"({'%g–%g' % (self.db_vmin,self.db_vmax) if self.use_fixed_db else 'span=%g dB' % self.db_span})")
        print(f"  AoA method  : {self.aoa_method.upper()}  "
              f"CFAR: {self.cfar_method.upper()}  "
              f"Moving gate: {'ON' if self.moving_objects_only else 'OFF'}")
        print("=" * 62)


# ---------------------------------------------------------------------------
# .cfg parser
# ---------------------------------------------------------------------------

def _tok(text: str, kw: str):
    for raw in text.splitlines():
        line = raw.split('%')[0].strip()
        if line and line.split()[0].lower() == kw.lower():
            return [float(x) for x in line.split()[1:]]
    return None


def parse_cfg_file(cfg_path: str) -> RadarConfig:
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(cfg_path)
    with open(cfg_path, 'r', errors='replace') as f:
        text = f.read()
    cfg = RadarConfig()
    cfg.calib_cfg_path = cfg_path

    # ── Core waveform parameters (original parser) ───────────────────────────
    prof = _tok(text, "profileCfg")
    frm  = _tok(text, "frameCfg")
    buf  = _tok(text, "adcbufCfg")
    if not prof or len(prof) < 11: raise ValueError("profileCfg missing")
    if not frm  or len(frm)  < 3:  raise ValueError("frameCfg missing")
    if not buf  or len(buf)  < 3:  raise ValueError("adcbufCfg missing")
    cfg.start_freq_ghz   = float(prof[1])
    cfg.idle_time_us     = float(prof[2])
    cfg.ramp_end_us      = float(prof[4])
    cfg.slope_mhz_us     = float(prof[7])
    cfg.num_samples      = int(prof[9])
    cfg.sample_rate_ksps = float(prof[10])
    cfg.num_tx           = int(frm[1]) - int(frm[0]) + 1
    cfg.num_loops        = int(frm[2])
    cfg.iq_swap          = bool(int(buf[2]))
    cfg.range_fft_size   = _next_pow2(cfg.num_samples)
    cfg.doppler_fft_size = _next_pow2(cfg.num_loops)
    print(f"[CFG] {os.path.basename(cfg_path)}: "
          f"slope={cfg.slope_mhz_us:.3f} MHz/µs  fs={cfg.sample_rate_ksps:.0f} ksps  "
          f"N={cfg.num_samples}  loops={cfg.num_loops}  "
          f"dop_max=±{cfg.dop_max:.3f} m/s")

    # ── C4: Extended parsing ─────────────────────────────────────────────────
    # Parse cfarCfg, compressionCfg, channelCfg, chirpCfg.
    # These were previously invisible to the pipeline — it always used
    # RadarConfig defaults regardless of what the cfg file specified.
    # Mismatches between cfg file and pipeline defaults are now flagged
    # explicitly so the operator knows the pipeline is not matching firmware.
    #
    # IMPORTANT: cfarCfg threshold_dB is NOT mapped to cfar_margin_db.
    # The firmware uses a fixed dB threshold; the pipeline uses FA-rate
    # thresholding + an additive margin.  These have different semantics.
    # threshold_dB is stored as cfar_threshold_db (read-only reference)
    # and printed for comparison only.  To match firmware CFAR behaviour,
    # use --cfar caso --cfar-guard 0 0 --cfar-train 16 16 on the CLI.

    _parse_issues = []   # collect non-fatal mismatches for summary

    # ── cfarCfg (two lines: procDirection=1 Doppler, procDirection=0 Range) ──
    # TI syntax: cfarCfg <subframe> <procDir> <mode> <noiseWin> <guardLen>
    #            <divShift> <cyclicMode> <threshold_dB> <peakGrouping> ...
    # procDir: 0=range, 1=Doppler
    _cfar_range = None
    _cfar_dop   = None
    for raw in text.splitlines():
        line = raw.split('%')[0].strip()
        if not line:
            continue
        parts = line.split()
        if parts[0].lower() != 'cfarcfg' or len(parts) < 9:
            continue
        try:
            proc_dir = int(float(parts[2]))
            mode     = int(float(parts[3]))
            noise_win= int(float(parts[4]))
            guard_len= int(float(parts[5]))
            thr_db   = float(parts[8])
        except (ValueError, IndexError):
            continue
        if proc_dir == 0:
            _cfar_range = (mode, noise_win, guard_len, thr_db)
        elif proc_dir == 1:
            _cfar_dop   = (mode, noise_win, guard_len, thr_db)

    print(f"[CFG-PARSE] cfarCfg:")
    _cfar_method_map = {0: "ca", 1: "cago", 2: "ca", 3: "caso", 4: "os"}

    if _cfar_range:
        mode, nw, gl, thr = _cfar_range
        fw_method = _cfar_method_map.get(mode, f"mode{mode}")
        print(f"  range  : method={fw_method}(mode={mode})  "
              f"noiseWin={nw}  guardLen={gl}  threshold={thr:.1f} dB")
        # Store firmware threshold for reference — NOT applied to cfar_margin_db
        cfg.cfar_threshold_db = thr
        if fw_method != cfg.cfar_method:
            _parse_issues.append(
                f"cfarCfg range method={fw_method} but pipeline "
                f"cfar_method={cfg.cfar_method!r} — pass --cfar {fw_method} "
                f"to match firmware")
    else:
        print(f"  range  : not found in cfg")

    if _cfar_dop:
        mode, nw, gl, thr = _cfar_dop
        fw_method = _cfar_method_map.get(mode, f"mode{mode}")
        print(f"  doppler: method={fw_method}(mode={mode})  "
              f"noiseWin={nw}  guardLen={gl}  threshold={thr:.1f} dB")
    else:
        print(f"  doppler: not found in cfg")

    # Guard and train hints use firmware values for BOTH dimensions together.
    # This avoids a mixed hint (e.g. --cfar-guard 0 6) that would only partially
    # match firmware.  Hints are only generated when both range and Doppler are
    # parsed so the full two-value argument is always correct.
    _fw_gl_r  = _cfar_range[2] if _cfar_range else None
    _fw_gl_d  = _cfar_dop[2]   if _cfar_dop   else None
    _fw_nw_r  = _cfar_range[1] if _cfar_range else None
    _fw_nw_d  = _cfar_dop[1]   if _cfar_dop   else None

    if _fw_gl_r is not None and _fw_gl_d is not None:
        if _fw_gl_r != cfg.cfar_guard[0] or _fw_gl_d != cfg.cfar_guard[1]:
            _parse_issues.append(
                f"cfarCfg guardLen: range={_fw_gl_r}, doppler={_fw_gl_d} but "
                f"pipeline cfar_guard={cfg.cfar_guard} — "
                f"pass --cfar-guard {_fw_gl_r} {_fw_gl_d} to match firmware")

    if _fw_nw_r is not None and _fw_nw_d is not None:
        if _fw_nw_r != cfg.cfar_train[0] or _fw_nw_d != cfg.cfar_train[1]:
            _parse_issues.append(
                f"cfarCfg noiseWin: range={_fw_nw_r}, doppler={_fw_nw_d} but "
                f"pipeline cfar_train={cfg.cfar_train} — "
                f"pass --cfar-train {_fw_nw_r} {_fw_nw_d} to match firmware")

    print(f"  pipeline: method={cfg.cfar_method!r}  "
          f"guard={cfg.cfar_guard}  train={cfg.cfar_train}  "
          f"fa_rate={cfg.cfar_fa_rate:g}  margin_db={cfg.cfar_margin_db:.1f}")
    print(f"  NOTE: firmware uses fixed threshold_dB; pipeline uses FA-rate "
          f"thresholding + margin_db — semantics differ, no auto-override applied")

    # ── compressionCfg ────────────────────────────────────────────────────────
    # TI syntax: compressionCfg <subframe> <enabled> <method> <ratio> <bitWidth>
    _comp = _tok(text, "compressionCfg")
    print(f"[CFG-PARSE] compressionCfg:")
    if _comp and len(_comp) >= 2:
        comp_enabled = int(_comp[1])
        comp_method  = int(_comp[2]) if len(_comp) > 2 else 0
        comp_ratio   = float(_comp[3]) if len(_comp) > 3 else 0.0
        comp_bits    = int(_comp[4]) if len(_comp) > 4 else 0
        if comp_enabled:
            # The AM273x cascade firmware has no compression path: the .cfg is
            # not sent to the device and raw ADC always reaches LVDS
            # (firmware investigation 2026-10-10). Only the cfg text is wrong.
            print(f"  enabled=1 in the cfg (method={comp_method} ratio={comp_ratio:.2f} "
                  f"bitWidth={comp_bits}) - IGNORED: the cascade firmware has no "
                  f"compression; data is raw ADC. Fix the cfg text to compressionCfg -1 0 ...")
            _parse_issues.append(
                "compressionCfg enabled=1 in the cfg, but the cascade firmware never "
                "compresses (raw ADC) - the cfg is out of date.")
        else:
            print(f"  disabled (enabled=0)  - raw ADC  [OK]")
    else:
        print(f"  not found — compression state unknown (assuming disabled)")

    # ── channelCfg ────────────────────────────────────────────────────────────
    # TI syntax: channelCfg <rxEn> <txEn> <cascade> <rxEn_dev2> <txEn_dev2>
    # rxEn is a bitmask: bit count gives num_rx_per_dev
    # txEn is a bitmask: bit count gives num_tx_per_dev
    _chan = _tok(text, "channelCfg")
    print(f"[CFG-PARSE] channelCfg:")
    if _chan and len(_chan) >= 2:
        rx_en = int(_chan[0])
        tx_en = int(_chan[1])
        cascade = int(_chan[2]) if len(_chan) > 2 else 0
        n_rx_per_dev = bin(rx_en).count('1')
        n_tx_per_dev = bin(tx_en).count('1')
        n_devices    = 1 + cascade
        n_rx_total   = n_rx_per_dev * n_devices
        n_tx_total   = n_tx_per_dev * n_devices
        print(f"  rxEn=0b{rx_en:04b}({rx_en})  txEn=0b{tx_en:03b}({tx_en})  "
              f"cascade={cascade}")
        print(f"  → {n_rx_per_dev} RX/device × {n_devices} device(s) = "
              f"{n_rx_total} RX total")
        print(f"  → {n_tx_per_dev} TX/device × {n_devices} device(s) = "
              f"{n_tx_total} TX total")
        # Validate against cfg.num_tx derived from frameCfg
        if n_tx_total != cfg.num_tx:
            _parse_issues.append(
                f"channelCfg implies {n_tx_total} TX but frameCfg implies "
                f"num_tx={cfg.num_tx} — geometry mismatch, check cfg file")
        else:
            print(f"  TX count consistent with frameCfg num_tx={cfg.num_tx}  [OK]")
        # Update num_rx_per_dev if it can be inferred (cascade=1 → 2 devices)
        if cascade >= 1:
            cfg.num_devices = n_devices
    else:
        print(f"  not found")

    # ── chirpCfg ─────────────────────────────────────────────────────────────
    # Count distinct chirp configs and validate against num_tx
    _chirp_count = sum(
        1 for raw in text.splitlines()
        if raw.split('%')[0].strip().lower().startswith('chirpcfg')
    )
    print(f"[CFG-PARSE] chirpCfg: {_chirp_count} chirp definition(s) found")
    if _chirp_count != cfg.num_tx:
        _parse_issues.append(
            f"chirpCfg count={_chirp_count} does not match num_tx={cfg.num_tx} "
            f"— TDM chirp-to-TX assignment may be wrong")
    else:
        print(f"  chirp count matches num_tx={cfg.num_tx}  [OK]")

    # ── Mismatch summary ──────────────────────────────────────────────────────
    if _parse_issues:
        print(f"[CFG-PARSE] {len(_parse_issues)} pipeline/firmware mismatch(es):")
        for i, issue in enumerate(_parse_issues, 1):
            print(f"  [{i}] {issue}")
    else:
        print(f"[CFG-PARSE] no pipeline/firmware mismatches detected")

    return cfg


def apply_raw_iq_order_override(cfg: RadarConfig, mode: str) -> RadarConfig:
    """
    Match capture/test.py raw IQ override semantics.

    "cfg"     -- keep iq_swap as parsed from adcbufCfg SampleSwap
    "i-first" -- raw pairs are [I, Q]
    "q-first" -- raw pairs are [Q, I]
    """
    mode = str(mode).lower()
    if mode == "cfg":
        return cfg
    if mode == "i-first":
        cfg.iq_swap = False
        return cfg
    if mode == "q-first":
        cfg.iq_swap = True
        return cfg
    raise ValueError(f"Unknown raw IQ order override: {mode!r}")


def parse_calibration(cfg_path: str, num_tx: int = 6, num_rx: int = 8,
                      enabled: bool = True) -> np.ndarray:
    identity = np.ones((num_tx, num_rx), dtype=np.complex64)

    if not enabled:
        print("[CALIB] Calibration DISABLED — identity matrix applied to all channels")
        return identity

    if not os.path.isfile(cfg_path):
        print(f"[CALIB] WARNING: cfg not found ({cfg_path}) — identity matrix applied")
        return identity

    with open(cfg_path, 'r', errors='replace') as f:
        text = f.read()

    flat = []
    found = []
    for cmd in ('antennacalibparams1', 'antennacalibparams2', 'antennacalibparams3'):
        t = _tok(text, cmd)
        if t and len(t) == 32:
            flat.extend(t)
            found.append(cmd)
        else:
            flat.extend([1.0, 0.0] * 16)
            if t is None:
                print(f"[CALIB] WARNING: {cmd} not found in cfg — identity used for those 16 channels")
            else:
                print(f"[CALIB] WARNING: {cmd} has {len(t)} values (expected 32) — identity used for those 16 channels")

    if len(flat) != 96:
        print(f"[CALIB] ERROR: expected 96 calibration floats, got {len(flat)} — identity matrix applied")
        return identity

    # ── Build correction matrix ───────────────────────────────────────────────
    # FORMAT: antennaCalibParams values are [Re, Im] pairs (rectangular complex).
    # Confirmed by C5 sanity check: 24/48 channels had negative first values,
    # which is impossible for magnitudes.  Re + j*Im is the correct interpretation.
    # Previous code used magnitude * exp(j * deg2rad(phase)) — WRONG for this data.
    corr = identity.copy()
    for k in range(num_tx * num_rx):
        tx, rx = k // num_rx, k % num_rx
        corr[tx, rx] = float(flat[2*k]) + 1j * float(flat[2*k+1])

    # ── C5: Sanity check — [Re, Im] plausibility ─────────────────────────────
    # Now that format is confirmed as [Re, Im], the correct checks are:
    #   |Re| and |Im|   : both in [-5.0, 5.0]  — component bound
    #   |Re + j*Im|     : in [0.05, 5.0]        — correction gain bound
    # A correction magnitude below 0.05 effectively zeros the channel.
    # A correction magnitude above 5.0 applies > 14 dB gain — implausible trim.

    re_vals  = np.array([float(flat[2*k])   for k in range(num_tx * num_rx)])
    im_vals  = np.array([float(flat[2*k+1]) for k in range(num_tx * num_rx)])
    comp_mag = np.abs(re_vals + 1j * im_vals)

    COMP_MIN, COMP_MAX = -5.0, 5.0
    MAG_MIN,  MAG_MAX  =  0.05, 5.0

    bad_re  = np.where((re_vals < COMP_MIN) | (re_vals > COMP_MAX))[0]
    bad_im  = np.where((im_vals < COMP_MIN) | (im_vals > COMP_MAX))[0]
    bad_mag = np.where((comp_mag < MAG_MIN) | (comp_mag > MAG_MAX))[0]

    n_channels = num_tx * num_rx
    print(f"[CALIB] Loaded from: {os.path.basename(cfg_path)}  "
          f"({len(found)}/3 antennaCalibParams blocks found)")
    print(f"[CALIB] Format: [Re, Im] rectangular complex  "
          f"(confirmed — negative Re values rule out [magnitude, phase_deg])")
    print(f"[CALIB] Channels: {n_channels}  "
          f"Re=[{re_vals.min():.4f}, {re_vals.max():.4f}]  "
          f"Im=[{im_vals.min():.4f}, {im_vals.max():.4f}]  "
          f"|corr|=[{comp_mag.min():.4f}, {comp_mag.max():.4f}]")

    if len(bad_re) > 0 or len(bad_im) > 0:
        print(f"[CALIB] WARNING: {len(bad_re)} Re and {len(bad_im)} Im component(s) "
              f"outside [{COMP_MIN}, {COMP_MAX}] — likely a parse error")
    else:
        print(f"[CALIB] Component bounds check: OK  "
              f"(all Re and Im values in [{COMP_MIN}, {COMP_MAX}])")

    if len(bad_mag) > 0:
        offenders = [(k // num_rx, k % num_rx, comp_mag[k]) for k in bad_mag]
        print(f"[CALIB] WARNING: {len(bad_mag)} channel(s) have implausible correction "
              f"magnitude (outside [{MAG_MIN}, {MAG_MAX}]):")
        for tx, rx, m in offenders[:8]:
            print(f"         TX{tx} RX{rx}: |corr|={m:.6f}  "
                  f"Re={re_vals[tx*num_rx+rx]:.6f}  Im={im_vals[tx*num_rx+rx]:.6f}")
        if len(offenders) > 8:
            print(f"         ... and {len(offenders)-8} more")
    else:
        print(f"[CALIB] Correction magnitude check: OK  "
              f"(all {n_channels} channels |corr| in [{MAG_MIN}, {MAG_MAX}])")

    return corr


def _next_pow2(n: int) -> int:
    p = 1
    while p < n: p <<= 1
    return p


def shortrange_config() -> RadarConfig:
    return RadarConfig()
