"""
processing/accumulate.py  —  Phase 2 + 3: Temporal accumulation with motion
compensation and DBSCAN single-subject clustering.

Phase 2 — TemporalAccumulator:
    Motion-compensated ring buffer that co-registers each frame's cloud to
    the current frame's estimated subject position.  See class docstring.

Phase 3 — DBSCAN clustering (added to TemporalAccumulator.update()):
    After accumulation, DBSCAN is run on the merged X–Y cloud to find the
    dominant body cluster.  The cluster centroid replaces the raw median
    centroid, eliminating large jumps caused by single ghost points.

    Parameters:
        dbscan_eps (float): neighbourhood radius (m). Default 0.50 m
            connects all points within a shoulder-width of each other.
            A wall ghost at x=+3 m is never connected to the body cluster.
        dbscan_min_samples (int): minimum pts to form a core point.
            Default 3 — body cluster always has >> 3 pts; single ghosts
            are classified as noise (label=-1).

    If sklearn is unavailable the module falls back to the Phase 2 raw
    median centroid with a warning printed once.
"""
from __future__ import annotations
import math
import warnings
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# Try to import sklearn DBSCAN — graceful fallback if unavailable
try:
    from sklearn.cluster import DBSCAN as _DBSCAN
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
    warnings.warn(
        "[accumulate] sklearn not found — Phase 3 DBSCAN clustering disabled. "
        "Install with: pip install scikit-learn",
        ImportWarning, stacklevel=2)


@dataclass
class AccumFrame:
    """One frame's contribution to the accumulation window."""
    cloud:      np.ndarray   # [N, 7] xyz_cloud (copy)
    centroid_y: float        # median y at time of capture
    centroid_x: float        # median x at time of capture
    frame_idx:  int          # global frame counter


@dataclass
class SubjectCluster:
    """One DBSCAN cluster, retained with a stable identity (Stage 3 / P0-2).

    Previously _cluster() kept ONLY the largest cluster:

        best_label = max(unique, key=lambda l: (labels == l).sum())
        body_pts   = pts[labels == best_label]

    DBSCAN already separates multiple people correctly - the segmentation is
    in `labels`. Discarding every other cluster destroyed person B one call
    before the display layer, which is why the viewer could not distinguish
    one subject from two, and why the tracked oval SNAPPED between people
    when they separated (whichever had more points won).

    track_id is assigned by nearest-centroid association across frames, so a
    subject keeps the same id - and therefore the same colour and trail - for
    as long as it is tracked.
    """
    cloud:        np.ndarray    # [K, 7] points in this cluster
    centroid_x:   float
    centroid_y:   float
    n_pts:        int
    radius_x:     float
    radius_y:     float
    track_id:     int  = -1     # stable across frames; -1 = unassigned
    age:          int  = 0      # consecutive frames matched
    coast:        int  = 0      # consecutive frames unmatched (predicted only)
    wide:         bool = False  # radius_x exceeded max_cluster_radius


@dataclass
class ClusterResult:
    """Phase 3 DBSCAN output.

    centroid_*/n_pts/radius_* describe the PRIMARY (largest) cluster and are
    unchanged, so every existing caller keeps working. Stage 3 adds `subjects`
    carrying ALL clusters with stable ids.
    """
    cloud:          np.ndarray   # [K, 7] pts belonging to body cluster
    centroid_x:     float        # cluster centroid x (m)
    centroid_y:     float        # cluster centroid y (m)
    n_pts:          int          # pts in body cluster
    n_noise:        int          # pts classified as noise
    radius_x:       float        # half-width of cluster in x (m)
    radius_y:       float        # half-width of cluster in y (m)
    cluster_found:  bool         # False if DBSCAN found no valid cluster
    # Audit A-2: True when a cluster WAS found but was discarded because
    # radius_x exceeded max_cluster_radius. Distinguishes "no cluster at all"
    # from "cluster rejected as implausibly wide" — previously indistinguishable,
    # so wide-cluster frame drops were invisible in frames.csv.
    wide_rejected:  bool = False
    # Stage 3 / P0-2: every cluster DBSCAN found, each with a stable track_id.
    # The primary cluster above is subjects[primary_idx] when subjects is
    # non-empty. Defaults keep older construction sites valid.
    subjects:       list = field(default_factory=list)
    primary_idx:    int  = 0


@dataclass
class GaitFeatures:
    """
    Phase 6: per-frame micro-Doppler / gait feature descriptor.

    Computed from the DBSCAN body cluster's Doppler column (col 4).
    All values are in m/s unless otherwise noted.

    Fields
    ------
    dop_centroid   : Doppler centroid of the cluster (power-weighted mean).
                     Positive = subject moving away; negative = approaching.
                     Equals the trunk/body velocity projected onto the radial axis.
    dop_spread     : Std of cluster Doppler values.
                     Higher spread = more limb motion in this frame.
    dop_max        : Maximum |Doppler| in the cluster (fastest-moving scatter).
    n_torso_pts    : Cluster pts with |dop| < dop_limb_thr (≈ trunk returns).
    n_limb_pts     : Cluster pts with |dop| >= dop_limb_thr (≈ limb returns).
    limb_frac      : n_limb_pts / cluster_pts  (0.0–1.0, gait activity index).
    dop_limb_thr   : Threshold used (m/s), stored for reproducibility.
    valid          : False if the cluster had fewer than 3 pts (feature unreliable).
    """
    dop_centroid:  float
    dop_spread:    float
    dop_max:       float
    n_torso_pts:   int
    n_limb_pts:    int
    limb_frac:     float
    dop_limb_thr:  float
    valid:         bool


@dataclass
class AccumResult:
    """Output of TemporalAccumulator.update()."""
    cloud:          np.ndarray           # [M, 7] motion-compensated accumulated cloud
    centroid_x:     float                # smoothed centroid x — Phase 3: from cluster
    centroid_y:     float                # smoothed centroid y — Phase 3: from cluster
    velocity_y:     float                # estimated radial velocity (m/s)
    n_frames:       int                  # frames in accumulation window
    compensated:    bool                 # True if motion compensation was applied
    cluster:        Optional[ClusterResult] = None   # Phase 3 DBSCAN result
    subject_cloud:  Optional[np.ndarray] = None      # Phase 4: body-local (dx,dy,dz,…)
    gait:           Optional[GaitFeatures] = None    # Phase 6: micro-Doppler features
    # Motion-compensation residual diagnostics (audit Stage 4 input).
    # comp_residual_x/y = spread of the per-frame centroids AFTER
    # compensation. If compensation were perfect these would be ~0; the
    # observed value is the trail length DBSCAN chains over, and is what
    # actually drives wide clusters. Without this, eps cannot be chosen
    # on evidence.
    comp_residual_x: float = 0.0
    comp_residual_y: float = 0.0
    comp_step_x:     float = 0.0     # max consecutive compensated step
    # Lag-corrected CURRENT position estimate (P0-6 follow-up).
    # centroid_x/y are derived from the EMA, which trails the true position
    # by (1-alpha)/alpha frames — 2.33 frames at alpha=0.3, i.e. 0.58 m at a
    # measured 0.25 m/frame. Feeding that stale value to the subject gate as
    # a "current position" consumed most of the 0.90 m radius budget and
    # caused a 34.6% bypass rate (r=+0.855 against subject speed).
    # predicted_x/y add velocity x lag_time to undo that.
    velocity_x:      float = 0.0
    predicted_x:     float = 0.0
    predicted_y:     float = 0.0
    # Stage 3.3: lag-corrected position of EVERY confirmed track, for the
    # multi-track subject gate. Single-subject scenes yield one entry, so
    # callers can pass this list unconditionally.
    track_hints:     list = field(default_factory=list)
    # Multi-subject state ACTUALLY USED for this frame's compensation
    # decision (one frame behind n_subjects - see _build_result note).
    multi_used:      bool = False
    # Raw DBSCAN cluster count BEFORE the age>=2 confirmation filter.
    # If this is routinely >1 while n_subjects stays 1, tracks are being
    # created and lost before they can confirm - i.e. association is the
    # bottleneck, not detection.
    n_raw_clusters:  int = 0


def subject_frame(accum_result: AccumResult) -> Optional[np.ndarray]:
    """
    Phase 4: Subject-centric coordinate normalisation.

    Translates the accumulated cluster cloud so the body centroid sits at
    the origin (0, 0, 0).  Returns a [K, 7] array where columns 0–2 are
    body-local offsets:

        dx = x - cluster_cx   (lateral offset from body centre, m)
        dy = y - cluster_cy   (range offset from body centre, m)
        dz = z - cluster_cz   (elevation offset from body centre, m)

    Columns 3–6 (pwr_db, dop_mps, rng_m, az_deg) are carried through
    unchanged so the body-frame cloud can still be coloured by Doppler,
    etc.

    The plan says:
      - Subtract track centroid from each accumulated point.
      - Do NOT rotate by heading yet — walking toward/away ⇒ heading is
        ~radial; rotation adds noise for a single-sequence dataset.
      - Keep dx lateral, dy radial — no axis remapping needed.

    Returns None if no valid cluster is available (subject_frame_en=False
    or Phase 3 disabled).

    Acceptance criterion M5: normalised lateral spread (std of dx) should
    converge toward ~0.15–0.25 m in active segments and should NOT scale
    with the subject's range — it will be approximately the same whether
    the subject is at 3 m or 10 m.

    Parameters
    ----------
    accum_result : AccumResult
        Output of TemporalAccumulator.update().  Must have a valid .cluster
        with cluster_found=True.

    Returns
    -------
    np.ndarray [K, 7] in body-local coordinates, or None.
    """
    if accum_result.cluster is None or not accum_result.cluster.cluster_found:
        return None

    body_pts = accum_result.cluster.cloud   # [K, 7] — body cluster only
    if body_pts is None or len(body_pts) == 0:
        return None

    cx = accum_result.cluster.centroid_x
    cy = accum_result.cluster.centroid_y
    # Elevation centroid: median z of cluster pts (z = col 2)
    cz = float(np.median(body_pts[:, 2]))

    normalised = body_pts.copy()
    normalised[:, 0] = body_pts[:, 0] - cx   # dx
    normalised[:, 1] = body_pts[:, 1] - cy   # dy
    normalised[:, 2] = body_pts[:, 2] - cz   # dz
    # cols 3-6 unchanged: pwr_db, dop_mps, rng_m, az_deg

    return normalised


def fit_body_ellipse(sf_cloud: np.ndarray,
                     max_dx: float = 0.50,
                     max_dy: float = 0.50,
                     n_pts:  int   = 64) -> Optional[dict]:
    """
    Phase 5: Fit a PCA ellipse to the (dx, dy) body-frame cloud.

    Uses Principal Component Analysis on the (dx, dy) columns of the
    subject-frame cloud to find the orientation and extent of the body
    footprint.  The 2-sigma PCA ellipse is clamped to human body bounds
    before being returned so that ghost-driven stretching is suppressed.

    Plan says: clamp envelope size to plausible human bounds to avoid
    ghost-driven stretching.

    Parameters
    ----------
    sf_cloud : np.ndarray [K, 7]
        Subject-centric cloud from ``subject_frame()``.
        Columns 0–1 are (dx, dy) in metres.
    max_dx : float
        Maximum allowed half-width of the ellipse in dx (default 0.50 m).
    max_dy : float
        Maximum allowed half-height in dy (default 0.50 m).
    n_pts : int
        Number of points on the ellipse outline (default 64).

    Returns
    -------
    dict with keys: cx, cy, rx, ry, angle_deg, x (outline), y (outline)
    or None if the cloud is too small to fit.
    """
    if sf_cloud is None or len(sf_cloud) < 4:
        return None

    dx = sf_cloud[:, 0].astype(float)
    dy = sf_cloud[:, 1].astype(float)

    cx = float(np.median(dx))
    cy = float(np.median(dy))

    # Robust outlier removal: keep pts within 2.5σ of the median distance
    dx_c = dx - cx;  dy_c = dy - cy
    dist  = np.sqrt(dx_c**2 + dy_c**2)
    thr   = float(np.std(dist)) * 2.5 + 0.05
    keep  = dist < thr
    if keep.sum() < 4:
        keep = np.ones(len(dx), dtype=bool)

    pts = np.column_stack([dx[keep], dy[keep]])
    cov = np.cov(pts.T)
    if cov.ndim < 2 or np.any(~np.isfinite(cov)):
        return None

    # PCA: eigenvectors = principal axes, sqrt(eigenvalues) = std along axis
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-6)

    # 2-sigma extents: larger eigenvalue → major axis
    rx_pca = 2.0 * float(np.sqrt(eigvals[1]))
    ry_pca = 2.0 * float(np.sqrt(eigvals[0]))

    # Clamp to human body bounds (plan: "clamp to plausible human bounds")
    rx = float(np.clip(rx_pca, 0.05, max_dx))
    ry = float(np.clip(ry_pca, 0.05, max_dy))

    # Rotation angle of the major axis (in radians)
    angle_rad = float(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    angle_deg = float(np.degrees(angle_rad))

    # Generate ellipse outline (64 pts)
    t      = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    ex     = rx * np.cos(t);   ey = ry * np.sin(t)
    cos_a  = np.cos(angle_rad); sin_a = np.sin(angle_rad)
    x_out  = cx + cos_a * ex - sin_a * ey
    y_out  = cy + sin_a * ex + cos_a * ey

    return {
        'cx':        cx,
        'cy':        cy,
        'rx':        rx,
        'ry':        ry,
        'angle_deg': angle_deg,
        'x':         x_out,
        'y':         y_out,
    }


def gait_features(cluster_cloud: np.ndarray,
                  dop_limb_thr: float = 0.30,
                  dop_limb_mode: str = 'adaptive',
                  dop_limb_offset: float = 0.35,
                  dop_limb_scale: float = 0.0,
                  dop_limb_k: float = 1.0) -> GaitFeatures:
    """
    Phase 6: Extract micro-Doppler / gait features from the DBSCAN body cluster.

    Operates on the raw cluster cloud (col 4 = dop_mps) rather than the
    subject-frame cloud so that Doppler values are unchanged from the
    original RD detection.

    Parameters
    ----------
    cluster_cloud : np.ndarray [K, 7]
        Body cluster cloud from ClusterResult.cloud.
        Column 4 must be dop_mps (radial velocity, m/s).
    dop_limb_thr : float
        Doppler threshold separating trunk from limb returns (m/s).
        Trunk: |dop| < dop_limb_thr.  Limb: |dop| >= dop_limb_thr.
        Default 0.30 m/s — between trunk (~0.1–0.3 m/s radial component)
        and swinging limbs (~0.5–1.5 m/s).

    Returns
    -------
    GaitFeatures with valid=False if cluster has fewer than 3 pts.

    Physical interpretation
    -----------------------
    dop_centroid  → signed trunk velocity (radial component of walking speed)
    dop_spread    → intra-frame velocity diversity (gait activity level)
    limb_frac     → fraction of cluster pts that are limb-scale scatterers
    n_limb_pts    → absolute count of limb returns this frame

    The micro-Doppler signature of walking:
      Stance phase  : most pts near trunk Doppler, low spread, limb_frac ≈ 0.1
      Swing phase   : some pts at ±limb Doppler, higher spread, limb_frac ≈ 0.4
    Cadence can be estimated from the periodicity of limb_frac over time.
    """
    _empty = GaitFeatures(
        dop_centroid=0.0, dop_spread=0.0, dop_max=0.0,
        n_torso_pts=0, n_limb_pts=0, limb_frac=0.0,
        dop_limb_thr=dop_limb_thr, valid=False)

    if cluster_cloud is None or len(cluster_cloud) < 3:
        return _empty

    dop = cluster_cloud[:, 4].astype(float)
    pwr = cluster_cloud[:, 3].astype(float)

    # Power-weighted Doppler centroid — trunk velocity estimate
    pwr_lin  = 10.0 ** (pwr / 10.0)
    pwr_lin  = np.maximum(pwr_lin, 1e-6)
    dop_cen  = float(np.sum(dop * pwr_lin) / np.sum(pwr_lin))

    dop_std  = float(np.std(dop))
    dop_max  = float(np.max(np.abs(dop)))

    # ── Torso / limb separation (Stage 4.2 / P0-3) ───────────────────────────
    # LEGACY 'absolute' mode:  is_limb = |dop| >= dop_limb_thr (0.30 m/s)
    #   The MTI gate's minimum passed Doppler is 0.234 m/s and normal walking
    #   Doppler is 0.4-0.8 m/s, so during radial walking nearly EVERY point
    #   exceeded the threshold. limb_frac saturated at ~1.0, its variance
    #   collapsed, and _estimate_cadence() hit its own
    #   `if lf_zm.std() < 1e-4: return nan` guard - failing on precisely the
    #   sessions with the cleanest gait (A_S01 100%, D_S01 96.2%).
    #
    # 'relative' mode (default): a limb is a scatterer moving DIFFERENTLY FROM
    #   THE TORSO, which is the physically meaningful question:
    #       is_limb = |dop - dop_centroid| >= dop_limb_offset
    #   This also restores the gait OSCILLATION the absolute form could not
    #   produce: at mid-stance limb velocities sit near the torso velocity so
    #   few exceed the offset; at mid-swing they reach their extremes and many
    #   do. That oscillation is what the cadence autocorrelation consumes.
    #
    #   Predicted limb_frac from the measured per-session dop_spread:
    #       A_S01 0.77   A_S02 0.62   D_S01 0.68   outdoor 0.74   B_S02 0.39
    #   - all clear of saturation.
    _mode = str(dop_limb_mode).lower()
    if _mode == 'absolute':
        # LEGACY. Saturates at ~1.0 during radial walking (P0-3).
        is_limb  = np.abs(dop) >= dop_limb_thr
        thr_used = float(dop_limb_thr)
    elif _mode == 'relative':
        # Fixed offset from the torso. Better than absolute, but a FIXED
        # constant cannot sit inside a limb velocity distribution whose width
        # varies with walking speed and range: too large and limb_frac -> 0,
        # too small and it -> 1. Measured on the A_S01 condition:
        #   absolute 0.30 -> limb_frac 1.0000 (std 0.00000, cadence NaN)
        #   relative 0.35 -> limb_frac 0.0109 (std 0.02572, cadence 0.33 Hz)
        # Both saturated, in opposite directions.
        is_limb  = np.abs(dop - dop_cen) >= dop_limb_offset
        thr_used = float(dop_limb_offset)
    else:
        # ADAPTIVE (default). dop_limb_scale is an EMA of the per-frame median
        # |dop - dop_centroid|, supplied by the caller. Because the reference
        # is SMOOTHED across frames while the instantaneous deviation
        # oscillates with the gait cycle, limb_frac sits near 0.5 and swings
        # with stance/swing - the variation the cadence autocorrelation needs.
        # It also self-adjusts to speed, range and subject, so it needs no
        # per-session tuning.
        _scale = float(dop_limb_scale)
        if _scale <= 1e-6:                     # bootstrap: no history yet
            _scale = float(np.median(np.abs(dop - dop_cen)))
        thr_used = max(float(dop_limb_k) * _scale, 1e-3)
        is_limb  = np.abs(dop - dop_cen) >= thr_used
    n_limb       = int(is_limb.sum())
    n_torso      = len(dop) - n_limb
    limb_frac    = float(n_limb / len(dop))

    return GaitFeatures(
        dop_centroid=round(dop_cen, 4),
        dop_spread=round(dop_std, 4),
        dop_max=round(dop_max, 4),
        n_torso_pts=n_torso,
        n_limb_pts=n_limb,
        limb_frac=round(limb_frac, 4),
        dop_limb_thr=thr_used,
        valid=True,
    )


class TemporalAccumulator:
    """
    Motion-compensated temporal accumulator with optional DBSCAN clustering.

    Phase 2 (accumulation):
        Maintains a ring buffer of the last `window` frames and translates
        each historical frame's Y coordinates so that a radially walking
        subject is co-registered to the current position.

    Phase 3 (clustering, enabled when dbscan_eps > 0 and sklearn available):
        After building the accumulated cloud, DBSCAN clusters the X–Y
        coordinates.  The largest cluster is taken as the body.  Its centroid
        replaces the raw median, eliminating ghost-induced centroid jumps.

    Usage::

        acc = TemporalAccumulator(window=16, max_jump_m=0.6, alpha=0.3,
                                  frame_rate_hz=4.0,
                                  dbscan_eps=0.50, dbscan_min_samples=3)

        for xyz in frame_clouds:
            result = acc.update(xyz, frame_idx)
            # result.cluster.centroid_x/y — stable body position
            # result.cloud                — full accumulated cloud
    """

    def __init__(self,
                 window:              int   = 16,
                 max_jump_m:          float = 0.6,
                 alpha:               float = 0.3,
                 frame_rate_hz:       float = 4.0,
                 dbscan_eps:          float = 0.80,
                 dbscan_min_samples:  int   = 3,
                 max_cluster_radius:  float = 0.65,
                 subject_frame_en:    bool  = False,
                 gait_features_en:    bool  = False,
                 dop_limb_thr:        float = 0.30,
                 dop_limb_mode:       str   = 'adaptive',
                 dop_limb_offset:     float = 0.35,
                 dop_limb_k:          float = 1.0,
                 dop_limb_scale_alpha: float = 0.15):
        self._window             = int(window)
        self._max_jump           = float(max_jump_m)
        self._alpha              = float(np.clip(alpha, 0.0, 1.0))
        self._fps                = float(frame_rate_hz)
        self._eps                = float(dbscan_eps)
        self._min_samp           = int(dbscan_min_samples)
        self._max_radius         = float(max_cluster_radius)
        self._subject_frame_en   = bool(subject_frame_en)
        self._gait_features_en   = bool(gait_features_en)
        self._dop_limb_thr       = float(dop_limb_thr)
        self._dop_limb_mode      = str(dop_limb_mode)
        self._dop_limb_offset    = float(dop_limb_offset)
        self._dop_limb_k         = float(dop_limb_k)
        self._limb_scale_alpha   = float(dop_limb_scale_alpha)
        # EMA of median |dop - dop_centroid|. Slow (alpha=0.15, ~6-frame
        # memory) so it tracks the subject's speed but not the gait cycle -
        # if it followed the cycle it would cancel the very oscillation the
        # cadence estimator consumes.
        self._limb_scale: float = 0.0
        self._dbscan_warn        = False

        self._buf: deque[AccumFrame] = deque(maxlen=self._window)
        self._cx:      Optional[float] = None
        self._cy:      Optional[float] = None
        self._prev_cy: Optional[float] = None
        self._prev_cx: Optional[float] = None
        # Ghost-gate state (audit P0-5 fix).
        # _last_raw_cy is the last ACCEPTED per-frame centroid — the gate must
        # compare against this, NOT against the lagging EMA self._cy.
        self._last_raw_cy: Optional[float] = None
        self._coast: int = 0
        self._max_coast: int = 3          # frames to coast before re-seeding
        self.n_ghost_gated: int = 0       # diagnostics
        # Stage 3.2 identity tracker: {track_id: (x, y, age, coast)}
        self._tracks: dict = {}
        self._next_track_id: int = 0
        # Stage 3 finding: motion compensation is inherently SINGLE-subject.
        # The shift is (ref_cx - frame.centroid_x), and with two subjects the
        # frame centroid sits BETWEEN them - a point belonging to neither. It
        # jumps whenever either subject appears, disappears or moves relative
        # to the other, so buffered frames get shifted by different amounts and
        # ONE physical person is smeared across several x positions. DBSCAN
        # then correctly reports them as separate clusters, inflating the
        # subject count.
        # While more than one subject is tracked we therefore accumulate
        # UNCOMPENSATED: positions stay true, at the cost of the intra-body
        # coherence gain compensation provides for a single subject.
        self._multi_subject: bool = False
        self.n_reseeds: int = 0

    # ── Public API ──────────────────────────────────────────────────────────

    def update(self, xyz_cloud: np.ndarray,
               frame_idx: int = -1) -> AccumResult:
        """Ingest one frame and return accumulated + clustered result."""
        if xyz_cloud is None or len(xyz_cloud) == 0:
            return self._empty_result()

        cur_cx = float(np.median(xyz_cloud[:, 0]))
        cur_cy = float(np.median(xyz_cloud[:, 1]))

        # Initialise on first frame
        if self._cy is None:
            self._cx = cur_cx
            self._cy = cur_cy
            self._prev_cy = cur_cy
            self._prev_cx = cur_cx
            self._last_raw_cy = cur_cy
            self._coast = 0
            self._buf.append(AccumFrame(
                cloud=xyz_cloud.copy(), centroid_y=cur_cy,
                centroid_x=cur_cx, frame_idx=frame_idx))
            return self._build_result(compensated=False)

        # ── Ghost gate (audit P0-5) ─────────────────────────────────────────
        # Previous implementation:
        #     if abs(cur_cy - self._cy) > self._max_jump: return stale result
        # Two defects:
        #  (a) it compared against self._cy, the EMA, which LAGS the true
        #      position by (v/fps)*(1-alpha)/alpha. At alpha=0.3, fps=4,
        #      max_jump=0.6 the lag exceeds the gate for any speed above
        #      1.03 m/s — i.e. normal walking tripped the gate with no ghost
        #      present at all.
        #  (b) self._cy was NOT updated on the reject path, so once tripped
        #      while the subject moved away, |cur_cy - _cy| only grew and the
        #      gate fired forever. Permanent deadlock, no recovery.
        # Fix: gate against the last ACCEPTED RAW centroid (frame-to-frame
        # displacement, which is what max_jump=0.6 m -> 2.4 m/s actually
        # describes), and coast for at most _max_coast frames before
        # re-seeding so the accumulator can never lock up permanently.
        ref_raw = self._last_raw_cy if self._last_raw_cy is not None else self._cy
        if abs(cur_cy - ref_raw) > self._max_jump:
            self._coast += 1
            self.n_ghost_gated += 1
            if self._coast <= self._max_coast:
                return self._build_result(compensated=True)
            # Coast exhausted: the new position is persistent, not a ghost.
            # Re-seed onto it rather than staying frozen.
            self._cx = cur_cx
            self._cy = cur_cy
            self._prev_cy = cur_cy
            self._prev_cx = cur_cx
            self._buf.clear()
            self._coast = 0
            self.n_reseeds += 1
            self._last_raw_cy = cur_cy
            self._buf.append(AccumFrame(
                cloud=xyz_cloud.copy(), centroid_y=cur_cy,
                centroid_x=cur_cx, frame_idx=frame_idx))
            return self._build_result(compensated=False)

        self._coast = 0
        self._last_raw_cy = cur_cy

        # EMA centroid update
        self._prev_cy = self._cy
        self._prev_cx = self._cx
        self._cx = self._alpha * cur_cx + (1 - self._alpha) * self._cx
        self._cy = self._alpha * cur_cy + (1 - self._alpha) * self._cy

        self._buf.append(AccumFrame(
            cloud=xyz_cloud.copy(), centroid_y=cur_cy,
            centroid_x=cur_cx, frame_idx=frame_idx))

        return self._build_result(compensated=True)

    def reset(self) -> None:
        self._buf.clear()
        self._cx = self._cy = self._prev_cy = self._prev_cx = None
        self._last_raw_cy = None
        self._coast = 0
        self._tracks = {}
        self._next_track_id = 0
        self._multi_subject = False
        self._limb_scale = 0.0

    @property
    def n_frames(self) -> int:
        return len(self._buf)

    @property
    def centroid(self) -> tuple[float, float]:
        return (self._cx or 0.0, self._cy or 0.0)

    # ── Internal ────────────────────────────────────────────────────────────

    def _build_result(self, compensated: bool) -> AccumResult:
        if not self._buf:
            return self._empty_result()

        # Suppress compensation in multi-subject scenes (see __init__ note).
        #
        # ORDERING: this reads _multi_subject BEFORE _cluster() runs on the
        # merged cloud, so it reflects the PREVIOUS frame's subject count.
        # The dependency is genuinely circular within a frame - compensating
        # needs the subject count, counting needs a cluster, clustering needs
        # the merged cloud, merging needs the compensation decision. One frame
        # of lag is the cheap resolution; the principled fix is per-subject
        # accumulation (Appendix Q).
        #
        # We record the value ACTUALLY USED so the audit does not re-read a
        # post-hoc value and report a discrepancy that is really this lag.
        _multi_used = bool(getattr(self, '_multi_subject', False))
        if _multi_used:
            compensated = False
        ref_cy = self._cy
        # X compensation: use EMA centroid (self._cx), which is robust against
        # single-frame ghost outliers. The raw last-frame centroid
        # (self._buf[-1].centroid_x) can be a ghost position (wall return)
        # that would incorrectly shift all historical pts to the ghost location.
        # The EMA (alpha=0.3) smooths over individual bad frames.
        # Note: rx will be ~0.35-0.45m for a radially-walking subject because
        # lateral drift accumulates across 16 frames — this is physically correct,
        # not a pipeline error. Body WIDTH (0.45m) is captured per-frame by
        # az multi-peak, not by temporal accumulation across frames.
        ref_cx = self._cx
        parts  = []
        for frame in self._buf:
            pts = frame.cloud.copy()
            if compensated and ref_cy is not None:
                pts[:, 1] = pts[:, 1] + (ref_cy - frame.centroid_y)   # Y: EMA ref
                if ref_cx is not None:
                    pts[:, 0] = pts[:, 0] + (ref_cx - frame.centroid_x)  # X: EMA ref
            parts.append(pts)

        merged = np.concatenate(parts, axis=0).astype(np.float32) \
                 if parts else np.empty((0, 7), dtype=np.float32)

        vel_y = 0.0
        if self._prev_cy is not None and self._cy is not None:
            vel_y = (self._cy - self._prev_cy) * self._fps
        vel_x = 0.0
        if self._prev_cx is not None and self._cx is not None:
            vel_x = (self._cx - self._prev_cx) * self._fps


        # ── Phase 3: DBSCAN clustering ────────────────────────────────────
        cluster = self._cluster(merged) if (
            self._eps > 0 and len(merged) >= self._min_samp) else None

        # Use cluster centroid if available, else EMA centroid
        cx_out = float(cluster.centroid_x) if (cluster and cluster.cluster_found) \
                 else (self._cx or 0.0)
        cy_out = float(cluster.centroid_y) if (cluster and cluster.cluster_found) \
                 else (self._cy or 0.0)

        # Undo the EMA's own lag so the subject gate receives a CURRENT
        # position. lag_frames = (1-alpha)/alpha; alpha=1.0 -> lag 0 -> the
        # prediction collapses to the centroid. Must be computed AFTER
        # cx_out/cy_out exist.
        _lag_s = (((1.0 - self._alpha) / self._alpha) / self._fps
                  if self._alpha > 0 else 0.0)
        _pred_x = cx_out + vel_x * _lag_s
        _pred_y = cy_out + vel_y * _lag_s

        # ── Phase 4: subject-centric normalisation ───────────────────────────
        sf_cloud = None
        if self._subject_frame_en and cluster is not None and cluster.cluster_found:
            sf_cloud = subject_frame(AccumResult(
                cloud=merged, centroid_x=cx_out, centroid_y=cy_out,
                velocity_y=vel_y, n_frames=len(self._buf),
                compensated=compensated, cluster=cluster))

        # ── Phase 6: gait features ───────────────────────────────────────────
        gait = None
        if self._gait_features_en and cluster is not None and cluster.cluster_found:
            gait = gait_features(cluster.cloud,
                                 dop_limb_thr=self._dop_limb_thr,
                                 dop_limb_mode=self._dop_limb_mode,
                                 dop_limb_offset=self._dop_limb_offset,
                                 dop_limb_scale=self._limb_scale,
                                 dop_limb_k=self._dop_limb_k)
            # Update the smoothed deviation scale AFTER classifying, so this
            # frame is judged against history rather than against itself.
            _d = cluster.cloud[:, 4].astype(np.float64)
            if _d.size:
                _inst = float(np.median(np.abs(_d - float(np.median(_d)))))
                a = self._limb_scale_alpha
                self._limb_scale = (_inst if self._limb_scale <= 1e-6
                                    else a * _inst + (1 - a) * self._limb_scale)

        # ── Cloud-spread diagnostics (audit Stage 4 input) ───────────────────
        # NOTE: measuring the spread of COMPENSATED CENTROIDS is meaningless —
        # compensation maps every centroid onto ref_cx/ref_cy exactly, so that
        # spread is identically zero. What DBSCAN actually chains over is the
        # merged POINT cloud. So measure:
        #   spread_x = full x-extent of the merged cloud (trail length)
        #   gap_x    = largest gap between consecutive sorted x values
        #              (DBSCAN links the cloud into one cluster iff every
        #               such gap <= eps, so gap_x vs eps decides chaining)
        _res_x = _res_y = _step_x = 0.0
        if len(merged) > 1:
            _xs = np.sort(merged[:, 0].astype(float))
            _res_x = float(_xs[-1] - _xs[0])
            _step_x = float(np.diff(_xs).max()) if len(_xs) > 1 else 0.0
            _ys = merged[:, 1].astype(float)
            _res_y = float(_ys.max() - _ys.min())

        # Per-track hints for the Stage 3.3 gate. Apply the same EMA-lag
        # correction used for predicted_x/y so every track is a CURRENT
        # position estimate rather than one trailing by ~2.3 frames.
        _hints = []
        if cluster is not None and getattr(cluster, 'subjects', None):
            for _sc in cluster.subjects:
                if _sc.age >= 2:                     # confirmed tracks only
                    _hints.append((float(_sc.centroid_x + vel_x * _lag_s),
                                   float(_sc.centroid_y + vel_y * _lag_s)))
        if not _hints:
            _hints = [(_pred_x, _pred_y)]

        return AccumResult(
            cloud=merged,
            centroid_x=cx_out,
            centroid_y=cy_out,
            velocity_y=vel_y,
            n_frames=len(self._buf),
            compensated=compensated,
            cluster=cluster,
            subject_cloud=sf_cloud,
            gait=gait,
            comp_residual_x=round(_res_x, 4),
            comp_residual_y=round(_res_y, 4),
            comp_step_x=round(_step_x, 4),
            velocity_x=round(vel_x, 4),
            predicted_x=round(_pred_x, 4),
            predicted_y=round(_pred_y, 4),
            track_hints=_hints,
            multi_used=_multi_used,
            n_raw_clusters=(len(cluster.subjects)
                            if cluster is not None
                            and getattr(cluster, 'subjects', None) else 0),
        )

    def _cluster(self, pts: np.ndarray) -> Optional[ClusterResult]:
        """Run DBSCAN on X–Y coordinates and return the body cluster."""
        if not _SKLEARN_OK:
            return None

        xy = pts[:, :2]   # use x (lateral) and y (range) only
        try:
            labels = _DBSCAN(
                eps=self._eps,
                min_samples=self._min_samp,
                algorithm='ball_tree',
                metric='euclidean',
            ).fit_predict(xy)
        except Exception:
            return None

        n_noise = int((labels == -1).sum())

        # Find largest non-noise cluster
        unique = [l for l in np.unique(labels) if l >= 0]
        if not unique:
            return ClusterResult(
                cloud=pts, centroid_x=self._cx or 0.0,
                centroid_y=self._cy or 0.0,
                n_pts=0, n_noise=n_noise,
                radius_x=0.0, radius_y=0.0,
                cluster_found=False)

        # ── Stage 3 / P0-2: build EVERY cluster, not just the largest ───────
        subs = []
        for lab in unique:
            m  = labels == lab
            bp = pts[m]
            rx_i = float((bp[:, 0].max() - bp[:, 0].min()) / 2)
            ry_i = float((bp[:, 1].max() - bp[:, 1].min()) / 2)
            subs.append(SubjectCluster(
                cloud=bp,
                centroid_x=float(np.median(bp[:, 0])),
                centroid_y=float(np.median(bp[:, 1])),
                n_pts=int(m.sum()),
                radius_x=rx_i,
                radius_y=ry_i,
                wide=bool(self._max_radius > 0 and rx_i > self._max_radius),
            ))

        # Assign stable ids BEFORE choosing a primary, so identity does not
        # flip when the largest cluster changes hands between two people.
        self._assign_track_ids(subs)

        # Primary = largest, preserving the previous behaviour for all the
        # existing single-subject callers.
        primary_idx = max(range(len(subs)), key=lambda i: subs[i].n_pts)
        best        = subs[primary_idx]
        body_pts    = best.cloud
        cx, cy      = best.centroid_x, best.centroid_y
        rx, ry      = best.radius_x, best.radius_y

        # Reject the PRIMARY cluster if it is physically implausible (too wide
        # in X). Note the other subjects are still carried in `subjects`, so a
        # wide primary no longer blanks the whole frame for multi-subject use.
        if self._max_radius > 0 and rx > self._max_radius:
            return ClusterResult(
                cloud=pts, centroid_x=self._cx or 0.0,
                centroid_y=self._cy or 0.0,
                n_pts=0, n_noise=len(pts),
                radius_x=rx, radius_y=ry,
                cluster_found=False,
                wide_rejected=True,
                subjects=subs, primary_idx=primary_idx)

        return ClusterResult(
            cloud=body_pts,
            centroid_x=cx,
            centroid_y=cy,
            n_pts=best.n_pts,
            n_noise=n_noise,
            radius_x=rx,
            radius_y=ry,
            cluster_found=True,
            subjects=subs,
            primary_idx=primary_idx,
        )

    # ── Stage 3.2: frame-to-frame identity association ──────────────────────
    def _assign_track_ids(self, subs: list) -> None:
        """Give each cluster a track_id that persists across frames.

        Greedy nearest-centroid matching against the previous frame's tracks,
        gated by _max_jump (0.6 m at 4 fps = 2.4 m/s, the same physical bound
        the ghost gate uses). Greedy is adequate here: clusters are few (1-3)
        and well separated relative to the gate, so the optimal assignment and
        the greedy one coincide except in contrived cases.

        Unmatched previous tracks COAST for _max_coast frames before being
        dropped, so a subject briefly missed by DBSCAN keeps its identity
        rather than being reissued a new id on reappearance.
        """
        prev = self._tracks            # {track_id: (x, y, age, coast)}
        used = set()

        pairs = []
        for i, sc in enumerate(subs):
            for tid, (px, py, _age, _co) in prev.items():
                d = math.hypot(sc.centroid_x - px, sc.centroid_y - py)
                if d <= self._max_jump:
                    pairs.append((d, i, tid))
        pairs.sort(key=lambda t: t[0])

        for _d, i, tid in pairs:
            if subs[i].track_id >= 0 or tid in used:
                continue
            subs[i].track_id = tid
            subs[i].age      = prev[tid][2] + 1
            subs[i].coast    = 0
            used.add(tid)

        for sc in subs:
            if sc.track_id < 0:
                sc.track_id = self._next_track_id
                self._next_track_id += 1
                sc.age = 1

        new_tracks = {sc.track_id: (sc.centroid_x, sc.centroid_y, sc.age, 0)
                      for sc in subs}
        # Coast the tracks that were not matched this frame.
        for tid, (px, py, age, co) in prev.items():
            if tid in new_tracks:
                continue
            if co + 1 <= self._max_coast:
                new_tracks[tid] = (px, py, age, co + 1)
        self._tracks = new_tracks
        # Count only CONFIRMED tracks (age >= 2) so a one-frame DBSCAN
        # fragment cannot flip the pipeline into multi-subject mode.
        self._multi_subject = sum(1 for sc in subs if sc.age >= 2) > 1

    def _empty_result(self) -> AccumResult:
        return AccumResult(
            cloud=np.empty((0, 7), dtype=np.float32),
            centroid_x=self._cx or 0.0,
            centroid_y=self._cy or 0.0,
            velocity_y=0.0,
            n_frames=len(self._buf),
            compensated=False,
            cluster=None,
        )
