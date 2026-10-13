"""
processing/microdoppler.py  —  Phase 6: Doppler-time spectrogram and cadence.

Builds a range-gated Doppler-time spectrogram from the cluster Doppler
values accumulated by TemporalAccumulator / gait_features.

The plan says: "for the tracked range bin(s), collect the Doppler spectrum each
frame and build a Doppler-time spectrogram (range-gated to the track). Extract
simple gait features: torso Doppler (spectral centroid), limb spread
(bandwidth), cadence (periodicity via autocorrelation)."

Architecture
────────────
MicroDopplerBuffer is a stateful ring buffer that:
  - Accepts one frame's cluster Doppler values per call to .update()
  - Builds a 2D histogram (Doppler bins × time frames) — the spectrogram
  - Exposes .spectrogram  [n_dop_bins, time_frames] float32 for rendering
  - Exposes .cadence_hz   float — estimated gait cadence from autocorrelation
    of the limb_frac time series

Spectrogram axes
────────────────
  Row axis:  Doppler velocity (−v_max … +v_max), n_dop_bins rows
             positive = subject moving away, negative = approaching
  Col axis:  time (oldest left, newest right), time_frames columns
  Cell value: number of cluster pts in that (Doppler bin, frame) cell
             normalised to [0,1] per column for display

Cadence estimation
──────────────────
  limb_frac time series is maintained in a parallel ring buffer.
  Autocorrelation of the last time_frames values finds the dominant
  periodicity.  The peak lag (excluding lag=0) gives the cycle period T.
  cadence_hz = 1 / (T / frame_rate_hz).
  Returns NaN when fewer than 2*min_lag frames are available or when no
  clear peak is found above a minimum prominence.
"""
from __future__ import annotations
import numpy as np
from collections import deque
from typing import Optional


class MicroDopplerBuffer:
    """
    Doppler-time spectrogram ring buffer with cadence estimation.

    Usage::

        mdb = MicroDopplerBuffer(time_frames=64, v_max=1.5,
                                 n_dop_bins=40, frame_rate_hz=4.0)

        # Each frame — call with cluster Doppler values
        mdb.update(cluster_dop_mps, limb_frac)

        # Render
        img = mdb.spectrogram          # [n_dop_bins, time_frames] float32
        cad = mdb.cadence_hz           # float or nan
        torso = mdb.torso_line         # [time_frames] float — dop_centroid history
    """

    def __init__(self,
                 time_frames:   int   = 64,
                 v_max:         float = 1.5,
                 n_dop_bins:    int   = 40,
                 frame_rate_hz: float = 4.0,
                 min_cadence_hz: float = 0.3,
                 max_cadence_hz: float = 3.0):
        self._T       = int(time_frames)
        self._v_max   = float(v_max)
        self._N       = int(n_dop_bins)
        self._fps     = float(frame_rate_hz)
        self._min_cad = float(min_cadence_hz)
        self._max_cad = float(max_cadence_hz)

        # Doppler bin edges: n_dop_bins+1 edges spanning -v_max to +v_max
        self._dop_edges = np.linspace(-v_max, v_max, n_dop_bins + 1,
                                      dtype=np.float32)

        # Spectrogram buffer: [n_dop_bins, time_frames]
        self._spec = np.zeros((n_dop_bins, time_frames), dtype=np.float32)
        self._col  = 0    # current write column (ring)
        self._filled = 0  # how many columns have been written

        # Torso line (dop_centroid history) and limb_frac history
        self._torso  = np.full(time_frames, np.nan, dtype=np.float32)
        self._lfrac  = np.full(time_frames, np.nan, dtype=np.float32)
        self._cadence_hz = float('nan')

    # ── Public API ──────────────────────────────────────────────────────────

    def update(self, cluster_dop_mps: Optional[np.ndarray],
               limb_frac: float = float('nan'),
               dop_centroid: float = float('nan')) -> None:
        """
        Ingest one frame's cluster Doppler values and advance the buffer.

        Parameters
        ----------
        cluster_dop_mps : np.ndarray [K] or None
            Radial velocity of each cluster point (m/s, col 4 of cluster.cloud).
        limb_frac : float
            Fraction of cluster pts with |dop| ≥ dop_limb_thr (from GaitFeatures).
        dop_centroid : float
            Power-weighted Doppler centroid (from GaitFeatures).
        """
        col = self._col

        # Build Doppler histogram for this frame
        if cluster_dop_mps is not None and len(cluster_dop_mps) > 0:
            counts, _ = np.histogram(cluster_dop_mps, bins=self._dop_edges)
            self._spec[:, col] = counts.astype(np.float32)
        else:
            self._spec[:, col] = 0.0

        self._torso[col] = float(dop_centroid) if not np.isnan(dop_centroid) else np.nan
        self._lfrac[col] = float(limb_frac)    if not np.isnan(limb_frac)    else np.nan

        # Advance ring pointer
        self._col    = (col + 1) % self._T
        self._filled = min(self._filled + 1, self._T)

        # Update cadence every 8 frames (cheap)
        if self._filled >= 8 and self._filled % 8 == 0:
            self._cadence_hz = self._estimate_cadence()

    def reset(self) -> None:
        self._spec[:] = 0.0
        self._torso[:] = np.nan
        self._lfrac[:] = np.nan
        self._col    = 0
        self._filled = 0
        self._cadence_hz = float('nan')

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def spectrogram(self) -> np.ndarray:
        """
        Return the spectrogram [n_dop_bins, time_frames] in time order
        (oldest column first, newest column last), normalised to [0,1].
        """
        # Roll so that column 0 = oldest, column T-1 = newest
        rolled = np.roll(self._spec, -self._col, axis=1)
        col_max = rolled.max(axis=0, keepdims=True)
        col_max[col_max == 0] = 1.0
        return (rolled / col_max).astype(np.float32)

    @property
    def torso_line(self) -> np.ndarray:
        """Dop centroid history [time_frames], time-ordered (oldest first)."""
        return np.roll(self._torso, -self._col)

    @property
    def limb_frac_history(self) -> np.ndarray:
        """limb_frac history [time_frames], time-ordered (oldest first)."""
        return np.roll(self._lfrac, -self._col)

    @property
    def cadence_hz(self) -> float:
        return self._cadence_hz

    @property
    def n_frames_filled(self) -> int:
        return self._filled

    @property
    def dop_axis(self) -> np.ndarray:
        """Centre of each Doppler bin (m/s), length n_dop_bins."""
        return 0.5 * (self._dop_edges[:-1] + self._dop_edges[1:])

    @property
    def v_max(self) -> float:
        return self._v_max

    # ── Internal ────────────────────────────────────────────────────────────

    def _estimate_cadence(self) -> float:
        """
        Estimate gait cadence from the limb_frac time series via
        normalised autocorrelation.  Returns NaN if no clear peak.

        Human gait cadence: 0.3–3.0 Hz (slow shuffle to fast run).
        At 4 fps, lag 1 = 0.25s → max resolvable = 2 Hz (Nyquist).
        Minimum detectable = 1 full cycle in T frames:
          min_cad = fps/T = 4/64 = 0.0625 Hz (well below 0.3 Hz threshold).
        """
        # Use the last filled frames only
        n = min(self._filled, self._T)
        if n < 8:
            return float('nan')

        # Get time-ordered limb_frac
        lf = np.roll(self._lfrac, -self._col)[-n:]
        valid = ~np.isnan(lf)
        if valid.sum() < 8:
            return float('nan')

        # Linearly interpolate NaNs
        lf_clean = lf.copy()
        x = np.arange(n)
        lf_clean[~valid] = np.interp(x[~valid], x[valid], lf[valid])

        # Zero-mean
        lf_zm = lf_clean - lf_clean.mean()
        if lf_zm.std() < 1e-4:
            return float('nan')  # constant → no periodicity

        # Normalised autocorrelation
        ac = np.correlate(lf_zm, lf_zm, mode='full')
        ac = ac[len(ac)//2:]   # keep lags 0, 1, 2, …
        ac = ac / max(ac[0], 1e-6)

        # Valid lag range: corresponds to cadence_min to cadence_max
        lag_min = max(1, int(np.ceil(self._fps / self._max_cad)))
        lag_max = min(n // 2, int(np.floor(self._fps / self._min_cad)))

        if lag_min >= lag_max or lag_max >= len(ac):
            return float('nan')

        ac_window = ac[lag_min:lag_max + 1]
        peak_lag  = int(np.argmax(ac_window)) + lag_min

        # Require the peak to be a genuine local maximum above 0.2
        if ac[peak_lag] < 0.20:
            return float('nan')

        # Sanity: must be a local max (not just the edge of the window)
        if peak_lag > lag_min and peak_lag < lag_max:
            if ac[peak_lag] <= ac[peak_lag - 1] and \
               ac[peak_lag] <= ac[peak_lag + 1]:
                return float('nan')

        return round(float(self._fps / peak_lag), 2)
