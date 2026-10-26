"""
Per-track radar features: motion-compensated accumulation and micro-Doppler.

They replace the old chain's single-subject accumulate.py and microdoppler.py
(retired 2026-10-25) with versions that run for every confirmed track at once.

Accumulation: the detections assigned to a track (its cluster) are kept for the
last `accum_s` seconds and shifted by the track's velocity to the current time,
p + v (t_now - t_k). A single frame has a handful of points per object; the
accumulated cloud has tens, for shape and for annotation.

Micro-Doppler: each frame, the Doppler spectrum at the track's cells (range
+-range_bins, azimuth +-az_bins around the track) of the angle-resolved
Doppler cube, normalised to its peak, goes into a rolling spectrogram per
track. From it:

    v_torso      power-weighted mean radial velocity (m/s, + receding)
    spread       power-weighted standard deviation of the radial velocity (m/s):
                 the limbs; near 0 for a rigid object (a vehicle, a wall image)
    cadence_hz   the dominant period of the spread over the last window_s,
                 from the autocorrelation (walking ~1.5-2.2 Hz); None when the
                 window is short or there is no clear period

Frame rate 8 Hz: periods up to 4 Hz are measurable.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .core.conventions import processing_azimuth


class TrackFeatures:
    def __init__(self, range_axis, az_axis_proc, dop_axis, accum_s: float = 1.0, window_s: float = 4.0,
                 range_bins: int = 2, az_bins: int = 3, frame_hz: float = 8.0):
        self.range_axis = np.asarray(range_axis, float)
        self.az_axis = np.asarray(az_axis_proc, float)           # processing azimuth (+ right), ascending
        self.dop_axis = np.asarray(dop_axis, float)
        self.res = float(self.range_axis[1] - self.range_axis[0])
        self.accum_s, self.window_s, self.frame_hz = accum_s, window_s, frame_hz
        self.rb, self.ab = range_bins, az_bins
        self.n_win = int(round(window_s * frame_hz))
        self.state = {}                                          # track id -> dict

    def reset(self):
        self.state = {}

    def _cells(self, x: float, y: float):
        r = float(np.hypot(x, y))
        ri = int(round(r / self.res))
        ai = int(np.clip(np.searchsorted(self.az_axis, processing_azimuth(np.degrees(np.arctan2(y, x)))),
                         0, len(self.az_axis) - 1))
        return slice(max(ri - self.rb, 0), ri + self.rb + 1), slice(max(ai - self.ab, 0), ai + self.ab + 1)

    def update(self, t: float, tracks, det: np.ndarray, labels: np.ndarray, p_az=None):
        """tracks: Tracker.tracks after update (meas_idx = the cluster assigned this step); det,
        labels: this frame's detections and their cluster label (cluster_detections); p_az:
        [doppler, range, az] linear power (torch or numpy) for micro-Doppler, or None.
        Returns {track id: features} for the tracks that were assigned this step."""
        from .radar import C
        live = {tr.id for tr in tracks}
        for tid in [k for k in self.state if k not in live]:
            del self.state[tid]
        profiles = {}
        if p_az is not None:
            want = [(tr.id, *self._cells(tr.x[0], tr.x[1])) for tr in tracks if tr.meas_idx >= 0]
            if want:
                try:
                    import torch
                    is_t = isinstance(p_az, torch.Tensor)
                except ImportError:
                    is_t = False
                stack = [p_az[:, rs, az_].sum(dim=(1, 2)) if is_t else p_az[:, rs, az_].sum(axis=(1, 2))
                         for _, rs, az_ in want]
                if is_t:
                    import torch
                    stack = torch.stack(stack).cpu().numpy()
                for (tid, _, _), prof in zip(want, np.asarray(stack, float)):
                    profiles[tid] = prof
        out = {}
        for tr in tracks:
            st = self.state.setdefault(tr.id, {"pts": deque(), "spec": deque(maxlen=self.n_win),
                                               "spread": deque(maxlen=self.n_win), "t": deque(maxlen=self.n_win)})
            if tr.meas_idx < 0:
                continue
            m = labels == tr.meas_idx
            if m.any():
                st["pts"].append((t, det[m][:, [C["x"], C["y"], C["z"], C["v_r"], C["power_db"]]].astype(np.float32)))
            while st["pts"] and t - st["pts"][0][0] > self.accum_s:
                st["pts"].popleft()
            prof = profiles.get(tr.id)
            if prof is not None and prof.max() > 0:
                w = prof / prof.sum()
                v = float(w @ self.dop_axis)
                spread = float(np.sqrt(max(w @ (self.dop_axis - v) ** 2, 0.0)))
                st["spec"].append((10 * np.log10(prof / prof.max() + 1e-6)).astype(np.float32))
                st["spread"].append(spread)
                st["t"].append(t)
                out[tr.id] = {"v_torso": v, "spread": spread, "cadence_hz": self._cadence(st)}
        return out

    def _cadence(self, st):
        s = np.asarray(st["spread"], float)
        if len(s) < max(16, self.n_win // 2):
            return None
        tt = np.asarray(st["t"], float)
        if np.ptp(tt) < 0.6 * self.window_s:                    # frames missing: no regular sampling
            return None
        s = s - s.mean()
        if s.std() < 1e-3:
            return None
        ac = np.correlate(s, s, mode="full")[len(s) - 1:]
        ac /= ac[0]
        lo, hi = int(self.frame_hz / 4.0), int(self.frame_hz / 0.8)   # periods for 0.8-4 Hz
        seg = ac[lo:min(hi + 1, len(ac))]
        if len(seg) < 3:
            return None
        k = int(np.argmax(seg))
        if seg[k] < 0.2 or k in (0, len(seg) - 1):              # no clear periodic peak
            return None
        y0, y1, y2 = seg[k - 1], seg[k], seg[k + 1]             # sub-frame period: parabola through the peak
        den = y0 - 2 * y1 + y2
        lag = lo + k + (0.5 * (y0 - y2) / den if abs(den) > 1e-9 else 0.0)
        return float(self.frame_hz / lag)

    def accumulated(self, tid: int, t_now: float, velocity) -> np.ndarray:
        """The track's accumulated detections [N, 5] (x, y, z, v_r, power_db), shifted by the track's
        velocity (vx, vy) to t_now."""
        st = self.state.get(tid)
        if not st or not st["pts"]:
            return np.zeros((0, 5), np.float32)
        out = []
        for tk, pts in st["pts"]:
            q = pts.copy()
            q[:, 0] += velocity[0] * (t_now - tk)
            q[:, 1] += velocity[1] * (t_now - tk)
            out.append(q)
        return np.concatenate(out)

    def spectrogram(self, tid: int) -> np.ndarray:
        """[frames, doppler] dB (0 = the frame's peak), oldest first."""
        st = self.state.get(tid)
        return np.zeros((0, len(self.dop_axis)), np.float32) if not st or not st["spec"] else np.stack(st["spec"])
