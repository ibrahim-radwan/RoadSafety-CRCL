"""
Multi-target tracker shared by the radar and LiDAR detectors.

Constant-velocity Kalman filter in the bird's-eye plane (x forward, y left),
gated association solved with the Hungarian algorithm, and M-of-N
confirmation: a track is CONFIRMED after `confirm_hits` hits within its
first `confirm_window` updates, and deleted after `max_misses` consecutive
misses. Reliability comes from this step: a single-frame false alarm never
becomes a confirmed track.

Radar: pass the measured radial velocity (Doppler) of each measurement as
`radial`. It is then part of the measurement (EKF: v_r = (x vx + y vy) / r)
and of the association cost, and a track must stay kinematically consistent
to be reported: the range rate of its measured positions over the last ~1 s
must agree with its mean measured Doppler within `max_vr_innov` m/s.
Multipath images stitched into a track move in ways their Doppler
contradicts; real objects do not. (The EKF's own Doppler innovation cannot
show this: it trusts the precise Doppler and absorbs the contradiction.)

Measurements are N x 2 (x, y) plus an optional per-measurement payload
(anything: class, size, radial velocity) that the track keeps from its last
hit.
"""
from __future__ import annotations

import numpy as np

H_POS = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)


class Track:
    __slots__ = ("id", "x", "P", "hits", "misses", "age", "confirmed", "payload", "t", "history", "vr_innov",
                 "meas")

    def __init__(self, tid: int, xy, t: float, pos_var: float, vel_var: float, payload=None, vr=None):
        self.id = tid
        self.x = np.array([xy[0], xy[1], 0.0, 0.0])
        if vr is not None:                                  # start with the radial velocity along the line of sight
            u = np.asarray(xy, float) / max(np.hypot(*xy), 1e-6)
            self.x[2:] = vr * u
        self.P = np.diag([pos_var, pos_var, vel_var, vel_var])
        self.hits, self.misses, self.age, self.confirmed = 1, 0, 1, False
        self.payload, self.t = payload, t
        self.history = [(t, float(xy[0]), float(xy[1]))]
        self.vr_innov = 0.0
        self.meas = [(t, float(np.hypot(*xy)), np.nan if vr is None else float(vr))]   # (t, range, v_r) as measured


class Tracker:
    def __init__(self, gate_m: float = 1.5, meas_std_m: float = 0.3, accel_std: float = 3.0,
                 confirm_hits: int = 3, confirm_window: int = 5, max_misses: int = 3,
                 init_vel_std: float = 3.0, history: int = 40, vr_std: float = 0.25,
                 max_vr_innov: float = 0.6, vr_gate: float = 1.2, merge_m: float = 0.0, merge_dv: float = 1.0,
                 consist_n: int = 8):
        self.gate = gate_m
        self.merge_m, self.merge_dv = merge_m, merge_dv
        self.consist_n = consist_n
        self.R = np.eye(2) * meas_std_m ** 2
        self.q = accel_std ** 2
        self.confirm_hits, self.confirm_window, self.max_misses = confirm_hits, confirm_window, max_misses
        self.init_P = (meas_std_m ** 2, init_vel_std ** 2)
        self.hist_len = history
        self.vr_var, self.max_vr_innov, self.vr_gate = vr_std ** 2, max_vr_innov, vr_gate
        self.tracks = []
        self._next = 1
        self.t = None

    def reset(self):
        self.tracks, self.t = [], None

    def _predict(self, dt: float):
        Fm = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        Q = G @ G.T * self.q
        for tr in self.tracks:
            tr.x = Fm @ tr.x
            tr.P = Fm @ tr.P @ Fm.T + Q

    @staticmethod
    def _vr_row(x):
        """Predicted radial velocity and its Jacobian row at state x."""
        px, py, vx, vy = x
        r = max(np.hypot(px, py), 1e-3)
        vr = (px * vx + py * vy) / r
        return vr, np.array([vx / r - px * vr / r ** 2, vy / r - py * vr / r ** 2, px / r, py / r])

    def update(self, t: float, meas: np.ndarray, payloads=None, radial=None):
        """Advance to time t with measurements meas (N x 2) and optionally their radial velocities
        (N, m/s, positive = receding). Returns the live tracks."""
        from scipy.optimize import linear_sum_assignment
        meas = np.asarray(meas, float).reshape(-1, 2)
        payloads = payloads if payloads is not None else [None] * len(meas)
        radial = None if radial is None else np.asarray(radial, float).reshape(-1)
        if self.t is not None:
            dt = t - self.t
            if dt < 0 or dt > 2.0:                       # a seek or a gap: start over
                self.reset()
            elif dt > 0:
                self._predict(dt)
        self.t = t
        assigned = set()
        if self.tracks and len(meas):
            cost = np.full((len(self.tracks), len(meas)), 1e6)
            for i, tr in enumerate(self.tracks):
                S = H_POS @ tr.P @ H_POS.T + self.R
                d = meas - tr.x[:2]
                m2 = np.einsum("ni,ij,nj->n", d, np.linalg.inv(S), d)          # Mahalanobis^2
                eu = np.hypot(d[:, 0], d[:, 1])
                ok = (m2 < 9.21) | (eu < self.gate)                             # chi2(2) 99% or metric gate
                c = eu.copy()
                if radial is not None and tr.hits >= 2:                         # velocity known: use Doppler too
                    dvr = np.abs(radial - self._vr_row(tr.x)[0])
                    ok &= dvr < self.vr_gate
                    c = c + 0.5 * dvr
                cost[i, ok] = c[ok]
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] >= 1e6:
                    continue
                tr = self.tracks[i]
                if radial is not None:
                    vr_pred, hv = self._vr_row(tr.x)
                    Hm = np.vstack([H_POS, hv])
                    R = np.diag([self.R[0, 0], self.R[1, 1], self.vr_var])
                    z = np.r_[meas[j], radial[j]]
                    zp = np.r_[tr.x[:2], vr_pred]
                    tr.meas.append((t, float(np.hypot(*meas[j])), float(radial[j])))
                    del tr.meas[:-self.consist_n]
                    if len(tr.meas) >= 6:                # measured range rate vs measured Doppler
                        mt, mr, mv = (np.array(c) for c in zip(*tr.meas))
                        if np.ptp(mt) > 0.3:
                            tr.vr_innov = abs(np.polyfit(mt - mt[-1], mr, 1)[0] - float(np.mean(mv)))
                else:
                    Hm, R, z, zp = H_POS, self.R, meas[j], tr.x[:2]
                S = Hm @ tr.P @ Hm.T + R
                K = tr.P @ Hm.T @ np.linalg.inv(S)
                tr.x = tr.x + K @ (z - zp)
                tr.P = (np.eye(4) - K @ Hm) @ tr.P
                tr.hits += 1
                tr.misses = 0
                tr.payload, tr.t = payloads[j], t
                tr.history.append((t, float(tr.x[0]), float(tr.x[1])))
                del tr.history[:-self.hist_len]
                assigned.add(j)
                tr.age += 1
        live = []
        for tr in self.tracks:
            if tr.t != t:                                # not updated this step
                tr.misses += 1
                tr.age += 1
            consistent = radial is None or tr.vr_innov < self.max_vr_innov
            if (not tr.confirmed and tr.hits >= self.confirm_hits and consistent
                    and (tr.age <= self.confirm_window or 2 * tr.hits >= tr.age)):   # first time, or re-confirm
                tr.confirmed = True
            if tr.confirmed and not consistent:          # lost kinematic consistency: back to tentative
                tr.confirmed = False
            dead = tr.misses > self.max_misses or (not tr.confirmed and tr.age >= self.confirm_window
                                                    and tr.hits < self.confirm_hits)
            if not dead:
                live.append(tr)
        if self.merge_m > 0 and len(live) > 1:          # duplicates: one object, two tracks -> keep the older
            live.sort(key=lambda tr_: -tr_.hits)
            kept = []
            for tr in live:
                if any(np.hypot(*(tr.x[:2] - k_.x[:2])) < self.merge_m
                       and np.hypot(*(tr.x[2:] - k_.x[2:])) < self.merge_dv for k_ in kept):
                    continue
                kept.append(tr)
            live = kept
        self.tracks = live
        for j in range(len(meas)):
            if j not in assigned:
                self.tracks.append(Track(self._next, meas[j], t, *self.init_P, payload=payloads[j],
                                         vr=None if radial is None else radial[j]))
                self._next += 1
        return self.tracks

    def reported(self, max_coast: int = 2):
        """Confirmed tracks updated within the last max_coast steps: the tracker's output. Longer
        coasting tracks stay alive (to bridge short gaps) but are not reported."""
        return [tr for tr in self.tracks if tr.confirmed and tr.misses <= max_coast]

    def confirmed(self):
        return [tr for tr in self.tracks if tr.confirmed]
