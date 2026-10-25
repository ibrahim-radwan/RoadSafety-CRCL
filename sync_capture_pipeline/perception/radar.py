"""
GPU radar detection for the cascaded AWR2243 (PyTorch), independent of the
team's process_frame.

One raw frame -> display products (range-Doppler, range-azimuth) and a
detection list, all on the GPU in a few milliseconds:

    1. calibration (the .cfg antennaCalibParams, times an optional in-situ
       per-RX phase correction, see below), range FFT (Hann)
    2. MTI (mean over chirp loops removed) and Doppler FFT (Hann) of every
       virtual channel, then TDM-MIMO Doppler compensation: a moving target's
       phase advances between the 6 TX slots of a loop, by 2 pi f_d t_slot
       per slot; each channel is rotated back by its TX slot
    3. virtual arrays: 38 uniform azimuth positions (overlaps averaged) and the
       8-channel elevation row (TX5), paired with the azimuth row at shared
       positions
    4. azimuth FFT of every range-Doppler cell (256 bins, uniform in sin az);
       detection map = the angle-maximum power of each cell (coherent gain of
       the whole aperture, ~16 dB over a single channel)
    5. CA-CFAR in dB on that map (circular in Doppler), zero-Doppler band
       excluded, 3x3 local maxima kept
    6. per detection: the strongest azimuth peak (max_peaks > 1 adds peaks
       within peak_db; off by default: with the current calibration the beam
       still has a split lobe ~13 deg away, ~4 dB down, which would become a
       second, false target), parabolic interpolation in sin az, range
       interpolation,
       elevation from the top/bottom pair phase (UNVERIFIED geometry)
    7. multipath ghosts flagged (column "ghost" = 1, see flag_multipath): a
       weaker detection at the azimuth of a stronger one, either at n x its
       range with n x its radial velocity (n = 2, 3; radar -> target -> rig
       -> target -> radar) or farther with the same radial velocity (via a
       static reflector). On stage3_02 the detections the LiDAR does not
       support cluster at range ratio 2.0-2.2 / velocity ratio 1.9 and at
       velocity ratio 1.0 behind the walker

Conventions (verified against the LiDAR on freewalk_ref_02, see
docs/knowledge/03_radar_firmware_and_dca.md):
    * The FFT bin -> angle mapping is the same as the team's processing, whose
      azimuth is positive to the RIGHT (their README: "x = lateral, positive =
      right of boresight"). The pipeline frame is y LEFT, so the world azimuth
      is the negated processing azimuth. This is a frame conversion, not a
      correction of an error.
    * Radial velocity: the team's Doppler axis (cfg.doppler_axis); positive =
      receding (99% sign agreement with the LiDAR range rate).

In-situ RX phase correction (calibration/radar_rx_phase.json, per firmware
variant, from estimate_rx_phase.py): the .cfg calibration was measured with
the old HPF setting and leaves a stable per-RX phase error (up to ~70 deg) on
the v2 images, which splits the azimuth beam. It is PROVISIONAL: replace it
with the team's recalibration when it arrives (pass rx_phase_deg=None).

Output columns of `detections` (float32, N x len(DET_COLS)): see DET_COLS.
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RADAR_PKG = HERE.parent.parent / "cascaded_radar_capture_pipeline"

DET_COLS = ("x", "y", "z", "range", "az_deg", "el_deg", "v_r", "power_db", "snr_db", "ghost")
C = {k: i for i, k in enumerate(DET_COLS)}


RX_PHASE_FILE = HERE.parent / "calibration" / "radar_rx_phase.json"

# Tracker settings for radar objects (tuned on freewalk_ref_02 against the LiDAR walker: the walker
# has a reported track 77% of the time, 1.1 other reported tracks per frame; see run_perception.py).
# Clusters: eps 1.5 m with Doppler weighted 1 m per 4 m/s, so a walker's limbs (micro-Doppler)
# stay one object.
RADAR_TRACKER = dict(gate_m=2.0, meas_std_m=0.6, max_misses=6, max_vr_innov=1.0, accel_std=2.0, merge_m=1.0)


def rx_phase_for(variant: str):
    """The in-situ per-RX phase correction (deg, 8 values) for a firmware variant, or None."""
    import json
    if not RX_PHASE_FILE.is_file():
        return None
    entry = json.loads(RX_PHASE_FILE.read_text()).get("variants", {}).get(variant)
    return None if entry is None else entry["rx_phase_deg"]


def load_radar_config(cfg_path):
    """(cfg, calibration matrix [tx, rx]) from the team's parser, silenced."""
    if str(RADAR_PKG) not in sys.path:
        sys.path.insert(0, str(RADAR_PKG))
    from processing.radar_config import parse_calibration, parse_cfg_file
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = parse_cfg_file(str(cfg_path))
        corr = parse_calibration(str(cfg_path), cfg.num_tx, cfg.num_rx)
    return cfg, np.asarray(corr)


class RadarDetector:
    """Radar products and detections for one raw frame (int16 I/Q, I first unless cfg.iq_swap)."""

    def __init__(self, cfg_path, device: str = "cuda", *, cfar_guard=(2, 2), cfar_train=(4, 8),
                 threshold_db: float = 14.0, min_range_m: float = 1.0, max_range_m: float = None,
                 fov_deg: float = 70.0, peak_db: float = 6.0, max_peaks: int = 1, zero_doppler_bins: int = 1,
                 max_detections: int = 512, rx_phase_deg=None, doppler_comp: bool = True):
        import torch
        self.torch = torch
        cfg, corr = load_radar_config(cfg_path)
        self.cfg = cfg
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_tx, self.n_rx, self.n_loops, self.n_s = cfg.num_tx, cfg.num_rx, cfg.num_loops, cfg.num_samples
        self.iq_swap = bool(cfg.iq_swap)
        ph = None if rx_phase_deg is None else np.asarray(rx_phase_deg, float)
        if ph is not None:                       # per RX (8) or per channel (tx x rx, 6 x 8)
            corr = corr * (np.exp(-1j * np.radians(ph))[None, :] if ph.ndim == 1 else np.exp(-1j * np.radians(ph)))
        self.rx_phase_deg = ph
        self.corr = torch.as_tensor(corr, dtype=torch.complex64, device=self.dev)
        ch_az, ch_el, _ = cfg.build_channel_table()
        self.n_pos = n_pos = max(c["az"] for c in ch_az) + 1
        self.ch_tx = torch.tensor([c["tx"] for c in ch_az], device=self.dev)
        self.ch_rx = torch.tensor([c["rx"] for c in ch_az], device=self.dev)
        self.ch_pos = torch.tensor([c["az"] for c in ch_az], device=self.dev)
        counts = np.bincount([c["az"] for c in ch_az], minlength=n_pos).astype(np.float32)
        self.pos_w = torch.as_tensor(1.0 / np.maximum(counts, 1), device=self.dev)
        # elevation row: top channel and the azimuth position it shares with the bottom row
        self.el_tx = torch.tensor([c["tx"] for c in ch_el], device=self.dev)
        self.el_rx = torch.tensor([c["rx"] for c in ch_el], device=self.dev)
        self.el_pos = torch.tensor([c["az"] for c in ch_el], device=self.dev)
        self.d_el = float(cfg.d_el_lambda)
        self.win_r = torch.hann_window(self.n_s, periodic=False, device=self.dev)
        self.win_d = torch.hann_window(self.n_loops, periodic=False, device=self.dev)
        self.win_a = torch.hann_window(n_pos, periodic=False, device=self.dev)
        self.n_az = int(cfg.az_fft_size)
        self.range_res = float(cfg.range_res_m)
        self.n_range = int(cfg.n_range)
        self.range_axis = np.arange(self.n_range) * self.range_res
        self.az_axis = np.asarray(cfg.az_axis_deg)               # processing azimuth (+ = right), ascending
        self.sin_axis = np.linspace(-1.0, 1.0, self.n_az, endpoint=False)
        self.dop_axis = np.asarray(cfg.doppler_axis)
        self.dop_axis_t = torch.as_tensor(self.dop_axis, dtype=torch.float32, device=self.dev)
        # TDM Doppler compensation, [tx, doppler bin]: rotate TX slot t back by 2 pi f_d t t_slot
        t_slot = (cfg.idle_time_us + cfg.ramp_end_us) * 1e-6
        fd = 2.0 * self.dop_axis / cfg.lambda_m
        ph = np.exp(-1j * 2 * np.pi * np.outer(np.arange(self.n_tx), fd) * t_slot) if doppler_comp \
            else np.ones((self.n_tx, len(fd)))
        self.tdm_comp = torch.as_tensor(ph, dtype=torch.complex64, device=self.dev)
        self.doppler_comp = bool(doppler_comp)
        # detection parameters
        self.guard, self.train = tuple(cfar_guard), tuple(cfar_train)
        self.threshold_db, self.peak_db, self.max_peaks = float(threshold_db), float(peak_db), int(max_peaks)
        self.min_range = float(min_range_m)
        self.max_range = float(max_range_m) if max_range_m else float(self.range_axis[-1])
        self.fov_sin = float(np.sin(np.radians(fov_deg)))
        self.zero_bins = int(zero_doppler_bins)
        self.max_det = int(max_detections)
        gd, gr = self.guard
        td, tr = gd + self.train[0], gr + self.train[1]
        k = torch.ones((2 * td + 1, 2 * tr + 1), device=self.dev)
        k[td - gd:td + gd + 1, tr - gr:tr + gr + 1] = 0
        self.cfar_kernel = (k / k.sum())[None, None]
        self.cfar_pad = (td, tr)
        rmask = (self.range_axis >= self.min_range) & (self.range_axis <= self.max_range)
        self.range_ok = torch.as_tensor(rmask, device=self.dev)
        mid = self.n_loops // 2
        dmask = np.ones(self.n_loops, bool)
        dmask[mid - self.zero_bins:mid + self.zero_bins + 1] = False
        self.dop_ok = torch.as_tensor(dmask, device=self.dev)
        self.sin_t = torch.as_tensor(self.sin_axis, dtype=torch.float32, device=self.dev)
        self.fov_ok = self.sin_t.abs() <= self.fov_sin
        fov_i = np.flatnonzero(np.abs(self.sin_axis) <= self.fov_sin)
        self.fov_sl = slice(int(fov_i[0]), int(fov_i[-1]) + 1)                 # contiguous azimuth FOV bins
        self.last = {}

    # ── stages ─────────────────────────────────────────────────────────────
    def range_cube(self, raw: np.ndarray):
        """[tx, loop, rx, range] complex, calibrated, range-FFT'd."""
        torch = self.torch
        x = torch.as_tensor(raw, device=self.dev).view(self.n_loops, self.n_tx, self.n_rx, self.n_s, 2).float()
        iq = torch.complex(x[..., 1], x[..., 0]) if self.iq_swap else torch.complex(x[..., 0], x[..., 1])
        cube = iq.permute(1, 0, 2, 3) * self.corr[:, None, :, None]       # TDM: chirp = loop * n_tx + tx
        return torch.fft.fft(cube * self.win_r, dim=-1)[..., :self.n_range]

    def process(self, raw: np.ndarray, moving_only: bool = True, detect: bool = True):
        """Returns dict: rd [doppler, range] dB, ra [az, range] dB (processing azimuth order),
        and, if detect, detections (N x len(DET_COLS)) and noise_db (frame median of the
        detection map)."""
        torch = self.torch
        rng = self.range_cube(raw)
        virt = rng[self.ch_tx, :, self.ch_rx, :]                                 # [ch, loop, range]
        el = rng[self.el_tx, :, self.el_rx, :]                                   # [8, loop, range]
        if moving_only:
            virt = virt - virt.mean(dim=1, keepdim=True)
            el = el - el.mean(dim=1, keepdim=True)
        wd = self.win_d[None, :, None]
        dch = torch.fft.fftshift(torch.fft.fft(virt * wd, dim=1), dim=1)         # [ch, dop, range]
        el_dop = torch.fft.fftshift(torch.fft.fft(el * wd, dim=1), dim=1)       # [8, dop, range]
        if self.doppler_comp:
            dch = dch * self.tdm_comp[self.ch_tx][:, :, None]
            el_dop = el_dop * self.tdm_comp[self.el_tx][:, :, None]
        dop = torch.zeros((self.n_pos,) + dch.shape[1:], dtype=dch.dtype, device=self.dev)
        dop.index_add_(0, self.ch_pos, dch)
        dop = dop * self.pos_w[:, None, None]                                    # uniform array [pos, dop, range]
        rd = 10 * torch.log10((dop.abs() ** 2).sum(dim=0) + 1e-6)               # [dop, range]
        # azimuth on the last (contiguous) axis: unstrided cuFFT, contiguous per-cell spectra
        az = torch.fft.fft((dop * self.win_a[:, None, None]).permute(1, 2, 0), n=self.n_az, dim=-1)
        p_az = torch.fft.fftshift(az.real ** 2 + az.imag ** 2, dim=-1)          # [dop, range, az]
        ra = 10 * torch.log10(p_az.sum(dim=0).T + 1e-6)                          # [az, range]
        out = {"rd": rd, "ra": ra}
        if detect:
            dets, noise = self._detect(p_az, dop, el_dop, moving_only)
            out.update(detections=dets, noise_db=noise)
        self.last = out
        return out

    def _detect(self, p_az, dop, el_dop, moving_only):
        torch = self.torch
        F = torch.nn.functional
        det_map = 10 * torch.log10(p_az[..., self.fov_sl].max(dim=-1).values + 1e-6)   # [dop, range]
        pd, pr = self.cfar_pad
        x = det_map[None, None]
        xp = F.pad(F.pad(x, (0, 0, pd, pd), mode="circular"), (pr, pr, 0, 0), mode="replicate")
        noise = F.conv2d(xp, self.cfar_kernel)[0, 0]
        snr = det_map - noise
        hit = snr > self.threshold_db
        hit &= self.range_ok[None, :]
        if moving_only:
            hit &= self.dop_ok[:, None]
        # keep 3x3 local maxima of the detection map (circular in Doppler)
        mp = F.max_pool2d(F.pad(x, (0, 0, 1, 1), mode="circular"), 3, stride=1, padding=(0, 1))[0, 0]
        hit &= det_map >= mp
        di, ri = torch.nonzero(hit, as_tuple=True)
        if di.numel() > self.max_det:
            keep = torch.topk(snr[di, ri], self.max_det).indices
            di, ri = di[keep], ri[keep]
        noise_db = float(det_map.median())
        if di.numel() == 0:
            return np.zeros((0, len(DET_COLS)), np.float32), noise_db
        # azimuth peaks per detection cell
        spec = 10 * torch.log10(p_az[di, ri] + 1e-6)                             # [N, az]
        spec = torch.where(self.fov_ok[None, :], spec, torch.full_like(spec, -1e9))
        left, right = torch.roll(spec, 1, dims=1), torch.roll(spec, -1, dims=1)
        is_pk = (spec >= left) & (spec >= right) & (spec >= spec.max(dim=1, keepdim=True).values - self.peak_db)
        val = torch.where(is_pk, spec, torch.full_like(spec, -1e9))
        top_v, top_i = torch.topk(val, self.max_peaks, dim=1)                   # [N, max_peaks]
        ok = top_v > -1e8
        n_idx = torch.arange(spec.shape[0], device=self.dev)[:, None].expand_as(top_i)
        y0 = spec[n_idx, (top_i - 1).clamp(0, self.n_az - 1)]
        y2 = spec[n_idx, (top_i + 1).clamp(0, self.n_az - 1)]
        den = (y0 - 2 * top_v + y2)
        delta = torch.where(den.abs() > 1e-6, 0.5 * (y0 - y2) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)
        sin_proc = (self.sin_t[top_i] + delta * (2.0 / self.n_az)).clamp(-1, 1)
        # range interpolation on the detection map
        r0 = det_map[di, (ri - 1).clamp(0, self.n_range - 1)]
        r1 = det_map[di, ri]
        r2 = det_map[di, (ri + 1).clamp(0, self.n_range - 1)]
        rden = r0 - 2 * r1 + r2
        rdel = torch.where(rden.abs() > 1e-6, 0.5 * (r0 - r2) / rden, torch.zeros_like(rden)).clamp(-0.5, 0.5)
        rng_m = (ri.float() + rdel) * self.range_res
        # elevation: phase of top x conj(bottom) over the pairs, at this range-Doppler cell (unverified)
        top = el_dop[:, di, ri]                                                  # [8, N]
        bot = dop[self.el_pos][:, di, ri]
        phase = torch.angle((top * bot.conj()).sum(dim=0))
        sin_el = (phase / (2 * np.pi * self.d_el)).clamp(-1, 1)
        # assemble: one row per (cell, azimuth peak)
        sel = ok.flatten()
        k = n_idx.flatten()[sel]
        az_w = -torch.rad2deg(torch.asin(sin_proc.flatten()[sel]))             # world azimuth, + = left
        el_d = torch.rad2deg(torch.asin(sin_el))[k]
        r = rng_m[k]
        ca, ce = torch.cos(torch.deg2rad(az_w)), torch.cos(torch.deg2rad(el_d))
        cols = torch.stack([r * ce * ca, r * ce * torch.sin(torch.deg2rad(az_w)), r * torch.sin(torch.deg2rad(el_d)),
                            r, az_w, el_d, self.dop_axis_t[di[k]], top_v.flatten()[sel],
                            top_v.flatten()[sel] - noise[di[k], ri[k]], torch.zeros_like(r)], dim=1)
        det = cols.float().cpu().numpy()
        det[:, C["ghost"]] = flag_multipath(det, float(self.dop_axis[-1] - self.dop_axis[0] + self.dop_axis[1]
                                                       - self.dop_axis[0]))
        return det, noise_db


def flag_multipath(det: np.ndarray, v_span: float, az_tol_deg: float = 6.0, rel_tol: float = 0.2,
                   min_drop_db: float = 3.0, behind_drop_db: float = 6.0) -> np.ndarray:
    """1.0 for detections explained as a multipath image of a stronger detection j at the same azimuth:
      n-bounce (n = 2, 3): range ~ n r_j, radial velocity ~ n v_j (modulo the Doppler span v_span),
                           radar -> target -> rig -> target -> radar;
      via a static reflector: farther than j (> 0.5 m) with j's radial velocity and >= behind_drop_db
                           weaker, radar -> target -> wall/fence -> radar.
    Limitation: two road users in single file at the same radial speed, the far one >= 6 dB weaker,
    would flag the far one."""
    n = len(det)
    ghost = np.zeros(n, np.float32)
    if n < 2:
        return ghost
    r, az, v, pw = det[:, C["range"]], det[:, C["az_deg"]], det[:, C["v_r"]], det[:, C["power_db"]]
    same_az = np.abs(az[:, None] - az[None, :]) < az_tol_deg                  # [candidate i, source j]
    drop = pw[None, :] - pw[:, None]
    for m in (2.0, 3.0):
        ratio = r[:, None] / np.maximum(r[None, :], 1e-3)
        dv = (v[:, None] - m * v[None, :] + v_span / 2) % v_span - v_span / 2
        hit = same_az & (np.abs(ratio - m) < rel_tol * m) & (np.abs(dv) < 0.3 + 0.15 * np.abs(m * v[None, :])) \
            & (drop > min_drop_db)
        ghost[hit.any(axis=1)] = 1.0
    behind = same_az & (r[:, None] > r[None, :] + 0.5) & (np.abs(v[:, None] - v[None, :]) < 0.2 + 0.1 * np.abs(v[None, :])) \
        & (drop > behind_drop_db) & (np.abs(v[None, :]) > 0.2)
    ghost[behind.any(axis=1)] = 1.0
    return ghost


def cluster_detections(det: np.ndarray, eps_m: float = 1.5, v_scale: float = 4.0):
    """Group detections of one frame into objects: single-linkage clusters (connected components of
    the eps graph, i.e. DBSCAN with min_samples = 1) on x, y and v_r / v_scale (m per m/s). Multipath
    ghosts are left out. scipy's csgraph, not sklearn: sklearn's per-call threadpool introspection
    holds the GIL for tens of ms and starved the replay GUI.
    Returns (labels per detection, -1 = ghost; objects N x 8: x, y, range, az_deg, v_r, power_db, n, extent_m)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    lab = np.full(len(det), -1, int)
    real = np.flatnonzero(det[:, C["ghost"]] == 0) if len(det) else np.zeros(0, int)
    if len(real) == 0:
        return lab, np.zeros((0, 8), np.float32)
    d = det[real]
    f = np.c_[d[:, C["x"]], d[:, C["y"]], d[:, C["v_r"]] / max(v_scale, 1e-6)]
    adj = np.sqrt(((f[:, None, :] - f[None, :, :]) ** 2).sum(-1)) < eps_m
    _, comp = connected_components(csr_matrix(adj), directed=False)
    lab[real] = comp
    objs = []
    for L in np.unique(comp):
        m = comp == L
        w = 10 ** ((d[m, C["power_db"]] - d[m, C["power_db"]].max()) / 10)
        w = w / w.sum()
        x, y = float(w @ d[m, C["x"]]), float(w @ d[m, C["y"]])
        ext = float(np.hypot(np.ptp(d[m, C["x"]]), np.ptp(d[m, C["y"]])))
        objs.append([x, y, np.hypot(x, y), np.degrees(np.arctan2(y, x)), float(w @ d[m, C["v_r"]]),
                     float(d[m, C["power_db"]].max()), int(m.sum()), ext])
    return lab, np.asarray(objs, np.float32).reshape(-1, 8)
