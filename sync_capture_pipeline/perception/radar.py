"""
GPU radar detection for the cascaded AWR2243 (PyTorch), independent of the
team's process_frame.

One raw frame -> display products (range-Doppler, range-azimuth) and a
detection list, all on the GPU in a few milliseconds:

    1. range FFT (Hann); measured layout: the slave chip's phase (a constant
       plus a ramp in range, calibration/radar_array.json) removed
    2. MTI (mean over chirp loops removed) and Doppler FFT (Hann) of every
       virtual channel, then TDM-MIMO Doppler compensation: a moving target's
       phase advances between the 6 TX slots of a loop, by 2 pi f_d t_slot
       per slot; each channel is rotated back by its TX slot
    3. virtual array (ARRAY LAYOUT below): measured, two groups of 11 uniform
       positions, one per chip (the 4 azimuth chirps x 4 RX, overlaps averaged)
    4. azimuth FFT of every range-Doppler cell and group (256 bins, uniform in
       sin az, Hann); the groups' powers add (incoherent: no fringes from the
       gap between them, sidelobes ~20 dB down); detection map = the
       angle-maximum power of each cell
    5. CA-CFAR in dB on that map (circular in Doppler), zero-Doppler band
       excluded, 3x3 local maxima kept
    6. per detection: the strongest azimuth peak (max_peaks > 1 adds peaks
       within peak_db; off by default), parabolic interpolation in sin az, then
       (measured layout) the angle refined coherently across both chips: the
       fringe nearest the peak, when it beats the others by refine_gate_db
       (_refine); range interpolation; elevation: 0 with the measured layout
       (its elevated TX positions are not measured yet), the top/bottom pair
       phase with the team layout (UNVERIFIED geometry)
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

ARRAY LAYOUT (layout=, 2026-11-07; docs/knowledge/03_radar_firmware_and_dca.md,
"Off-axis azimuth"):
    "measured" (default)  calibration/radar_array.json, from estimate_array_layout.py:
                          chirps 0, 1, 2, 5 are azimuth TX at 4, 2, 0, 7 (lambda/2) in each
                          chip's group, chirps 3 and 4 are elevated; RX 0-3 (master) and 4-7
                          (slave) 19 apart. No .cfg antennaCalibParams, no in-situ phase: they
                          made it worse. On the 2026-11-03 corner reflector: 0.4 deg spread
                          against the camera (refined coherently: ~0.1), none off by > 1 deg
    "team"                radar_config.py's table (TX at 0, 4, 7, 11, 15, chirp 5 elevated),
                          the .cfg calibration and, if given, the in-situ phase
                          (calibration/radar_rx_phase.json, estimate_rx_phase.py). It does not
                          match the board: a split beam and 5-16 deg errors off boresight.
                          Kept for comparisons and the old chain (radar_chain.py)

Output columns of `detections` (float32, N x len(DET_COLS)): see DET_COLS.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import radar_dsp as dsp
from .core.calibration import (ARRAY_FILE, RX_PHASE_FILE, apply_phase, array_for, array_layout,  # noqa: F401
                               load_radar_config, rx_phase_for, variant_of_cfg)
from .core.conventions import AZIMUTH_SIGN
from .core.gpu import GPU_LOCK

HERE = Path(__file__).resolve().parent

DET_COLS = ("x", "y", "z", "range", "az_deg", "el_deg", "v_r", "power_db", "snr_db", "ghost")
C = {k: i for i, k in enumerate(DET_COLS)}

# Tracker settings for radar objects (tuned on freewalk_ref_02 against the LiDAR walker: the walker
# has a reported track 77% of the time, 1.1 other reported tracks per frame; see run_perception.py).
# Clusters: eps 1.5 m with Doppler weighted 1 m per 4 m/s, so a walker's limbs (micro-Doppler)
# stay one object.
RADAR_TRACKER = dict(gate_m=2.0, meas_std_m=0.6, max_misses=6, max_vr_innov=1.0, accel_std=2.0, merge_m=1.0)


class RadarDetector:
    """Radar products and detections for one raw frame (int16 I/Q, I first unless cfg.iq_swap)."""

    def __init__(self, cfg_path, device: str = "cuda", *, cfar_guard=(2, 2), cfar_train=(4, 8),
                 threshold_db: float = 14.0, min_range_m: float = 1.0, max_range_m: float = None,
                 fov_deg: float = 70.0, peak_db: float = 6.0, max_peaks: int = 1, zero_doppler_bins: int = 1,
                 max_detections: int = 512, rx_phase_deg=None, doppler_comp: bool = True,
                 static_guard=(2, 2), static_train=(4, 8), static_db: float = 15.0, static_sidelobe_db: float = 10.0,
                 static_bins: int = 1, static_max_range_m: float = 60.0, cuda_graph: bool = True,
                 layout: str = "measured", cfg_calibration: bool = None, inter_chip: bool = True,
                 refine_gate_db: float = 1.0):
        import torch
        self.torch = torch
        cfg, corr = load_radar_config(cfg_path)
        self.cfg = cfg
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_tx, self.n_rx, self.n_loops, self.n_s = cfg.num_tx, cfg.num_rx, cfg.num_loops, cfg.num_samples
        self.iq_swap = bool(cfg.iq_swap)
        self.variant = variant_of_cfg(cfg_path)
        lay = array_layout() if layout == "measured" else None
        if layout == "measured" and lay is None:
            raise FileNotFoundError(f"layout='measured' needs a layout in {ARRAY_FILE}")
        if layout not in ("measured", "team"):
            raise ValueError(f"layout {layout!r}: 'measured' or 'team'")
        self.layout = layout
        # the .cfg antenna calibration belongs to the team layout (it makes the measured one worse)
        if cfg_calibration is None:
            cfg_calibration = layout == "team"
        if not cfg_calibration:
            corr = np.ones_like(corr)
        corr = apply_phase(corr, rx_phase_deg)   # in-situ correction: per RX (8) or per channel (6 x 8)
        self.rx_phase_deg = None if rx_phase_deg is None else np.asarray(rx_phase_deg, float)
        self.corr = torch.as_tensor(corr, dtype=torch.complex64, device=self.dev)
        self.range_res = float(cfg.range_res_m)
        self.n_range = int(cfg.n_range)
        self.range_axis = np.arange(self.n_range) * self.range_res
        self.chip_corr = None                    # [rx, range] inter-chip phase, applied after the range FFT
        if lay is None:
            ch_az, ch_el, _ = cfg.build_channel_table()
            self.n_pos = n_pos = max(c["az"] for c in ch_az) + 1
            self.ch_tx = torch.tensor([c["tx"] for c in ch_az], device=self.dev)
            self.ch_rx = torch.tensor([c["rx"] for c in ch_az], device=self.dev)
            self.ch_pos_np = np.array([c["az"] for c in ch_az], float)          # index in the uniform array
            self.ch_phys_np = self.ch_pos_np.copy()                             # aperture position, lambda/2
            # elevation row: top channel and the azimuth position it shares with the bottom row
            self.el_tx = torch.tensor([c["tx"] for c in ch_el], device=self.dev)
            self.el_rx = torch.tensor([c["rx"] for c in ch_el], device=self.dev)
            self.el_pos = torch.tensor([c["az"] for c in ch_el], device=self.dev)
            self.has_el = len(ch_el) > 0
            self.n_groups, self.group = 1, n_pos
            self.win_a = dsp.window("hann", n_pos, device=self.dev)
            self.coherent, self.chip_offset = False, 0
        else:
            # measured: each chip's channels form one contiguous group of `group` positions; the two
            # groups are chip_offset apart (a gap between them). Beamformed per group, the groups'
            # powers add (incoherent: no fringes from the gap); the angle is then refined coherently
            # across both (see _cell_rows). Elevated TX (None) are not in the azimuth row.
            txp, rxc, rxp = lay["chirp_tx_pos"], lay["rx_chip"], lay["rx_pos"]
            chans = [(t, r) for t in range(self.n_tx) if txp[t] is not None for r in range(self.n_rx)]
            self.group = G = max(p for p in txp if p is not None) + max(rxp) + 1
            self.n_groups = int(max(rxc)) + 1
            idx = np.array([rxc[r] * G + txp[t] + rxp[r] for t, r in chans], float)
            if len(np.unique(idx)) != self.n_groups * G:
                raise ValueError(f"{ARRAY_FILE}: the groups are not contiguous ({len(np.unique(idx))} positions)")
            self.chip_offset = int(lay["chip_offset"])
            self.ch_tx = torch.tensor([t for t, _ in chans], device=self.dev)
            self.ch_rx = torch.tensor([r for _, r in chans], device=self.dev)
            self.ch_pos_np = idx                                                 # index in the [group x position] array
            self.ch_phys_np = np.array([rxc[r] * self.chip_offset + txp[t] + rxp[r] for t, r in chans], float)
            self.n_pos = n_pos = self.n_groups * G
            self.has_el = False                                                  # elevated TX positions not measured yet
            self.el_tx = self.el_rx = self.el_pos = None
            self.win_a = torch.as_tensor(np.hanning(G + 2)[1:-1].astype(np.float32), device=self.dev)
            A = array_for(self.variant) if inter_chip else None
            self.coherent = A is not None and self.n_groups == 2
            if self.coherent:
                ph = np.zeros((self.n_rx, self.n_range))
                ph[np.asarray(rxc) == 1] = (float(A["slave_phase_rad_per_m"]) * self.range_axis
                                            + np.radians(float(A["slave_phase_deg"])))[None, :]
                self.chip_corr = torch.as_tensor(np.exp(-1j * ph), dtype=torch.complex64, device=self.dev)
        self.d_el = float(cfg.d_el_lambda)
        self.win_r = dsp.window("hann", self.n_s, device=self.dev)
        self.win_d = dsp.window("hann", self.n_loops, device=self.dev)
        self.n_az = int(cfg.az_fft_size)
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
        # coherent refinement across the two chips (measured layout): the gap makes fringes 2 / D apart
        # in sin (D = chip_offset); the fringe nearest the incoherent peak (within 1 / D) is taken when
        # it beats every other fringe within 3 / D by refine_gate_db, else the incoherent angle stays.
        self.refine_gate_db = float(refine_gate_db)
        if self.coherent:
            step = 2.0 / self.n_az
            self.ref_near = int(round(1.0 / self.chip_offset / step))
            self.ref_offs = torch.arange(-3 * self.ref_near, 3 * self.ref_near + 1, device=self.dev)
            self.chip_steer = torch.exp(-1j * np.pi * self.chip_offset * self.sin_t.double()).to(torch.complex64)
        self.static_guard, self.static_train = tuple(static_guard), tuple(static_train)
        self.static_db, self.static_sidelobe_db, self.static_bins = float(static_db), float(static_sidelobe_db), int(static_bins)
        self.static_range_ok = self.range_ok & torch.as_tensor(self.range_axis <= static_max_range_m, device=self.dev)
        fov_i = np.flatnonzero(np.abs(self.sin_axis) <= self.fov_sin)
        self.fov_sl = slice(int(fov_i[0]), int(fov_i[-1]) + 1)                 # contiguous azimuth FOV bins
        gr, ga = self.static_guard
        tr_, ta = gr + self.static_train[0], ga + self.static_train[1]
        k = torch.ones((2 * tr_ + 1, 2 * ta + 1), device=self.dev)
        k[tr_ - gr:tr_ + gr + 1, ta - ga:ta + ga + 1] = 0
        self.static_kernel, self.static_pad = (k / k.sum())[None, None], (tr_, ta)
        ri = torch.arange(self.n_range, device=self.dev)
        self.r_idx_f, self.r_prev, self.r_next = ri.float(), (ri - 1).clamp(0, self.n_range - 1), \
            (ri + 1).clamp(0, self.n_range - 1)
        self.range_m_t = ri.float() * self.range_res
        self.st_az_w = AZIMUTH_SIGN * torch.rad2deg(torch.asin(self.sin_t[self.fov_sl]))
        self.static_az_deg = self.st_az_w.cpu().numpy()          # world azimuth of static_map's columns
        self.last = {}
        # an empty-scene background (set_background): its static part [range, channel], subtracted
        # coherently from every frame's virtual channels while use_background is on
        self.bg_virt, self.use_background = None, True
        # CUDA graph of the fixed-shape front end (_front), one per (moving_only, detect, static, background)
        self.cuda_graph = bool(cuda_graph) and self.dev.type == "cuda"
        self._graphs, self._x, self._pin, self._copied = {}, None, None, None

    # ── stages ─────────────────────────────────────────────────────────────
    def range_cube(self, raw: np.ndarray):
        """[tx, loop, rx, range] complex, calibrated, range-FFT'd."""
        return self._range_cube(self.torch.as_tensor(raw, device=self.dev))

    def _range_cube(self, x):
        torch = self.torch
        x = x.view(self.n_loops, self.n_tx, self.n_rx, self.n_s, 2).float()
        iq = torch.complex(x[..., 1], x[..., 0]) if self.iq_swap else torch.complex(x[..., 0], x[..., 1])
        cube = iq.permute(1, 0, 2, 3) * self.corr[:, None, :, None]       # TDM: chirp = loop * n_tx + tx
        rng = dsp.range_fft(cube, self.win_r, self.n_s)[..., :self.n_range]
        if self.chip_corr is not None:                                   # the slave's phase: constant + range ramp
            rng = rng * self.chip_corr[None, None]
        return rng

    def _beam(self, x):
        """Measured layout: virtual channels [..., channel] -> (power [..., n_az] linear, the groups'
        powers added, and the groups' spectra [..., groups, n_az] complex)."""
        g, _ = dsp.unique_aperture(x, self.ch_pos_np)                    # [..., groups * group]
        B = dsp.az_spectrum(g.reshape(g.shape[:-1] + (self.n_groups, self.group)), self.win_a, self.n_az)
        return (B.real ** 2 + B.imag ** 2).sum(dim=-2), B, g

    def _front(self, x, moving_only: bool, detect: bool, static: bool, background: bool = False) -> dict:
        """Everything of one frame with fixed shapes, from the raw frame x (int16, on the device)
        up to the CFAR noise maps: no host sync, so it can be a CUDA graph. Thresholds and the
        variable-size steps (nonzero onwards) stay in _detect / _detect_static. With background, the
        background's static part (set_background) is subtracted from the virtual channels first."""
        torch = self.torch
        F = torch.nn.functional
        rng = self._range_cube(x)
        virt = dsp.virtual_array(rng, self.ch_tx, self.ch_rx)                   # [loop, range, ch]
        if background:
            virt = virt - self.bg_virt[None]
        dch = dsp.doppler_fft(virt, self.win_d, self.n_loops, remove_clutter=moving_only)    # [dop, range, ch]
        if self.doppler_comp:
            dch = dch * self.tdm_comp[self.ch_tx].T[:, None, :]
        el_dop = grp = None
        if self.has_el:
            el = dsp.virtual_array(rng, self.el_tx, self.el_rx)                 # [loop, range, 8]
            el_dop = dsp.doppler_fft(el, self.win_d, self.n_loops, remove_clutter=moving_only)   # [dop, range, 8]
            if self.doppler_comp:
                el_dop = el_dop * self.tdm_comp[self.el_tx].T[:, None, :]
        if self.layout == "team":
            dop, _ = dsp.unique_aperture(dch, self.ch_pos_np)                  # uniform array [dop, range, pos]
            az = dsp.az_spectrum(dop, self.win_a, self.n_az)                    # [dop, range, az]
            p_az = az.real ** 2 + az.imag ** 2
        else:
            p_az, grp, dop = self._beam(dch)                                     # [dop, range, az], [.., groups, az]
        rd = dsp.power_db(dop, dim=-1)                                           # [dop, range]
        ra = 10 * torch.log10(p_az.sum(dim=0).T + 1e-6)                          # [az, range]
        f = {"rd": rd, "ra": ra, "p_az": p_az}               # p_az: [doppler, range, az] linear power (GPU)
        if detect:
            det_map = 10 * torch.log10(p_az[..., self.fov_sl].max(dim=-1).values + 1e-6)   # [dop, range]
            pd, pr = self.cfar_pad
            xm = det_map[None, None]
            xp = F.pad(F.pad(xm, (0, 0, pd, pd), mode="circular"), (pr, pr, 0, 0), mode="replicate")
            f.update(dop=dop, el_dop=el_dop, det_map=det_map, det_noise=F.conv2d(xp, self.cfar_kernel)[0, 0],
                     # 3x3 local maxima of the detection map (circular in Doppler)
                     det_max=F.max_pool2d(F.pad(xm, (0, 0, 1, 1), mode="circular"), 3, stride=1, padding=(0, 1))[0, 0],
                     det_median=det_map.median())
            f.update(self._cell_rows(p_az, dop, el_dop, det_map, f["det_noise"], grp))
        if static:
            b = self.static_bins
            d0 = dsp.doppler_band(virt, self.win_d, self.n_loops, b)    # = doppler_fft(...)[n/2 - b:n/2 + b + 1]
            m = 10 * torch.log10(self._static_power(d0)[:, self.fov_sl] + 1e-6)  # [range, az_fov]
            f.update(self._static_stats(m))
            f["st_band"] = d0
        return f

    def _static_power(self, band):
        """Azimuth power [range, n_az] (linear) of a zero-Doppler band [band, range, channel], summed
        over the band (and, measured layout, over the chips: incoherent)."""
        if self.layout == "team":
            cube, _ = dsp.unique_aperture(band, self.ch_pos_np)                 # [band, range, pos]
            az0 = dsp.az_spectrum(cube, self.win_a, self.n_az)
            return (az0.real ** 2 + az0.imag ** 2).sum(dim=0)
        return self._beam(band)[0].sum(dim=0)

    def _static_stats(self, m):
        """The static CFAR's maps of a [range, az_fov] dB static map m (no host sync)."""
        torch = self.torch
        F = torch.nn.functional
        tr_, ta = self.static_pad
        xs = m[None, None]
        f = dict(st_map=m, st_noise=F.conv2d(F.pad(xs, (ta, ta, tr_, tr_), mode="replicate"), self.static_kernel)[0, 0],
                 st_max=F.max_pool2d(F.pad(xs, (1, 1, 1, 1), mode="replicate"), 3, stride=1)[0, 0],
                 st_rowmax=m.max(dim=1, keepdim=True).values)
        snr = m - f["st_noise"]
        r = self.range_m_t[:, None]
        az_w = self.st_az_w[None, :]
        zero = torch.zeros_like(m)
        f.update(st_snr=snr, st_rows=torch.stack(       # one DET_COLS row per range-azimuth cell
            [r * torch.cos(torch.deg2rad(az_w)) + zero, r * torch.sin(torch.deg2rad(az_w)) + zero, zero,
             r + zero, az_w + zero, zero, zero, m, snr, zero], dim=-1))
        return f

    def detect_static_map(self, m, interpolate: bool = True) -> np.ndarray:
        """Stationary reflectors of ANY static map (dB, [range, az_fov], e.g. the mean power of the
        last frames' static_map, which is steadier than one frame): the same CFAR, local maxima,
        sidelobe prune and multipath flag as process(static=True). With interpolate, each
        detection's range and azimuth are refined by a parabola through its neighbours (sub-cell),
        and x, y follow."""
        torch = self.torch
        with torch.no_grad():
            m = torch.as_tensor(m, device=self.dev, dtype=torch.float32)
            f = self._static_stats(m)
            det = self._detect_static(f)
            if not interpolate or not len(det):
                return det
            mm = m.cpu().numpy()
        h, w = mm.shape
        az = np.asarray(self.static_az_deg, float)
        for row in det:
            ri = int(round(row[C["range"]] / self.range_res))
            ai = int(np.argmin(np.abs(az - row[C["az_deg"]])))
            fr = fa = 0.0
            if 0 < ri < h - 1:
                d = mm[ri - 1, ai] - 2 * mm[ri, ai] + mm[ri + 1, ai]
                fr = float(np.clip(0.5 * (mm[ri - 1, ai] - mm[ri + 1, ai]) / d, -0.5, 0.5)) if d < 0 else 0.0
            if 0 < ai < w - 1:
                d = mm[ri, ai - 1] - 2 * mm[ri, ai] + mm[ri, ai + 1]
                fa = float(np.clip(0.5 * (mm[ri, ai - 1] - mm[ri, ai + 1]) / d, -0.5, 0.5)) if d < 0 else 0.0
            r = (ri + fr) * self.range_res
            a = float(np.interp(ai + fa, np.arange(w), az))
            row[C["range"]], row[C["az_deg"]] = r, a
            row[C["x"]], row[C["y"]] = r * np.cos(np.radians(a)), r * np.sin(np.radians(a))
        return det

    def _cell_rows(self, p_az, dop, el_dop, det_map, noise, grp=None) -> dict:
        """The moving-target detection row (DET_COLS) of EVERY range-Doppler cell and azimuth peak,
        [dop, range, max_peaks, cols], with det_ok [dop, range, max_peaks] (a real peak): elementwise,
        so a detected cell's row is what computing it for that cell alone gives, but as a few large
        kernels in the graph instead of ~100 small ones per frame. _detect gathers the hit cells."""
        torch = self.torch
        A, R = self.n_az, self.n_range
        # azimuth peaks per cell
        if self.max_peaks == 1:
            # the strongest bin within the FOV is always a peak (its neighbours are <= it, or outside the
            # FOV), and its level is the detection map: only its two neighbours are needed
            _, im = p_az[..., self.fov_sl].max(dim=-1)
            top_i = (im + self.fov_sl.start)[..., None]                          # [dop, range, 1]
            top_v = det_map[..., None]

            def spec_at(idx):                                                    # dB, -1e9 outside the FOV
                v = 10 * torch.log10(torch.gather(p_az, -1, idx) + 1e-6)
                return torch.where(self.fov_ok[idx], v, torch.full_like(v, -1e9))
            y0, y2 = spec_at((top_i - 1).clamp(0, A - 1)), spec_at((top_i + 1).clamp(0, A - 1))
        else:
            spec = 10 * torch.log10(p_az + 1e-6)                                # [dop, range, az]
            spec = torch.where(self.fov_ok, spec, torch.full_like(spec, -1e9))
            left, right = torch.roll(spec, 1, dims=-1), torch.roll(spec, -1, dims=-1)
            is_pk = (spec >= left) & (spec >= right) & (spec >= spec.max(dim=-1, keepdim=True).values - self.peak_db)
            val = torch.where(is_pk, spec, torch.full_like(spec, -1e9))
            top_v, top_i = torch.topk(val, self.max_peaks, dim=-1)              # [dop, range, max_peaks]
            y0 = torch.gather(spec, -1, (top_i - 1).clamp(0, A - 1))
            y2 = torch.gather(spec, -1, (top_i + 1).clamp(0, A - 1))
        den = (y0 - 2 * top_v + y2)
        delta = torch.where(den.abs() > 1e-6, 0.5 * (y0 - y2) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)
        sin_proc = (self.sin_t[top_i] + delta * (2.0 / A)).clamp(-1, 1)
        if self.coherent and grp is not None:
            sin_proc = self._refine(grp, top_i, sin_proc)
        # range interpolation on the detection map
        r0 = det_map[:, self.r_prev]
        r2 = det_map[:, self.r_next]
        rden = r0 - 2 * det_map + r2
        rdel = torch.where(rden.abs() > 1e-6, 0.5 * (r0 - r2) / rden, torch.zeros_like(rden)).clamp(-0.5, 0.5)
        rng_m = (self.r_idx_f[None, :] + rdel) * self.range_res                # [dop, range]
        # elevation: phase of top x conj(bottom) over the pairs, at each range-Doppler cell (unverified;
        # team layout only: the measured layout's elevated TX positions are not measured yet -> 0)
        if self.has_el:
            phase = torch.angle((el_dop * dop[..., self.el_pos].conj()).sum(dim=-1))
            sin_el = (phase / (2 * np.pi * self.d_el)).clamp(-1, 1)
        else:
            sin_el = torch.zeros_like(rng_m)
        # assemble: one row per (cell, azimuth peak)
        az_w = AZIMUTH_SIGN * torch.rad2deg(torch.asin(sin_proc))              # world azimuth, + = left
        el_d = torch.rad2deg(torch.asin(sin_el))[..., None].expand_as(az_w)
        r = rng_m[..., None].expand_as(az_w)
        ca, ce = torch.cos(torch.deg2rad(az_w)), torch.cos(torch.deg2rad(el_d))
        rows = torch.stack([r * ce * ca, r * ce * torch.sin(torch.deg2rad(az_w)), r * torch.sin(torch.deg2rad(el_d)),
                            r, az_w, el_d, self.dop_axis_t[:, None, None].expand_as(az_w), top_v,
                            top_v - noise[..., None], torch.zeros_like(r)], dim=-1)
        return {"det_rows": rows, "det_ok": top_v > -1e8}

    def _refine(self, grp, top_i, sin_inc):
        """Coherent angle across the two chips (measured layout), per cell and peak: beam(s) =
        |B0(s) + B1(s) exp(-j pi D s)|^2 has fringes 2 / D apart under the incoherent peak; take the
        fringe nearest the peak (within 1 / D bins), parabolically interpolated, when it is
        refine_gate_db above every other fringe within 3 / D, else keep sin_inc. grp [dop, range,
        2, az] complex, top_i [dop, range, P] incoherent peak bins, sin_inc [dop, range, P]."""
        torch = self.torch
        A, n = self.n_az, self.ref_near
        P = top_i.shape[-1]
        idx = (top_i[..., None] + self.ref_offs).clamp(0, A - 1)                 # [dop, range, P, W]
        flat = idx.reshape(idx.shape[0], idx.shape[1], -1)
        b0 = torch.gather(grp[..., 0, :], -1, flat).reshape(idx.shape)
        b1 = torch.gather(grp[..., 1, :], -1, flat).reshape(idx.shape)
        coh = (b0 + b1 * self.chip_steer[idx]).abs() ** 2
        coh = torch.where(self.fov_ok[idx], coh, torch.zeros_like(coh))
        near = self.ref_offs.abs() <= n                                          # [W]
        ci = torch.where(near, coh, torch.zeros_like(coh)).argmax(dim=-1, keepdim=True)   # [dop, range, P, 1]
        cmax = torch.gather(coh, -1, ci)
        off_c = self.ref_offs[ci]                                                # offset of the chosen fringe
        other = (self.ref_offs - off_c).abs() > n
        omax = torch.where(other, coh, torch.zeros_like(coh)).max(dim=-1, keepdim=True).values
        ok = cmax >= omax * (10 ** (self.refine_gate_db / 10))
        W = coh.shape[-1]
        c0 = torch.gather(coh, -1, (ci - 1).clamp(0, W - 1))
        c2 = torch.gather(coh, -1, (ci + 1).clamp(0, W - 1))
        y0, y1, y2 = (10 * torch.log10(v + 1e-12) for v in (c0, cmax, c2))
        den = y0 - 2 * y1 + y2
        dl = torch.where(den.abs() > 1e-6, 0.5 * (y0 - y2) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)
        bin_c = torch.gather(idx, -1, ci).float() + dl
        sin_c = (-1.0 + 2.0 * bin_c / A).clamp(-1, 1)[..., 0]
        return torch.where(ok[..., 0] & (cmax[..., 0] > 0), sin_c, sin_inc)

    def _graphed(self, raw: np.ndarray, key: tuple) -> dict:
        """_front(raw, *key) as a CUDA graph replay: the frame goes through a pinned staging
        buffer into the graph's fixed input; returned maps are copies (the graph's own outputs
        are overwritten by the next frame)."""
        torch = self.torch
        raw = np.asarray(raw).reshape(-1)
        if self._x is None or self._x.numel() != raw.size:
            self._graphs = {}
            self._x = torch.zeros(raw.size, dtype=torch.int16, device=self.dev)
            self._pin = torch.empty(raw.size, dtype=torch.int16).pin_memory()
            self._copied = torch.cuda.Event()
        self._copied.synchronize()                      # the previous frame's upload has left the buffer
        self._pin.numpy()[:] = raw
        self._x.copy_(self._pin, non_blocking=True)
        self._copied.record()
        if key not in self._graphs:
            self._graphs[key] = self._capture(key)
        g, f = self._graphs[key]
        g.replay()
        return {k: (v.clone() if k in ("rd", "ra", "p_az", "st_map", "st_band") else v) for k, v in f.items()}

    def _capture(self, key: tuple):
        """Warm up (cuFFT plans, cached tables) on a side stream, then capture _front, holding
        GPU_LOCK so that no other thread's CUDA work breaks the capture."""
        torch = self.torch
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._front(self._x, *key)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with GPU_LOCK:
            with torch.cuda.graph(g):
                f = self._front(self._x, *key)
        return g, f

    def process(self, raw: np.ndarray, moving_only: bool = True, detect: bool = True, static: bool = False):
        """Returns dict: rd [doppler, range] dB, ra [az, range] dB (processing azimuth order),
        p_az [doppler, range, az] linear power, and, if detect, detections (N x len(DET_COLS)) and
        noise_db (frame median of the detection map); if static, static_detections
        (N x len(DET_COLS), v_r = 0): see _detect_static, and static_map, the zero-Doppler
        range-azimuth map without clutter removal [range, azimuth within the FOV] dB (world
        azimuths: static_az_deg), and static_band, its zero-Doppler band (static_band; for
        refine_azimuth). With a background set (set_background) and use_background on, every map
        is of the change from it (the moving-target path is unaffected: clutter removal takes
        each frame's static part out anyway)."""
        key = (bool(moving_only), bool(detect), bool(static), self.background_active)
        f = self._graphed(raw, key) if self.cuda_graph else \
            self._front(self.torch.as_tensor(raw, device=self.dev), *key)
        out = {"rd": f["rd"], "ra": f["ra"], "p_az": f["p_az"]}
        if detect:
            dets, noise = self._detect(f, moving_only)
            out.update(detections=dets, noise_db=noise)
        if static:
            out["static_detections"] = self._detect_static(f)
            out["static_map"] = f["st_map"]
            out["static_band"] = f["st_band"]
        self.last = out
        return out

    @property
    def background_active(self) -> bool:
        return self.bg_virt is not None and bool(self.use_background)

    def set_background(self, raws) -> int:
        """An empty-scene background from raw frames (the same firmware): the mean over its frames and
        loops of the virtual channels [range, channel] (complex), i.e. its static part. process() then
        subtracts it from every frame, coherently, before beamforming, so the static maps show only
        what changed (a corner reflector without the walls' sidelobes). Returns the frames used."""
        torch = self.torch
        acc, n = None, 0
        with torch.no_grad():
            for raw in raws:
                virt = dsp.virtual_array(self._range_cube(torch.as_tensor(raw, device=self.dev)), self.ch_tx, self.ch_rx)
                m = virt.mean(dim=0)
                acc = m if acc is None else acc + m
                n += 1
        if n == 0:
            raise ValueError("no background frames")
        self.bg_virt = (acc / n).contiguous()
        self._graphs = {k: v for k, v in self._graphs.items() if not k[3]}   # graphs hold the old buffer
        return n

    def clear_background(self) -> None:
        self.bg_virt = None
        self._graphs = {k: v for k, v in self._graphs.items() if not k[3]}

    def static_band(self, raw: np.ndarray):
        """The zero-Doppler band (+-static_bins) of the virtual channels, no clutter removal:
        complex [2 static_bins + 1, range, channel] on the device, what static_map is made of. Two
        bands can be subtracted COHERENTLY (a position minus an empty-scene background, e.g.
        calib_reflector.py) before static_map_of: the change is then beamformed on its own, without
        the sidelobes of the walls around it."""
        torch = self.torch
        with torch.no_grad():
            virt = dsp.virtual_array(self._range_cube(torch.as_tensor(raw, device=self.dev)), self.ch_tx, self.ch_rx)
            return dsp.doppler_band(virt, self.win_d, self.n_loops, self.static_bins)

    def static_map_of(self, band) -> np.ndarray:
        """[range, azimuth in the FOV] dB map of a static_band (or a difference of bands), exactly as
        process(static=True)["static_map"] (columns at static_az_deg, world + left)."""
        torch = self.torch
        with torch.no_grad():
            p = self._static_power(band)[:, self.fov_sl]
            return (10 * torch.log10(p + 1e-6)).float().cpu().numpy()

    def refine_azimuth(self, bands, range_m: float, az_deg: float) -> tuple:
        """Coherent azimuth of ONE static reflector (measured layout), from its zero-Doppler bands (e.g.
        a position minus the background, static_band): the beams of the two chips combined coherently,
        summed over the bands, at the reflector's range bin; the fringe nearest az_deg (world, + left;
        from the incoherent map) is taken under the same gate as _refine. Returns (az_deg, refined):
        az_deg unchanged and False when there is no inter-chip phase or the gate fails. On the
        2026-11-03 corner reflector it brings the spread against the camera from ~0.4 to ~0.1 deg."""
        az, ok = self.refine_azimuths(bands, [range_m], [az_deg])
        return float(az[0]), bool(ok[0])

    def refine_azimuths(self, bands, ranges_m, az_deg) -> tuple:
        """refine_azimuth for several reflectors at once (one GPU pass): (az_deg array, refined array)."""
        az_deg = np.asarray(az_deg, float).reshape(-1)
        out, ok = az_deg.copy(), np.zeros(len(az_deg), bool)
        if not self.coherent or not len(bands) or not len(az_deg):
            return out, ok
        torch = self.torch
        ri = torch.as_tensor(np.clip(np.round(np.asarray(ranges_m, float) / self.range_res), 0, self.n_range - 1)
                             .astype(np.int64), device=self.dev)
        coh = torch.zeros(len(az_deg), self.n_az, device=self.dev)
        with torch.no_grad():
            for b in bands:
                _, B, _ = self._beam(torch.as_tensor(b, device=self.dev)[:, ri])       # [band, rows, 2, az]
                coh += ((B[..., 0, :] + B[..., 1, :] * self.chip_steer).abs() ** 2).sum(dim=0)
        coh = np.where(self.fov_ok.cpu().numpy()[None], coh.cpu().numpy(), 0.0)
        A, n = self.n_az, self.ref_near
        for j in range(len(az_deg)):
            c_ = coh[j]
            i0 = int(np.argmin(np.abs(self.sin_axis - np.sin(np.radians(AZIMUTH_SIGN * az_deg[j])))))
            lo, hi = max(i0 - n, 0), min(i0 + n + 1, A)
            ci = lo + int(np.argmax(c_[lo:hi]))
            lobe = np.zeros(A, bool)
            lobe[max(i0 - 3 * n, 0):min(i0 + 3 * n + 1, A)] = True
            other = lobe & (np.abs(np.arange(A) - ci) > n)
            omax = c_[other].max() if other.any() else 0.0
            if c_[ci] <= 0 or c_[ci] < omax * 10 ** (self.refine_gate_db / 10):
                continue
            d = 0.0
            if 0 < ci < A - 1:
                y0, y1, y2 = (10 * np.log10(c_[k] + 1e-12) for k in (ci - 1, ci, ci + 1))
                den = y0 - 2 * y1 + y2
                d = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) if abs(den) > 1e-6 else 0.0
            sn = float(np.clip(-1.0 + 2.0 * (ci + d) / A, -1, 1))
            out[j], ok[j] = AZIMUTH_SIGN * np.degrees(np.arcsin(sn)), True
        return out, ok

    def _detect_static(self, f):
        """Stationary reflectors, which the moving-target path removes: the zero-Doppler band
        (+-static_bins) of the cube WITHOUT static-clutter removal, beamformed in azimuth into a
        static range-azimuth map (within the FOV), then a 2D CA-CFAR in dB over range x azimuth,
        3x3 local maxima, and a sidelobe prune per range row (within static_sidelobe_db of the row's
        strongest cell: this array's sidelobes are only 5-10 dB down), within static_max_range_m
        (beyond ~60 m the static map shows artifacts even indoors). Multipath images (n x the range
        of a stronger detection at the same azimuth) are flagged in the "ghost" column. Columns as
        DET_COLS, v_r = 0, z = 0 (no elevation). The maps come from _front."""
        torch = self.torch
        m, snr = f["st_map"], f["st_snr"]
        hit = (snr > self.static_db) & self.static_range_ok[:, None]
        hit &= m >= f["st_max"]
        hit &= m >= f["st_rowmax"] - self.static_sidelobe_db
        ri, ai = torch.nonzero(hit, as_tuple=True)
        if ri.numel() > self.max_det:
            keep = torch.topk(snr[ri, ai], self.max_det).indices
            ri, ai = ri[keep], ai[keep]
        if ri.numel() == 0:
            return np.zeros((0, len(DET_COLS)), np.float32)
        cols = f["st_rows"][ri, ai]
        det = cols.float().cpu().numpy()
        det[:, C["ghost"]] = flag_multipath(det, float(self.dop_axis[-1] - self.dop_axis[0] + self.dop_axis[1]
                                                       - self.dop_axis[0]))     # n-bounce images (v = 0)
        return det

    def _detect(self, f, moving_only):
        torch = self.torch
        det_map = f["det_map"]
        snr = det_map - f["det_noise"]
        hit = snr > self.threshold_db
        hit &= self.range_ok[None, :]
        if moving_only:
            hit &= self.dop_ok[:, None]
        hit &= det_map >= f["det_max"]
        di, ri = torch.nonzero(hit, as_tuple=True)
        if di.numel() > self.max_det:
            keep = torch.topk(snr[di, ri], self.max_det).indices
            di, ri = di[keep], ri[keep]
        noise_db = float(f["det_median"])
        if di.numel() == 0:
            return np.zeros((0, len(DET_COLS)), np.float32), noise_db
        rows, ok = f["det_rows"][di, ri], f["det_ok"][di, ri]                   # [N, max_peaks, cols], [N, max_peaks]
        cols = rows[ok]                                                          # one row per (cell, azimuth peak)
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
