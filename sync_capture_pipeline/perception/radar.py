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

from pathlib import Path

import numpy as np

from . import radar_dsp as dsp
from .core.calibration import RX_PHASE_FILE, apply_phase, load_radar_config, rx_phase_for  # noqa: F401
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
                 static_bins: int = 1, static_max_range_m: float = 60.0, cuda_graph: bool = True):
        import torch
        self.torch = torch
        cfg, corr = load_radar_config(cfg_path)
        self.cfg = cfg
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_tx, self.n_rx, self.n_loops, self.n_s = cfg.num_tx, cfg.num_rx, cfg.num_loops, cfg.num_samples
        self.iq_swap = bool(cfg.iq_swap)
        corr = apply_phase(corr, rx_phase_deg)   # in-situ correction: per RX (8) or per channel (6 x 8)
        self.rx_phase_deg = None if rx_phase_deg is None else np.asarray(rx_phase_deg, float)
        self.corr = torch.as_tensor(corr, dtype=torch.complex64, device=self.dev)
        ch_az, ch_el, _ = cfg.build_channel_table()
        self.n_pos = n_pos = max(c["az"] for c in ch_az) + 1
        self.ch_tx = torch.tensor([c["tx"] for c in ch_az], device=self.dev)
        self.ch_rx = torch.tensor([c["rx"] for c in ch_az], device=self.dev)
        self.ch_pos_np = np.array([c["az"] for c in ch_az], float)              # aperture position per channel
        # elevation row: top channel and the azimuth position it shares with the bottom row
        self.el_tx = torch.tensor([c["tx"] for c in ch_el], device=self.dev)
        self.el_rx = torch.tensor([c["rx"] for c in ch_el], device=self.dev)
        self.el_pos = torch.tensor([c["az"] for c in ch_el], device=self.dev)   # = index in the uniform array
        self.d_el = float(cfg.d_el_lambda)
        self.win_r = dsp.window("hann", self.n_s, device=self.dev)
        self.win_d = dsp.window("hann", self.n_loops, device=self.dev)
        self.win_a = dsp.window("hann", n_pos, device=self.dev)
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
        # CUDA graph of the fixed-shape front end (_front), one per (moving_only, detect, static)
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
        return dsp.range_fft(cube, self.win_r, self.n_s)[..., :self.n_range]

    def _front(self, x, moving_only: bool, detect: bool, static: bool) -> dict:
        """Everything of one frame with fixed shapes, from the raw frame x (int16, on the device)
        up to the CFAR noise maps: no host sync, so it can be a CUDA graph. Thresholds and the
        variable-size steps (nonzero onwards) stay in _detect / _detect_static."""
        torch = self.torch
        F = torch.nn.functional
        rng = self._range_cube(x)
        virt = dsp.virtual_array(rng, self.ch_tx, self.ch_rx)                   # [loop, range, ch]
        el = dsp.virtual_array(rng, self.el_tx, self.el_rx)                     # [loop, range, 8]
        dch = dsp.doppler_fft(virt, self.win_d, self.n_loops, remove_clutter=moving_only)    # [dop, range, ch]
        el_dop = dsp.doppler_fft(el, self.win_d, self.n_loops, remove_clutter=moving_only)   # [dop, range, 8]
        if self.doppler_comp:
            dch = dch * self.tdm_comp[self.ch_tx].T[:, None, :]
            el_dop = el_dop * self.tdm_comp[self.el_tx].T[:, None, :]
        dop, _ = dsp.unique_aperture(dch, self.ch_pos_np)                      # uniform array [dop, range, pos]
        rd = dsp.power_db(dop, dim=-1)                                           # [dop, range]
        az = dsp.az_spectrum(dop, self.win_a, self.n_az)                        # [dop, range, az]
        p_az = az.real ** 2 + az.imag ** 2
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
            f.update(self._cell_rows(p_az, dop, el_dop, det_map, f["det_noise"]))
        if static:
            b = self.static_bins
            d0 = dsp.doppler_band(virt, self.win_d, self.n_loops, b)    # = doppler_fft(...)[n/2 - b:n/2 + b + 1]
            cube, _ = dsp.unique_aperture(d0, self.ch_pos_np)                   # [band, range, pos]
            az0 = dsp.az_spectrum(cube, self.win_a, self.n_az)
            m = 10 * torch.log10((az0.real ** 2 + az0.imag ** 2).sum(dim=0)[:, self.fov_sl] + 1e-6)   # [range, az_fov]
            tr_, ta = self.static_pad
            xs = m[None, None]
            f.update(st_map=m, st_noise=F.conv2d(F.pad(xs, (ta, ta, tr_, tr_), mode="replicate"), self.static_kernel)[0, 0],
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

    def _cell_rows(self, p_az, dop, el_dop, det_map, noise) -> dict:
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
        # range interpolation on the detection map
        r0 = det_map[:, self.r_prev]
        r2 = det_map[:, self.r_next]
        rden = r0 - 2 * det_map + r2
        rdel = torch.where(rden.abs() > 1e-6, 0.5 * (r0 - r2) / rden, torch.zeros_like(rden)).clamp(-0.5, 0.5)
        rng_m = (self.r_idx_f[None, :] + rdel) * self.range_res                # [dop, range]
        # elevation: phase of top x conj(bottom) over the pairs, at each range-Doppler cell (unverified)
        phase = torch.angle((el_dop * dop[..., self.el_pos].conj()).sum(dim=-1))
        sin_el = (phase / (2 * np.pi * self.d_el)).clamp(-1, 1)
        # assemble: one row per (cell, azimuth peak)
        az_w = AZIMUTH_SIGN * torch.rad2deg(torch.asin(sin_proc))              # world azimuth, + = left
        el_d = torch.rad2deg(torch.asin(sin_el))[..., None].expand_as(az_w)
        r = rng_m[..., None].expand_as(az_w)
        ca, ce = torch.cos(torch.deg2rad(az_w)), torch.cos(torch.deg2rad(el_d))
        rows = torch.stack([r * ce * ca, r * ce * torch.sin(torch.deg2rad(az_w)), r * torch.sin(torch.deg2rad(el_d)),
                            r, az_w, el_d, self.dop_axis_t[:, None, None].expand_as(az_w), top_v,
                            top_v - noise[..., None], torch.zeros_like(r)], dim=-1)
        return {"det_rows": rows, "det_ok": top_v > -1e8}

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
        return {k: (v.clone() if k in ("rd", "ra", "p_az", "st_map") else v) for k, v in f.items()}

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
        azimuths: static_az_deg)."""
        key = (bool(moving_only), bool(detect), bool(static))
        f = self._graphed(raw, key) if self.cuda_graph else \
            self._front(self.torch.as_tensor(raw, device=self.dev), *key)
        out = {"rd": f["rd"], "ra": f["ra"], "p_az": f["p_az"]}
        if detect:
            dets, noise = self._detect(f, moving_only)
            out.update(detections=dets, noise_db=noise)
        if static:
            out["static_detections"] = self._detect_static(f)
            out["static_map"] = f["st_map"]
        self.last = out
        return out

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
