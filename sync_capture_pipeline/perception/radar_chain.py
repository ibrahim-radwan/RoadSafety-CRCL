"""
The old chain's front end (process_frame stages 1-7 and 9: calibration to the range-Doppler,
CFAR, range-azimuth and range-elevation maps) composed from the PyTorch stages in radar_dsp.py.

It reproduces the old chain's golden frames on both of its paths (cfar = "caso": the CuPy path,
as process_frame requests; cfar = "os": the CPU fallback); tests/test_baseline.py checks this.
The parameters are the old chain's (RadarConfig from the .cfg); each is an argument, so a pipeline
can change one (for example max_range_m: the old chain stops at 16 m) without touching the stages.

Not ported here (merge step 3, pending a decision): stage 8, the single-subject "body-shape"
gates that turn the maps into ra_det and the xyz point cloud (an angle gate around the peak
nearest boresight, a cross-cluster gate, a subject gate around the tracked person).

    chain = RadarChain(cfg, corr, cfar="caso")
    out = chain.process(raw_int16)      # dict of tensors: RD_raw, RD_mti, rd_thresh, rd_det, RA, RE, ...
"""
from __future__ import annotations

import numpy as np
import torch

from . import radar_dsp as dsp
from .core.radar_config import RadarConfig


def _elevation_pairs(ch_az: list, ch_el: list):
    """Top-row elevation channels and the bottom-row azimuth channel at the same position."""
    first = {}
    for i, c in enumerate(ch_az):
        first.setdefault(int(c["az"]), i)
    bot, top = [], []
    for e, c in enumerate(ch_el):
        if int(c["az"]) in first:
            bot.append(first[int(c["az"])])
            top.append(e)
    return np.array(bot, int), np.array(top, int)


class RadarChain:
    def __init__(self, cfg: RadarConfig, corr: np.ndarray, cfar: str = None, device: str = "cuda"):
        self.cfg = cfg
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.cfar = (cfar or cfg.cfar_method).lower()
        self.corr = torch.as_tensor(np.asarray(corr), dtype=torch.complex64, device=self.dev)
        ch_az, ch_el, _ = cfg.build_channel_table()
        self.az_tx = torch.tensor([c["tx"] for c in ch_az], device=self.dev)
        self.az_rx = torch.tensor([c["rx"] for c in ch_az], device=self.dev)
        self.el_tx = torch.tensor([c["tx"] for c in ch_el], device=self.dev)
        self.el_rx = torch.tensor([c["rx"] for c in ch_el], device=self.dev)
        self.az_pos = np.array([c["az"] for c in ch_az], float)
        bot, top = _elevation_pairs(ch_az, ch_el)
        self.bot_idx = torch.as_tensor(bot, device=self.dev)
        self.top_idx = torch.as_tensor(top, device=self.dev)
        a = cfg.chebyshev_atten_db
        self.win_r = dsp.window(cfg.range_window, cfg.num_samples, a, self.dev)
        self.win_d = dsp.window(cfg.doppler_window, cfg.num_loops, a, self.dev)
        self.uniq_pos = np.unique(self.az_pos) if getattr(cfg, "music_use_unique_aperture", True) else self.az_pos
        self.win_a = dsp.window(cfg.az_window, len(self.uniq_pos), a, self.dev)
        self.win_e = dsp.window(cfg.el_window, len(top), a, self.dev)
        self.n_r = int(cfg.n_range)
        self.n_dop = int(cfg.doppler_fft_size or cfg.num_loops)
        self.range_axis = torch.as_tensor(np.asarray(cfg.range_axis, np.float32), device=self.dev)
        self.keep_r = dsp.range_keep_mask(self.range_axis[:self.n_r], cfg.min_range_m, cfg.max_range_m)

    def cube(self, raw: np.ndarray) -> torch.Tensor:
        """Raw int16 frame -> complex cube [tx, loop, rx, sample] (the old reader's layout)."""
        c = self.cfg
        x = torch.as_tensor(np.asarray(raw), device=self.dev).view(c.num_loops * c.num_tx, c.num_rx, c.num_samples, 2)
        x = x.float()
        iq = torch.complex(x[..., 1], x[..., 0]) if c.iq_swap else torch.complex(x[..., 0], x[..., 1])
        return iq.view(c.num_loops, c.num_tx, c.num_rx, c.num_samples).permute(1, 0, 2, 3)

    def process(self, raw: np.ndarray) -> dict:
        c, N_r = self.cfg, self.n_r
        cube = self.cube(raw)
        if c.apply_calibration:
            cube = cube * self.corr[:, None, :, None]
        rng = dsp.range_fft(cube, self.win_r, int(c.range_fft_size or c.num_samples))
        va_az = dsp.virtual_array(rng, self.az_tx, self.az_rx)               # [loop, range, 40]
        va_el = dsp.virtual_array(rng, self.el_tx, self.el_rx)               # [loop, range, 8]
        rd_raw = dsp.doppler_fft(va_az, self.win_d, self.n_dop, remove_clutter=False)
        rd_mti = dsp.doppler_fft(va_az, self.win_d, self.n_dop, remove_clutter=c.clutter_removal)
        el_top = dsp.doppler_fft(va_el[:, :, self.top_idx], self.win_d, self.n_dop, c.clutter_removal)
        el_bot = dsp.doppler_fft(va_az[:, :, self.bot_idx], self.win_d, self.n_dop, c.clutter_removal)
        rd_mti = dsp.zero_dc_notch(rd_mti, c.zero_dc_bins)
        el_top = dsp.zero_dc_notch(el_top, c.zero_dc_bins)
        el_bot = dsp.zero_dc_notch(el_bot, c.zero_dc_bins)
        RD_raw = dsp.power_db(rd_raw[:, :N_r], dim=-1)
        RD_mti = dsp.power_db(rd_mti[:, :N_r], dim=-1)
        if getattr(c, "doppler_stripe_removal", False):
            RD_raw = dsp.suppress_doppler_stripes(RD_raw)
            RD_mti = dsp.suppress_doppler_stripes(RD_mti)
        if c.gaussian_sigma > 0:
            raise NotImplementedError("gaussian_sigma > 0 is not ported (0 in every shipped .cfg)")
        det, thr = dsp.cfar_2d(10.0 ** (RD_mti / 10.0), c.cfar_guard, c.cfar_train, self.cfar, c.cfar_fa_rate,
                               getattr(c, "cfar_os_rank", 0.75), c.cfar_margin_db,
                               getattr(c, "cfar_min_power_db", 0.0), c.nms_size)
        rd_thresh = 10.0 * torch.log10(torch.clamp(thr, min=1e-12))
        if getattr(c, "apply_sidelobe_filters", False):
            det = dsp.prune_rd_sidelobes(det, RD_mti, float(getattr(c, "rd_sidelobe_db", 0.0)))
        det &= self.keep_r[None, :]
        det = dsp.consistency_mask(det, rd_mti[:, :N_r], int(getattr(c, "consistency_k", 2)))
        focus = dsp.focus_mask(RD_mti, det)
        cube_az, _ = dsp.unique_aperture(rd_mti[:, :N_r], self.az_pos) if len(self.uniq_pos) < len(self.az_pos) \
            else (rd_mti[:, :N_r], self.az_pos)
        method = str(c.aoa_method).lower()
        if method == "capon":
            RA = dsp.ra_capon(cube_az, focus, self.uniq_pos, self.win_a, c.az_fft_size,
                              float(getattr(c, "music_diag_load", 0.0)), bool(getattr(c, "music_forward_backward_avg", False)))
        elif method == "music":
            RA = dsp.ra_music(cube_az, focus, self.uniq_pos, self.win_a, c.az_fft_size, int(c.music_num_sources),
                              float(getattr(c, "music_diag_load", 0.0)), bool(getattr(c, "music_forward_backward_avg", False)))
        elif method == "bartlett":
            RA = dsp.ra_bartlett(cube_az, focus, self.uniq_pos, self.win_a, c.az_fft_size,
                                 bool(getattr(c, "music_forward_backward_avg", False)))
        else:
            RA = dsp.focused_ra(cube_az, focus, self.win_a, c.az_fft_size)
        RE = dsp.elevation_map(el_top[:, :N_r], el_bot[:, :N_r], float(c.d_el_lambda), self.win_e, c.el_fft_size)
        return {"RD_raw": RD_raw, "RD_mti": RD_mti, "rd_thresh": rd_thresh, "rd_det": det, "RA": RA[:N_r],
                "RE": RE, "focus": focus, "rd_mti_channels": rd_mti[:, :N_r]}
