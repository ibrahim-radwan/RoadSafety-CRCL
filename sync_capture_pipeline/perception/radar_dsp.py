"""
Radar signal-processing stages, in PyTorch (GPU or CPU, the same code).

Ported from the old chain (cascaded_radar_capture_pipeline/processing: radar_processing.py and its
CuPy copy gpu_accel.py) in merge step 3, each stage tested for equivalence against the old numpy
function on real frames (tests/test_radar_dsp.py) and, composed in radar_chain.py, against the
golden frames of both of the old chain's paths (tests/test_baseline.py).

Shapes follow the old chain: a Doppler cube is [doppler, range, channel]; maps are [doppler, range]
(RD), [range, azimuth] (RA) and [range, elevation] (RE); power maps are in dB unless a name says
"lin". Azimuth bins are the processing's (+ right, see core/conventions.py).

Where the old code relied on a library detail, the port reproduces it:
  * scipy.ndimage.convolve flips its kernel (CASO's left/right masks are not symmetric);
  * ndimage mode "mirror" = torch "reflect"; mode "nearest" = "replicate";
  * numpy's median of an even count averages the two middle values (torch.median does not).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_WIN = {}
_APERTURE = {}
_DFT = {}


# ── basics ──────────────────────────────────────────────────────────────────
def window(name: str, n: int, atten: float = 60.0, device="cpu") -> torch.Tensor:
    """The old chain's windows (scipy, symmetric) as float32 tensors: hann, hamming, blackman,
    chebyshev (atten dB), none/rect."""
    key = (name.lower(), int(n), float(atten), str(device))
    if key not in _WIN:
        from scipy.signal import windows as sw
        nm = key[0]
        w = (np.ones(n) if nm in ("none", "rect") else sw.hann(n) if nm in ("hann", "hanning")
             else sw.hamming(n) if nm == "hamming" else sw.blackman(n) if nm == "blackman"
             else sw.chebwin(n, atten) if nm in ("chebyshev", "chebwin") else np.ones(n))
        _WIN[key] = torch.as_tensor(w.astype(np.float32), device=device)
    return _WIN[key]


def median(x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """numpy-style median (the mean of the two middle values for an even count)."""
    return torch.quantile(x, 0.5, dim=dim, keepdim=keepdim)


def power_db(x: torch.Tensor, dim: int = -1, ref: float = 1e-12) -> torch.Tensor:
    """Non-coherent integration: sum |x|^2 over `dim`, in dB."""
    return 10.0 * torch.log10(torch.clamp((x.real ** 2 + x.imag ** 2).sum(dim=dim), min=ref))


# ── range, virtual arrays, Doppler ──────────────────────────────────────────
def range_fft(cube: torch.Tensor, win: torch.Tensor, n: int) -> torch.Tensor:
    """[..., samples] -> [..., n] range spectrum (window on the samples)."""
    return torch.fft.fft(cube * win, n=n, dim=-1)


def virtual_array(range_cube: torch.Tensor, tx: torch.Tensor, rx: torch.Tensor) -> torch.Tensor:
    """range_cube [tx, loop, rx, range] -> [loop, range, channel] for the (tx, rx) channel list."""
    return range_cube[tx, :, rx, :].permute(1, 2, 0)


def doppler_fft(x: torch.Tensor, win: torch.Tensor, n: int, remove_clutter: bool = False) -> torch.Tensor:
    """[loop, range, channel] -> [doppler, range, channel], fftshifted. remove_clutter: subtract the
    mean over the loops first (MTI: the static scene)."""
    if remove_clutter:
        x = x - x.mean(dim=0, keepdim=True)
    return torch.fft.fftshift(torch.fft.fft(x * win[:, None, None], n=n, dim=0), dim=0)


def doppler_band(x: torch.Tensor, win: torch.Tensor, n: int, half: int) -> torch.Tensor:
    """The 2 half + 1 Doppler bins around zero of doppler_fft(x, win, n) (no clutter removal),
    [loop, range, channel] -> [2 half + 1, range, channel], as one small DFT matmul instead of
    the full FFT."""
    key = (int(n), int(half), x.shape[0], str(x.device))
    if key not in _DFT:
        k = np.arange(-half, half + 1)[:, None]                                  # fftshifted bins
        e = np.exp(-2j * np.pi * k * np.arange(x.shape[0])[None, :] / n)
        _DFT[key] = torch.as_tensor(e, dtype=torch.complex64, device=x.device) * win[None, :]
    L, R, C = x.shape
    return (_DFT[key] @ x.reshape(L, R * C)).reshape(-1, R, C)


def zero_dc_notch(x: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Zero the 2 n_bins + 1 Doppler rows around zero velocity (axis 0, fftshifted)."""
    if n_bins <= 0:
        return x
    mid = x.shape[0] // 2
    out = x.clone()
    out[max(0, mid - n_bins):min(x.shape[0], mid + n_bins + 1)] = 0
    return out


def suppress_doppler_stripes(rd_db: torch.Tensor) -> torch.Tensor:
    """Remove each Doppler row's median over range (stripes from phase noise / vibration), keeping
    the global level: rd - median_row + median(median_row)."""
    base = median(rd_db, dim=1, keepdim=True)
    return rd_db - base + median(base.flatten(), dim=0)


# ── CFAR ────────────────────────────────────────────────────────────────────
def _cfar_masks(guard, train):
    g0, g1 = int(guard[0]), int(guard[1])
    t0, t1 = int(train[0]), int(train[1])
    msz = (2 * (g0 + t0) + 1, 2 * (g1 + t1) + 1)
    mask = np.ones(msz, np.float32)
    mask[t0:t0 + 2 * g0 + 1, t1:t1 + 2 * g1 + 1] = 0.0
    split = t1 + g1
    left, right = mask.copy(), mask.copy()
    left[:, split + 1:] = 0.0
    right[:, :split] = 0.0
    return mask, left, right, msz


def _ndconvolve_mirror(x: torch.Tensor, kernel: np.ndarray) -> torch.Tensor:
    """scipy.ndimage.convolve(x, kernel, mode='mirror') for odd kernels: a correlation with the
    flipped kernel over reflect padding."""
    k = torch.as_tensor(np.ascontiguousarray(kernel[::-1, ::-1]), device=x.device)[None, None]
    ph, pw = kernel.shape[0] // 2, kernel.shape[1] // 2
    xp = F.pad(x[None, None], (pw, pw, ph, ph), mode="reflect")
    return F.conv2d(xp, k)[0, 0]


def cfar_2d(heat_lin: torch.Tensor, guard=(4, 6), train=(8, 16), method: str = "caso", fa_rate: float = 3e-5,
            os_rank: float = 0.75, margin_db: float = 0.0, min_power_db: float = 0.0, nms=(3, 3)):
    """2D CFAR on a linear-power map [doppler, range] -> (detections bool, threshold linear).

    method: "ca" (cell averaging), "caso"/"cago" (smallest/greatest of the left and right halves,
    split along range), "os" (ordered statistic: the os_rank quantile of the training cells).
    Threshold = alpha * noise, alpha = N (fa_rate^(-1/N) - 1). Then: margin_db above the threshold,
    min_power_db absolute, and non-maximum suppression over an nms window (dB, 0.05 dB tolerance)."""
    heat = heat_lin.float()
    mask, left, right, msz = _cfar_masks(guard, train)
    n_tc = int(np.count_nonzero(mask))
    method = method.lower()
    if method == "os":
        fill = float(heat[torch.isfinite(heat)].min()) if bool(torch.isfinite(heat).any()) else 0.0
        ph, pw = msz[0] // 2, msz[1] // 2
        xp = F.pad(heat[None, None], (pw, pw, ph, ph), mode="constant", value=fill)
        cols = F.unfold(xp, msz)[0]                                   # [win cells, H*W]
        sel = torch.as_tensor(mask.flatten().astype(bool), device=heat.device)
        train_cells = cols[sel]                                       # [n_tc, H*W]
        rank = int(np.clip(np.ceil(float(os_rank) * n_tc) - 1, 0, n_tc - 1))
        noise = torch.kthvalue(train_cells, rank + 1, dim=0).values.view_as(heat)
        alpha = n_tc * (fa_rate ** (-1.0 / max(n_tc, 1)) - 1.0)
    elif method in ("caso", "cago"):
        n_l, n_r = float(np.count_nonzero(left)) or 1.0, float(np.count_nonzero(right)) or 1.0
        ln = _ndconvolve_mirror(heat, left / n_l)
        rn = _ndconvolve_mirror(heat, right / n_r)
        noise = torch.minimum(ln, rn) if method == "caso" else torch.maximum(ln, rn)
        cells = max(min(n_l, n_r), 1.0)
        alpha = cells * (fa_rate ** (-1.0 / cells) - 1.0)
    else:
        cells = float(n_tc) or 1.0
        noise = _ndconvolve_mirror(heat, mask / cells)
        alpha = cells * (fa_rate ** (-1.0 / cells) - 1.0)
    thresh = alpha * noise
    det = heat > thresh
    if margin_db > 0:
        ratio = 10.0 * torch.log10(torch.clamp(heat, min=1e-12) / torch.clamp(thresh, min=1e-12))
        det &= ratio >= margin_db
    heat_db = 10.0 * torch.log10(torch.clamp(heat, min=1e-12))
    if min_power_db > 0:
        det &= heat_db >= float(min_power_db)
    nk = tuple(max(1, int(k)) for k in nms)
    xp = F.pad(heat[None, None], (nk[1] // 2, nk[1] - 1 - nk[1] // 2, nk[0] // 2, nk[0] - 1 - nk[0] // 2),
               mode="replicate")
    lmx = F.max_pool2d(xp, nk, stride=1)[0, 0]
    det &= heat_db >= 10.0 * torch.log10(torch.clamp(lmx, min=1e-12)) - 0.05
    return det, thresh


# ── detection filters ───────────────────────────────────────────────────────
def prune_rd_sidelobes(det: torch.Tensor, rd_db: torch.Tensor, rel_db: float) -> torch.Tensor:
    """Keep detections within rel_db of their range column's strongest Doppler bin."""
    if rel_db <= 0:
        return det
    return det & (rd_db >= rd_db.max(dim=0, keepdim=True).values - rel_db)


def range_keep_mask(range_axis: torch.Tensor, min_m: float, max_m: float) -> torch.Tensor:
    """Range bins within [min_m, max_m] (max_m <= 0: no upper limit)."""
    hi = float(range_axis[-1]) if max_m <= 0 else max_m
    return (range_axis >= min_m) & (range_axis <= hi)


def consistency_mask(det: torch.Tensor, channels: torch.Tensor, k: int, pct: float = 99.9) -> torch.Tensor:
    """Keep detections whose power exceeds the per-channel pct-th percentile noise floor in at least
    k virtual channels (a real target appears in every channel; a single-channel spike in one).
    channels: [doppler, range, channel] complex."""
    if k <= 0 or not bool(det.any()):
        return det
    pwr = channels.real ** 2 + channels.imag ** 2
    floor = torch.quantile(pwr.reshape(-1, pwr.shape[-1]).double(), pct / 100.0, dim=0).float()
    votes = (pwr > floor).sum(dim=-1)
    return det & (votes >= k)


def focus_mask(rd_db: torch.Tensor, det: torch.Tensor, expand: int = 1) -> torch.Tensor:
    """Per range bin, the Doppler bins to form the range-azimuth map from: its detections, or, with
    none, the bins >= max(median + 6 dB, peak - 8 dB) (top 3 if none pass); then +-expand bins."""
    n_dop, n_r = rd_db.shape
    has = det.any(dim=0)
    safe = torch.where(torch.isfinite(rd_db), rd_db, torch.full_like(rd_db, -float("inf")))
    med = median(safe, dim=0)
    thr = torch.maximum(med + 6.0, safe.max(dim=0).values - 8.0)
    fb = safe >= thr[None, :]
    none = ~fb.any(dim=0)
    if bool(none.any()):
        top = torch.zeros_like(fb)
        idx = torch.topk(safe, min(3, n_dop), dim=0).indices
        top.scatter_(0, idx, True)
        fb |= top & none[None, :]
    mask = torch.where(has[None, :], det, fb)
    for _ in range(expand):
        m = mask.clone()
        m[:-1] |= mask[1:]
        m[1:] |= mask[:-1]
        mask = m
    return mask


# ── azimuth ─────────────────────────────────────────────────────────────────
def unique_aperture(channels: torch.Tensor, positions: np.ndarray) -> tuple:
    """Average the channels that share an aperture position: [..., channel] -> [..., unique],
    with the unique positions (ascending)."""
    key = (np.asarray(positions, float).tobytes(), str(channels.device))
    if key not in _APERTURE:                      # once per array layout and device
        uniq, inv = np.unique(np.asarray(positions, float), return_inverse=True)
        _APERTURE[key] = (uniq, torch.as_tensor(inv, device=channels.device),
                          torch.as_tensor(np.bincount(inv).astype(np.float32), device=channels.device))
    uniq, inv_t, cnt = _APERTURE[key]
    if len(uniq) == len(positions):
        return channels, uniq
    out = torch.zeros(channels.shape[:-1] + (len(uniq),), dtype=channels.dtype, device=channels.device)
    out.index_add_(-1, inv_t, channels)
    return out / cnt, uniq


def az_spectrum(snapshots: torch.Tensor, win: torch.Tensor, n_az: int) -> torch.Tensor:
    """[..., position] uniform-array snapshots -> [..., n_az] complex azimuth spectrum (fftshifted;
    bin k <-> sin = -1 + 2k / n_az, processing azimuth)."""
    return torch.fft.fftshift(torch.fft.fft(snapshots * win, n=n_az, dim=-1), dim=-1)


def focused_ra(cube: torch.Tensor, focus: torch.Tensor, win: torch.Tensor, n_az: int) -> torch.Tensor:
    """Range-azimuth map [range, n_az] dB: per range, the mean azimuth power over its focus Doppler
    bins (all bins where the focus column is empty). cube: [doppler, range, position]."""
    n_dop, n_r, _ = cube.shape
    f = focus[:, :n_r] | ~focus[:, :n_r].any(dim=0, keepdim=True)
    di, ri = torch.nonzero(f, as_tuple=True)
    spec = az_spectrum(cube[di, ri], win, n_az)
    pwr = spec.real ** 2 + spec.imag ** 2
    sums = torch.zeros((n_r, n_az), device=cube.device).index_add_(0, ri, pwr)
    cnt = torch.bincount(ri, minlength=n_r).clamp(min=1).float()
    return 10.0 * torch.log10(torch.clamp(sums / cnt[:, None], min=1e-12))


def steering(positions: np.ndarray, n_az: int, device, dtype=torch.complex64) -> torch.Tensor:
    """[n_az, positions] steering vectors at the azimuth bins (positions in lambda/2 units)."""
    s = np.linspace(-1.0, 1.0, n_az, endpoint=False)
    return torch.as_tensor(np.exp(1j * np.pi * np.outer(s, np.asarray(positions, float))), device=device).to(dtype)


def _covariances(cube, focus, taper, diag_load, fb_avg):
    """Per range bin, the sample covariance [range, P, P] of the tapered focus snapshots."""
    n_dop, n_r, P = cube.shape
    f = (focus[:, :n_r] | ~focus[:, :n_r].any(dim=0, keepdim=True)).to(cube.dtype)
    X = cube * taper.to(cube.dtype)                                 # [D, R, P]
    R = torch.einsum("drp,drq,dr->rpq", X, X.conj(), f) / f.real.sum(dim=0).clamp(min=1)[:, None, None]
    R = 0.5 * (R + R.conj().transpose(-1, -2))
    if fb_avg and P > 1:
        J = torch.flip(torch.eye(P, dtype=R.dtype, device=R.device), dims=[1])
        R = 0.5 * (R + J @ R.conj() @ J)
    if diag_load > 0:                                                # load * trace / P (1 if the trace is 0)
        tr = torch.diagonal(R, dim1=-2, dim2=-1).real.sum(dim=-1)
        scale = torch.where(tr > 0, tr / P, torch.ones_like(tr))
        R = R + (diag_load * scale)[:, None, None].to(R.dtype) * torch.eye(P, dtype=R.dtype, device=R.device)
    return R


def ra_capon(cube, focus, positions, taper, n_az, diag_load=1e-6, fb_avg=False):
    """Capon (MVDR) range-azimuth map [range, n_az] dB: 1 / (a^H R^-1 a)."""
    sv = steering(positions, n_az, cube.device, cube.dtype) * taper.to(cube.dtype)
    Ri = torch.linalg.inv(_covariances(cube, focus, taper, max(diag_load, 1e-6), fb_avg))
    den = torch.einsum("ki,rij,kj->rk", sv.conj(), Ri, sv).real
    return 20.0 * torch.log10(1.0 / den.clamp(min=1e-20) + 1e-9)


def ra_music(cube, focus, positions, taper, n_az, n_sources=2, diag_load=0.0, fb_avg=False):
    """MUSIC pseudo-spectrum [range, n_az] dB: 1 / ||En^H a||^2 with n_sources signals."""
    P = cube.shape[-1]
    ns = int(np.clip(n_sources, 1, max(min(P, cube.shape[0]) - 1, 1)))
    sv = steering(positions, n_az, cube.device, cube.dtype) * taper.to(cube.dtype)
    _, V = torch.linalg.eigh(_covariances(cube, focus, taper, diag_load, fb_avg))
    En = V[..., :max(P - ns, 1)]                                     # noise subspace
    den = (torch.einsum("rpn,kp->rnk", En.conj(), sv).abs() ** 2).sum(dim=1)
    return 20.0 * torch.log10(1.0 / den.clamp(min=1e-20) + 1e-9)


def ra_bartlett(cube, focus, positions, taper, n_az, fb_avg=False):
    """Bartlett (conventional) beamformer [range, n_az] dB: a^H R a."""
    sv = steering(positions, n_az, cube.device, cube.dtype) * taper.to(cube.dtype)
    R = _covariances(cube, focus, taper, 0.0, fb_avg)
    spec = torch.einsum("ki,rij,kj->rk", sv.conj(), R, sv).real
    return 20.0 * torch.log10(spec.clamp(min=1e-20) + 1e-9)


# ── elevation ───────────────────────────────────────────────────────────────
def elevation_map(top: torch.Tensor, bot: torch.Tensor, d_el: float, win: torch.Tensor, n_el: int) -> torch.Tensor:
    """Range-elevation map [range, n_el] dB from the matched top/bottom channel pairs
    [doppler, range, pair]: per elevation bin, mean |bot + top e^{-j 2 pi d sin}|^2 over Doppler and
    pairs. sin spans +-min(1 / (2 d), 1)."""
    n = min(top.shape[-1], bot.shape[-1])
    if n == 0:
        return torch.zeros((top.shape[1], n_el), device=top.device)
    top, bot = top[..., :n] * win, bot[..., :n] * win
    lim = min(1.0 / (2.0 * d_el), 1.0)
    s = np.linspace(-lim, lim, n_el, endpoint=False, dtype=np.float32)
    st = torch.as_tensor(np.exp(-1j * 2.0 * np.pi * d_el * s).astype(np.complex64), device=top.device)
    comb = bot[..., None] + top[..., None] * st
    return 10.0 * torch.log10(torch.clamp((comb.real ** 2 + comb.imag ** 2).mean(dim=(0, 2)), min=1e-12))
