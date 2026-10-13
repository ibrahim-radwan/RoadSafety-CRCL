"""
radar_plots.py  —  AWR2243-2X cascade visualisation
=====================================================

Layout  (2-row, portrait top row)
---------------------------------
  Row 0 :  [Range-Doppler]  [Range-Elevation]  [Range-Azimuth rect]
  Row 1 :  [RA Polar]       [3-D CFAR point cloud]

Axis conventions
  Range-Doppler   : x = Doppler (m/s)    y = Range (m)
  Range-Elevation : x = Elevation (°)    y = Range (m)
  Range-Azimuth   : x = Azimuth (°)      y = Range (m)
  RA Polar        : standard polar, theta=azimuth, r=range, fixed −90°…+90°
  3-D point cloud : x = Azimuth (°)  y = Range (m)  z = Elevation (°)
                    Points = CFAR detections coloured black by power.

Colour scale
  ALL heatmaps share ONE (vmin, vmax) from compute_global_scale():
    adaptive: vmax = 99th-pct of {RD, RA, RE},  vmin = vmax − db_span
    fixed:    cfg.db_vmin, cfg.db_vmax
"""

from __future__ import annotations

import os
import glob
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from scipy.ndimage import map_coordinates

from .radar_config     import RadarConfig
from .radar_processing import ProcessingResult

_JET = plt.get_cmap("jet")

_RC = {
    "figure.facecolor": "white", "axes.facecolor":  "white",
    "axes.edgecolor":   "black", "axes.labelcolor": "black",
    "xtick.color":      "black", "ytick.color":     "black",
    "text.color":       "black", "grid.color":      "#cccccc",
    "grid.linewidth":   0.5,     "font.size":        10,
}


# ============================================================================
# Global colour scale
# ============================================================================

def compute_global_scale(result: ProcessingResult,
                          cfg:    RadarConfig) -> tuple:
    """
    ONE (vmin, vmax) shared by all heatmap subplots.
    Fixed mode  : (cfg.db_vmin, cfg.db_vmax)
    Adaptive    : vmax = 99th-pct of {RD_mti, RA, RE},  vmin = vmax − db_span
    """
    if getattr(cfg, 'use_fixed_db', False):
        return float(cfg.db_vmin), float(cfg.db_vmax)
    maps = [m for m in (result.RD_mti, result.RA, result.RE)
            if m is not None and np.isfinite(m).any()]
    if not maps:
        return 55.0, 90.0
    all_v = np.concatenate([m[np.isfinite(m)].ravel() for m in maps])
    vmax  = float(np.percentile(all_v, getattr(cfg, 'cmap_pct_high', 99.0)))
    vlow  = float(np.percentile(all_v, getattr(cfg, 'cmap_pct_low', 1.0)))
    vmin = max(vlow, vmax - float(getattr(cfg, 'db_span', 35.0)))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmax = float(np.max(all_v))
        vmin = float(np.min(all_v))
    return vmin, vmax


# ============================================================================
# Helpers
# ============================================================================

def _imshow_jet(ax, data: np.ndarray, extent,
                vmin: float, vmax: float):
    safe = np.where(np.isfinite(data), data, vmin)
    return ax.imshow(safe, origin="lower", aspect="auto",
                     extent=extent, cmap="jet",
                     vmin=vmin, vmax=vmax, interpolation="nearest")


def _cbar(ax, im, label: str = "Power (dB)", **kw):
    cb = plt.colorbar(im, ax=ax, label=label, **kw)
    cb.locator = mticker.MultipleLocator(5)
    cb.ax.yaxis.set_tick_params(labelsize=8)
    cb.update_ticks()
    return cb


def _style(ax, grid_alpha: float = 0.3):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_color("black"); sp.set_linewidth(0.8)
    ax.grid(True, alpha=grid_alpha, color="#cccccc", linewidth=0.5)


def _rmax(cfg: RadarConfig, r_ax: np.ndarray) -> float:
    return float(cfg.max_range_m) if cfg.max_range_m > 0 else float(r_ax[-1])


def _ra_display_map(result: ProcessingResult, cfg: RadarConfig) -> Optional[np.ndarray]:
    RA = result.RA
    if RA is None:
        return None
    disp = np.asarray(RA, dtype=np.float32).copy()
    if disp.size == 0:
        return disp

    ra_thresh = result.ra_thresh if result.ra_thresh is not None and result.ra_thresh.shape == disp.shape else None
    rd_det = result.rd_det
    keep_db = max(float(getattr(cfg, "ra_sidelobe_db", 8.0)), 6.0)
    snr_db = float(getattr(cfg, "ra_display_snr_db", 4.0))
    floor_db = float(getattr(cfg, "ra_display_floor_db", 6.0))

    for ri in range(disp.shape[0]):
        row = disp[ri]
        finite = np.isfinite(row)
        if not finite.any():
            continue
        vals = row[finite]
        med = float(np.median(vals))
        peak = float(np.max(vals))
        keep = row >= max(med + snr_db, peak - keep_db)
        gate = med + snr_db
        if ra_thresh is not None:
            thr_row = ra_thresh[ri]
            thr_vals = thr_row[np.isfinite(thr_row)]
            if thr_vals.size:
                gate = max(gate, float(np.median(thr_vals)) - 1.5)
                keep |= row >= (thr_row - 1.5)
        if rd_det is not None and ri < rd_det.shape[1] and not bool(np.any(rd_det[:, ri])):
            keep = row >= (peak - max(keep_db * 0.7, 4.0))
            gate = peak - max(keep_db + floor_db, 12.0)
        floor = min(gate - floor_db, peak - max(keep_db + 2.0, 8.0))
        row[~keep] = floor
        disp[ri] = row
    return disp


# ============================================================================
# 1. Range-Doppler  —  x = Doppler,  y = Range  (portrait)
# ============================================================================

def plot_range_doppler(result: ProcessingResult, cfg: RadarConfig,
                        vmin: float = None, vmax: float = None,
                        ax=None) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 10))
    else:
        fig = ax.get_figure()

    rd = result.RD_mti
    if rd is None:
        ax.text(0.5,0.5,"No RD data",ha="center",transform=ax.transAxes); return fig

    d_ax = result.doppler_axis   # ±dop_max from waveform config
    r_ax = result.range_axis
    d0, d1 = float(d_ax[0]), float(d_ax[-1])
    r0, r1 = float(r_ax[0]), float(r_ax[-1])

    # Transpose [N_dop,N_range] → [N_range,N_dop]:  rows=Range(y),  cols=Doppler(x)
    im = _imshow_jet(ax, rd.T, extent=[d0, d1, r0, r1], vmin=vmin, vmax=vmax)
    ax.axvline(0.0, color="white", alpha=0.35, ls="--", lw=0.9)
    _cbar(ax, im, pad=0.02)

    # D3: annotate title with strongest cell margin (CUT - effective threshold).
    # This makes the CFAR operating point visible in the figure window without
    # needing to read the terminal log.
    rd_det    = result.rd_det
    rd_thresh = result.rd_thresh
    _margin_str = ""
    if rd_thresh is not None and np.isfinite(rd.flatten()).any():
        _peak_idx  = np.unravel_index(np.nanargmax(rd), rd.shape)
        _cut_db    = float(rd[_peak_idx])
        _thr_db    = float(rd_thresh[_peak_idx])
        _margin_db = _cut_db - (_thr_db + float(getattr(cfg, 'cfar_margin_db', 0.0)))
        _det_mark  = "✓" if (rd_det is not None and rd_det[_peak_idx]) else "✗"
        _margin_str = f"  margin={_margin_db:+.1f} dB {_det_mark}"

    ax.set_xlim(d0, d1)
    ax.set_ylim(r0, _rmax(cfg, r_ax))
    ax.set_xlabel("Doppler (m/s)", fontsize=9)
    ax.set_ylabel("Range (m)", fontsize=9)
    ax.set_title(f"Range–Doppler{_margin_str}", fontsize=10)
    _style(ax)
    return fig


# ============================================================================
# 2. Range-Elevation  —  x = Elevation,  y = Range  (portrait)
# ============================================================================

def plot_range_elevation(result: ProcessingResult, cfg: RadarConfig,
                          vmin: float = None, vmax: float = None,
                          ax=None) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 10))
    else:
        fig = ax.get_figure()

    RE = result.RE
    if RE is None:
        ax.text(0.5,0.5,"No RE data",ha="center",transform=ax.transAxes); return fig

    el_ax = result.el_axis; r_ax = result.range_axis
    e0, e1 = float(el_ax[0]), float(el_ax[-1])
    r0, r1 = float(r_ax[0]), float(r_ax[-1])

    # RE [N_range,N_el]: rows=Range(y), cols=El(x)
    im = _imshow_jet(ax, RE, extent=[e0, e1, r0, r1], vmin=vmin, vmax=vmax)
    _cbar(ax, im, pad=0.02)
    ax.axvline(0.0, color="#444444", ls="--", lw=0.8, alpha=0.5)

    ax.set_xlim(e0, e1)
    ax.set_ylim(r0, _rmax(cfg, r_ax))
    ax.set_xlabel("Elevation (°)", fontsize=9)
    ax.set_ylabel("Range (m)", fontsize=9)
    ax.set_title("Range–Elevation", fontsize=10)
    _style(ax)
    return fig


# ============================================================================
# 3. Range-Azimuth rect  —  x = Azimuth,  y = Range  (landscape-data)
# ============================================================================

def plot_range_azimuth(result: ProcessingResult, cfg: RadarConfig,
                        vmin: float = None, vmax: float = None,
                        ax=None) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    RA = _ra_display_map(result, cfg)
    if RA is None:
        ax.text(0.5,0.5,"No RA data",ha="center",transform=ax.transAxes); return fig

    az_ax = result.az_axis; r_ax = result.range_axis
    az0, az1 = float(az_ax[0]), float(az_ax[-1])
    r0,  r1  = float(r_ax[0]), float(r_ax[-1])

    # RA [N_range,N_az]: rows=Range(y), cols=Az(x)
    im = _imshow_jet(ax, RA, extent=[az0, az1, r0, r1], vmin=vmin, vmax=vmax)
    _cbar(ax, im, pad=0.02)

    # D1: CFAR detection overlay — white cross markers at surviving RA cells.
    # ra_det shape is [N_range, N_az].  Convert cell indices to axis coordinates
    # using the actual az_axis and range_axis arrays so markers land precisely
    # on the bin centres regardless of FFT size or axis scaling.
    ra_det = result.ra_det
    if ra_det is not None and ra_det.any():
        det_r_idx, det_az_idx = np.where(ra_det)
        det_az_deg = az_ax[det_az_idx]
        det_r_m    = result.range_axis[det_r_idx]
        ax.scatter(det_az_deg, det_r_m,
                   marker='+', s=120, linewidths=1.8,
                   color='white', alpha=0.95, zorder=5,
                   label=f"CFAR ({ra_det.sum()} cells)")
        ax.legend(loc='upper right', fontsize=7,
                  framealpha=0.6, facecolor='black', labelcolor='white',
                  markerscale=0.8)

    ax.set_xlim(az0, az1)
    ax.set_ylim(r0, _rmax(cfg, r_ax))
    ax.set_xlabel("Azimuth (°)", fontsize=9)
    ax.set_ylabel("Range (m)", fontsize=9)
    ax.set_title(f"Range–Azimuth  ({result.aoa_method.upper()})", fontsize=10)
    _style(ax)
    return fig


# ============================================================================
# 4. RA Polar  —  FIXED −90° … +90°  (full half-circle, no empty white half)
# ============================================================================


def plot_range_azimuth_polar(result: ProcessingResult, cfg: RadarConfig,
                             vmin: float = None, vmax: float = None,
                             ax=None) -> plt.Figure:
    """
    Range-Azimuth polar fan in DEGREES, limited to -60° ... +60°.

    Robust behavior:
      - If ax is None -> create a polar axis
      - If ax is a normal Cartesian axis -> replace it with a polar axis
      - If ax is already polar -> reuse it

    This avoids:
      AttributeError: 'Axes' object has no attribute 'set_theta_zero_location'
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # ------------------------------------------------------------------
    # Ensure we really have a polar axis
    # ------------------------------------------------------------------
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="polar")
    else:
        fig = ax.get_figure()
        if not hasattr(ax, "set_theta_zero_location"):
            # incoming axis is Cartesian -> replace with polar
            pos = ax.get_position()
            ax.remove()
            ax = fig.add_axes(pos, projection="polar")

    RA   = _ra_display_map(result, cfg)
    r_ax = result.range_axis
    az   = result.az_axis   # expected in degrees

    if RA is None or r_ax is None or az is None:
        ax.text(0.5, 0.5, "No RA data", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    # ------------------------------------------------------------------
    # Force displayed azimuth range to -60° ... +60°
    # ------------------------------------------------------------------
    az_min = -60.0
    az_max =  60.0

    mask = (az >= az_min) & (az <= az_max)
    if not np.any(mask):
        ax.text(0.5, 0.5, "No azimuth bins in [-60°, 60°]",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    az_sel = az[mask]
    RA_sel = RA[:, mask]

    # Convert degrees -> radians for polar plotting
    theta = np.deg2rad(az_sel)

    # Range limit
    r_max = float(_rmax(cfg, r_ax))
    valid_r = r_ax <= r_max
    r_sel = r_ax[valid_r]
    RA_sel = RA_sel[valid_r, :]

    # Build polar mesh
    # RA_sel shape = [range, az]
    T, R = np.meshgrid(theta, r_sel, indexing="xy")

    # Color scaling
    vmin_p = vmin if vmin is not None else float(np.nanpercentile(RA_sel, 1))
    vmax_p = vmax if vmax is not None else float(np.nanpercentile(RA_sel, 99))

    # Clear axis before redraw (important in live viewer)
    ax.clear()

    # Plot
    pcm = ax.pcolormesh(
        T, R, RA_sel,
        cmap="jet",
        shading="auto",
        vmin=vmin_p,
        vmax=vmax_p
    )

    # ------------------------------------------------------------------
    # Polar axis formatting
    # ------------------------------------------------------------------
    ax.set_theta_zero_location("N")   # 0° straight ahead
    ax.set_theta_direction(-1)        # clockwise positive to the right
    ax.set_thetamin(az_min)
    ax.set_thetamax(az_max)
    ax.set_ylim(0.0, r_max)

    # Degree ticks
    tick_deg = np.arange(-60, 61, 20)
    ax.set_thetagrids(tick_deg, labels=[f"{d}°" for d in tick_deg])

    # Radial ticks
    ax.yaxis.set_major_locator(mticker.MaxNLocator(6))

    # Labels / title
    ax.set_title("Range–Azimuth Heatmap (-60° to 60°)", fontsize=10, pad=14)

    # Grid styling
    ax.grid(True, alpha=0.30, color="#bbbbbb", linewidth=0.6)

    # Colorbar
    cb = plt.colorbar(pcm, ax=ax, pad=0.10, fraction=0.046)
    cb.set_label("Power (dB)")
    cb.locator = mticker.MultipleLocator(5)
    cb.update_ticks()

    return fig

# ============================================================================
# 5. 3-D CFAR point cloud
#    x = Azimuth (°)   y = Range (m)   z = Elevation (°)
#    High-power CFAR detections plotted as BLACK points sized by power
# ============================================================================

def plot_pc3d(result: ProcessingResult, cfg: RadarConfig,
              vmin: float = None, vmax: float = None,
              ax=None) -> plt.Figure:
    """
    3-D CFAR point cloud.

    Coordinates:
      x = Azimuth  (°)      horizontal spread
      y = Range    (m)      depth
      z = Elevation (°)     vertical  (from RE argmax per range bin)

    Colour: black points.  Size ∝ power above noise floor.

    Source: CFAR detections from ra_det [N_range, N_az].
    Fallback: if CFAR gives 0 detections, use top-25% of RA power.
    """
    if ax is None:
        fig = plt.figure(figsize=(12, 7))
        fig.patch.set_facecolor("white")
        ax  = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    RA   = result.RA
    xyz  = result.xyz_cloud
    r_ax = result.range_axis

    # ── Axis styling: light grey background so black points are visible ───────
    ax.set_facecolor("#e8e8e8")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("#e8e8e8")
        axis.pane.set_alpha(0.9)
        axis.pane.set_edgecolor("#aaaaaa")
        axis.label.set_color("black")
    ax.grid(color="#aaaaaa", alpha=0.4, linewidth=0.4)
    ax.tick_params(colors="black", labelsize=7)

    r_max  = float(_rmax(cfg, r_ax))
    fov    = 90.0
    el_lim = float(np.degrees(np.arcsin(min(1.0, 1.0 / (2.0 * cfg.d_el_lambda)))))

    ax.set_xlim(-fov, fov)
    ax.set_ylim(0.0, r_max)
    ax.set_zlim(-el_lim, el_lim)
    ax.set_xlabel("Azimuth (°)",   labelpad=6, fontsize=9, color="black")
    ax.set_ylabel("Range (m)",     labelpad=6, fontsize=9, color="black")
    ax.set_zlabel("Elevation (°)", labelpad=6, fontsize=9, color="black")

    if RA is None or r_ax is None:
        ax.set_title("3-D Cloud  (no data)", fontsize=9, color="black")
        return fig

    # ── Select cells to plot ─────────────────────────────────────────────────
    src = "filtered"
    if xyz is not None and len(xyz) > 0:
        rng    = xyz[:, 5].astype(np.float32)
        az_deg = xyz[:, 6].astype(np.float32)
        pwr    = xyz[:, 3].astype(np.float32)
        dop_mps = xyz[:, 4].astype(np.float32)
        el_deg = np.degrees(
            np.arcsin(np.clip(xyz[:, 2].astype(np.float32) / np.maximum(rng, 1e-6), -1.0, 1.0))
        ).astype(np.float32)
    else:
        ra_det = result.ra_det
        az_ax = result.az_axis
        el_ax = result.el_axis
        RE = result.RE
        RD_mti = result.RD_mti
        dop_ax = result.doppler_axis
        if ra_det is not None and ra_det.sum() > 0 and az_ax is not None:
            ri_idx, ai_idx = np.where(ra_det)
            rng    = r_ax[np.clip(ri_idx, 0, len(r_ax)-1)].astype(np.float32)
            az_deg = az_ax[np.clip(ai_idx, 0, len(az_ax)-1)].astype(np.float32)
            pwr    = RA[ri_idx, ai_idx].astype(np.float32)
            el_deg = np.zeros(len(ri_idx), dtype=np.float32)
            dop_mps = np.zeros(len(ri_idx), dtype=np.float32)
            if RE is not None and el_ax is not None and RE.shape[0] > 0:
                peak_el = np.argmax(np.where(np.isfinite(RE), RE, -np.inf), axis=1)
                el_deg = el_ax[np.clip(peak_el[np.clip(ri_idx, 0, len(peak_el)-1)], 0, len(el_ax)-1)].astype(np.float32)
            if RD_mti is not None and dop_ax is not None and RD_mti.shape[1] > 0:
                for k, ri in enumerate(ri_idx):
                    if ri < RD_mti.shape[1]:
                        dop_mps[k] = float(dop_ax[int(np.argmax(RD_mti[:, ri]))])
            src = "cfar-mask"
        else:
            fv = RA[np.isfinite(RA)]
            if fv.size == 0:
                ax.set_title("3-D Cloud  (no points)", fontsize=9, color="black")
                return fig
            thr = float(np.percentile(fv, 99.5))
            ri_idx, ai_idx = np.where(np.isfinite(RA) & (RA >= thr))
            if ri_idx.size == 0:
                ax.set_title("3-D Cloud  (no points)", fontsize=9, color="black")
                return fig
            rng    = r_ax[np.clip(ri_idx, 0, len(r_ax)-1)].astype(np.float32)
            az_deg = result.az_axis[np.clip(ai_idx, 0, len(result.az_axis)-1)].astype(np.float32)
            pwr    = RA[ri_idx, ai_idx].astype(np.float32)
            el_deg = np.zeros(len(ri_idx), dtype=np.float32)
            dop_mps = np.zeros(len(ri_idx), dtype=np.float32)
            src = "power-fallback"

    # Range gate
    mask = (rng >= cfg.min_range_m) & (rng <= r_max) & \
           (az_deg >= -fov) & (az_deg <= fov)
    rng    = rng[mask];    az_deg = az_deg[mask]
    el_deg = el_deg[mask]; pwr    = pwr[mask]; dop_mps = dop_mps[mask]

    if len(pwr) == 0:
        ax.set_title("3-D Cloud  (all points outside range gate)", fontsize=9)
        return fig

    # Subsample: keep highest-power points (max 20 000)
    MAX_PTS = 20000
    if len(pwr) > MAX_PTS:
        keep   = np.argpartition(pwr, -MAX_PTS)[-MAX_PTS:]
        rng    = rng[keep];    az_deg = az_deg[keep]
        el_deg = el_deg[keep]; pwr    = pwr[keep]; dop_mps = dop_mps[keep]

    # Marker size: 8–80 pt² proportional to power above 10th-pct
    p10 = float(np.nanpercentile(pwr, 10))
    above = np.clip(pwr - p10, 0.0, None)
    s_min, s_max = 8.0, 80.0
    sizes = (s_min + (s_max - s_min) * above / max(above.max(), 1e-6)).astype(np.float32)

    # BLACK points — dense cloud = strong reflector cluster
    sc = ax.scatter(
        az_deg, rng, el_deg,
        s=sizes.astype(float),
        c=dop_mps.astype(float),
        cmap="jet",
        vmin=-float(cfg.dop_max),
        vmax=float(cfg.dop_max),
        alpha=0.85,
        edgecolors="none",
        depthshade=False,   # False = uniform shade → easier to see density
        zorder=6,
    )
    cb = plt.colorbar(sc, ax=ax, pad=0.04, fraction=0.03)
    cb.set_label("Doppler (m/s)")
    cb.ax.yaxis.set_tick_params(labelsize=8)

    ax.set_title(f"4-D Cloud  [{len(pwr)} pts  src={src}]",
                 fontsize=9, color="black", pad=8)
    try:
        ax.set_box_aspect((3.8, 2.7, 0.9))
    except Exception:
        pass
    ax.view_init(elev=18, azim=220)
    return fig


# ============================================================================
# Improved 3-D spatial scene
# ============================================================================

def _build_scene_cloud(result: ProcessingResult, cfg: RadarConfig,
                       max_points: int = 3500) -> np.ndarray:
    RA = result.RA
    RE = result.RE
    RD = result.RD_mti
    if RA is None or RE is None or result.range_axis is None or result.az_axis is None or result.el_axis is None:
        return np.empty((0, 5), dtype=np.float32)

    r_ax = result.range_axis
    az_ax = result.az_axis
    el_ax = result.el_axis
    dop_ax = result.doppler_axis
    r_max = float(_rmax(cfg, r_ax))
    dop_thr = cfg.min_abs_doppler_mps if cfg.min_abs_doppler_mps > 0 else 0.5 * cfg.dop_res

    finite_ra = RA[np.isfinite(RA)]
    if finite_ra.size == 0:
        return np.empty((0, 5), dtype=np.float32)
    power_floor = float(np.percentile(finite_ra, 93.0))
    rows = np.where((r_ax >= cfg.min_range_m) & (r_ax <= r_max))[0]
    records = []
    for ri in rows:
        row = RA[ri]
        candidates = np.where(np.isfinite(row) & (row >= power_floor))[0]
        if candidates.size == 0:
            continue
        local_order = np.argsort(row[candidates])[::-1][:3]
        el_idx = int(np.argmax(RE[ri])) if ri < RE.shape[0] else 0
        el_deg = float(el_ax[min(el_idx, len(el_ax) - 1)])
        rng = float(r_ax[ri])
        dop = 0.0
        if RD is not None and dop_ax is not None and ri < RD.shape[1]:
            dop = float(dop_ax[int(np.argmax(RD[:, ri]))])
        for ai in candidates[local_order]:
            az_deg = float(az_ax[min(ai, len(az_ax) - 1)])
            az_r = np.deg2rad(az_deg)
            el_r = np.deg2rad(el_deg)
            ce = np.cos(el_r)
            records.append([
                float(rng * ce * np.sin(az_r)),
                float(rng * ce * np.cos(az_r)),
                float(rng * np.sin(el_r)),
                float(row[ai]),
                dop,
            ])
    if not records:
        return np.empty((0, 5), dtype=np.float32)
    cloud = np.array(records, dtype=np.float32)
    if len(cloud) > max_points:
        keep = np.argpartition(cloud[:, 3], -max_points)[-max_points:]
        cloud = cloud[keep]
    return cloud[np.argsort(cloud[:, 3])[::-1]]


def plot_pc3d(result: ProcessingResult, cfg: RadarConfig,
              vmin: float = None, vmax: float = None,
              ax=None) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
    else:
        fig = ax.get_figure()

    RA = result.RA
    xyz = result.xyz_cloud
    r_ax = result.range_axis
    rd_det = result.rd_det
    if RA is None or r_ax is None:
        ax.set_title("PC2D Cloud  (no data)", fontsize=9, color="black")
        return fig

    ax.set_facecolor("#e8e8e8")
    ax.grid(color="#aaaaaa", alpha=0.4, linewidth=0.4)
    ax.tick_params(colors="black", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("black")
        sp.set_linewidth(1.2)

    r_max = float(_rmax(cfg, r_ax))
    dop_thr = cfg.min_abs_doppler_mps if cfg.min_abs_doppler_mps > 0 else 0.5 * cfg.dop_res
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(0.0, r_max)
    ax.set_xlabel("X lateral (m)", labelpad=6, fontsize=9, color="black")
    ax.set_ylabel("Y forward (m)", labelpad=6, fontsize=9, color="black")
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0.0, color="black", linewidth=1.2, alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=1.2, alpha=0.85)

    def _fallback_from_ra_power():
        az_ax = result.az_axis
        el_ax = result.el_axis
        RE = result.RE
        if az_ax is None or RA is None:
            return None
        r_mask = (r_ax >= cfg.min_range_m) & (r_ax <= r_max)
        if not np.any(r_mask):
            return None
        RA_work = np.array(RA, dtype=np.float32, copy=True)
        RA_work[~np.isfinite(RA_work)] = -np.inf
        RA_work[~r_mask, :] = -np.inf
        if not np.isfinite(RA_work).any():
            return None
        n_ranges = min(max(int(getattr(cfg, "pc2d_ranges_to_show", 24)), 1), int(np.count_nonzero(r_mask)))
        range_scores = np.max(RA_work, axis=1)
        valid_ranges = np.flatnonzero(np.isfinite(range_scores))
        if valid_ranges.size == 0:
            return None
        pick_ranges = valid_ranges[np.argsort(range_scores[valid_ranges])[-n_ranges:]]
        fx, fy, fz, fp, fd = [], [], [], [], []
        max_per_range = max(
            int(getattr(cfg, "max_az_detections_per_range", 1)),
            int(getattr(cfg, "pc2d_points_per_range", 4)),
            1,
        )
        for ri in pick_ranges:
            row = RA_work[ri]
            cand = np.argpartition(row, -max_per_range)[-max_per_range:]
            cand = cand[np.argsort(row[cand])[::-1]]
            rng_val = float(r_ax[min(int(ri), len(r_ax) - 1)])
            el_deg = 0.0
            if RE is not None and el_ax is not None and ri < RE.shape[0]:
                ei = int(np.argmax(np.where(np.isfinite(RE[ri]), RE[ri], -np.inf)))
                el_deg = float(el_ax[min(ei, len(el_ax) - 1)])
            el_r = np.deg2rad(el_deg)
            ce = np.cos(el_r)
            for ai in cand:
                if not np.isfinite(row[ai]):
                    continue
                az_deg = float(az_ax[min(int(ai), len(az_ax) - 1)])
                az_r = np.deg2rad(az_deg)
                fx.append(float(rng_val * ce * np.sin(az_r)))
                fy.append(float(rng_val * ce * np.cos(az_r)))
                fz.append(float(rng_val * np.sin(el_r)))
                fp.append(float(RA[ri, ai]))
                fd.append(0.0)
        if not fp:
            return None
        return (
            np.asarray(fx, dtype=np.float32),
            np.asarray(fy, dtype=np.float32),
            np.asarray(fz, dtype=np.float32),
            np.asarray(fp, dtype=np.float32),
            np.asarray(fd, dtype=np.float32),
            "ra-power",
        )

    src = "filtered"
    if xyz is not None and len(xyz) > 0:
        x_m = xyz[:, 0].astype(np.float32)
        y_m = xyz[:, 1].astype(np.float32)
        z_m = xyz[:, 2].astype(np.float32)
        pwr = xyz[:, 3].astype(np.float32)
        dop_mps = xyz[:, 4].astype(np.float32)
        if rd_det is not None and result.doppler_axis is not None and rd_det.size:
            rng_col = np.clip(np.round(xyz[:, 5] / max(float(cfg.range_res_m), 1e-6)).astype(int), 0, rd_det.shape[1] - 1)
            mov_rows = np.abs(result.doppler_axis) >= dop_thr
            for k, ri in enumerate(rng_col):
                if abs(float(dop_mps[k])) >= dop_thr:
                    continue
                det_mask = rd_det[:, ri].astype(bool) & mov_rows
                if not det_mask.any() or result.RD_mti is None or ri >= result.RD_mti.shape[1]:
                    continue
                col = result.RD_mti[:, ri].astype(np.float32).copy()
                col[~det_mask] = -np.inf
                if np.isfinite(col).any():
                    dop_mps[k] = float(result.doppler_axis[int(np.argmax(col))])
    else:
        ra_det = result.ra_det
        az_ax = result.az_axis
        el_ax = result.el_axis
        RE = result.RE
        RD_mti = result.RD_mti
        dop_ax = result.doppler_axis
        ri_idx = np.empty(0, dtype=int)
        ai_idx = np.empty(0, dtype=int)
        dop_seed = np.empty(0, dtype=np.float32)
        if ra_det is not None and az_ax is not None and ra_det.sum() > 0:
            ri_idx, ai_idx = np.where(ra_det)
            dop_seed = np.zeros(len(ri_idx), dtype=np.float32)
            src = "filtered-mask"
        else:
            if az_ax is None or rd_det is None or dop_ax is None or RD_mti is None:
                fallback = _fallback_from_ra_power()
                if fallback is None:
                    ax.set_title("PC2D Cloud  (no filtered cloud)", fontsize=9, color="black")
                    return fig
                x_m, y_m, z_m, pwr, dop_mps, src = fallback
                rng = np.sqrt(x_m * x_m + y_m * y_m + z_m * z_m)
                mask = (rng >= cfg.min_range_m) & (rng <= r_max)
                x_m = x_m[mask]
                y_m = y_m[mask]
                z_m = z_m[mask]
                pwr = pwr[mask]
                dop_mps = dop_mps[mask]
                moving_mask = np.ones(len(pwr), dtype=bool)
                if len(pwr) == 0:
                    ax.set_title("PC2D Cloud  (no filtered cloud)", fontsize=9, color="black")
                    return fig
                # Skip the RA/RD reconstruction path below; we already have display points.
                ri_idx = np.empty(0, dtype=int)
                ai_idx = np.empty(0, dtype=int)
                goto_ready = True
            else:
                goto_ready = False
            if not goto_ready:
                mov_rows = np.abs(dop_ax) >= dop_thr
                rd_move = rd_det & mov_rows[:, np.newaxis]
                moving_ranges = np.flatnonzero(rd_move.any(axis=0))
                if moving_ranges.size == 0:
                    fallback = _fallback_from_ra_power()
                    if fallback is None:
                        ax.set_title("PC2D Cloud  (no moving reflections)", fontsize=9, color="black")
                        return fig
                    x_m, y_m, z_m, pwr, dop_mps, src = fallback
                else:
                    max_per_range = max(
                        int(getattr(cfg, "max_az_detections_per_range", 1)),
                        int(getattr(cfg, "pc2d_points_per_range", 4)),
                        1,
                    )
                    ri_list = []
                    ai_list = []
                    dop_list = []
                    for ri in moving_ranges:
                        if ri >= RA.shape[0] or ri >= RD_mti.shape[1]:
                            continue
                        row = RA[ri].astype(np.float32)
                        finite = np.isfinite(row)
                        if not finite.any():
                            continue
                        row_work = row.copy()
                        row_work[~finite] = -np.inf
                        take = min(max_per_range, row_work.size)
                        cand = np.argpartition(row_work, -take)[-take:]
                        cand = cand[np.argsort(row_work[cand])[::-1]]
                        col = RD_mti[:, ri].astype(np.float32).copy()
                        col[~rd_move[:, ri]] = -np.inf
                        if not np.isfinite(col).any():
                            continue
                        dop_val = float(dop_ax[int(np.argmax(col))])
                        for ai in cand:
                            if not np.isfinite(row_work[ai]):
                                continue
                            ri_list.append(int(ri))
                            ai_list.append(int(ai))
                            dop_list.append(dop_val)
                    if not ri_list:
                        fallback = _fallback_from_ra_power()
                        if fallback is None:
                            ax.set_title("PC2D Cloud  (no filtered cloud)", fontsize=9, color="black")
                            return fig
                        x_m, y_m, z_m, pwr, dop_mps, src = fallback
                    else:
                        ri_idx = np.asarray(ri_list, dtype=int)
                        ai_idx = np.asarray(ai_list, dtype=int)
                        dop_seed = np.asarray(dop_list, dtype=np.float32)
                        src = "rd-confirmed"
                        rng = r_ax[np.clip(ri_idx, 0, len(r_ax) - 1)].astype(np.float32)
                        az_deg = az_ax[np.clip(ai_idx, 0, len(az_ax) - 1)].astype(np.float32)
                        pwr = RA[ri_idx, ai_idx].astype(np.float32)
                        el_deg = np.zeros(len(ri_idx), dtype=np.float32)
                        dop_mps = dop_seed.copy()
                        if RE is not None and el_ax is not None and RE.shape[0] > 0:
                            peak_el = np.argmax(np.where(np.isfinite(RE), RE, -np.inf), axis=1)
                            el_deg = el_ax[np.clip(peak_el[np.clip(ri_idx, 0, len(peak_el) - 1)], 0, len(el_ax) - 1)].astype(np.float32)
                        if RD_mti is not None and dop_ax is not None and RD_mti.shape[1] > 0:
                            for k, ri in enumerate(ri_idx):
                                if ri < RD_mti.shape[1]:
                                    col = RD_mti[:, ri].astype(np.float32)
                                    mov_bins = np.abs(dop_ax) >= dop_thr
                                    det_mask = None
                                    if rd_det is not None and ri < rd_det.shape[1]:
                                        det_mask = rd_det[:, ri].astype(bool)
                                        if np.any(mov_bins):
                                            det_mask = det_mask & mov_bins
                                        if det_mask.any():
                                            col = col.copy()
                                            col[~det_mask] = -np.inf
                                            if np.isfinite(col).any():
                                                dop_mps[k] = float(dop_ax[int(np.argmax(col))])
                                            continue
                                    if np.any(mov_bins):
                                        col = col.copy()
                                        col[~mov_bins] = -np.inf
                                        if np.isfinite(col).any():
                                            dop_mps[k] = float(dop_ax[int(np.argmax(col))])
                        az_r = np.deg2rad(az_deg)
                        el_r = np.deg2rad(el_deg)
                        ce = np.cos(el_r)
                        x_m = (rng * ce * np.sin(az_r)).astype(np.float32)
                        y_m = (rng * ce * np.cos(az_r)).astype(np.float32)
                        z_m = (rng * np.sin(el_r)).astype(np.float32)

    rng = np.sqrt(x_m * x_m + y_m * y_m + z_m * z_m)
    mask = (rng >= cfg.min_range_m) & (rng <= r_max)
    x_m = x_m[mask]
    y_m = y_m[mask]
    z_m = z_m[mask]
    pwr = pwr[mask]
    dop_mps = dop_mps[mask]
    if len(pwr) == 0:
        ax.set_title("PC2D Cloud  (all points outside range gate)", fontsize=9)
        return fig

    min_display_pts = max(
        int(getattr(cfg, "pc2d_min_points", 240)),
        int(getattr(cfg, "pc2d_ranges_to_show", 24)),
        int(getattr(cfg, "pc2d_points_per_range", 4)) * 8,
    )
    if len(pwr) < min_display_pts:
        fallback = _fallback_from_ra_power()
        if fallback is not None:
            fx, fy, fz, fp, fd, _ = fallback
            if len(fp) > 0:
                frng = np.sqrt(fx * fx + fy * fy + fz * fz)
                fmask = (frng >= cfg.min_range_m) & (frng <= r_max)
                fx = fx[fmask]
                fy = fy[fmask]
                fz = fz[fmask]
                fp = fp[fmask]
                fd = fd[fmask]
                if len(fp) > 0:
                    x_m = np.concatenate([x_m, fx.astype(np.float32)])
                    y_m = np.concatenate([y_m, fy.astype(np.float32)])
                    z_m = np.concatenate([z_m, fz.astype(np.float32)])
                    pwr = np.concatenate([pwr, fp.astype(np.float32)])
                    dop_mps = np.concatenate([dop_mps, fd.astype(np.float32)])
                    src = f"{src}+ra"

    moving_mask = np.abs(dop_mps) >= dop_thr
    if not np.any(moving_mask):
        az_full = result.az_axis
        el_full = result.el_axis
        RE_full = result.RE
        RD_full = result.RD_mti
        dop_full = result.doppler_axis
        if RA is not None and RD_full is not None and dop_full is not None and az_full is not None:
            relaxed_thr = max(1e-3, 0.5 * dop_thr)
            mov_bins = np.abs(dop_full) >= relaxed_thr
            if np.any(mov_bins):
                rd_energy = np.max(
                    np.where(mov_bins[:, np.newaxis], RD_full, -np.inf),
                    axis=0,
                ).astype(np.float32)
                finite = np.isfinite(rd_energy)
                if np.any(finite):
                    energy_vals = rd_energy[finite]
                    energy_med = float(np.median(energy_vals))
                    energy_peak = float(np.max(energy_vals))
                    energy_thr = max(energy_med + 3.0, energy_peak - 10.0)
                    moving_ranges = np.flatnonzero(finite & (rd_energy >= energy_thr))
                    if moving_ranges.size == 0:
                        finite_idx = np.flatnonzero(finite)
                        top_k = min(max(int(getattr(cfg, "max_az_detections_per_range", 1)) * 3, 3), finite_idx.size)
                        if top_k > 0:
                            moving_ranges = finite_idx[np.argsort(rd_energy[finite_idx])[-top_k:]]
                    if moving_ranges.size > 0:
                        max_per_range = max(
                            int(getattr(cfg, "max_az_detections_per_range", 1)),
                            int(getattr(cfg, "pc2d_points_per_range", 4)),
                            1,
                        )
                        fx = []
                        fy = []
                        fz = []
                        fp = []
                        fd = []
                        for ri in moving_ranges:
                            if ri >= RA.shape[0] or ri >= RD_full.shape[1]:
                                continue
                            row = RA[ri].astype(np.float32)
                            finite_row = np.isfinite(row)
                            if not finite_row.any():
                                continue
                            row_work = row.copy()
                            row_work[~finite_row] = -np.inf
                            take = min(max_per_range, row_work.size)
                            cand = np.argpartition(row_work, -take)[-take:]
                            cand = cand[np.argsort(row_work[cand])[::-1]]
                            col = RD_full[:, ri].astype(np.float32).copy()
                            col[~mov_bins] = -np.inf
                            if not np.isfinite(col).any():
                                continue
                            dop_val = float(dop_full[int(np.argmax(col))])
                            rng_val = float(r_ax[min(int(ri), len(r_ax) - 1)])
                            el_deg = 0.0
                            if RE_full is not None and el_full is not None and ri < RE_full.shape[0]:
                                ei = int(np.argmax(np.where(np.isfinite(RE_full[ri]), RE_full[ri], -np.inf)))
                                el_deg = float(el_full[min(ei, len(el_full) - 1)])
                            az_sel = az_full[np.clip(cand, 0, len(az_full) - 1)].astype(np.float32)
                            az_r = np.deg2rad(az_sel)
                            el_r = np.deg2rad(el_deg)
                            ce = np.cos(el_r)
                            for ai, az_deg_val, az_rad in zip(cand, az_sel, az_r):
                                fx.append(float(rng_val * ce * np.sin(az_rad)))
                                fy.append(float(rng_val * ce * np.cos(az_rad)))
                                fz.append(float(rng_val * np.sin(el_r)))
                                fp.append(float(RA[ri, ai]))
                                fd.append(dop_val)
                        if fp:
                            x_m = np.asarray(fx, dtype=np.float32)
                            y_m = np.asarray(fy, dtype=np.float32)
                            z_m = np.asarray(fz, dtype=np.float32)
                            pwr = np.asarray(fp, dtype=np.float32)
                            dop_mps = np.asarray(fd, dtype=np.float32)
                            moving_mask = np.ones(len(pwr), dtype=bool)
                            src = "rd-energy"
    if not np.any(moving_mask):
        if len(pwr) > 0:
            keep_n = min(max(int(getattr(cfg, "max_az_detections_per_range", 1)) * 4, 8), len(pwr))
            keep_idx = np.argsort(pwr)[-keep_n:]
            moving_mask = np.zeros(len(pwr), dtype=bool)
            moving_mask[keep_idx] = True
            src = "weak-motion"
        else:
            ax.set_title("PC2D Cloud  (no moving reflections)", fontsize=9, color="black")
            return fig

    x_m = x_m[moving_mask]
    y_m = y_m[moving_mask]
    pwr = pwr[moving_mask]
    dop_mps = dop_mps[moving_mask]

    max_cloud_pts = max(int(getattr(cfg, "point_cloud_max_points", 30000)), 1000)
    if len(pwr) > max_cloud_pts:
        keep = np.argpartition(pwr, -max_cloud_pts)[-max_cloud_pts:]
        x_m = x_m[keep]
        y_m = y_m[keep]
        z_m = z_m[keep]
        pwr = pwr[keep]
        dop_mps = dop_mps[keep]

    p_lo = float(np.nanpercentile(pwr, 10))
    p_hi = float(np.nanpercentile(pwr, 95))
    denom = max(p_hi - p_lo, 1e-6)
    sizes = (5.0 + 28.0 * np.clip((pwr - p_lo) / denom, 0.0, 1.0)).astype(np.float32)
    cloud_all = pwr
    if getattr(cfg, "pc2d_dynamic_db", True):
        pc_vmax = float(np.nanpercentile(cloud_all, float(getattr(cfg, "pc2d_cmap_pct_high", 99.0))))
        pc_low = float(np.nanpercentile(cloud_all, float(getattr(cfg, "pc2d_cmap_pct_low", 5.0))))
        pc_vmin = max(pc_low, pc_vmax - float(getattr(cfg, "pc2d_db_span", 24.0)))
    else:
        fixed_pc_vmin = getattr(cfg, "pc2d_db_vmin", None)
        fixed_pc_vmax = getattr(cfg, "pc2d_db_vmax", None)
        pc_vmin = float(fixed_pc_vmin if fixed_pc_vmin is not None else (vmin if vmin is not None else np.nanpercentile(cloud_all, 5.0)))
        pc_vmax = float(fixed_pc_vmax if fixed_pc_vmax is not None else (vmax if vmax is not None else np.nanpercentile(cloud_all, 99.0)))
    if not np.isfinite(pc_vmin) or not np.isfinite(pc_vmax) or pc_vmin >= pc_vmax:
        pc_vmin = float(np.nanmin(cloud_all))
        pc_vmax = float(np.nanmax(cloud_all) + 1e-3)

    strong_power_thr = float(np.percentile(pwr, 85.0)) if len(pwr) > 0 else float("inf")
    high_reflect_mask = pwr >= strong_power_thr
    base_mask = ~high_reflect_mask
    if np.any(base_mask):
        ax.scatter(x_m[base_mask], y_m[base_mask],
                   s=np.maximum(5.0, 0.70 * sizes[base_mask]).astype(float),
                   c=pwr[base_mask].astype(float), cmap="jet",
                   vmin=pc_vmin, vmax=pc_vmax,
                   alpha=0.82, edgecolors="none", zorder=4)

    sc = ax.scatter(x_m[high_reflect_mask], y_m[high_reflect_mask],
                    s=np.maximum(18.0, 1.35 * sizes[high_reflect_mask]).astype(float),
                    c=pwr[high_reflect_mask].astype(float), cmap="jet",
                    vmin=pc_vmin, vmax=pc_vmax,
                    alpha=0.98, edgecolors="none", zorder=6)
    cb = plt.colorbar(sc, ax=ax, pad=0.04, fraction=0.03)
    cb.set_label("Power (dB)")
    cb.ax.yaxis.set_tick_params(labelsize=8)

    ax.set_title(
        f"PC2D Cloud  [{len(pwr)} moving pts, strong={int(np.count_nonzero(high_reflect_mask))} src={src}]",
        fontsize=9, color="black", pad=8
    )
    return fig


# ============================================================================
# RA Sector fan  (standalone, not in composite)
# ============================================================================

def _ra_sector(RA: np.ndarray, r_ax, az_ax_deg,
               out_h: int = 500, out_w: int = 900):
    ra  = np.asarray(RA, dtype=np.float32)
    r   = np.asarray(r_ax, dtype=np.float32)
    az  = np.asarray(az_ax_deg, dtype=np.float32)
    rmax   = float(r[-1])
    fov    = max(abs(float(az.min())), abs(float(az.max())))
    x_arr  = np.linspace(-rmax*np.sin(np.deg2rad(fov)), rmax*np.sin(np.deg2rad(fov)),
                         out_w, dtype=np.float32)
    y_arr  = np.linspace(0.0, rmax, out_h, dtype=np.float32)
    X, Y   = np.meshgrid(x_arr, y_arr)
    R_grid = np.sqrt(X*X + Y*Y).astype(np.float32)
    A_grid = np.degrees(np.arctan2(X, Y)).astype(np.float32)
    valid  = (R_grid >= r[0]) & (R_grid <= rmax) & \
             (A_grid >= float(az.min())) & (A_grid <= float(az.max()))
    ri = np.interp(R_grid.ravel(), r,  np.arange(len(r),  dtype=np.float32)).reshape(R_grid.shape)
    ai = np.interp(A_grid.ravel(), az, np.arange(len(az), dtype=np.float32)).reshape(A_grid.shape)
    ri_s = np.where(valid, ri, 0.0).astype(np.float32)
    ai_s = np.where(valid, ai, 0.0).astype(np.float32)
    img  = map_coordinates(ra, [ri_s, ai_s], order=1, mode='nearest').astype(np.float32)
    img[~valid] = np.nan
    return img, x_arr, y_arr


def plot_ra_sector(result: ProcessingResult, cfg: RadarConfig,
                   vmin: float = None, vmax: float = None,
                   ax=None) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.get_figure()
    RA=result.RA; r_ax=result.range_axis; az=result.az_axis
    if RA is None: return fig
    img, x_arr, y_arr = _ra_sector(RA, r_ax, az)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgba = _JET(norm(np.where(np.isfinite(img), img, np.nan)))
    rgba[~np.isfinite(img)] = [1,1,1,0]
    ax.imshow(rgba, origin="lower", aspect="equal",
              extent=[float(x_arr[0]),float(x_arr[-1]),float(y_arr[0]),float(y_arr[-1])],
              interpolation="bilinear")
    sm = plt.cm.ScalarMappable(norm=norm, cmap="jet"); sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, label="Power (dB)", pad=0.02, fraction=0.046)
    cb.locator = mticker.MultipleLocator(5); cb.update_ticks()
    ax.axvline(0.0, color="#555555", ls="--", lw=0.8, alpha=0.7)
    az0,az1=float(az.min()),float(az.max())
    ax.set_title(f"Range–Azimuth  Fan  ({az0:.0f}°,{az1:.0f}°)")
    ax.set_xlabel("Azimuth (m)"); ax.set_ylabel("Range (m)")
    _style(ax); return fig


# ============================================================================
# Composite figure  — portrait subplots (height > width)
# ============================================================================

def _make_axes(fig):
    """
    Layout:

      Row 0 : [RD] [RE] [RA]   ← all portrait (height > width)
      Row 1 : [Fan] [3-D cloud spanning most of the row]

    Figure size and row ratios are chosen so the top-row axes remain taller
    than they are wide, while the 3-D view keeps enough horizontal room.
    """
    gs = gridspec.GridSpec(
        2, 8, figure=fig,
        height_ratios=[2.8, 1.7],
        hspace=0.34, wspace=0.34,
        left=0.06, right=0.97,
        top=0.93, bottom=0.04,
    )
    ax_rd  = fig.add_subplot(gs[0, 0:2])               # portrait
    ax_re  = fig.add_subplot(gs[0, 3:5])               # portrait
    ax_ra  = fig.add_subplot(gs[0, 6:8])               # portrait
    ax_fan = fig.add_subplot(gs[1, 0:2])               # portrait-ish polar fan
    ax_3d  = fig.add_subplot(gs[1, 2:8])
    return ax_rd, ax_re, ax_ra, ax_fan, ax_3d


def plot_all(result: ProcessingResult, cfg: RadarConfig,
             title: str = "", save_path: Optional[str] = None,
             dpi: int = 120) -> plt.Figure:
    matplotlib.rcParams.update(_RC)

    fig = plt.figure(figsize=(18, 18), facecolor="white", num="AWR2243-2X Radar")
    fig.clf()
    ax_rd, ax_re, ax_ra, ax_fan, ax_3d = _make_axes(fig)

    vmin, vmax = compute_global_scale(result, cfg)
    scale_tag  = (f"FIXED {vmin:.0f}–{vmax:.0f} dB"
                  if getattr(cfg, 'use_fixed_db', False)
                  else f"ADAPTIVE {vmin:.0f}–{vmax:.0f} dB")

    # D2: build detection status string for the suptitle so the figure window
    # shows the same counts as the terminal without needing to read logs.
    rd, ra, cloud = _det_counts(result)
    _det_status = _det_label(rd, ra, cloud)

    fig.suptitle(
        (title or "AWR2243-2X Cascade Radar") +
        f"\n[{scale_tag}  |  dop_max=±{cfg.dop_max:.2f} m/s  |  {_det_status}]",
        fontsize=11, fontweight="600", color="black", y=0.98)

    plot_range_doppler(result, cfg, vmin=vmin, vmax=vmax, ax=ax_rd)
    plot_range_elevation(result, cfg, vmin=vmin, vmax=vmax, ax=ax_re)
    plot_range_azimuth(result, cfg, vmin=vmin, vmax=vmax, ax=ax_ra)
    plot_range_azimuth_polar(result, cfg, vmin=vmin, vmax=vmax, ax=ax_fan)
    plot_pc3d(result, cfg, vmin=vmin, vmax=vmax, ax=ax_3d)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"[PLOT] Saved → {save_path}")
    return fig


# ============================================================================
# C1: Detection count helper
# ============================================================================

# D5: Rolling colour scale state — shared across all frames in a viewer session.
# Smooths the adaptive vmax over the last _VMAX_WINDOW frames using a median
# so the colour scale does not jump frame-to-frame as the target RCS changes.
_VMAX_WINDOW   = 8    # number of frames in the rolling window
_vmax_history: list[float] = []


def _smooth_vmax(raw_vmax: float) -> float:
    """
    Append raw_vmax to the rolling history and return the median.
    Resets if the history grows beyond _VMAX_WINDOW.
    """
    _vmax_history.append(raw_vmax)
    if len(_vmax_history) > _VMAX_WINDOW:
        _vmax_history.pop(0)
    return float(np.median(_vmax_history))

def _det_counts(result: "ProcessingResult") -> tuple:
    """
    Return (rd_cfar, ra_cfar, cloud_pts) safely from a ProcessingResult.

    None-safe: any field not populated returns 0 rather than raising
    AttributeError or TypeError.  Used by both run_folder_viewer and
    run_frame so the guard logic is not duplicated.
    """
    rd    = int(result.rd_det.sum())   if result.rd_det   is not None else 0
    ra    = int(result.ra_det.sum())   if result.ra_det   is not None else 0
    cloud = int(len(result.xyz_cloud)) if result.xyz_cloud is not None else 0
    return rd, ra, cloud


def _det_label(rd: int, ra: int, cloud: int) -> str:
    """Format the per-frame detection count label.

    Fields:
      RD-CFAR  = Range-Doppler CFAR survivors (rd_det.sum())
      az-pts   = RD-confirmed multi-peak azimuth detections (ra_det.sum())
                 NOTE: previously printed as 'RA-CFAR' — renamed to reflect
                 that there is no independent RA-CFAR stage; this count is
                 the output of the prominence-filtered az peak search in Stage 8.
      cloud    = xyz_cloud points after all gates and extract_xyz filters

    Example: 'RD-CFAR:12  az-pts:5  cloud:3'
    """
    return f"RD-CFAR:{rd}  az-pts:{ra}  cloud:{cloud}"
    """
    Format detection counts as a compact stage-labelled string.
    Example: 'RD-CFAR:12  RA-CFAR:5  cloud:3'
    Used in both the figure suptitle and the terminal print line.
    """
    return f"RD-CFAR:{rd}  RA-CFAR:{ra}  cloud:{cloud}"


# ============================================================================
# Sequential folder viewer
# ============================================================================

def run_folder_viewer(bin_dir: str, cfg: RadarConfig,
                      correction: "np.ndarray",
                      frame_pattern: str = "*.bin",
                      pause_s: float = 0.15,
                      save_dir: Optional[str] = None,
                      verbose: bool = True):
    from .radar_reader     import read_frame, count_frames
    from .radar_processing import process_frame

    bin_files = ([bin_dir] if os.path.isfile(bin_dir)
                 else sorted(glob.glob(os.path.join(bin_dir, frame_pattern))))
    if not bin_files:
        print(f"[VIEWER] No files in {bin_dir}"); return
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.ion(); matplotlib.rcParams.update(_RC)
    fig = plt.figure("AWR2243-2X Radar", figsize=(18, 18), facecolor="white")
    gidx = 0

    # D5: reset rolling vmax history at the start of each viewer session
    # so a previous run's history does not bleed into a new capture.
    _vmax_history.clear()

    # Pre-allocate CuPy memory pool before the first frame is processed.
    # Eliminates period-3 ~300ms GC synchronization spikes in the offline viewer.
    try:
        from .gpu_accel import warmup_gpu_memory_pool
        warmup_gpu_memory_pool(cfg)
    except Exception:
        pass   # GPU not available or gpu_accel not importable — silent

    for bp in bin_files:
        n = count_frames(bp, cfg)
        if n == 0: continue
        if verbose: print(f"[VIEWER] {os.path.basename(bp)} → {n} frame(s)")
        for fi in range(n):
            try:
                cube   = read_frame(bp, cfg, fi)
                result = process_frame(cube, cfg, correction, verbose=False)
            except Exception as e:
                print(f"[VIEWER] frame {gidx}: {e}"); gidx += 1; continue

            rd, ra, cloud = _det_counts(result)
            raw_vmin, raw_vmax = compute_global_scale(result, cfg)
            # D5: smooth vmax over rolling window; recompute vmin from smoothed vmax
            smooth_vmax = _smooth_vmax(raw_vmax)
            vmin = smooth_vmax - float(getattr(cfg, 'db_span', 24.0))
            vmax = smooth_vmax
            label = (f"Frame {fi}  |  {os.path.basename(bp)}  |  "
                     f"{_det_label(rd, ra, cloud)}  |  "
                     f"{vmin:.0f}–{vmax:.0f} dB")
            if verbose: print(f"  frame {fi}/{n-1} — {label}")

            sp = os.path.join(save_dir, f"frame_{gidx:05d}.png") if save_dir else None
            fig.clf()
            ax_rd, ax_re, ax_ra, ax_fan, ax_3d = _make_axes(fig)
            fig.suptitle(label, fontsize=9, fontweight="600", color="black", y=0.98)
            plot_range_doppler(result, cfg, vmin=vmin, vmax=vmax, ax=ax_rd)
            plot_range_elevation(result, cfg, vmin=vmin, vmax=vmax, ax=ax_re)
            plot_range_azimuth(result, cfg, vmin=vmin, vmax=vmax, ax=ax_ra)
            plot_range_azimuth_polar(result, cfg, vmin=vmin, vmax=vmax, ax=ax_fan)
            plot_pc3d(result, cfg, vmin=vmin, vmax=vmax, ax=ax_3d)
            if sp:
                fig.savefig(sp, dpi=100, bbox_inches="tight", facecolor="white")
            plt.pause(pause_s); gidx += 1

    plt.ioff(); plt.show()
