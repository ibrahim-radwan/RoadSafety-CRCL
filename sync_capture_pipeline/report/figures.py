"""Figures for the session report (matplotlib, rendered to PNG and embedded).

Conventions, the same in every figure:
  - sensor identity: radar = blue, LiDAR = orange, camera = aqua (the first
    three slots of the reference categorical palette, validated all-pairs in
    light mode), Altos V4 radar = yellow (slot 4; validated against its
    neighbours only, so it is never one of more than three series in one
    scatter plot); always with a text label, never colour alone;
  - magnitude (heatmaps): one-hue blue ramp, light -> dark;
  - time axis: seconds since the first radar frame;
  - bird's-eye views: forward up, vehicle-left on the LEFT (as in the camera image);
  - one y-scale per panel (related measures are stacked panels, never dual axes).
"""
from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
import numpy as np                                  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
RADAR, LIDAR, CAMERA, ALTOS = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SEQ = LinearSegmentedColormap.from_list("blue_ramp", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5",
                                                      "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "font.family": "DejaVu Sans",
})


def png(fig, dpi=110) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _rel(D, t):
    return np.asarray(t, dtype=float) - D["t0"]


def _per_second(t, t0, t1):
    edges = np.arange(np.floor(t0), np.ceil(t1) + 1)
    c, _ = np.histogram(t, bins=edges)
    return edges[:-1], c


# ── overview ────────────────────────────────────────────────────────────────
def timeline(D):
    R, L, C, A = D["radar"], D["lidar"], D["camera"], D.get("altos")
    lg = D["log"]
    t_end = D["duration"]
    na = 1 if A is not None else 0
    fig, ax = plt.subplots(5 + na, 1, figsize=(11, 8.6 + 1.5 * na), sharex=True,
                           gridspec_kw={"height_ratios": [1.3 + 0.3 * na, 1, 1, 1] + [1] * na + [0.9]})
    # coverage strips
    a = ax[0]
    a.grid(False)
    rows = [("Radar frames", RADAR, _rel(D, R["t"])), ("LiDAR sweeps", LIDAR, _rel(D, L["sweep_t"])),
            ("Camera frames", CAMERA, _rel(D, C["t_ref"][np.isfinite(C["t_ref"])]))]
    if A is not None:
        rows.append(("Altos frames", ALTOS, _rel(D, A["t"])))
    for k, (name, col, t) in enumerate(rows):
        y = 2 - k
        a.hlines(y, t.min(), t.max(), color=col, lw=7, alpha=0.85)
        a.text(t.min() - 0.01 * t_end, y, name, ha="right", va="center", fontsize=8, color=INK2)
    gaps = C["missing"] > 0
    if gaps.any():
        a.vlines(_rel(D, C["t_ref"][gaps]), -0.35, 0.35, color=INK, lw=1)
    nob = np.array([int(r["boresight_points"]) == 0 for r in L["timeline"]]) if L["timeline"] else np.array([])
    if nob.any() and len(L["sweep_t"]) == len(nob):
        # sweeps without boresight points have no boresight time: use the sweep's own time
        a.vlines(_rel(D, np.asarray(L["sweep_t"])[nob]), 0.65, 1.35, color=INK, lw=1)
    if A is not None:
        gap = np.diff(A["t"]) > 0.15                        # > 2 frame periods: Altos frames missing
        if gap.any():
            a.vlines(_rel(D, A["t"][1:][gap]), -1.35, -0.65, color=INK, lw=1)
    a.set_yticks([])
    a.set_ylim(-0.7 - na, 2.7)
    a.set_title("Coverage  (black ticks: camera gaps on the camera row; LiDAR sweeps with no points in the "
                "radar boresight on the LiDAR row" + ("; Altos frames missing on the Altos row)" if A is not None else ")"))
    # radar packets/s (receiver's per-second count, from radar start)
    # the receiver counts seconds from its first data packet = the first radar frame
    rp = lg["radar_pps"]
    if len(rp):
        ax[1].plot(rp[:, 0] - 0.5, rp[:, 1] / 1e3, color=RADAR, drawstyle="steps-mid")
    ax[1].set_ylabel("radar\nk packets/s")
    ax[1].set_ylim(bottom=0)
    # camera fps per second
    tc = C["t_ref"][np.isfinite(C["t_ref"])]
    s, c = _per_second(_rel(D, tc), _rel(D, tc).min(), _rel(D, tc).max())
    ax[2].plot(s[1:-1] + 0.5, c[1:-1], color=CAMERA, drawstyle="steps-mid")
    ax[2].set_ylabel("camera\nframes/s")
    ax[2].set_ylim(0, max(35, c.max() + 2))
    # LiDAR packets/s
    ax[3].plot(_rel(D, L["stats_t"]), L["stats_msop"], color=LIDAR, drawstyle="steps-mid")
    ax[3].set_ylabel("LiDAR\nMSOP packets/s")
    ax[3].set_ylim(bottom=0)
    # Altos frames per second
    if A is not None:
        ta = _rel(D, A["t"])
        s_, c_ = _per_second(ta, ta.min(), ta.max())
        ax[4].plot(s_[1:-1] + 0.5, c_[1:-1], color=ALTOS, drawstyle="steps-mid")
        ax[4].set_ylabel("Altos\nframes/s")
        ax[4].set_ylim(0, max(17, c_.max() + 2))
    # board temperature (5 s samples from radar start; approximate time)
    b = lg["board"]
    bx = ax[4 + na]
    if len(b) > 1:
        # first sample = preflight; then every 5 s from the start of the run loop (≈ radar start)
        t_radar_start = D["meta"]["clocks"].get("radar_start", {}).get("monotonic_ns")
        off = (t_radar_start / 1e9 - D["t0"]) if t_radar_start else 0.0
        bx.plot(off + 5.0 * np.arange(len(b) - 1), b[1:, 0], color=INK2)
    bx.set_ylabel("Jetson Tj\n°C")
    bx.set_xlabel("time since first radar frame (s)")
    fig.align_ylabels(ax)
    fig.tight_layout(h_pad=0.6)
    return png(fig)


# ── synchronisation ─────────────────────────────────────────────────────────
def intervals(D):
    R, L, C, A = D["radar"], D["lidar"], D["camera"], D.get("altos")
    fig, ax = plt.subplots(1, 3 + (A is not None), figsize=(11 + 3 * (A is not None), 2.8))
    dr = np.diff(R["t_arrival"]) * 1e3
    tc = np.sort(C["t_rx"])
    dc = np.diff(tc) * 1e3
    tlt = np.array([float(r["t_ref"]) if r["t_ref"] not in ("", "nan") else np.nan for r in L["timeline"]])
    tlt = tlt[np.isfinite(tlt)]
    dl = np.diff(tlt) * 1e3
    panels = [(ax[0], dr, RADAR, "Radar frame interval (arrival)", 125),
              (ax[1], dc, CAMERA, "Camera frame interval (host receipt)", 1000 / 30),
              (ax[2], dl, LIDAR, "LiDAR sweep interval (boresight time)", 100)]
    if A is not None:
        panels.append((ax[3], np.diff(A["t_arrival"]) * 1e3, ALTOS, "Altos frame interval (arrival)", 1000 / 15))
    for a, d, col, name, nom in panels:
        d = d[np.isfinite(d)]
        if len(d):
            lo, hi = np.percentile(d, [0.1, 99.9])
            a.hist(d, bins=80, range=(min(lo, nom * 0.5), max(hi, nom * 1.6)), color=col, log=True)
            a.axvline(nom, color=INK2, lw=0.8)
        a.set_title(name)
        a.set_xlabel("ms")
    ax[0].set_ylabel("count (log)")
    fig.tight_layout()
    return png(fig)


def pairing(D):
    P = D["pairs"]
    ce = np.array([float(p["camera_error_ms"]) if p["camera_error_ms"] else np.nan for p in P])
    le = np.array([float(p["lidar_error_ms"]) if p["lidar_error_ms"] else np.nan for p in P])
    tr = _rel(D, D["radar"]["t"])
    rows = [(ce, CAMERA, "Camera", 1000 / 60), (le, LIDAR, "LiDAR", 50)]
    if D.get("altos") is not None:
        ae = np.array([float(p["altos_error_ms"]) if p.get("altos_error_ms") else np.nan for p in P])
        rows.append((ae, ALTOS, "Altos", 1000 / 30))
    fig, ax = plt.subplots(len(rows), 2, figsize=(11, 2.3 * len(rows)), gridspec_kw={"width_ratios": [1, 2.2]})
    for k, (e, col, name, half) in enumerate(rows):
        ok = np.isfinite(e)
        ax[k, 0].hist(e[ok], bins=80, color=col)
        for s in (-half, half):
            ax[k, 0].axvline(s, color=INK2, lw=0.8)
        ax[k, 0].set_title(f"{name} − radar time, per radar frame")
        ax[k, 1].plot(tr[ok], e[ok], ".", ms=1.5, color=col)
        for s in (-half, half):
            ax[k, 1].axhline(s, color=INK2, lw=0.8)
        ax[k, 1].set_title(f"{name} pairing error over time  (lines: ± half a {name.lower()} period)")
        ax[k, 1].set_ylabel("ms")
    ax[-1, 0].set_xlabel("ms")
    ax[-1, 1].set_xlabel("time since first radar frame (s)")
    fig.tight_layout()
    return png(fig)


def velocity_check(D, which="radar"):
    """which = "radar" (the TI radar) or "altos": the same check for either radar."""
    V = D["vel_timing"] if which == "radar" else D.get("altos_vel_timing") or []
    R, L = (D["radar"] if which == "radar" else D["altos"]), D["lidar"]
    COL = RADAR if which == "radar" else ALTOS
    NAME = "Radar" if which == "radar" else "Altos"
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1, 1.6]})
    ok = [v for v in V if v.get("status") == "ok"]
    x = [(v["start"] + v["end"]) / 2 for v in ok]
    lim = 150
    inside = [v for v in ok if abs(v["residual_ms"]) <= lim]
    outside = [v for v in ok if abs(v["residual_ms"]) > lim]
    if inside:
        ax[0].errorbar([(v["start"] + v["end"]) / 2 for v in inside], [v["residual_ms"] for v in inside],
                       yerr=[min(v["sigma_ms"], 2 * lim) for v in inside], fmt="o", color=COL, ms=5, lw=1.2,
                       capsize=2, mec=SURFACE, mew=1.5)
    for v in outside:
        y = 0.94 * lim * np.sign(v["residual_ms"])
        ax[0].plot((v["start"] + v["end"]) / 2, y, "^" if y > 0 else "v", color=COL, ms=7, mec=SURFACE, mew=1.5)
        ax[0].annotate(f"{v['residual_ms']:+.0f}", ((v["start"] + v["end"]) / 2, y), xytext=(6, 0),
                       textcoords="offset points", va="center", fontsize=7, color=INK2)
    for v in V:
        if v.get("status") != "ok":
            ax[0].axvspan(v["start"], v["end"], color=GRID, alpha=0.6, lw=0)
    ax[0].axhline(0, color=INK2, lw=0.8)
    ax[0].set_ylim(-lim, lim)
    ax[0].set_title(f"{NAME} − LiDAR timing residual per 30 s window")
    ax[0].set_ylabel("ms (offsets applied)")
    ax[0].set_xlabel("window centre (s)   grey: not enough radial motion")
    # best window: velocity overlay
    best = min(ok, key=lambda v: v["sigma_ms"]) if ok else None
    a = ax[1]
    if best:
        w0, w1 = best["start"], best["end"]
        tr = _rel(D, R["t"])
        s = (tr >= w0) & (tr < w1)
        a.plot(tr[s], R["vel"][s], color=COL, label=f"{NAME} Doppler velocity at the LiDAR target range")
        tl = _rel(D, L["track_t"])
        sl = (tl >= w0) & (tl < w1) & np.isfinite(L["track_r"])
        if sl.sum() > 7:
            from scipy.signal import savgol_filter
            rr = savgol_filter(L["track_r"][sl], 7, 2)
            a.plot(tl[sl], np.gradient(rr, tl[sl]), color=LIDAR, label="LiDAR range rate (target track)")
        a.set_title(f"Example window {w0:.0f}–{w1:.0f} s: the two velocity traces the residual is fitted on")
        a.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2)
    a.set_ylabel("m/s")
    a.set_xlabel("time since first radar frame (s)")
    fig.tight_layout()
    return png(fig)


# ── radar ───────────────────────────────────────────────────────────────────
def radar_range_time(D):
    R, L = D["radar"], D["lidar"]
    rt = R["rt"].astype(float)
    rel = rt - np.nanmedian(rt, axis=0)
    t = _rel(D, R["t"])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    im = ax.imshow(np.clip(rel, 0, 30).T, origin="lower", aspect="auto", cmap=SEQ, interpolation="nearest",
                   extent=[t[0], t[-1], R["range_axis"][0], R["range_axis"][-1]])
    tl, lr = _rel(D, L["track_t"]), L["track_r"]
    ax.plot(tl, lr, ".", ms=0.8, color=LIDAR, label="LiDAR target range (nearest surface)")
    floor = D["fw"].get("hpf_floor_m")
    if floor:
        ax.axhline(floor, color=INK2, lw=0.8)
        ax.text(t[0] + 2, floor + 0.3, f"high-pass floor {floor} m", fontsize=7, color=INK2)
    ax.set_ylabel("range (m)")
    ax.set_xlabel("time since first radar frame (s)")
    ax.set_title("Radar moving returns: range vs time  (per-frame static clutter removed; dB above each "
                 "range bin's median)")
    ax.grid(False)
    ax.legend(loc="upper right", markerscale=8)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("dB", fontsize=8)
    fig.tight_layout()
    return png(fig)


def radar_doppler_target(D):
    R = D["radar"]
    dt = R["dop_target"].astype(float)
    ok = np.isfinite(dt).all(axis=1)
    if ok.sum() < 10:
        return None
    t = _rel(D, R["t"])
    rel = dt - np.nanmedian(dt[ok], axis=0)
    rel[~ok] = np.nan
    fig, ax = plt.subplots(figsize=(11, 3.4))
    v = R["v_axis"]
    im = ax.imshow(np.clip(rel, 0, 25).T, origin="lower", aspect="auto", cmap=SEQ, interpolation="nearest",
                   extent=[t[0], t[-1], v[0], v[-1]])
    ax.set_ylabel("radial velocity (m/s)")
    ax.set_xlabel("time since first radar frame (s)")
    ax.set_title("Radar micro-Doppler at the LiDAR target's range (blank: no single LiDAR target)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("dB", fontsize=8)
    fig.tight_layout()
    return png(fig)


def radar_adc(D):
    R = D["radar"]
    t = _rel(D, R["t"])
    fig = plt.figure(figsize=(11, 7.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.2, 1], width_ratios=[1, 1.6])
    a = fig.add_subplot(gs[0, 0])
    h = R["amp_hist"].astype(float)
    nz = np.nonzero(h)[0]
    a.loglog(nz[nz > 0], h[nz[nz > 0]], color=RADAR)
    a.axvline(32767, color=INK2, lw=0.8)
    a.set_title("Raw ADC |sample| distribution (every 10th frame)")
    a.set_xlabel("|value| (counts)")
    a.set_ylabel("count")
    b = fig.add_subplot(gs[0, 1])
    b.plot(t, R["peak"], color=RADAR, lw=1)
    b.set_ylim(0, max(4000, R["peak"].max() * 1.1))
    b.set_title(f"Peak |sample| per frame  (full scale 32767; clipped samples in session: "
                f"{int(R['clipped'].sum())})")
    b.set_ylabel("counts")
    c = fig.add_subplot(gs[1, :])
    ch = R["ch_rms"].astype(float)
    step = max(1, len(ch) // 600)
    im = c.imshow(20 * np.log10(ch[::step].T + 1e-6), aspect="auto", origin="lower", cmap=SEQ,
                  interpolation="nearest", extent=[t[0], t[-1], -0.5, 47.5])
    c.set_yticks([0, 8, 16, 24, 32, 40, 47])
    c.set_ylabel("virtual channel\n(tx·8 + dev·4 + rx)")
    c.set_title("Per-channel RMS level over time (dB counts)")
    c.grid(False)
    fig.colorbar(im, ax=c, pad=0.01)
    d = fig.add_subplot(gs[2, 0])
    d.plot(t, 10 * np.log10(R["iq"][:, 0]), color=RADAR, lw=1)
    d.set_title("I/Q power ratio (dB)")
    d.set_xlabel("time since first radar frame (s)")
    e = fig.add_subplot(gs[2, 1])
    e.plot(t, R["iq"][:, 1], color=RADAR, lw=1)
    e.set_title("I–Q correlation coefficient (0 = orthogonal)")
    e.set_xlabel("time since first radar frame (s)")
    fig.tight_layout()
    return png(fig)


def radar_profiles(D):
    R = D["radar"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
    ra = R["static_range_axis"]
    ax[0].plot(ra, 10 * np.log10(R["static_profile"] + 1e-12), color=RADAR, label="static scene (all Doppler)")
    mv = 10 ** (R["rt"].astype(float) / 10)
    ax[0].plot(R["range_axis"], 10 * np.log10(np.nanmedian(mv, axis=0) + 1e-12), color=INK2,
               label="moving-only, session median (≈ noise floor)")
    floor = D["fw"].get("hpf_floor_m")
    if floor:
        ax[0].axvline(floor, color=INK2, lw=0.8)
    ax[0].set_xlim(0, R["static_range_axis"][-1])
    ax[0].set_xlabel("range (m)")
    ax[0].set_ylabel("dB")
    ax[0].set_title("Range profiles")
    ax[0].legend()
    m = np.nanmean(R["ch_rms"].astype(float), axis=0)
    ax[1].plot(np.arange(48), 20 * np.log10(m + 1e-6), "o", ms=5, color=RADAR, mec=SURFACE, mew=1.5)
    for k in range(8, 48, 8):
        ax[1].axvline(k - 0.5, color=GRID, lw=0.8)
    ax[1].set_xlabel("virtual channel (tx·8 + dev·4 + rx)")
    ax[1].set_ylabel("dB counts")
    ax[1].set_title("Mean RMS level per virtual channel (blocks of 8 = one TX)")
    fig.tight_layout()
    return png(fig)


# ── LiDAR ───────────────────────────────────────────────────────────────────
def lidar_maps(D):
    L = D["lidar"]
    acc = L["acc"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
    bev = acc["bev"] / max(1, acc["n_acc"])
    # rows: x (-40..40), cols: y (-40..40). Display forward up, vehicle-left on the left.
    ax[0].imshow(bev, origin="lower", cmap=SEQ, norm=LogNorm(vmin=0.01, vmax=max(1, bev.max())),
                 extent=[-40, 40, -40, 40], interpolation="nearest")      # horizontal = y (left +)
    ax[0].set_xlim(25, -25)
    ax[0].set_ylim(-10, 30)
    for s in (1, -1):                                       # radar azimuth FOV +/-60 deg
        ax[0].plot([0, s * 30 * np.tan(np.radians(60))], [0, 30], color=RADAR, lw=0.8)
    ax[0].set_xlabel("lateral (m)  ← vehicle left   |   right →")
    ax[0].set_ylabel("forward (m)")
    ax[0].set_title("LiDAR points, bird's-eye (mean per sweep, log)")
    ax[0].grid(False)
    ar = acc["az_range"] / max(1, acc["n_acc"])
    im = ax[1].imshow(ar.T, origin="lower", aspect="auto", cmap=SEQ, norm=LogNorm(vmin=0.01, vmax=max(1, ar.max())),
                      extent=[-180, 180, 0, 60], interpolation="nearest")
    ax[1].set_xlabel("azimuth (deg, + = left, 0 = forward)")
    ax[1].set_ylabel("range (m)")
    ax[1].set_title("Points by direction and range (blocked sectors appear empty beyond the blockage)")
    ax[1].grid(False)
    fig.colorbar(im, ax=ax[1], pad=0.01, label="points per sweep per cell")
    fig.tight_layout()
    return png(fig)


def lidar_stats(D):
    L = D["lidar"]
    acc = L["acc"]
    t = _rel(D, L["sweep_t"])
    fig, ax = plt.subplots(2, 2, figsize=(11, 5))
    def roll(v, n=10):                       # 1 s rolling median (10 sweeps)
        v = np.asarray(v, dtype=float)
        if len(v) < n:
            return v
        w = np.lib.stride_tricks.sliding_window_view(v, n)
        return np.r_[np.full(n // 2, np.nan), np.nanmedian(w, axis=1), np.full(n - 1 - n // 2, np.nan)]
    ax[0, 0].plot(t, roll(L["sweep_points"]) / 1e3, color=LIDAR)
    grid = np.median(L["sweep_grid_points"]) if "sweep_grid_points" in L else None
    ax[0, 0].set_title("Valid returns per sweep (1 s median)" + (f"  — of a {grid:,.0f}-point grid" if grid else ""))
    ax[0, 0].set_ylabel("thousand")
    ax[0, 1].plot(t, 100 * roll(L["sweep_near_frac"]), color=LIDAR)
    ax[0, 1].set_title("Share of valid returns closer than 1 m (mount / nearby obstruction; 1 s median)")
    ax[0, 1].set_ylabel("%")
    for a in ax[0]:
        a.set_xlabel("time since first radar frame (s)")
    rb = np.linspace(0, 100, 401)
    ax[1, 0].bar(rb[:-1], acc["range"] / max(1, acc["n_acc"]), width=0.25, align="edge", color=LIDAR)
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_xlim(0, 80)
    ax[1, 0].set_title("Range distribution (points per sweep per 25 cm)")
    ax[1, 0].set_xlabel("range (m)")
    ax[1, 1].bar(np.arange(256), acc["intensity"] / max(1, acc["n_acc"]), width=1, color=LIDAR)
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_title("Intensity distribution (points per sweep)")
    ax[1, 1].set_xlabel("intensity")
    fig.tight_layout()
    return png(fig)


# ── camera ──────────────────────────────────────────────────────────────────
def camera_thumbs(D):
    th = D["camera"]["thumbs"]
    if not th:
        return None
    items = list(th.items())
    n = len(items)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(11, 1.35 * rows * 1.0 + 0.2))
    ax = np.atleast_2d(ax)
    for k, a in enumerate(ax.ravel()):
        a.axis("off")
        if k < n:
            t, im = items[k]
            a.imshow(im)
            a.set_title(f"{t - D['t0']:.0f} s", fontsize=8, color=INK2, loc="center", fontweight="normal")
    fig.tight_layout(pad=0.3)
    return png(fig, dpi=120)


def camera_quality(D):
    C = D["camera"]
    t = _rel(D, C["bright_t"])
    b = C["bright"]
    fig, ax = plt.subplots(3, 1, figsize=(11, 6.2), sharex=True)
    ax[0].fill_between(t, b[:, 1], b[:, 2], color=CAMERA, alpha=0.12, lw=0)
    ax[0].plot(t, b[:, 0], color=CAMERA, label="mean")
    ax[0].set_ylim(0, 255)
    ax[0].set_ylabel("grey level")
    ax[0].set_title("Image brightness (left image; band = 5th–95th percentile of pixels)")
    ax[1].plot(t, 100 * b[:, 3], color=CAMERA, label="saturated (≥ 250)")
    ax[1].plot(t, 100 * b[:, 4], color=INK2, label="black (≤ 5)")
    ax[1].set_ylabel("% of pixels")
    ax[1].set_title("Saturated and black pixels")
    ax[1].legend(loc="upper right")
    g = C["missing"] > 0
    tg = _rel(D, C["t_ref"][g])
    ax[2].vlines(tg, 0, C["missing"][g], color=CAMERA, lw=1.5)
    ax[2].set_ylim(0, max(3, C["missing"].max() + 1))
    ax[2].set_ylabel("frames")
    ax[2].set_title(f"Dropped camera frames ({int(C['missing'].sum())} in {int(g.sum())} gaps)")
    ax[2].set_xlabel("time since first radar frame (s)")
    fig.tight_layout()
    return png(fig)


def camera_imu(D):
    C = D["camera"]
    if not len(C["imu_t"]):
        return None
    t = _rel(D, C["imu_t"])
    acc = np.linalg.norm(C["imu_acc"], axis=1)
    gyr = np.degrees(np.linalg.norm(C["imu_gyr"], axis=1))
    fig, ax = plt.subplots(2, 1, figsize=(11, 3.8), sharex=True)
    ax[0].plot(t, acc, color=CAMERA, lw=0.6)
    ax[0].set_ylabel("m/s²")
    ax[0].set_title("ZED IMU: acceleration magnitude (≈ 9.81 when static)")
    ax[1].plot(t, gyr, color=CAMERA, lw=0.6)
    ax[1].set_ylabel("deg/s")
    ax[1].set_title("ZED IMU: angular rate magnitude")
    ax[1].set_xlabel("time since first radar frame (s)")
    fig.tight_layout()
    return png(fig)


# ── cross-sensor ────────────────────────────────────────────────────────────
def range_overlay(D):
    R, L, A = D["radar"], D["lidar"], D.get("altos")
    fig, axs = plt.subplots(1 + (A is not None), 1, figsize=(11, 3.4 * (1 + (A is not None))), sharex=True,
                            squeeze=False)
    ax = axs[0, 0]
    ax.plot(_rel(D, L["track_t"]), L["track_r"], ".", ms=1.2, color=LIDAR, label="LiDAR target (nearest surface)")
    ax.plot(_rel(D, R["t"]), R["dominant_range"], ".", ms=1.2, color=RADAR, label="radar strongest moving return")
    ax.set_ylabel("range (m)")
    ax.set_title("Range of the moving target: radar vs LiDAR (paired times, Stage 3 offsets applied)")
    ax.legend(loc="upper right", markerscale=6)
    if A is not None:                                   # a panel of its own: at most two series per plot
        b = axs[1, 0]
        b.plot(_rel(D, L["track_t"]), L["track_r"], ".", ms=1.2, color=LIDAR, label="LiDAR target (nearest surface)")
        b.plot(_rel(D, A["t"]), A["target_range"], ".", ms=1.2, color=ALTOS,
               label="Altos moving points at the LiDAR target (median range)")
        b.set_ylabel("range (m)")
        b.set_title("Range of the moving target: Altos vs LiDAR")
        b.legend(loc="upper right", markerscale=6)
    axs[-1, 0].set_xlabel("time since first radar frame (s)")
    fig.tight_layout()
    return png(fig)


def snapshots(D):
    R, L, C, A = D["radar"], D["lidar"], D["camera"], D.get("altos")
    idx = D["snap_radar_idx"]
    n = len(idx)
    nc = 4 + (A is not None)
    fig, ax = plt.subplots(n, nc, figsize=(12 + 2.6 * (A is not None), 2.9 * n),
                           gridspec_kw={"width_ratios": [1.45, 1, 1, 1] + [1] * (A is not None)})
    ax = np.atleast_2d(ax)
    bx, by = R["bev_x"], R["bev_y"]
    for row, i in enumerate(idx):
        t = R["t"][i]
        p = D["pairs"][i]
        # camera
        ci = D["snap_cam_idx"][row]
        a = ax[row, 0]
        a.axis("off")
        if ci in C["snaps"]:
            a.imshow(C["snaps"][ci])
        err = p.get("camera_error_ms") or "–"
        a.set_title(f"t = {t - D['t0']:.1f} s   camera frame {ci} (Δ {err} ms)", fontsize=8,
                    fontweight="normal", color=INK2)
        # LiDAR BEV (nearest stored sweep)
        a = ax[row, 1]
        if L["snap_sweeps"]:
            ks = np.array(list(L["snap_sweeps"].keys()))
            k = ks[np.argmin(np.abs(ks - t))]
            pts = L["snap_sweeps"][k].astype(np.float32)
            sel = (pts[:, 2] > -1.2) & (pts[:, 2] < 2.2)
            a.plot(pts[sel, 1], pts[sel, 0], ".", ms=1.0, color=LIDAR, alpha=0.5)   # horizontal = y (left +)
            a.set_title(f"LiDAR (Δ {1e3 * (k - t):+.0f} ms)", fontsize=8, fontweight="normal", color=INK2)
        tl, lr, la = L["track_t"], L["track_r"], L["track_az"]
        j = np.argmin(np.abs(tl - t)) if len(tl) else None
        if j is not None and np.isfinite(lr[j]) and abs(tl[j] - t) < 0.2:
            for aa in (ax[row, 1], ax[row, 2]):
                aa.plot(lr[j] * np.sin(np.radians(la[j])), lr[j] * np.cos(np.radians(la[j])), "o", ms=9,
                        mfc="none", mec=INK, mew=1)
        # radar BEV in the pipeline frame (+ left)
        a2 = ax[row, 2]
        s = R["snaps"].get(i)
        if s is not None:
            bev = s["bev"].astype(float)
            floor = np.nanmedian(bev)                       # this frame's noise floor
            vmax = max(floor + 20, np.nanpercentile(bev, 99.8))
            a2.imshow(bev, origin="lower", cmap=SEQ, vmin=floor + 3, vmax=vmax, interpolation="nearest",
                      extent=[by[0], by[-1], bx[0], bx[-1]], aspect="auto")
            a2.set_title("Radar BEV (moving, + left)", fontsize=8, fontweight="normal", color=INK2)
            a3 = ax[row, 3]
            rd = s["rd"].astype(float)
            kmax = int(np.searchsorted(R["static_range_axis"], 25.0))
            a3.imshow(rd[:, :kmax], origin="lower", aspect="auto", cmap=SEQ,
                      vmin=np.percentile(rd[:, :kmax], 50), vmax=rd[:, :kmax].max(), interpolation="nearest",
                      extent=[0, R["static_range_axis"][kmax - 1], R["v_axis"][0], R["v_axis"][-1]])
            a3.set_title("Radar range–Doppler (moving)", fontsize=8, fontweight="normal", color=INK2)
            a3.set_xlabel("range (m)", fontsize=7)
            a3.set_ylabel("m/s", fontsize=7)
            a3.grid(False)
        if A is not None:
            a4 = ax[row, 4]
            sa = A["snaps"].get(i)
            if sa is not None and len(sa["xyz_v_snr"]):
                q = sa["xyz_v_snr"].astype(float)
                mv = np.abs(q[:, 3]) >= A["moving_v"]
                a4.scatter(q[~mv, 1], q[~mv, 0], s=4, color=MUTED, lw=0, label="still")
                a4.scatter(q[mv, 1], q[mv, 0], s=9, color=ALTOS, lw=0, label="moving")
                a4.set_title(f"Altos (Δ {sa['error_ms'] or '–'} ms; own frame)", fontsize=8, fontweight="normal",
                             color=INK2)
            else:
                a4.set_title("Altos (no frame)", fontsize=8, fontweight="normal", color=INK2)
            if row == 0:
                a4.legend(loc="upper right", fontsize=7, markerscale=1.5)
            a4.set_xlim(12, -12)
            a4.set_ylim(0, 25)
            a4.set_aspect("equal")
            a4.grid(False)
            a4.tick_params(labelsize=7)
        for aa in (ax[row, 1], ax[row, 2]):
            aa.set_xlim(12, -12)                  # +y (vehicle left) drawn on the left
            aa.set_ylim(0, 25)
            aa.set_aspect("equal")
            aa.grid(False)
            aa.tick_params(labelsize=7)
        ax[row, 1].set_ylabel("forward (m)", fontsize=7)
    fig.tight_layout(h_pad=1.0)
    return png(fig, dpi=100)


# ── Altos V4 ────────────────────────────────────────────────────────────────
def altos_points(D):
    A = D["altos"]
    t = _rel(D, A["t"])
    fig, ax = plt.subplots(2, 1, figsize=(11, 4.4), sharex=True)

    def roll(v, n=15):                       # 1 s rolling median (15 frames)
        v = np.asarray(v, dtype=float)
        if len(v) < n:
            return v
        w = np.lib.stride_tricks.sliding_window_view(v, n)
        return np.r_[np.full(n // 2, np.nan), np.nanmedian(w, axis=1), np.full(n - 1 - n // 2, np.nan)]
    ax[0].plot(t, roll(A["points_per_frame"]), color=ALTOS)
    ax[0].set_ylim(bottom=0)
    ax[0].set_title("Altos points per frame (1 s median)")
    ax[0].set_ylabel("points")
    ax[1].plot(t, roll(A["moving_per_frame"]), color=ALTOS)
    ax[1].set_ylim(bottom=0)
    ax[1].set_title(f"Altos moving points per frame (|Doppler| ≥ {A['moving_v']} m/s; 1 s median)")
    ax[1].set_ylabel("points")
    ax[1].set_xlabel("time since first radar frame (s)")
    fig.tight_layout()
    return png(fig)


def altos_distributions(D):
    A = D["altos"]
    H = A["hist"]
    fig, ax = plt.subplots(2, 3, figsize=(11, 5.2))
    spec = [("range", "range (m)", "Range", 0), ("azimuth", "azimuth (deg, + = left)", "Azimuth", 0),
            ("elevation", "elevation (deg, + = up)", "Elevation", 0), ("snr_db", "SNR (dB)", "SNR", 0),
            ("doppler", "Doppler (m/s, + = away)", "Doppler (radial velocity)", 1)]
    for a, (key, xl, title, logy) in zip(ax.ravel(), spec):
        c, e = H[key]
        a.bar(e[:-1], c / max(1, len(A["t"])), width=np.diff(e), align="edge", color=ALTOS, lw=0)
        if logy:
            a.set_yscale("log")
        a.set_title(f"{title} (points per frame)")
        a.set_xlabel(xl)
    a = ax[1, 2]
    a.axis("off")
    inv = 100 * A["invalid"] / max(1, A["total"])
    a.text(0, 0.9, f"{A['total']:,} points in {len(A['t']):,} frames", fontsize=9, color=INK2, va="top")
    a.text(0, 0.72, f"{A['invalid']:,} with NaN values ({inv:.2f}%), left out", fontsize=9, color=INK2, va="top")
    a.text(0, 0.54, "Elevation: + up (the raw value is z-down)", fontsize=9, color=INK2, va="top")
    fig.tight_layout()
    return png(fig)


def altos_maps(D):
    A = D["altos"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1, 1.3]})
    bev = A["bev"]
    ax[0].imshow(bev, origin="lower", cmap=SEQ, norm=LogNorm(vmin=0.005, vmax=max(0.01, bev.max())),
                 extent=[-20, 20, 0, 40], interpolation="nearest")
    ax[0].set_xlim(20, -20)                                 # vehicle-left on the left
    for s_ in (1, -1):                                      # the Altos's +/-75 deg field of view
        ax[0].plot([0, s_ * 40 * np.tan(np.radians(75))], [0, 40], color=ALTOS, lw=0.8)
    ax[0].set_ylim(0, 40)
    ax[0].set_xlabel("y (m, + left)")
    ax[0].set_ylabel("x (m, forward)")
    ax[0].set_title("Altos points, top view (per frame, log; own frame)")
    ax[0].grid(False)
    side = A["side"]
    im = ax[1].imshow(side.T, origin="lower", aspect="auto", cmap=SEQ,
                      norm=LogNorm(vmin=0.005, vmax=max(0.01, side.max())), extent=[0, 40, -5, 5],
                      interpolation="nearest")
    for s_ in (1, -1):                                      # +/-15 deg vertical field of view
        ax[1].plot([0, 40], [0, s_ * 40 * np.tan(np.radians(15))], color=ALTOS, lw=0.8)
    ax[1].set_xlabel("x (m, forward)")
    ax[1].set_ylabel("z (m, + up)")
    ax[1].set_title("Altos points, side view (per frame, log)")
    ax[1].grid(False)
    fig.colorbar(im, ax=ax[1], pad=0.01, label="points per frame per cell")
    fig.tight_layout()
    return png(fig)


def altos_range_time(D):
    A, L = D["altos"], D["lidar"]
    t = _rel(D, A["t"])
    rt = A["rt"].astype(float)
    e = A["rt_range_edges"]
    fig, ax = plt.subplots(figsize=(11, 4.0))
    im = ax.imshow(rt.T, origin="lower", aspect="auto", cmap=SEQ, interpolation="nearest",
                   norm=LogNorm(vmin=1, vmax=max(2, rt.max())), extent=[t[0], t[-1], e[0], e[-1]])
    ax.plot(_rel(D, L["track_t"]), L["track_r"], ".", ms=0.8, color=LIDAR, label="LiDAR target range (nearest surface)")
    ax.set_ylim(e[0], e[-1])
    ax.set_ylabel("range (m)")
    ax.set_xlabel("time since first radar frame (s)")
    ax.set_title(f"Altos moving points (|Doppler| ≥ {A['moving_v']} m/s): range vs time")
    ax.grid(False)
    ax.legend(loc="upper right", markerscale=8)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("points per frame per 0.1 m", fontsize=8)
    fig.tight_layout()
    return png(fig)
