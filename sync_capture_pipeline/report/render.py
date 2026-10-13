"""Assemble the self-contained HTML session report.

Descriptive, not a verdict: the recording itself is judged by check_session.py.
The report summarises what was recorded and lists factual observations; the
readers (radar/LiDAR/camera specialists) draw their own conclusions.
"""
from __future__ import annotations

import html
import time

import numpy as np

from . import figures as F

CSS = """
:root { color-scheme: light; --surface:#fcfcfb; --surface-2:#f3f2ef; --ink:#0b0b0b; --ink-2:#52514e;
        --muted:#8a8984; --rule:#e4e3df; --radar:#2a78d6; --lidar:#eb6834; --camera:#1baf7a; }
* { box-sizing: border-box; }
body { margin:0; background:var(--surface); color:var(--ink);
       font:15px/1.5 -apple-system, "Segoe UI", Roboto, "DejaVu Sans", sans-serif; }
header { padding:28px 32px 12px; border-bottom:1px solid var(--rule); }
header h1 { margin:0 0 4px; font-size:24px; }
header .sub { color:var(--ink-2); }
nav { position:sticky; top:0; z-index:2; background:var(--surface); border-bottom:1px solid var(--rule);
      padding:8px 32px; display:flex; gap:18px; flex-wrap:wrap; font-size:13px; }
nav a { color:var(--ink-2); text-decoration:none; } nav a:hover { color:var(--ink); }
main { max-width:1240px; margin:0 auto; padding:8px 32px 48px; }
section { margin-top:34px; }
h2 { font-size:19px; margin:0 0 6px; padding-bottom:6px; border-bottom:1px solid var(--rule); }
h3 { font-size:15px; margin:22px 0 4px; }
figure { margin:10px 0 18px; }
figure img { width:100%; height:auto; border:1px solid var(--rule); border-radius:4px; background:var(--surface); }
figcaption { color:var(--ink-2); font-size:13px; margin-top:4px; max-width:1050px; }
table { border-collapse:collapse; font-size:14px; font-variant-numeric:tabular-nums; }
td, th { padding:4px 14px 4px 0; text-align:left; vertical-align:top; border-bottom:1px solid var(--rule); }
th { color:var(--ink-2); font-weight:600; }
.grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:4px 36px; }
.obs { background:var(--surface-2); border:1px solid var(--rule); border-radius:6px; padding:12px 18px; }
.obs li { margin:4px 0; }
.key { display:inline-flex; gap:14px; font-size:13px; color:var(--ink-2); }
.sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; }
.caveat { border-left:3px solid var(--muted); padding:6px 12px; color:var(--ink-2); background:var(--surface-2);
          font-size:13px; margin:8px 0; }
.small { font-size:13px; color:var(--ink-2); }
@media (max-width:700px) { main, header, nav { padding-left:16px; padding-right:16px; } }
"""


def _fmt(v, nd=1):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:,.{nd}f}" if isinstance(v, (float, np.floating)) else f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)


def observations(D) -> list:
    R, L, C = D["radar"], D["lidar"], D["camera"]
    al = D["align"]
    obs = []
    # radar
    pps = D["log"]["radar_pps"]
    if len(pps) > 3:
        v = pps[2:, 1]
        obs.append(f"Radar packet rate {v.min():,.0f}–{v.max():,.0f} packets/s after start-up "
                   f"(median {np.median(v):,.0f}); expected ≈ {3145728 / 1456 / (D['fw'].get('frame_periodicity_ms', 125) / 1000):,.0f}.")
    obs.append(f"Radar raw data: {100 * R['stale_fraction']:.2f}% of samples identical in every frame; "
               f"peak |sample| {int(R['peak'].max()):,} of 32,767; {int(R['clipped'].sum()):,} clipped samples.")
    rms = np.nanmean(R["ch_rms"].astype(float), axis=0)
    lo, hi = 20 * np.log10(rms.min()), 20 * np.log10(rms.max())
    obs.append(f"Radar channel levels span {hi - lo:.1f} dB across the 48 virtual channels "
               f"(lowest: channel {int(np.argmin(rms))}, highest: channel {int(np.argmax(rms))}).")
    # sync
    src = al.get("radar", {}).get("timing_source", "")
    if src.startswith("NOMINAL"):
        late = al["radar"].get("residual_ms", {}).get("max")
        obs.append("Radar frame times are NOMINAL (first frame + k × frame period): the DCA1000 delivered frames late "
                   f"and unevenly (backlog; up to {late / 1e3:.0f} s late). Frame content is unaffected; data after the "
                   "DCA buffer filled is missing." if late else "Radar frame times are NOMINAL (DCA1000 backlog).")
    off = al.get("offsets", {})
    obs.append(f"Time offsets applied: radar {off.get('radar_ms', 0):+.2f} ms, camera {off.get('camera_ms', 0):+.2f} ms "
               f"(LiDAR reference){'' if al.get('offsets_calibrated') else ' — NOT calibrated'}.")
    pr = al.get("pairing", {})
    if pr:
        obs.append(f"Pairing: {pr.get('pairs'):,} radar frames; camera error p01/p99 "
                   f"{pr['camera_error_ms']['p01']:+.1f}/{pr['camera_error_ms']['p99']:+.1f} ms, "
                   f"LiDAR error p01/p99 {pr['lidar_error_ms']['p01']:+.1f}/{pr['lidar_error_ms']['p99']:+.1f} ms; "
                   f"{pr.get('flagged', 0)} pairs flagged ({pr.get('camera_gap_pairs', 0)} in a camera gap, "
                   f"{pr.get('lidar_gap_pairs', 0)} in a LiDAR gap).")
    vt = [v for v in D["vel_timing"] if v.get("status") == "ok" and v["sigma_ms"] < 15]
    if vt:
        r = np.array([v["residual_ms"] for v in vt]); s = np.array([v["sigma_ms"] for v in vt])
        w = 1 / np.maximum(s, 1) ** 2
        obs.append(f"Velocity-based radar−LiDAR timing check: {len(vt)} window(s) with σ < 15 ms, weighted mean "
                   f"{(w * r).sum() / w.sum():+.1f} ± {1 / np.sqrt(w.sum()):.1f} ms (0 = consistent with the Stage 3 "
                   f"offsets; the method's own systematic error is ≈ ±10 ms).")
    else:
        obs.append("Velocity-based timing check: no window had enough radial motion of a single target.")
    # camera
    g = C["missing"] > 0
    obs.append(f"Camera: {len(C['frame_index']):,} frames, {int(C['missing'].sum())} dropped in {int(g.sum())} gaps "
               f"(longest {int(C['missing'].max()) if g.any() else 0} frame{'' if g.any() and int(C['missing'].max()) == 1 else 's'}). Mean brightness "
               f"{np.nanmin(C['bright'][:, 0]):.0f}–{np.nanmax(C['bright'][:, 0]):.0f} grey levels; saturated pixels up to "
               f"{100 * np.nanmax(C['bright'][:, 3]):.1f}%.")
    # LiDAR
    tinfo = L["track_info"]
    nob = sum(1 for r in L["timeline"] if int(r["boresight_points"]) == 0)
    grid = f" of a {np.median(L['sweep_grid_points']):,.0f}-point grid" if "sweep_grid_points" in L else ""
    obs.append(f"LiDAR: {len(L['sweep_t']):,} sweeps, median {np.median(L['sweep_points']):,.0f} valid returns{grid} "
               f"({100 * np.nanmedian(L['sweep_near_frac']):.0f}% of them within 1 m); "
               f"{nob} sweeps with no points in the radar boresight cone; clock drift {L['clock_fit']['drift_ppm']:+.1f} ppm.")
    ar = L["acc"]["az_range"] / max(1, L["acc"]["n_acc"])
    beyond = ar[:, 6:].sum(axis=1)           # beyond 3 m, per 2-degree azimuth bin
    empty = beyond < 0.05 * np.median(beyond[beyond > 0]) if (beyond > 0).any() else np.zeros(len(beyond), bool)
    if empty.any():
        az = np.linspace(-180, 180, 181)
        runs, start = [], None
        for k, e in enumerate(np.r_[empty, False]):
            if e and start is None:
                start = k
            if not e and start is not None:
                runs.append((az[start], az[k])); start = None
        big = [r for r in runs if r[1] - r[0] >= 10]
        if big:
            obs.append("LiDAR sees almost nothing beyond 3 m in azimuth sector(s) " +
                       ", ".join(f"{a:+.0f}…{b:+.0f}°" for a, b in big) +
                       " (blocked by the mount/vehicle, or open sky/ground); check against the mounting.")
    obs.append(f"LiDAR single-target track present in {tinfo['sweeps_with_target']:,} of {tinfo['sweeps']:,} sweeps "
               f"(within ±30° of forward, {D['fw'].get('hpf_floor_m', '?')}–25 m).")
    if R.get("mirror_azimuth"):
        obs.append("Radar azimuth: the processing's angle sign is mirrored relative to the world (verified on "
                   "freewalk_ref_02 against camera and LiDAR); radar bird's-eye views here are mirror-corrected. "
                   "The TX/RX → azimuth table is not yet calibrated.")
    b = D["log"]["board"]
    if len(b):
        obs.append(f"Jetson junction temperature {b[:, 0].min():.1f}–{b[:, 0].max():.1f} °C.")
    return obs


def overview_table(D) -> str:
    M = D["meta"]
    fw = D["fw"]
    R, L, C = D["radar"], D["lidar"], D["camera"]
    rows = [
        ("Started", M.get("started_local")), ("Note", M.get("note") or "–"),
        ("Duration (radar)", f"{D['duration']:.1f} s ({len(R['t']):,} frames)"),
        ("Sensors", ", ".join(M.get("sensors", []))), ("Stop reason", M.get("stop_reason")),
        ("Recording check", D["check_line"] or "not run"),
    ]
    radar = [
        ("Firmware", f"{D['fw_variant']} ({D['fw_source']})"),
        ("Frame period", f"{fw.get('frame_periodicity_ms', '–')} ms"),
        ("Range window", f"{fw.get('hpf_floor_m', '–')} – {fw.get('max_range_m', '–')} m"),
        ("Range resolution", f"{_fmt(fw.get('range_res_m'), 3)} m"),
        ("Velocity", f"±{_fmt(fw.get('v_max_mps'), 2)} m/s, resolution {_fmt(fw.get('v_res_mps'), 3)} m/s"),
        ("Frame structure", "64 loops × 6 TX (TDM), 256 samples, 6 TX × 8 RX = 48 virtual channels"),
    ]
    other = [
        ("Camera", f"ZED 2i, {len(C['frame_index']):,} frames, {C['status'].get('fps', '–')} fps nominal"),
        ("LiDAR", f"RS-Fairy, {len(L['sweep_t']):,} sweeps"),
        ("Offsets file", (D['align'].get('offsets_file') or '–').split('/')[-1]),
        ("Report generated", time.strftime("%Y-%m-%d %H:%M")),
    ]

    def tab(title, rr):
        body = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in rr)
        return f"<div><h3>{title}</h3><table>{body}</table></div>"
    return f'<div class="grid">{tab("Session", rows)}{tab("Radar waveform", radar)}{tab("Other sensors", other)}</div>'


def fig_html(b64, caption) -> str:
    if not b64:
        return f'<p class="small">({html.escape(caption.split(".")[0])}: not available for this session)</p>'
    return f'<figure><img alt="{html.escape(caption[:120])}" src="data:image/png;base64,{b64}">' \
           f'<figcaption>{caption}</figcaption></figure>'


def build(D) -> str:
    key = ('<span class="key"><span><span class="sw" style="background:var(--radar)"></span>radar</span>'
           '<span><span class="sw" style="background:var(--lidar)"></span>LiDAR</span>'
           '<span><span class="sw" style="background:var(--camera)"></span>camera</span></span>')
    S = []
    S.append(("overview", "Overview", overview_table(D) +
              "<h3>Observations</h3><ul class='obs'>" + "".join(f"<li>{html.escape(o)}</li>" for o in observations(D)) +
              "</ul><p class='small'>Factual notes computed from the data; no pass/fail judgement is made here. "
              "Recording completeness is checked separately by check_session.py.</p>"))
    S.append(("timeline", "Timeline", fig_html(F.timeline(D),
              "Coverage of each sensor over the session and the per-second data rates. Camera and LiDAR start "
              "before and stop after the radar by design.")))
    S.append(("sync", "Synchronisation",
              fig_html(F.intervals(D), "Frame/sweep interval distributions (log count). Grey line: nominal "
                       "period (radar 125 ms at 8 fps, camera 33.3 ms, LiDAR 100 ms). The radar uses packet "
                       "arrival times here; pairing uses the de-jittered fit.") +
              fig_html(F.pairing(D), "For every radar frame, the time difference to the paired camera frame "
                       "and LiDAR sweep (LiDAR: time its points in the radar boresight were scanned). Errors are "
                       "bounded by half the partner's period when nothing is missing.") +
              fig_html(F.velocity_check(D), "Independent check of the radar timing on this session's own "
                       "data: in each 30 s window with enough radial motion of one target, the radar Doppler "
                       "velocity at the LiDAR target's range is fitted to the LiDAR range rate. 0 means "
                       "consistent with the Stage 3 offsets. Velocity is used because range comparisons depend "
                       "on which body part each sensor sees.")))
    S.append(("radar", "Radar",
              fig_html(F.radar_range_time(D), "Moving returns over range and time. Static clutter is removed "
                       "inside each frame (mean over the 64 chirp loops), then each range bin is shown relative "
                       "to its session median. Orange dots: the LiDAR's single-target track.") +
              fig_html(F.radar_doppler_target(D), "Doppler spectrum at the range of the LiDAR-tracked target "
                       "(torso and limb micro-Doppler of a walking person). Velocity sign follows the radar "
                       "processing convention.") +
              fig_html(F.radar_adc(D), "Raw ADC statistics: amplitude distribution against full scale, "
                       "per-frame peak, per-virtual-channel RMS level over time, and I/Q balance.") +
              fig_html(F.radar_profiles(D), "Left: static-scene range profile (all Doppler) and the median "
                       "moving-only profile, which approximates the noise floor; vertical line: high-pass floor. "
                       "Right: mean level per virtual channel.")))
    S.append(("lidar", "LiDAR",
              fig_html(F.lidar_maps(D), "Left: accumulated bird's-eye view (forward up, vehicle-left on the "
                       "left; blue lines: radar ±60° field of view). Right: points by direction and range; a "
                       "sector that stays empty beyond a few metres is blocked.") +
              fig_html(F.lidar_stats(D), "Points per sweep, near-field share, and range/intensity "
                       "distributions (accumulated on every 5th sweep).")))
    S.append(("camera", "Camera",
              fig_html(F.camera_thumbs(D), "Left-camera frames spread evenly over the session.") +
              fig_html(F.camera_quality(D), "Image brightness, saturation and black-clipping (5 Hz sample), and "
                       "the position of dropped frames.") +
              fig_html(F.camera_imu(D), "ZED 2i IMU magnitudes; shows whether the rig was static.")))
    S.append(("cross", "Cross-sensor",
              fig_html(F.range_overlay(D), "Range of the moving target seen by radar (strongest moving return "
                       "above its range bin's floor) and LiDAR (single-target track), on the common time base.") +
              '<div class="caveat">Radar bird\'s-eye views use the current processing\'s azimuth with its sign '
              'mirrored (verified against camera and LiDAR on freewalk_ref_02). The TX/RX → azimuth table and '
              'antenna calibration are not yet verified, so radar angles are indicative only.</div>' +
              fig_html(F.snapshots(D), "Paired snapshots at six moments: camera frame, LiDAR sweep (top view, "
                       "−1.2…2.2 m height), radar bird's-eye view and radar range–Doppler map, all for the same "
                       "radar frame. Circle: the LiDAR target. Δ: time difference to the radar frame. Radar BEV colour "
                       "scale starts 3 dB above that frame's median (its noise floor). Note how a single person "
                       "appears as an arc spanning much of the field of view: the radar's angular focusing is weak "
                       "with the current (uncalibrated) channel processing, while its range is sharp.")))
    nav = "".join(f'<a href="#{i}">{t}</a>' for i, t, _ in S)
    body = "".join(f'<section id="{i}"><h2>{t}</h2>{c}</section>' for i, t, c in S)
    title = f"Session report – {D['session']}"
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'><title>{html.escape(title)}</title>"
            f"<style>{CSS}</style></head><body><header><h1>{html.escape(D['session'])}</h1>"
            f"<div class='sub'>Three-sensor recording report · {html.escape(str(D['meta'].get('started_local')))} · "
            f"{key}</div></header><nav>{nav}</nav><main>{body}</main></body></html>")
