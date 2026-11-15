#!/usr/bin/env python3
"""
analyse_altos.py — measure what the Altos V4 radar saw in a recording, against
a room and targets measured with a tape, and compare recordings made with
different settings.

    python analyse_altos.py dataset/altos_session_A                        # one recording
    python analyse_altos.py dataset/altos_session_A dataset/altos_session_B --plot
    python analyse_altos.py dataset/altos_session_* --scene my_scene.yaml --csv sweep.csv

The scene (room and targets, see scene_example.yaml) is taken from each
session's scene.yaml (the recorder copies it with --scene), else from
--scene. Without a scene only the general measures are reported.

WHAT IS MEASURED (per recording)
  General     frame rate, points per frame, NaN share, SNR spread, radar
              restarts, settings the radar reported at the start.
  Room        every point is put in one class:
                target         within a target's radius
                surface        within `tolerance` of a wall, the floor or the ceiling
                interior       inside the room, away from the surfaces (furniture, people, ...)
                beyond walls   outside the room: reflections of reflections (ghosts)
                below floor    under the floor: floor-bounce reflections (the
                               vendor driver can mirror these up: z -> -z - 2h)
              and how much of each surface the radar sees: the surface is cut
              into 25 cm cells; a cell inside the field of view counts as
              seen when points fall on it in at least 5% of frames.
  Targets     detection rate (frames with a point on the target), and the
              strongest point's range, azimuth and elevation against the tape
              (mean error and frame-to-frame spread). Range and azimuth
              profiles around each target: separate peaks = resolved objects
              (used for the resolution tests).
  Motion      for moving points (|doppler| >= 0.3 m/s): the radar's Doppler
              against how fast their range changes from frame to frame.
              slope ~ +1 means positive Doppler = moving away, as assumed.
              Also the moving points' azimuth at the start and end (+ = left).
  Comparison  a table across recordings. For recordings of the same scene,
              profile and FPS, one with threshold 0 is the reference: for
              each other threshold, the SNR cut on the reference that leaves
              the same number of points, against the lowest SNR the other
              run actually has. If they agree, the threshold works as an SNR
              cut and recording at 0 loses nothing.

OUTPUT
  printed report; <session>/analysis.json; with --plot <session>/analysis.png
  (top and side views, points coloured by class, room and targets drawn);
  with --csv a table of all recordings.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

import altos_v4 as av

HALF_HFOV, HALF_VFOV = 75.0, 15.0
CLASSES = ("target", "surface", "interior", "beyond walls", "below floor")
CLASS_COLOURS = {"target": "#e8a400", "surface": "#2a78d6", "interior": "#1baf7a",
                 "beyond walls": "#d6402a", "below floor": "#9b59b6"}
CELL = 0.25
SEEN_FRACTION = 0.05
MOVING_V = 0.3


# ---------------------------------------------------------------- loading

def load_scene(session: Path, fallback):
    path = session / "scene.yaml"
    if not path.exists():
        path = fallback
    if path is None:
        return None, None
    import yaml
    sc = yaml.safe_load(Path(path).read_text())
    room = sc["room"]
    for k in ("front", "back", "left", "right", "ceiling"):
        room[k] = float(room[k])
    sc["radar"]["height"] = float(sc["radar"]["height"])
    sc["tolerance"] = float(sc.get("tolerance", 0.15))
    sc["targets"] = sc.get("targets") or []
    return sc, str(path)


def load_session(session: Path):
    asm = av.FrameAssembler()
    frames = []
    for t_ns, pkt in av.read_pcap(session / "altos.pcap"):
        frames.extend(asm.feed(pkt, t_ns))
    frames.extend(asm.flush())
    meta_path = session / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return frames, asm, meta


def settings_of(meta: dict) -> dict:
    inst = (meta.get("radar_settings", {}).get("install") or [{}])[0]
    thr = (inst.get("rpvb_control_overrides") or {}).get("threshold_coeff")
    return {
        "profile": inst.get("profile"),
        "threshold": 250000.0 if thr is None else float(thr.get("Float", [thr])[0]),
        "fps_limit": inst.get("fps_limit"),
        "sync": inst.get("sync_mode"),
    }


# ---------------------------------------------------------------- geometry

def spherical(p):
    """range, azimuth deg (+ left), elevation deg (+ up) of vehicle-frame xyz."""
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return (np.sqrt(x * x + y * y + z * z), np.degrees(np.arctan2(y, x)),
            np.degrees(np.arctan2(z, np.hypot(x, y))))


def room_planes(sc):
    r, h = sc["room"], sc["radar"]["height"]
    lo = np.array([-r["back"], -r["right"], -h])
    hi = np.array([r["front"], r["left"], r["ceiling"]])
    # name, axis, value; the back wall is behind the radar, out of its view
    planes = [("front wall", 0, hi[0]), ("left wall", 1, hi[1]), ("right wall", 1, lo[1]),
              ("floor", 2, lo[2]), ("ceiling", 2, hi[2])]
    return lo, hi, planes


def classify(xyz, sc):
    """Class index per point (CLASSES), nearest-surface index (into room_planes, -1 none)."""
    lo, hi, planes = room_planes(sc)
    tol = sc["tolerance"]
    n = len(xyz)
    cls = np.full(n, CLASSES.index("interior"), np.int8)
    below = xyz[:, 2] < lo[2] - tol
    outside = np.any((xyz < lo - tol) | (xyz > hi + tol), axis=1)
    cls[outside] = CLASSES.index("beyond walls")
    cls[below] = CLASSES.index("below floor")
    # distance to each viewed surface (inside the box limits of the other two axes)
    d = np.full((n, len(planes)), np.inf)
    for k, (_name, ax, val) in enumerate(planes):
        others = [a for a in range(3) if a != ax]
        within = np.all((xyz[:, others] >= lo[others] - tol) & (xyz[:, others] <= hi[others] + tol), axis=1)
        d[within, k] = np.abs(xyz[within, ax] - val)
    near = d.min(axis=1)
    nearest = np.where(np.isfinite(near), d.argmin(axis=1), -1)
    on_surface = (near <= tol) & ~below
    cls[on_surface] = CLASSES.index("surface")
    for t in sc["targets"]:
        c = np.array([t["x"], t["y"], t["z"]], float)
        cls[np.linalg.norm(xyz - c, axis=1) <= float(t.get("radius", 0.25))] = CLASSES.index("target")
    nearest[cls != CLASSES.index("surface")] = -1
    return cls, nearest


def surface_coverage(xyz, frame_idx, nearest, n_frames, sc):
    """Per surface: cells in the field of view, cells seen in >= SEEN_FRACTION of frames."""
    lo, hi, planes = room_planes(sc)
    out = {}
    for k, (name, ax, val) in enumerate(planes):
        u, v = [a for a in range(3) if a != ax]
        nu = max(1, int(round((hi[u] - lo[u]) / CELL)))
        nv = max(1, int(round((hi[v] - lo[v]) / CELL)))
        cu = lo[u] + (np.arange(nu) + 0.5) * (hi[u] - lo[u]) / nu
        cv = lo[v] + (np.arange(nv) + 0.5) * (hi[v] - lo[v]) / nv
        centres = np.zeros((nu, nv, 3))
        centres[..., ax] = val
        centres[..., u] = cu[:, None]
        centres[..., v] = cv[None, :]
        _r, az, el = spherical(centres)
        visible = (np.abs(az) <= HALF_HFOV) & (np.abs(el) <= HALF_VFOV) & (centres[..., 0] > 0.1)
        m = nearest == k
        seen = np.zeros((nu, nv), bool)
        if m.any():
            iu = np.clip(((xyz[m, u] - lo[u]) / (hi[u] - lo[u]) * nu).astype(int), 0, nu - 1)
            iv = np.clip(((xyz[m, v] - lo[v]) / (hi[v] - lo[v]) * nv).astype(int), 0, nv - 1)
            cell = iu * nv + iv
            pairs = np.unique(np.stack([cell, frame_idx[m]], axis=1), axis=0)   # one count per frame
            counts = np.bincount(pairs[:, 0], minlength=nu * nv).reshape(nu, nv)
            seen = counts >= SEEN_FRACTION * n_frames
        out[name] = {"cells_in_view": int(visible.sum()), "cells_seen": int((seen & visible).sum())}
    tot_v = sum(o["cells_in_view"] for o in out.values())
    tot_s = sum(o["cells_seen"] for o in out.values())
    out["all"] = {"cells_in_view": tot_v, "cells_seen": tot_s}
    return out


def peaks(values, bins, min_share=0.15, min_sep=2):
    """Peak positions and counts of a 1D sample (smoothed histogram)."""
    if len(values) == 0:
        return []
    h, edges = np.histogram(values, bins=bins)
    hs = np.convolve(h, [1, 2, 1], mode="same") / 4.0
    centres = (edges[:-1] + edges[1:]) / 2
    top = hs.max()
    cand = [i for i in range(len(hs)) if hs[i] >= min_share * top
            and hs[i] >= (hs[i - 1] if i else 0) and hs[i] > (hs[i + 1] if i + 1 < len(hs) else 0)]
    cand.sort(key=lambda i: -hs[i])
    kept = []
    for i in cand:
        if all(abs(i - j) >= min_sep for j in kept):
            kept.append(i)
    return [(float(centres[i]), int(h[max(0, i - 1):i + 2].sum())) for i in sorted(kept)[:5]]


# ---------------------------------------------------------------- analysis

def analyse(session: Path, scene_fallback):
    frames, asm, meta = load_session(session)
    if not frames:
        raise SystemExit("No frames in %s" % session)
    d = av.frames_to_arrays(frames)
    n = len(frames)
    valid = d["valid"]
    p = d["points"][valid]
    fidx = d["frame_idx"][valid]
    xyz = p[:, 5:8].astype(np.float64)
    t = d["t_host_last_ns"] * 1e-9
    dur = t[-1] - t[0] if n > 1 else 0.0
    snr_db = 10 * np.log10(np.maximum(p[:, 4], 1e-9))
    per_frame = np.bincount(fidx, minlength=n)

    res = {
        "session": str(session), "label": meta.get("label", ""), "settings": settings_of(meta),
        "frames": n, "duration_s": round(dur, 2), "fps": round((n - 1) / dur, 3) if dur > 0 else None,
        "radar_restarts": asm.restarts, "frames_lost": asm.missing_ids, "frames_incomplete": asm.incomplete,
        "points_per_frame": {"mean": float(per_frame.mean()), "p5": float(np.percentile(per_frame, 5)),
                             "median": float(np.median(per_frame)), "p95": float(np.percentile(per_frame, 95))},
        "nan_share": float((~valid).mean()) if len(valid) else 0.0,
        "snr_db": {q: float(np.percentile(snr_db, v)) for q, v in
                   (("p1", 1), ("p5", 5), ("p50", 50), ("p95", 95))} if len(p) else {},
    }

    sc, scene_path = load_scene(session, scene_fallback)
    cls = None
    if sc is not None:
        res["scene"] = scene_path
        cls, nearest = classify(xyz, sc)
        counts = np.bincount(cls, minlength=len(CLASSES))
        res["classes"] = {c: {"share": float(counts[i] / max(1, len(cls))),
                              "per_frame": float(counts[i] / n)} for i, c in enumerate(CLASSES)}
        res["coverage"] = surface_coverage(xyz, fidx, nearest, n, sc)
        res["targets"] = [target_report(t_, p, xyz, fidx, n) for t_ in sc["targets"]]
    res["motion"] = motion_report(p, xyz, fidx, t, cls)
    return res, (p, xyz, fidx, cls, sc)


def target_report(tg, p, xyz, fidx, n):
    c = np.array([tg["x"], tg["y"], tg["z"]], float)
    r_t, az_t, el_t = spherical(c)
    rad = float(tg.get("radius", 0.25))
    on = np.linalg.norm(xyz - c, axis=1) <= rad
    out = {"name": tg.get("name", "target"), "tape": {"range": float(r_t), "azimuth": float(az_t),
                                                       "elevation": float(el_t)}}
    frames_hit = np.unique(fidx[on])
    out["detection_rate"] = float(len(frames_hit) / n)
    out["points_per_frame"] = float(on.sum() / n)
    if on.any():
        # strongest point on the target, per frame
        idx = np.where(on)[0]
        order = np.lexsort((-p[idx, 4], fidx[idx]))
        idx = idx[order]
        first = np.r_[True, fidx[idx][1:] != fidx[idx][:-1]]
        best = idx[first]
        r, az, el = spherical(xyz[best])
        for name, val, ref in (("range", r, r_t), ("azimuth", az, az_t), ("elevation", el, el_t)):
            out[name] = {"mean": float(val.mean()), "error": float(val.mean() - ref), "spread": float(val.std())}
    # range and azimuth profiles around the target's bearing
    r_all, az_all, el_all = spherical(xyz)
    cone = (np.abs(az_all - az_t) <= 6) & (np.abs(el_all - el_t) <= 8)
    rng = cone & (r_all >= r_t - 1.0) & (r_all <= r_t + 1.5)
    out["range_peaks"] = peaks(r_all[rng], np.arange(r_t - 1.0, r_t + 1.5 + 1e-9, 0.02))
    ring = (np.abs(r_all - r_t) <= 0.3) & (np.abs(el_all - el_t) <= 8) & (np.abs(az_all - az_t) <= 30)
    out["azimuth_peaks"] = peaks(az_all[ring], np.arange(az_t - 30, az_t + 30 + 1e-9, 0.5))
    return out


def motion_report(p, xyz, fidx, t, cls):
    keep = (np.abs(p[:, 1]) >= MOVING_V) & (10 * np.log10(np.maximum(p[:, 4], 1e-9)) >= 10)
    if cls is not None:
        keep &= np.isin(cls, [CLASSES.index("interior"), CLASSES.index("target"), CLASSES.index("surface")])
    out = {"moving_points": int(keep.sum())}
    if keep.sum() < 30:
        return out
    r_all, az_all, _el = spherical(xyz)
    f = fidx[keep]
    fr = np.unique(f)
    r_med = np.array([np.median(r_all[keep][f == k]) for k in fr])
    v_med = np.array([np.median(p[keep, 1][f == k]) for k in fr])
    az_med = np.array([np.median(az_all[keep][f == k]) for k in fr])
    cnt = np.array([(f == k).sum() for k in fr])
    good = cnt >= 3
    fr, r_med, v_med, az_med, tt = fr[good], r_med[good], v_med[good], az_med[good], t[fr[good]]
    out["frames_with_motion"] = int(len(fr))
    if len(fr) < 10:
        return out
    # range rate: straight-line fit over 5 consecutive frames (frame ids consecutive)
    rr, vv = [], []
    for i in range(2, len(fr) - 2):
        if fr[i + 2] - fr[i - 2] != 4:
            continue
        sl = slice(i - 2, i + 3)
        rate = np.polyfit(tt[sl] - tt[i], r_med[sl], 1)[0]
        rr.append(rate)
        vv.append(v_med[i])
    rr, vv = np.array(rr), np.array(vv)
    clear = np.abs(rr) >= 0.3
    out["windows"] = int(len(rr))
    if clear.sum() >= 5:
        k = float(np.sum(rr[clear] * vv[clear]) / np.sum(rr[clear] ** 2))
        out["doppler_vs_range_rate"] = {
            "windows_used": int(clear.sum()),
            "slope": k,
            "sign_agreement": float(np.mean(np.sign(rr[clear]) == np.sign(vv[clear]))),
            "correlation": float(np.corrcoef(rr[clear], vv[clear])[0, 1]),
        }
    out["azimuth_start_end"] = [float(np.median(az_med[:5])), float(np.median(az_med[-5:]))]
    return out


# ---------------------------------------------------------------- output

def fmt_report(res) -> str:
    s = res["settings"]
    L = []
    L.append("=" * 100)
    L.append("%s   %s" % (Path(res["session"]).name, ("'%s'" % res["label"]) if res["label"] else ""))
    L.append("settings  profile %s | threshold %s | fps limit %s | sync %s" % (
        s["profile"], s["threshold"], s["fps_limit"] or "off", s["sync"]))
    ppf = res["points_per_frame"]
    L.append("frames    %d over %.1f s = %.2f fps | lost %d | incomplete %d | radar restarts %d" % (
        res["frames"], res["duration_s"], res["fps"] or 0, res["frames_lost"], res["frames_incomplete"],
        res["radar_restarts"]))
    L.append("points    per frame mean %.0f (p5 %.0f, median %.0f, p95 %.0f) | NaN %.2f%% | "
             "SNR dB p1 %.1f p5 %.1f p50 %.1f p95 %.1f" % (
                 ppf["mean"], ppf["p5"], ppf["median"], ppf["p95"], 100 * res["nan_share"],
                 res["snr_db"].get("p1", 0), res["snr_db"].get("p5", 0), res["snr_db"].get("p50", 0),
                 res["snr_db"].get("p95", 0)))
    if "classes" in res:
        L.append("room      " + " | ".join("%s %.1f%% (%.1f/frame)" % (c, 100 * v["share"], v["per_frame"])
                                         for c, v in res["classes"].items()))
        cov = res["coverage"]
        L.append("coverage  " + " | ".join(
            "%s %d/%d" % (k, v["cells_seen"], v["cells_in_view"]) for k, v in cov.items() if k != "all")
                 + " | ALL %.0f%% of %d cells in view" % (
                     100 * cov["all"]["cells_seen"] / max(1, cov["all"]["cells_in_view"]),
                     cov["all"]["cells_in_view"]))
        for tg in res["targets"]:
            tp = tg["tape"]
            line = "target    %s at %.2f m, az %+.1f, el %+.1f (tape): detected in %.0f%% of frames, %.1f pts/frame" % (
                tg["name"], tp["range"], tp["azimuth"], tp["elevation"], 100 * tg["detection_rate"],
                tg["points_per_frame"])
            L.append(line)
            if "range" in tg:
                L.append("          radar: range %.3f m (error %+.3f, spread %.3f) | az %+.2f deg (error %+.2f, "
                         "spread %.2f) | el %+.2f deg (error %+.2f, spread %.2f)" % (
                             tg["range"]["mean"], tg["range"]["error"], tg["range"]["spread"],
                             tg["azimuth"]["mean"], tg["azimuth"]["error"], tg["azimuth"]["spread"],
                             tg["elevation"]["mean"], tg["elevation"]["error"], tg["elevation"]["spread"]))
            L.append("          range peaks along its bearing: %s" % (
                ", ".join("%.2f m (%d)" % pk for pk in tg["range_peaks"]) or "-"))
            L.append("          azimuth peaks at its range:    %s" % (
                ", ".join("%+.1f deg (%d)" % pk for pk in tg["azimuth_peaks"]) or "-"))
    m = res["motion"]
    if "doppler_vs_range_rate" in m:
        dv = m["doppler_vs_range_rate"]
        L.append("motion    %d frames with movement | Doppler = %.2f x range rate (1.00 expected), "
                 "same sign in %.0f%% of %d windows | azimuth start %+.0f -> end %+.0f deg (+ = left)" % (
                     m["frames_with_motion"], dv["slope"], 100 * dv["sign_agreement"], dv["windows_used"],
                     m["azimuth_start_end"][0], m["azimuth_start_end"][1]))
    else:
        L.append("motion    %d moving points, %s frames with movement: not enough for the Doppler check" % (
            m["moving_points"], m.get("frames_with_motion", 0)))
    return "\n".join(L)


def threshold_equivalence(results, cache):
    """For runs sharing profile and FPS, compare each threshold with the threshold-0 run."""
    lines = []
    groups = {}
    for r in results:
        s = r["settings"]
        groups.setdefault((s["profile"], s["fps_limit"]), []).append(r)
    for (prof, fps), rs in groups.items():
        refs = [r for r in rs if r["settings"]["threshold"] == 0]
        others = [r for r in rs if r["settings"]["threshold"] != 0]
        if not refs or not others:
            continue
        ref = refs[0]
        ref_snr = 10 * np.log10(np.maximum(cache[ref["session"]][0][:, 4], 1e-9))
        n_ref = ref["frames"]
        lines.append("threshold check, profile %s, fps limit %s, reference %s (threshold 0)" % (
            prof, fps or "off", Path(ref["session"]).name))
        for r in sorted(others, key=lambda r: r["settings"]["threshold"]):
            want = r["points_per_frame"]["mean"] * n_ref           # same points per frame as the run
            srt = np.sort(ref_snr)[::-1]
            k = int(min(len(srt) - 1, max(0, round(want) - 1)))
            cut = float(srt[k])
            run_snr = 10 * np.log10(np.maximum(cache[r["session"]][0][:, 4], 1e-9))
            above = ref_snr[ref_snr >= cut]
            q_ref = np.percentile(above, [10, 50, 90]) if len(above) else [np.nan] * 3
            q_run = np.percentile(run_snr, [10, 50, 90]) if len(run_snr) else [np.nan] * 3
            lines.append("  threshold %-12g %.0f pts/frame: equal-count SNR cut on the reference %.1f dB | "
                         "run's lowest SNR (p1) %.1f dB | SNR p10/p50/p90 ref-above-cut %s vs run %s" % (
                             r["settings"]["threshold"], r["points_per_frame"]["mean"], cut,
                             r["snr_db"].get("p1", float("nan")),
                             "/".join("%.1f" % v for v in q_ref), "/".join("%.1f" % v for v in q_run)))
    return lines


def table(results):
    head = ["session", "label", "profile", "threshold", "fps_limit", "fps", "pts/frame", "SNR p5",
            "surface%", "interior%", "beyond%", "below%", "seen%"]
    rows = []
    for r in results:
        s = r["settings"]
        row = [Path(r["session"]).name, r["label"], s["profile"], "%g" % s["threshold"], s["fps_limit"] or "off",
               "%.2f" % (r["fps"] or 0), "%.0f" % r["points_per_frame"]["mean"], "%.1f" % r["snr_db"].get("p5", 0)]
        if "classes" in r:
            c, cov = r["classes"], r["coverage"]["all"]
            row += ["%.1f" % (100 * c[k]["share"]) for k in ("surface", "interior", "beyond walls", "below floor")]
            row.append("%.0f" % (100 * cov["cells_seen"] / max(1, cov["cells_in_view"])))
        else:
            row += ["-"] * 5
        for tg in r.get("targets", []):
            row.append("%s: det %.0f%% r_err %s az_err %s" % (
                tg["name"], 100 * tg["detection_rate"],
                "%+.3f" % tg["range"]["error"] if "range" in tg else "-",
                "%+.1f" % tg["azimuth"]["error"] if "azimuth" in tg else "-"))
        rows.append(row)
    return head, rows


def plot(res, data, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p, xyz, fidx, cls, sc = data
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1, 1.3]})
    s = res["settings"]
    fig.suptitle("%s  %s\nprofile %s, threshold %g, fps limit %s — all %d frames overlaid" % (
        Path(res["session"]).name, res["label"], s["profile"], s["threshold"], s["fps_limit"] or "off",
        res["frames"]))
    views = ((axes[0], 1, 0, "top view: forward up, radar-left on the left", "y (m, + left)", "x (m, forward)"),
             (axes[1], 0, 2, "side view: forward to the right, up up", "x (m, forward)", "z (m, + up)"))
    for ax, hcol, vcol, title, xl, yl in views:
        if cls is None:
            ax.scatter(xyz[:, hcol], xyz[:, vcol], s=2, c="#2a78d6", alpha=0.3, linewidths=0)
        else:
            for i, c in enumerate(CLASSES):
                m = cls == i
                if m.any():
                    ax.scatter(xyz[m, hcol], xyz[m, vcol], s=2, c=CLASS_COLOURS[c], alpha=0.35,
                               linewidths=0, label="%s (%.0f%%)" % (c, 100 * m.mean()))
        if sc is not None:
            lo, hi, _pl = room_planes(sc)
            bx = [lo[hcol], hi[hcol], hi[hcol], lo[hcol], lo[hcol]]
            by = [lo[vcol], lo[vcol], hi[vcol], hi[vcol], lo[vcol]]
            ax.plot(bx, by, color="k", lw=1.2)
            for tg in sc["targets"]:
                c = (tg["x"], tg["y"], tg["z"])
                ax.add_patch(plt.Circle((c[hcol], c[vcol]), float(tg.get("radius", 0.25)), fill=False,
                                        color=CLASS_COLOURS["target"], lw=1.5))
            span = max(hi - lo) + 1.0
        else:
            span = 8.0
        ax.plot([0], [0], marker="o", color="#2a78d6", ms=8)
        lim = HALF_HFOV if hcol == 1 else HALF_VFOV
        for sgn in (-1, 1):
            a = math.radians(sgn * lim)
            if hcol == 1:
                ax.plot([0, span * math.sin(a)], [0, span * math.cos(a)], color="#2a78d6", lw=0.8, ls="--")
            else:
                ax.plot([0, span * math.cos(a)], [0, span * math.sin(a)], color="#2a78d6", lw=0.8, ls="--")
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    # limits: the room (if any) and the bulk of the points (1st-99th percentile)
    if len(xyz):
        plo, phi = np.percentile(xyz, 1, axis=0), np.percentile(xyz, 99, axis=0)
    else:
        plo, phi = np.array([-1.0, -3, -1]), np.array([5.0, 3, 1])
    if sc is not None:
        lo, hi, _pl = room_planes(sc)
        plo, phi = np.minimum(plo, lo), np.maximum(phi, hi)
        axes[1].legend(loc="upper right", fontsize=8, markerscale=4)
    plo, phi = np.minimum(plo, 0) - 0.5, np.maximum(phi, 0) + 0.5
    axes[0].set_xlim(phi[1], plo[1])                      # left on the left
    axes[0].set_ylim(plo[0], phi[0])
    axes[1].set_xlim(plo[0], phi[0])
    axes[1].set_ylim(plo[2], phi[2])
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    ap.add_argument("sessions", type=Path, nargs="+")
    ap.add_argument("--scene", type=Path, help="scene file for sessions without their own scene.yaml")
    ap.add_argument("--plot", action="store_true", help="write <session>/analysis.png")
    ap.add_argument("--csv", type=Path, help="write the comparison table here")
    a = ap.parse_args()

    results, cache = [], {}
    for s in a.sessions:
        res, data = analyse(s, a.scene)
        results.append(res)
        cache[res["session"]] = data
        print(fmt_report(res))
        (s / "analysis.json").write_text(json.dumps(res, indent=2) + "\n")
        if a.plot:
            plot(res, data, s / "analysis.png")
            print("plot      %s" % (s / "analysis.png"))

    if len(results) > 1:
        print("=" * 100)
        head, rows = table(results)
        widths = [max(len(str(x)) for x in col) for col in zip(head, *[r[:len(head)] for r in rows])]
        print("  ".join(h.ljust(w) for h, w in zip(head, widths)))
        for r in rows:
            print("  ".join(str(x).ljust(w) for x, w in zip(r[:len(head)], widths)) + "  " + "  ".join(r[len(head):]))
        for line in threshold_equivalence(results, cache):
            print(line)
        if a.csv:
            with a.csv.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(head + ["targets"])
                for r in rows:
                    w.writerow(r[:len(head)] + [" | ".join(r[len(head):])])
            print("wrote %s" % a.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
