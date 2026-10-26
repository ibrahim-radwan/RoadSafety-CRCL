#!/usr/bin/env python3
"""
run_perception.py - our GPU detection + tracking over a whole session, saved
for the research team and scored against the LiDAR walker track.

    python run_perception.py --session /mnt/sdcard/sessions/<name> [--sensors radar,lidar]

Radar (perception/radar.py): every frame, in order -> detections, DBSCAN
objects, tracks. LiDAR (perception/lidar.py): the pcap is decoded at 3x real
time (the rs_driver limit) -> objects per sweep, tracks. Tracks are the
reliable output: a track is confirmed after 3 hits in its first 5 updates and
reported while it is updated (radar: and while its Doppler agrees with its motion).

Writes, in <session>/sync/:
    perception_radar.npz   frame_t [F], det_start [F+1], det [N, 9] (radar.DET_COLS),
                           obj_start [F+1], obj [M, 8] (x, y, range, az_deg, v_r, power_db, n, extent),
                           trk [T, 11] (frame, id, x, y, vx, vy, reported, hits, v_torso, spread,
                           cadence_hz): the last three are the track's micro-Doppler features
                           (perception/track_features.py; NaN when not measured this frame;
                           cadence is experimental), noise_db [F]
    perception_lidar.npz   sweep_t [S], obj_start [S+1], obj [M, 13] (lidar.OBJ_COLS),
                           trk [T, 10] (sweep, id, x, y, vx, vy, reported, hits, cls, moving: 1 moving,
                           0 static, NaN not yet known; sensor frame, see tracking.py), ground_z [S]
    perception.json        parameters, timings, and the scores below
Times are t_ref (CLOCK_MONOTONIC, Stage 3 offsets applied); frame x forward, y left, z up.

Scores (when sync/timing_lidar_track.npz exists, from make_report.py): for each
radar frame / LiDAR sweep with the single LiDAR target, whether a detection
or confirmed track lies on it, and the azimuth bias and spread of the radar
against it. The old chain's filtered-cloud scores are kept in
perception/baseline/scores.json. The LiDAR target is a
single-target reference: with two people it follows one of them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from perception.core.lidar_io import decode_sweeps                             # noqa: E402
from perception.core.session import Session, target_at                         # noqa: E402
from perception.lidar import CLASSES, LIDAR_TRACKER, O, LidarDetector          # noqa: E402
from perception.radar import (C, DET_COLS, RADAR_TRACKER, RadarDetector, cluster_detections,  # noqa: E402
                              rx_phase_for)
from perception.track_features import TrackFeatures                            # noqa: E402
from perception.tracking import Tracker                                         # noqa: E402


MOTION_CODE = {"moving": 1.0, "static": 0.0, "?": np.nan}


def _trk_rows(step: int, tracker, extra=None) -> list:
    """Live tracks of this step; 'confirmed' = reported (confirmed and updated within 2 steps)."""
    rep = {t.id for t in tracker.reported()}
    return [[step, t.id, t.x[0], t.x[1], t.x[2], t.x[3], float(t.id in rep), t.hits]
            + ([] if extra is None else list(extra(t))) for t in tracker.tracks]


def run_radar(S: Path, use_rx_phase: bool = True, log=print) -> dict:
    sess = Session(S)
    variant = sess.variant
    rx = rx_phase_for(variant) if use_rx_phase else None
    det = RadarDetector(sess.cfg_path, rx_phase_deg=rx)
    tl = sess.radar
    tracker = Tracker(**RADAR_TRACKER)
    feats = TrackFeatures(det.range_axis, det.az_axis, det.dop_axis)
    dets, objs, trks, noise = [], [], [], []
    t0 = time.monotonic()
    for k in range(len(tl)):
        out = det.process(tl.raw(k))
        d = out["detections"]
        lab, ob = cluster_detections(d)
        tracker.update(float(tl.t[k]), ob[:, :2], list(ob[:, 4]), radial=ob[:, 4])
        f = feats.update(float(tl.t[k]), tracker.tracks, d, lab, out["p_az"])
        dets.append(d); objs.append(ob); noise.append(out["noise_db"])
        nan3 = {"v_torso": np.nan, "spread": np.nan, "cadence_hz": None}
        trks += [row + [f.get(tr.id, nan3)["v_torso"], f.get(tr.id, nan3)["spread"],
                        np.nan if f.get(tr.id, nan3)["cadence_hz"] is None else f[tr.id]["cadence_hz"]]
                 for row, tr in zip(_trk_rows(k, tracker), tracker.tracks)]
        if k % 1000 == 0:
            log(f"  radar {k}/{len(tl)}  {time.monotonic() - t0:.0f} s")
    el = time.monotonic() - t0
    nd, no = np.array([len(x) for x in dets]), np.array([len(x) for x in objs])
    res = {"frame_t": tl.t.copy(),
           "det_start": np.r_[0, np.cumsum(nd)], "det": np.concatenate(dets) if nd.sum() else
           np.zeros((0, len(DET_COLS)), np.float32), "obj_start": np.r_[0, np.cumsum(no)],
           "obj": np.concatenate(objs) if no.sum() else np.zeros((0, 8), np.float32),
           "trk": np.asarray(trks, np.float32).reshape(-1, 11), "noise_db": np.array(noise)}
    res["meta"] = {"variant": variant, "rx_phase_deg": rx, "frames": len(tl), "seconds": round(el, 1),
                   "ms_per_frame": round(1e3 * el / max(len(tl), 1), 1),
                   "detections_per_frame": round(float(nd.mean()), 2), "objects_per_frame": round(float(no.mean()), 2)}
    log(f"  radar: {len(tl)} frames in {el:.0f} s ({res['meta']['ms_per_frame']} ms/frame incl. file reads), "
        f"{nd.mean():.1f} detections/frame")
    return res


def run_lidar(S: Path, log=print, rate: float = 3.0) -> dict:
    det = LidarDetector()
    tracker = Tracker(**LIDAR_TRACKER)
    out = {"t": [], "obj": [], "trk": [], "gz": []}
    ms = []
    t0 = time.monotonic()
    for t, p in decode_sweeps(S, rate):
        tic = time.perf_counter()
        o = det.detect(p, labels=False)
        ms.append(1e3 * (time.perf_counter() - tic))
        ob = o["objects"]
        movable = ob[ob[:, O["cls"]] != CLASSES.index("structure")]
        tracker.update(t, movable[:, [O["x"], O["y"]]], list(movable[:, O["cls"]]))
        k = len(out["t"])
        out["t"].append(t); out["obj"].append(ob); out["gz"].append(o["ground_z"])
        out["trk"] += _trk_rows(k, tracker, extra=lambda tr: [tr.payload, MOTION_CODE[tr.motion]])
        if len(out["t"]) % 1000 == 0:
            log(f"  lidar {len(out['t'])} sweeps  {time.monotonic() - t0:.0f} s")
    el = time.monotonic() - t0
    n = np.array([len(x) for x in out["obj"]])
    res = {"sweep_t": np.array(out["t"]), "obj_start": np.r_[0, np.cumsum(n)],
           "obj": np.concatenate(out["obj"]) if n.sum() else np.zeros((0, len(O)), np.float32),
           "trk": np.asarray(out["trk"], np.float32).reshape(-1, 10), "ground_z": np.array(out["gz"])}
    res["meta"] = {"sweeps": len(n), "seconds": round(el, 1), "detector_ms_median": round(float(np.median(ms)), 1),
                   "detector_ms_p95": round(float(np.percentile(ms, 95)), 1),
                   "objects_per_sweep": round(float(n.mean()), 2)}
    log(f"  lidar: {len(n)} sweeps in {el:.0f} s, detector {np.median(ms):.1f} ms median, "
        f"{n.mean():.1f} objects/sweep")
    return res


# ── scores against the LiDAR walker track ───────────────────────────────────
def score_radar_points(frame_t, start, rng, az, track, trk=None, gate_r=0.6, gate_az=10.0) -> dict:
    """Per radar frame with a LiDAR target: a detection within gate_r (range) and gate_az of it;
    azimuth bias/spread of the nearest-in-range one. trk: (frame, id, x, y, ..., confirmed) rows."""
    hit, errs, n_tgt, empty_conf, tgt_conf, extra_conf = 0, [], 0, [], 0, []
    by_frame = {}
    if trk is not None and len(trk):
        conf = trk[trk[:, 6] > 0]
        for k in np.unique(conf[:, 0]).astype(int):
            by_frame[k] = conf[conf[:, 0] == k][:, 2:4]
    for k, t in enumerate(frame_t):
        tg = target_at(track, t, 0.08)
        ct = by_frame.get(k, np.zeros((0, 2)))
        if tg is None:
            empty_conf.append(len(ct))
            continue
        n_tgt += 1
        a, b = start[k], start[k + 1]
        if b > a:
            dr = np.abs(rng[a:b] - tg[0])
            i = int(np.argmin(dr))
            if dr[i] < gate_r:
                e = az[a + i] - tg[1]
                errs.append(e)
                if abs(e) < gate_az:
                    hit += 1
        tx, ty = tg[0] * np.cos(np.radians(tg[1])), tg[0] * np.sin(np.radians(tg[1]))
        d = np.hypot(ct[:, 0] - tx, ct[:, 1] - ty) if len(ct) else np.zeros(0)
        tgt_conf += int((d < 1.5).any())
        extra_conf.append(int((d >= 1.5).sum()))
    e = np.array(errs)
    bias = float(np.median(e)) if len(e) else float("nan")
    s = {"frames_with_target": n_tgt, "detected": round(hit / max(n_tgt, 1), 3),
         "az_bias_deg": round(bias, 2),
         "az_spread_deg": round(float(np.median(np.abs(e - bias))), 2) if len(e) else None,
         "az_within_3deg_of_bias": round(float(np.mean(np.abs(e - bias) < 3)), 3) if len(e) else None}
    if trk is not None:
        s.update(confirmed_track_on_target=round(tgt_conf / max(n_tgt, 1), 3),
                 other_confirmed_tracks_per_frame=round(float(np.mean(extra_conf)), 2) if extra_conf else 0.0,
                 confirmed_tracks_per_empty_frame=round(float(np.mean(empty_conf)), 2) if empty_conf else None)
    return s


def score_lidar(res, track) -> dict:
    n_tgt, hit, conf_hit, cls = 0, 0, 0, []
    trk = res["trk"]
    by = {}
    if len(trk):
        conf = trk[trk[:, 6] > 0]
        for k in np.unique(conf[:, 0]).astype(int):
            by[k] = conf[conf[:, 0] == k][:, 2:4]
    for k, t in enumerate(res["sweep_t"]):
        tg = target_at(track, t, 0.06)
        if tg is None:
            continue
        n_tgt += 1
        tx, ty = tg[0] * np.cos(np.radians(tg[1])), tg[0] * np.sin(np.radians(tg[1]))
        ob = res["obj"][res["obj_start"][k]:res["obj_start"][k + 1]]
        if len(ob):
            d = np.hypot(ob[:, O["x"]] - tx, ob[:, O["y"]] - ty)
            i = int(np.argmin(d))
            if d[i] < 0.8:
                hit += 1
                cls.append(CLASSES[int(ob[i, O["cls"]])])
        ct = by.get(k, np.zeros((0, 2)))
        conf_hit += int(len(ct) and (np.hypot(ct[:, 0] - tx, ct[:, 1] - ty) < 0.8).any())
    u, c = np.unique(cls, return_counts=True)
    return {"sweeps_with_target": n_tgt, "detected": round(hit / max(n_tgt, 1), 3),
            "confirmed_track_on_target": round(conf_hit / max(n_tgt, 1), 3),
            "class_of_target": {k: round(v / max(len(cls), 1), 3) for k, v in zip(u, c)}}


def score_radar_vs_lidar(radar: dict, lidar: dict, gate_m: float = 1.5, fov_deg: float = 60.0) -> dict:
    """Confirmed radar tracks within the radar FOV: the share with a LiDAR object (any class, any
    size: the LiDAR sees every reflector the radar can) within gate_m at the nearest sweep.
    Unsupported tracks are the false-alarm proxy (no ground truth needed)."""
    trk = radar["trk"]
    conf = trk[trk[:, 6] > 0]
    if not len(conf) or not len(lidar["sweep_t"]):
        return {"confirmed_tracks": int(len(conf))}
    st, ob = lidar["obj_start"], lidar["obj"]
    sup, n, per_frame = 0, 0, {}
    for row in conf:
        k = int(row[0])
        x, y = row[2], row[3]
        if abs(np.degrees(np.arctan2(y, x))) > fov_deg or x <= 0:
            continue
        t = radar["frame_t"][k]
        j = int(np.argmin(np.abs(lidar["sweep_t"] - t)))
        o = ob[st[j]:st[j + 1]]
        ok = bool(len(o)) and bool((np.hypot(o[:, O["x"]] - x, o[:, O["y"]] - y) < gate_m).any())
        sup += ok
        n += 1
        per_frame[k] = per_frame.get(k, 0) + (not ok)
    nf = len(radar["frame_t"])
    return {"confirmed_tracks_in_fov": n, "supported_by_lidar": round(sup / max(n, 1), 3),
            "unsupported_per_frame": round(sum(per_frame.values()) / max(nf, 1), 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description="GPU detection + tracking for a session (radar, LiDAR)")
    ap.add_argument("--session", type=Path, nargs="+", required=True)
    ap.add_argument("--sensors", default="radar,lidar")
    ap.add_argument("--no-rx-phase", action="store_true", help="radar: without the in-situ RX phase correction")
    a = ap.parse_args()
    sensors = [s.strip() for s in a.sensors.split(",")]
    for S in a.session:
        print(f"=== {S}", flush=True)
        jp = S / "sync" / "perception.json"
        report = json.loads(jp.read_text()) if jp.is_file() else {}       # a partial run keeps the other sensor
        report.update(session=S.name, created=time.strftime("%Y-%m-%dT%H:%M:%S"))
        track, radar_res, lidar_res = Session(S).target, None, None
        if "radar" in sensors and (S / "sync" / "radar_timeline.csv").is_file():
            r = radar_res = run_radar(S, not a.no_rx_phase, log=lambda m: print(m, flush=True))
            meta = r.pop("meta")
            np.savez_compressed(S / "sync" / "perception_radar.npz", **r)
            report["radar"] = meta
            if track is not None:
                d = r["det"]
                report["radar"]["score"] = score_radar_points(r["frame_t"], r["det_start"], d[:, C["range"]],
                                                              d[:, C["az_deg"]], track, r["trk"])
        if "lidar" in sensors and (S / "lidar" / "lidar.pcap").is_file():
            r = lidar_res = run_lidar(S, log=lambda m: print(m, flush=True))
            meta = r.pop("meta")
            np.savez_compressed(S / "sync" / "perception_lidar.npz", **r)
            report["lidar"] = meta
            if track is not None:
                report["lidar"]["score"] = score_lidar(r, track)
        if radar_res is None and (S / "sync" / "perception_radar.npz").is_file():
            radar_res = dict(np.load(S / "sync" / "perception_radar.npz"))           # from an earlier run
        if lidar_res is None and (S / "sync" / "perception_lidar.npz").is_file():
            lidar_res = dict(np.load(S / "sync" / "perception_lidar.npz"))           # from an earlier run
        if radar_res is not None and lidar_res is not None:
            report["radar_tracks_vs_lidar"] = score_radar_vs_lidar(radar_res, lidar_res)
        jp.write_text(json.dumps(report, indent=2, default=float) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k not in ("session", "created")}, indent=2,
                         default=float), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
