#!/usr/bin/env python3
"""
calib_reflector.py - the corner reflector's points in the LiDAR and the radar, found as what CHANGED
from an empty-scene background, for each reflector position; saved with the frames they came from,
plus the LiDAR -> radar fit over all positions.

INPUT: a background and one or more positions, each either
    a recorded session (capture_session.py; aligned: sync/ from align_session.py), or
    a live-view calibration capture (the iPad page's "Save background" / "Save position":
    <sessions-root>/calibration/<set>/<name>/, see live/README.md).

    python calib_reflector.py --set /mnt/sdcard/sessions/calibration/<set>          # a live capture set
    python calib_reflector.py --background <bg session> --positions <s1> <s2> ...   # recorded sessions
        [--range 1 15] [--z -0.9 2.5] [--secs 8] [--out <dir>] [--no-figure] [--no-camera]

WHAT IT DOES (perception/core/reflector.py)
    LiDAR  the background's voxels (0.15 m, dilated); a position's new returns, in 0.3 m cells; the
           reflector is the cell group with the most returns present in >= 30% of the sweeps (people
           move, the reflector does not). Its points are the new returns within 0.45 m of it.
           Worked in the LiDAR's own frame; reported in the vehicle frame too, with the transform
           the session or capture used (calibration/lidar_extrinsic.json).
    radar  each frame's zero-Doppler channels minus the background's mean, subtracted coherently
           (complex) and then beamformed, so the change is seen without the walls' sidelobes; the
           largest rise >= 10 dB above the background's own frame-to-frame residual, interpolated
           in range and azimuth. Its cells are those within 3 dB of the peak, per frame. (Coherent
           subtraction holds across sessions: on 2026-11-03 the static scene left 6-15% after it.)
    camera the reflector in the left image: pixels changed from every background image in every
           image of the position, below the horizon; the largest blob is the tripod and its top the
           reflector; azimuth = atan((cx - u) / fx) (rectified intrinsics; no depth). This is the
           angle reference: LiDAR yaw = camera azimuth - LiDAR own-frame azimuth (robust: median, then
           the positions within 2 deg of it). At 8 m the tripod is too small (listed as outliers).
    Recorded sessions: the --secs window centred on the radar's recording; each radar frame is
    paired with the LiDAR sweep nearest in t_ref (the session's Stage 3 offsets apply); camera
    images at three times in the window.

OUTPUT (--out, default: the set folder, or <first position>/../reflector_calibration)
    reflector_summary.json   per position: LiDAR (own frame and vehicle frame), radar and camera
                             reflector, detection rates; fit: lidar_yaw_from_camera_deg (the one to
                             use), radar_minus_camera_az_deg, the LiDAR yaw from the radar positions
                             within 5 deg of its boresight, and a rigid LiDAR -> radar fit over all
                             positions (flagged: one wrong pick, e.g. the LiDAR at 8 m on 2026-11-03,
                             spoils it; see residual_m). The radar azimuth is refined coherently
                             across both chips (radar_az; radar_az_incoherent = the map's peak)
    reflector_pairs.csv      one row per position, the same numbers
    <name>_reflector.npz     per radar frame: its t, the paired sweep's t, and the reflector points of
                             both (LiDAR: x y z range intensity in the LiDAR frame, and x y z in the
                             vehicle frame; radar: x y range az power rise), frame index per row
    reflector_points.png     only the reflector points, every position, bird's-eye (vehicle frame)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from perception.core import reflector as rf                                    # noqa: E402
from perception.core.extrinsic import IDENTITY, describe, is_identity          # noqa: E402

LIDAR_COLOR, RADAR_COLOR = "#eb6834", "#2a78d6"


def to_lidar_frame(p: np.ndarray, ext: dict) -> np.ndarray:
    """Vehicle-frame points back into the LiDAR's own frame (inverse of lidar_points_to_vehicle)."""
    if is_identity(ext):
        return p
    q = np.array(p, np.float32, copy=True)
    ok = q[:, 3] > 0
    c, s = np.cos(np.radians(ext["yaw_deg"])), np.sin(np.radians(ext["yaw_deg"]))
    x, y = q[ok, 0] - ext["tx_m"], q[ok, 1] - ext["ty_m"]
    q[ok, 0], q[ok, 1] = c * x + s * y, -s * x + c * y
    q[ok, 3] = np.sqrt(q[ok, 0] ** 2 + q[ok, 1] ** 2 + q[ok, 2] ** 2)
    return q


def to_vehicle_xy(xy: np.ndarray, ext: dict) -> np.ndarray:
    xy = np.asarray(xy, float).reshape(-1, 2)
    if is_identity(ext):
        return xy
    c, s = np.cos(np.radians(ext["yaw_deg"])), np.sin(np.radians(ext["yaw_deg"]))
    return np.c_[c * xy[:, 0] - s * xy[:, 1] + ext["tx_m"], s * xy[:, 0] + c * xy[:, 1] + ext["ty_m"]]


# ── readers: one capture = LiDAR sweeps (own frame) + radar frames, with times ──
def is_live_capture(d: Path) -> bool:
    return (Path(d) / "capture.json").is_file()


def read_live(d: Path) -> dict:
    """A live-view calibration capture (live/radar_live.py and lidar_live.py write it)."""
    d = Path(d)
    cap = json.loads((d / "capture.json").read_text())
    out = {"name": cap.get("name", d.name), "kind": cap.get("kind"), "lidar": [], "lidar_t": np.zeros(0),
           "radar_raw": [], "radar_t": np.zeros(0), "ext": dict(IDENTITY), "variant": None}
    lz = d / "lidar" / "sweeps.npz"
    if lz.is_file():
        z = np.load(lz)
        cuts = np.cumsum(z["counts"])[:-1]
        out["lidar"] = np.split(z["points"], cuts)
        out["lidar_t"] = z["t"]
        meta = json.loads((d / "lidar" / "meta.json").read_text())
        out["ext"] = meta.get("extrinsic") or dict(IDENTITY)
    rz = d / "radar" / "frames.npy"
    if rz.is_file():
        out["radar_raw"] = list(np.load(rz))
        out["radar_t"] = np.load(d / "radar" / "t.npy")
        out["variant"] = json.loads((d / "radar" / "meta.json").read_text()).get("variant")
    return out


def read_session(s: Path, secs: float) -> dict:
    """A recorded session: the `secs` window centred on its radar recording."""
    from perception.core.extrinsic import extrinsic_for_session
    from perception.core.lidar_io import decode_sweeps
    from perception.core.session import Session
    sess = Session(s)
    tl = sess.radar
    mid = 0.5 * (tl.t[0] + tl.t[-1])
    t0, t1 = mid - secs / 2, mid + secs / 2
    ks = [k for k in range(len(tl)) if t0 <= tl.t[k] <= t1]
    ext = extrinsic_for_session(s)
    sweeps, st = [], []
    for t, p in decode_sweeps(s):                       # vehicle frame -> back to the LiDAR's own
        if t > t1:
            break
        if t >= t0:
            sweeps.append(to_lidar_frame(p[p[:, 3] > 0], ext))
            st.append(t)
    return {"name": Path(s).name, "kind": None, "lidar": sweeps, "lidar_t": np.array(st),
            "radar_raw": [tl.raw(k) for k in ks], "radar_t": tl.t[ks], "ext": ext, "variant": sess.variant}


def read_capture(d: Path, secs: float, camera: bool = True) -> dict:
    out = read_live(d) if is_live_capture(d) else read_session(d, secs)
    out["camera"], out["intrinsics"] = (read_camera(d, secs) if camera else ([], None))
    return out


def read_camera(d: Path, secs: float) -> tuple:
    """A few left images (RGB) of the capture and the rectified left intrinsics {fx, cx, cy, w, h}:
    a live capture's camera/*.jpg + camera/meta.json, or 3 frames of a session's SVO2 in the window."""
    d = Path(d)
    if is_live_capture(d):
        meta = d / "camera" / "meta.json"
        if not meta.is_file():
            return [], None
        import cv2
        imgs = [cv2.imread(str(f))[..., ::-1] for f in sorted((d / "camera").glob("frame_*.jpg"))]
        return [i for i in imgs if i is not None], json.loads(meta.read_text())["intrinsics"]
    cj = d / "camera" / "calibration" / "camera.json"
    svo = d / "camera" / "camera.svo2"
    if not (cj.is_file() and svo.is_file()):
        return [], None
    left = json.loads(cj.read_text())["rectified"]["left"]
    intr = {k: float(left[k]) for k in ("fx", "cx", "cy")}
    intr.update(w=int(left["width"]), h=int(left["height"]))
    try:
        import pyzed.sl as sl
    except ImportError:
        return [], None
    from perception.core.session import Session
    sess = Session(d)
    tl, ct, ci = sess.radar, sess.camera_t, sess.camera_index
    mid = 0.5 * (tl.t[0] + tl.t[-1])
    want = [int(ci[np.argmin(np.abs(ct - (mid + f * secs / 2)))]) for f in (-0.8, 0.0, 0.8)]
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.optional_settings_path = str(d / "camera" / "calibration")
    init.sdk_verbose = 0
    cam = sl.Camera()
    if cam.open(init) != sl.ERROR_CODE.SUCCESS:
        return [], None
    rt, mat, imgs = sl.RuntimeParameters(), sl.Mat(), []
    try:
        for k in want:
            cam.set_svo_position(k)
            if cam.grab(rt) == sl.ERROR_CODE.SUCCESS:
                cam.retrieve_image(mat, sl.VIEW.LEFT)
                imgs.append(mat.get_data()[..., 2::-1].copy())
    finally:
        cam.close()
    return imgs, intr


def camera_reflector(imgs: list, bg_imgs: list, intr: dict) -> dict:
    """The reflector in the camera: pixels changed from EVERY background image in EVERY image of the
    position (RGB difference > 60), below the horizon (the reflector stands lower than the camera;
    above it, ceiling lights and glass reflections flicker); the largest changed blob is the tripod,
    and its top 40 px the reflector. Azimuth (+ left) = atan((cx - u) / fx): no depth needed."""
    from scipy import ndimage
    if not imgs or not bg_imgs or intr is None or imgs[0].shape != bg_imgs[0].shape:
        return {"found": False}
    a = np.stack(imgs).astype(np.int16)
    b = np.stack(bg_imgs).astype(np.int16)
    diff = np.abs(a[:, None] - b[None]).sum(-1).min(axis=(0, 1))
    m = diff > 60
    m[: int(0.85 * intr["cy"])] = False
    m = ndimage.binary_opening(m, np.ones((3, 3)))
    lab, n = ndimage.label(ndimage.binary_dilation(m, np.ones((9, 9))))
    if n == 0:
        return {"found": False}
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    vv, uu = np.nonzero(m & (lab == 1 + int(np.argmax(sizes))))
    top = vv < vv.min() + 40 * intr["h"] / 1080.0
    u, v = float(np.median(uu[top])), float(np.median(vv[top]))
    return {"found": True, "u": u, "v": v, "az_deg": float(np.degrees(np.arctan2(intr["cx"] - u, intr["fx"]))),
            "blob_px": int(sizes.max())}


class RadarChange:
    """Radar static maps of the CHANGE from the background: each frame's zero-Doppler band minus the
    background's mean band, subtracted coherently (complex, per channel) and then beamformed
    (RadarDetector.static_band / static_map_of). The reference ("rise") is the background's own
    frame-to-frame residual. One detector per firmware variant."""

    def __init__(self):
        self.det, self.bg_band, self.ref = {}, None, None

    def detector(self, variant: str):
        import radar_firmware
        from perception.radar import RadarDetector, rx_phase_for
        if variant not in self.det:
            self.det[variant] = RadarDetector(radar_firmware.cfg_for_variant(variant), rx_phase_deg=rx_phase_for(variant))
        return self.det[variant]

    def set_background(self, raws: list, variant: str) -> None:
        det = self.detector(variant)
        bands = [det.static_band(r) for r in raws]
        self.bg_band = sum(bands) / len(bands)
        self.ref = rf.mean_db([det.static_map_of(b - self.bg_band) for b in bands])
        self.axes = (det.range_axis, np.asarray(det.static_az_deg, float))

    def maps(self, raws: list, variant: str) -> list:
        det = self.detector(variant)
        self.bands = [det.static_band(r) - self.bg_band for r in raws]
        self.variant = variant
        return [det.static_map_of(b) for b in self.bands]

    def refine(self, rng: float, az: float) -> tuple:
        """The reflector's coherent azimuth across both radar chips (RadarDetector.refine_azimuth) from
        the last maps() call's change bands: (az_deg, refined)."""
        return self.detector(self.variant).refine_azimuth(self.bands, rng, az)


# ── per position ─────────────────────────────────────────────────────────────
def analyse(pos: dict, bg_vox, bg_map, maps, r_axis, az_deg, r_lo, r_hi, z_lo=-3.0, z_hi=3.0, bg_cam=None,
            refine=None) -> tuple:
    lid = (rf.lidar_reflector(pos["lidar"], bg_vox, r_lo, r_hi, z_lo=z_lo, z_hi=z_hi)
           if pos["lidar"] and bg_vox is not None else {"found": False})
    rad = rf.radar_reflector(maps, bg_map, r_axis, az_deg, r_lo, r_hi) if maps and bg_map is not None else {"found": False}
    if rad["found"] and refine is not None:
        rf.refine_radar(rad, refine)
    row = {"name": pos["name"], "lidar_found": bool(lid["found"]), "radar_found": bool(rad["found"]),
           "lidar_sweeps": len(pos["lidar"]), "radar_frames": len(maps), "extrinsic": describe(pos["ext"])}
    if lid["found"]:
        v = to_vehicle_xy([lid["x"], lid["y"]], pos["ext"])[0]
        row.update(lidar_own_x=lid["x"], lidar_own_y=lid["y"], lidar_z=lid["z"], lidar_own_az=lid["az_deg"],
                   lidar_x=float(v[0]), lidar_y=float(v[1]), lidar_range=float(np.hypot(*v)),
                   lidar_az=float(np.degrees(np.arctan2(v[1], v[0]))), lidar_sd_m=lid["sd_xy"],
                   lidar_pts_per_sweep=lid["n_per_sweep"], lidar_present=lid["present"])
    if rad["found"]:
        row.update(radar_x=rad["x"], radar_y=rad["y"], radar_range=rad["range"], radar_az=rad["az_deg"],
                   radar_power_db=rad["power_db"], radar_rise_db=rad["rise_db"], radar_present=rad["present"],
                   radar_sd_az=rad["sd_az_deg"], radar_az_incoherent=rad.get("az_deg_incoherent"),
                   radar_az_refined=bool(rad.get("az_refined", False)))
    if lid["found"] and rad["found"]:
        row.update(radar_minus_lidar_range=rad["range"] - row["lidar_range"],
                   radar_minus_lidar_az=rad["az_deg"] - row["lidar_az"])
    cam = camera_reflector(pos.get("camera") or [], bg_cam or [], pos.get("intrinsics"))
    row["camera_found"] = bool(cam["found"])
    if cam["found"]:
        row.update(camera_u=cam["u"], camera_v=cam["v"], camera_az=cam["az_deg"])
        if lid["found"]:
            row["camera_minus_lidar_own_az"] = cam["az_deg"] - row["lidar_own_az"]
        if rad["found"]:
            row["radar_minus_camera_az"] = rad["az_deg"] - cam["az_deg"]
    return row, lid, rad


def save_points(path: Path, pos: dict, lid: dict, rad: dict) -> None:
    """Per radar frame, the paired sweep (nearest in time) and both sensors' reflector points."""
    rt, lt = np.asarray(pos["radar_t"]), np.asarray(pos["lidar_t"])
    pair = (np.abs(lt[None, :] - rt[:, None]).argmin(axis=1) if len(lt) and len(rt) else np.zeros(len(rt), int))
    lp = lid.get("points") or []
    lrows, lidx, rrows, ridx = [], [], [], []
    for k in range(len(rt)):
        if lp and len(lt):
            q = lp[pair[k]]
            if len(q):
                v = q.copy()
                v[:, :2] = to_vehicle_xy(q[:, :2], pos["ext"])
                lrows.append(np.c_[q[:, :5], v[:, :3]])
                lidx.append(np.full(len(q), k))
        rp = (rad.get("points") or [])
        if k < len(rp) and len(rp[k]):
            rrows.append(rp[k])
            ridx.append(np.full(len(rp[k]), k))
    cat = lambda a, n: np.concatenate(a).astype(np.float32) if a else np.zeros((0, n), np.float32)
    np.savez_compressed(path, radar_t=rt, lidar_t=lt[pair] if len(lt) else np.zeros(len(rt)),
                        lidar_points=cat(lrows, 8), lidar_frame=cat(lidx, 1).astype(np.int32).ravel(),
                        radar_points=cat(rrows, 6), radar_frame=cat(ridx, 1).astype(np.int32).ravel(),
                        lidar_columns=np.array("x y z range intensity (LiDAR frame) | x y z (vehicle frame)"),
                        radar_columns=np.array("x y range az_deg(+left) power_db rise_db (vehicle frame)"))


def robust(names: list, d) -> dict:
    """Median, then mean +- sd of the values within 2 deg of it; the rest are listed as outliers
    (a camera blob that is not the tripod, a radar lobe that is not the reflector)."""
    d = np.asarray(d, float)
    med = float(np.median(d))
    ok = np.abs(d - med) <= 2.0
    return {"mean": float(d[ok].mean()), "sd": float(d[ok].std()), "median": med, "n": int(ok.sum()),
            "outliers": {n: round(float(v), 2) for n, v, k in zip(names, d, ok) if not k},
            "per_position": {n: round(float(v), 2) for n, v in zip(names, d)}}


def fits(rows: list) -> dict:
    out = {}
    cam = [r for r in rows if "camera_minus_lidar_own_az" in r]
    if cam:
        out["lidar_yaw_from_camera_deg"] = robust([r["name"] for r in cam], [r["camera_minus_lidar_own_az"] for r in cam])
    rc = [r for r in rows if "radar_minus_camera_az" in r]
    if rc:
        out["radar_minus_camera_az_deg"] = robust([r["name"] for r in rc], [r["radar_minus_camera_az"] for r in rc])
    both = [r for r in rows if r["lidar_found"] and r["radar_found"]]
    out["n_positions"] = len(both)
    if len(both) >= 2:
        src = np.array([[r["lidar_own_x"], r["lidar_own_y"]] for r in both])
        dst = np.array([[r["radar_x"], r["radar_y"]] for r in both])
        f = rf.fit_rigid_2d(src, dst)
        f["residual_m"] = {r["name"]: round(e, 3) for r, e in zip(both, f["residual_m"])}
        f["warning"] = ("least squares over every position: one wrong LiDAR or radar pick (see residual_m) spoils "
                        "it; compare with lidar_yaw_from_camera_deg")
        out["lidar_to_radar_all"] = f
    bore = [r for r in both if abs(r["radar_az"]) <= 5.0]
    if bore:
        out["yaw_from_boresight_positions_deg"] = robust([r["name"] for r in bore],
                                                         [r["radar_az"] - r["lidar_own_az"] for r in bore])
    return out


def figure(path: Path, rows: list, results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 9))
    for r in rows:
        lid, rad, ext = results[r["name"]]
        if lid.get("found"):
            q = np.concatenate([p for p in lid["points"] if len(p)])
            v = to_vehicle_xy(q[:, :2], ext)
            ax.scatter(v[:, 1], v[:, 0], s=2, c=LIDAR_COLOR, alpha=0.5)
            ax.annotate(r["name"], (r["lidar_y"], r["lidar_x"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
        if rad.get("found"):
            m = rad["mean_points"]
            ax.scatter(m[:, 1], m[:, 0], s=10, c=RADAR_COLOR, marker="s", alpha=0.6)
            ax.plot(rad["y"], rad["x"], "+", c=RADAR_COLOR, ms=12, mew=2)
    ax.scatter([], [], s=8, c=LIDAR_COLOR, label="LiDAR reflector points (all sweeps)")
    ax.scatter([], [], s=12, c=RADAR_COLOR, marker="s", label="radar reflector cells (+ = interpolated peak)")
    ax.plot(0, 0, "k^", ms=8)
    ax.invert_xaxis()
    ax.set_xlabel("y (m, + left)")
    ax.set_ylabel("x (m, forward)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Reflector points only (change from the background), vehicle frame")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Corner-reflector points (change from a background) in LiDAR and radar")
    ap.add_argument("--set", type=Path, help="a live-view calibration set: background/ plus one folder per position")
    ap.add_argument("--background", type=Path, help="the empty-scene capture or session")
    ap.add_argument("--positions", type=Path, nargs="*", default=[], help="reflector positions (captures or sessions)")
    ap.add_argument("--range", type=float, nargs=2, default=(1.0, 15.0), metavar=("FROM", "TO"),
                    help="search range (m) for the reflector")
    ap.add_argument("--z", type=float, nargs=2, default=(-0.9, 2.5), metavar=("FROM", "TO"),
                    help="LiDAR: height window (m, LiDAR frame) for the reflector; the default leaves out the "
                         "floor of the 2026-11 rig (~1.1 m below the LiDAR)")
    ap.add_argument("--secs", type=float, default=8.0, help="recorded sessions: seconds used, centred on the radar")
    ap.add_argument("--out", type=Path, help="output folder")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--no-camera", action="store_true", help="skip the camera (no pyzed, or faster)")
    a = ap.parse_args()
    if a.set:
        bg = a.set / "background"
        positions = sorted(p for p in a.set.iterdir() if p.is_dir() and p.name != "background" and is_live_capture(p))
    else:
        bg, positions = a.background, list(a.positions)
    if bg is None or not positions:
        ap.error("give --set, or --background and --positions")
    out = a.out or (a.set if a.set else positions[0].parent / "reflector_calibration")
    out.mkdir(parents=True, exist_ok=True)
    r_lo, r_hi = a.range
    tic = time.monotonic()
    change = RadarChange()
    b = read_capture(bg, a.secs, not a.no_camera)
    if not b["lidar"] and not b["radar_raw"]:
        raise SystemExit(f"{bg}: no LiDAR sweeps and no radar frames")
    bg_vox = rf.background_voxels(b["lidar"]) if b["lidar"] else None
    bg_map, r_axis, az_deg = None, None, None
    if b["radar_raw"]:
        change.set_background(b["radar_raw"], b["variant"])
        bg_map, (r_axis, az_deg) = change.ref, change.axes
    print(f"background {bg.name}: {len(b['lidar'])} sweeps, {len(b['radar_raw'])} radar frames, "
          f"{len(b['camera'])} camera images")
    rows, results = [], {}
    for p in positions:
        pos = read_capture(p, a.secs, not a.no_camera)
        maps = change.maps(pos["radar_raw"], pos["variant"]) if pos["radar_raw"] and bg_map is not None else []
        row, lid, rad = analyse(pos, bg_vox, bg_map, maps, r_axis, az_deg, r_lo, r_hi, *a.z, bg_cam=b["camera"],
                                refine=change.refine if maps else None)
        rows.append(row)
        results[row["name"]] = (lid, rad, pos["ext"])
        save_points(out / f"{row['name']}_reflector.npz", pos, lid, rad)
        L = (f"LiDAR ({row['lidar_x']:+.2f}, {row['lidar_y']:+.2f}) z {row['lidar_z']:+.2f}, "
             f"{row['lidar_pts_per_sweep']:.0f} pts/sweep in {100 * row['lidar_present']:.0f}% of sweeps"
             if row["lidar_found"] else "LiDAR not found")
        R = (f"radar r {row['radar_range']:.2f} az {row['radar_az']:+.1f} (+{row['radar_rise_db']:.0f} dB, "
             f"{100 * row['radar_present']:.0f}% of frames)" if row["radar_found"] else "radar not found")
        Cm = f"camera az {row['camera_az']:+.1f}" if row["camera_found"] else "camera -"
        print(f"  {row['name']:18s} {L} | {R} | {Cm}")
    summary = {"made": time.strftime("%Y-%m-%dT%H:%M:%S"), "background": str(bg), "range_m": [r_lo, r_hi],
               "positions": rows, "fit": fits(rows),
               "note": "LiDAR own frame -> radar. Radar: measured array layout (calibration/radar_array.json, "
                       "2026-11-07), azimuth refined coherently across both chips. Provisional: the extrinsic "
                       "calibration is the team's."}
    (out / "reflector_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    keys = sorted({k for r in rows for k in r})
    with open(out / "reflector_pairs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name"] + [k for k in keys if k != "name"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    f = summary["fit"]
    if "lidar_yaw_from_camera_deg" in f:
        g = f["lidar_yaw_from_camera_deg"]
        print(f"LiDAR yaw from the camera ({g['n']} positions): {g['mean']:+.2f} +- {g['sd']:.2f} deg "
              f"(camera azimuth - LiDAR own-frame azimuth); outliers {g['outliers'] or 'none'}")
    if "radar_minus_camera_az_deg" in f:
        g = f["radar_minus_camera_az_deg"]
        print(f"radar - camera azimuth: {g['mean']:+.2f} +- {g['sd']:.2f} deg over {g['n']} positions; "
              f"outliers (off-axis lobes) {g['outliers'] or 'none'}")
    if "lidar_to_radar_all" in f:
        g = f["lidar_to_radar_all"]
        print(f"fit LiDAR -> radar over {f['n_positions']} positions: yaw {g['yaw_deg']:+.2f} deg, "
              f"t ({g['tx_m']:+.2f}, {g['ty_m']:+.2f}) m, residual {g['mean_m']:.2f} m mean, {g['max_m']:.2f} max")
    if "yaw_from_boresight_positions_deg" in f:
        g = f["yaw_from_boresight_positions_deg"]
        print(f"LiDAR yaw from the radar, {g['n']} positions within 5 deg of its boresight: "
              f"{g['mean']:+.2f} +- {g['sd']:.2f} deg; outliers {g['outliers'] or 'none'}")
    if not a.no_figure:
        figure(out / "reflector_points.png", rows, results)
    print(f"-> {out} ({time.monotonic() - tic:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
