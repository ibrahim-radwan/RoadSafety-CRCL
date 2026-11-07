"""
A corner reflector found as what CHANGED from an empty-scene background, in the LiDAR and in the
radar, and the 2D rigid fit between two sensors' reflector positions. numpy only: the live-view
workers, the web server and calib_reflector.py share it.

The reflector is not the LiDAR's brightest return (people, hi-vis vests and near objects saturate at
255; a reflector gives 10-120), nor always the radar's strongest stationary cell (walls). Against a
background recorded without it, it is the largest persistent change.

LiDAR (points N x >= 5: x, y, z, range, intensity, ...; any frame, as long as the background is in
the same frame)
    background_voxels(sweeps)      voxels (VOX m) occupied in any background sweep, dilated by one
    new_points(pts, ref)           mask of the returns in voxels the background did not have
    lidar_reflector(sweeps, ref)   the reflector (lidar_reflector_from_new: from returns already
                                   found new): new returns of all sweeps in CELL m cells; the
                                   cell group with the most returns that is present in at least
                                   MIN_PRESENT of the sweeps wins (people move, the reflector does
                                   not). Its points are the new returns within GROUP_M of its centre.

Radar (static maps: [range, azimuth] dB, zero Doppler, no clutter removal, RadarDetector
process(..., static=True)["static_map"], columns at RadarDetector.static_az_deg, world + left)
    radar_background(maps)         median map of the background frames
    radar_reflector(maps, ref)     the frames' mean power minus the background: the largest local
                                   maximum >= RISE_DB above it (and >= ROW_DB above its range row's
                                   median, so a background null is not a change), with parabolic
                                   interpolation in range and azimuth. Its cells (the "points") are
                                   those within 3 dB of that peak, per frame and on the mean.

fit_rigid_2d(src, dst)             yaw (deg) and t with dst ~ R(yaw) src + t, least squares.
"""
from __future__ import annotations

import numpy as np

VOX = 0.15                 # LiDAR background voxel (m), dilated by one voxel
CELL = 0.3                 # LiDAR change cells (m)
GROUP_M = 0.45             # LiDAR reflector points: new returns this close (xy) to its centre
MIN_PRESENT = 0.3          # ... the reflector's cell has new returns in at least this share of sweeps
NEAR_M = 0.5               # LiDAR returns nearer than this are the rig itself
RISE_DB = 10.0             # radar: a change is at least this far above the background ...
ROW_DB = 6.0               # ... and this far above its range row's median
CELL_DB = 3.0              # radar reflector cells: within this of the peak


# ── LiDAR ───────────────────────────────────────────────────────────────────
def _vkeys(xyz: np.ndarray, vox: float = VOX) -> np.ndarray:
    v = np.floor(np.asarray(xyz, float) / vox).astype(np.int64) + 2048            # +-300 m at 0.15 m
    return (v[:, 0] << 24) | (v[:, 1] << 12) | v[:, 2]


def background_voxels(sweeps: list, vox: float = VOX) -> np.ndarray:
    """Sorted voxel keys occupied in any background sweep, dilated by one voxel."""
    k = np.unique(np.concatenate([_vkeys(p[p[:, 3] > 0.3, :3], vox) for p in sweeps]))
    x, y, z = k >> 24, (k >> 12) & 4095, k & 4095
    out = [((x + dx) << 24) | ((y + dy) << 12) | (z + dz)
           for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    return np.unique(np.concatenate(out))


def new_points(pts: np.ndarray, ref: np.ndarray, near_m: float = NEAR_M, vox: float = VOX) -> np.ndarray:
    """Boolean mask: returns (beyond near_m) in voxels the background `ref` did not have."""
    ok = pts[:, 3] > near_m
    out = np.zeros(len(pts), bool)
    if ref is None or not ok.any():
        return out                                   # no background: nothing counts as new
    if len(ref) == 0:
        return ok                                    # an empty background: everything is new
    k = _vkeys(pts[ok, :3], vox)
    i = np.clip(np.searchsorted(ref, k), 0, len(ref) - 1)
    out[np.flatnonzero(ok)] = ref[i] != k
    return out


def _in_window(xy: np.ndarray, r_lo: float, r_hi: float, fov_deg: float) -> np.ndarray:
    r = np.hypot(xy[:, 0], xy[:, 1])
    az = np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))
    return (r >= r_lo) & (r <= r_hi) & (np.abs(az) <= fov_deg)


def lidar_reflector(sweeps: list, ref: np.ndarray, r_lo: float = 1.0, r_hi: float = 15.0,
                    fov_deg: float = 70.0, z_lo: float = -3.0, z_hi: float = 3.0) -> dict:
    """The reflector in a position's sweeps (see the module docstring). Returns
    found, x, y, z (median of its points), range, az_deg (+ left), sd_xy (m, of the per-sweep
    centroids), n_per_sweep (median), present (share of sweeps with >= 1 point), n_sweeps, and
    points: list (per sweep) of the reflector's points (rows of the input). Not found: found False."""
    return lidar_reflector_from_new([p[new_points(p, ref)] for p in sweeps], r_lo, r_hi, fov_deg, z_lo, z_hi)


def lidar_reflector_from_new(changed: list, r_lo: float = 1.0, r_hi: float = 15.0, fov_deg: float = 70.0,
                             z_lo: float = -3.0, z_hi: float = 3.0) -> dict:
    """lidar_reflector from each sweep's returns already found new against the background (e.g. the
    replay's background mask): the same window, cells, persistence rule and result."""
    news = [q[_in_window(q[:, :2], r_lo, r_hi, fov_deg) & (q[:, 2] >= z_lo) & (q[:, 2] <= z_hi)] for q in changed]
    sweeps = changed
    n = len(sweeps)
    res = {"found": False, "n_sweeps": n, "points": [np.zeros((0, sweeps[0].shape[1] if n else 6), np.float32)] * n}
    allq = [q for q in news if len(q)]
    if not allq:
        return res
    pool = np.concatenate(allq)
    sid = np.concatenate([np.full(len(q), i) for i, q in enumerate(news) if len(q)])
    g = np.floor(pool[:, :2] / CELL).astype(np.int64)
    cells, inv, cnt = np.unique(g, axis=0, return_inverse=True, return_counts=True)
    present = np.array([len(np.unique(sid[inv == c])) for c in range(len(cells))]) / max(n, 1)
    score = np.where(present >= MIN_PRESENT, cnt, 0)
    if score.max() <= 0:
        return res
    c = int(np.argmax(score))
    cxy = pool[inv == c, :2].mean(axis=0)
    near = np.hypot(pool[:, 0] - cxy[0], pool[:, 1] - cxy[1]) < GROUP_M
    cxy = np.median(pool[near, :2], axis=0)                          # re-centre on the whole group
    pts = [q[np.hypot(q[:, 0] - cxy[0], q[:, 1] - cxy[1]) < GROUP_M] for q in news]
    allp = np.concatenate([q for q in pts if len(q)])
    x, y, z = (float(v) for v in np.median(allp[:, :3], axis=0))
    cents = np.array([q[:, :2].mean(axis=0) for q in pts if len(q)])
    res.update(found=True, x=x, y=y, z=z, range=float(np.hypot(x, y)), az_deg=float(np.degrees(np.arctan2(y, x))),
               sd_xy=float(np.sqrt(np.mean(np.sum((cents - cents.mean(axis=0)) ** 2, axis=1)))) if len(cents) > 1 else 0.0,
               n_per_sweep=float(np.median([len(q) for q in pts])), present=float(np.mean([len(q) > 0 for q in pts])),
               points=pts)
    return res


# ── radar ───────────────────────────────────────────────────────────────────
def radar_background(maps: list) -> np.ndarray:
    """The background's static map: median over its frames (dB)."""
    return np.median(np.stack(maps), axis=0)


def mean_db(maps: list) -> np.ndarray:
    """The mean POWER of dB maps, in dB."""
    s = np.stack(maps).astype(np.float64)
    top = s.max(axis=0)
    return top + 10 * np.log10(np.mean(10 ** ((s - top) / 10), axis=0))        # mean power, stable


def _parabola(vm: float, v0: float, vp: float) -> float:
    d = vm - 2 * v0 + vp
    return 0.0 if d >= 0 else float(np.clip(0.5 * (vm - vp) / d, -0.5, 0.5))


def changed_peaks(m: np.ndarray, ref: np.ndarray, r_axis: np.ndarray, r_lo: float, r_hi: float,
                  az_deg: np.ndarray = None, fov_deg: float = 70.0, n: int = 8) -> list:
    """Local maxima of (m - ref) at least RISE_DB above the background and ROW_DB above the range
    row's median, inside [r_lo, r_hi] (and |azimuth| <= fov_deg if az_deg is given): (ri, ai) cell
    indices, largest rise first."""
    d = m - ref
    ok = (r_axis[:, None] >= r_lo) & (r_axis[:, None] <= r_hi)
    if az_deg is not None:
        ok = ok & (np.abs(np.asarray(az_deg))[None, :] <= fov_deg)
    ok = ok & (d >= RISE_DB) & (m >= np.median(m, axis=1, keepdims=True) + ROW_DB)
    pad = np.pad(d, 1, mode="constant", constant_values=-np.inf)
    h, w = d.shape
    peak = np.ones_like(ok)
    for di in (0, 1, 2):
        for dj in (0, 1, 2):
            if di != 1 or dj != 1:
                peak &= d >= pad[di:di + h, dj:dj + w]
    ri, ai = np.nonzero(ok & peak)
    order = np.argsort(-d[ri, ai])[:n]
    return list(zip(ri[order].tolist(), ai[order].tolist()))


def _cell_xy(r_axis, az_deg, ri, ai):
    r, az = np.asarray(r_axis)[ri], np.radians(np.asarray(az_deg)[ai])
    return np.c_[r * np.cos(az), r * np.sin(az)]


def radar_reflector(maps: list, ref: np.ndarray, r_axis: np.ndarray, az_deg: np.ndarray,
                    r_lo: float = 1.0, r_hi: float = 15.0, fov_deg: float = 70.0) -> dict:
    """The reflector in a position's static maps (see the module docstring). Returns found, range,
    az_deg (+ left), x, y (interpolated peak of the mean map), power_db, rise_db, present (share of
    frames whose own peak is within one cell of it), sd_az_deg (of the per-frame peaks), n_frames,
    and points: per frame, rows [x, y, range, az_deg, power_db, rise_db] of its cells within
    CELL_DB of that frame's peak, plus mean_points (the same on the mean map)."""
    r_axis, az_deg = np.asarray(r_axis, float), np.asarray(az_deg, float)
    res = {"found": False, "n_frames": len(maps), "points": [np.zeros((0, 6), np.float32)] * len(maps),
           "mean_points": np.zeros((0, 6), np.float32)}
    if not maps:
        return res
    mean = mean_db(maps)
    pk = changed_peaks(mean, ref, r_axis, r_lo, r_hi, az_deg, fov_deg, n=1)
    if not pk:
        return res
    ri, ai = pk[0]
    d = mean - ref
    h, w = d.shape
    fr = _parabola(d[ri - 1, ai], d[ri, ai], d[ri + 1, ai]) if 0 < ri < h - 1 else 0.0
    fa = _parabola(d[ri, ai - 1], d[ri, ai], d[ri, ai + 1]) if 0 < ai < w - 1 else 0.0
    rng = float(np.interp(ri + fr, np.arange(h), r_axis))
    az = float(np.interp(ai + fa, np.arange(w), az_deg))

    def cells(mm, i0, j0):
        dd = mm - ref
        lo_i, hi_i = max(i0 - 2, 0), min(i0 + 3, h)
        lo_j, hi_j = max(j0 - 3, 0), min(j0 + 4, w)
        sub = dd[lo_i:hi_i, lo_j:hi_j]
        ii, jj = np.nonzero(sub >= sub.max() - CELL_DB)
        ii, jj = ii + lo_i, jj + lo_j
        xy = _cell_xy(r_axis, az_deg, ii, jj)
        return np.c_[xy, r_axis[ii], az_deg[jj], mm[ii, jj], dd[ii, jj]].astype(np.float32)

    per, hits, azs = [], [], []
    for m in maps:
        dd = m - ref
        lo_i, hi_i = max(ri - 1, 0), min(ri + 2, h)
        lo_j, hi_j = max(ai - 2, 0), min(ai + 3, w)
        sub = dd[lo_i:hi_i, lo_j:hi_j]
        i0, j0 = np.unravel_index(int(np.argmax(sub)), sub.shape)
        i0, j0 = i0 + lo_i, j0 + lo_j
        ok = dd[i0, j0] >= RISE_DB and abs(i0 - ri) <= 1 and abs(j0 - ai) <= 1
        hits.append(ok)
        if ok:
            azs.append(float(az_deg[j0]))
        per.append(cells(m, i0, j0) if ok else np.zeros((0, 6), np.float32))
    res.update(found=True, range=rng, az_deg=az, x=rng * float(np.cos(np.radians(az))),
               y=rng * float(np.sin(np.radians(az))), power_db=float(mean[ri, ai]), rise_db=float(d[ri, ai]),
               present=float(np.mean(hits)), sd_az_deg=float(np.std(azs)) if len(azs) > 1 else 0.0,
               points=per, mean_points=cells(mean, ri, ai))
    return res


def refine_radar(res: dict, refine) -> dict:
    """Replace a found radar_reflector result's azimuth by refine(range, az_deg) -> (az_deg, refined)
    (RadarDetector.refine_azimuth: coherent across both chips) when it is refined; keeps the map's
    peak as az_deg_incoherent and sets az_refined. x, y follow."""
    az, ok = refine(res["range"], res["az_deg"])
    res["az_deg_incoherent"], res["az_refined"] = res["az_deg"], bool(ok)
    if ok:
        res["az_deg"] = float(az)
        res["x"], res["y"] = res["range"] * float(np.cos(np.radians(az))), res["range"] * float(np.sin(np.radians(az)))
    return res


# ── fit ─────────────────────────────────────────────────────────────────────
def fit_rigid_2d(src: np.ndarray, dst: np.ndarray, weights=None) -> dict:
    """dst ~ R(yaw) src + t (2D, least squares). Returns yaw_deg, tx_m, ty_m, residual_m (per point),
    mean_m, max_m. Needs >= 2 points."""
    src, dst = np.asarray(src, float)[:, :2], np.asarray(dst, float)[:, :2]
    w = np.ones(len(src)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    ms, md = w @ src, w @ dst
    a, b = src - ms, dst - md
    s = np.sum(w * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))
    c = np.sum(w * (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]))
    yaw = np.arctan2(s, c)
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    t = md - R @ ms
    res = np.hypot(*(dst - (src @ R.T + t)).T)
    return {"yaw_deg": float(np.degrees(yaw)), "tx_m": float(t[0]), "ty_m": float(t[1]),
            "residual_m": res.tolist(), "mean_m": float(res.mean()), "max_m": float(res.max())}
