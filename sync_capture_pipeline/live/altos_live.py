#!/usr/bin/env python3
"""
live/altos_live.py - the Altos V4 radar's live view: the newest frames as a top view (forward up,
vehicle-left on the left, the shared bird's-eye extent of common.bev_extent), plus the strongest
points, a few times a second. Nothing is written to disk, except a calibration capture on the
page's request (below).

INPUT
    live            the radar's UDP multicast (224.1.2.4:4040 on 192.168.3.1 by default), frames
                    put together by altos_standalone_capture/altos_v4.py. It can run while
                    nothing else listens; live view never runs during a recording.
    --from-session  a recorded session's altos/altos.pcap played at its recorded pace (looping).

VIEW
    The last TRAIL_FRAMES frames (the radar's point cloud is sparse) in the Altos's OWN frame (x
    forward, y left; its mounting on the rig is not calibrated). Still points grey; moving points
    (|Doppler| >= MOVING_V) blue when approaching, red when receding.

STRONGEST POINTS
    The newest frame's points binned in 0.5 m cells, scored by their summed linear SNR; the
    strongest cells (1.5 m apart) are sent as [x, y, range, azimuth (deg, + left), SNR dB of the
    strongest point, n points], strongest first. The page picks the strongest inside its range gate.

CALIBRATION CAPTURE (common.Capture)
    "Save background" / "Save position" write the next CAPTURE_FRAMES frames (2 s) to <dir>/altos/:
    frames.npz (points, counts per frame, t = CLOCK_MONOTONIC at receipt), meta.json; a position
    also reflector.json: the strongest cell inside the page's range gate that the set's saved
    background (0.3 m cells) did not have, if there is one, else the strongest cell.
"""
from __future__ import annotations

import argparse
import collections
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

from common import ROOT, ASPECT, Capture, Link, Pacer, Rate, add_args, bev_extent, jpeg, json_dump, log, rounded

ALTOS_DIR = ROOT / "altos_standalone_capture"
sys.path.insert(0, str(ALTOS_DIR))
import altos_v4 as av  # noqa: E402

PX = 600                       # image width (pixels across)
TRAIL_FRAMES = 5               # frames drawn together (1/3 s)
MOVING_V = 0.3                 # m/s
N_STRONG = 8
CAPTURE_FRAMES = 30            # frames per calibration capture (2 s)
BG_CELL = 0.3                  # background occupancy cell (m)


class Source(threading.Thread):
    """Frames (decoded points, receipt time) into a deque, from the radar or a recording."""

    def __init__(self, a, stop: threading.Event):
        super().__init__(name="altos-src", daemon=True)
        self.a, self.stop = a, stop
        self.frames = collections.deque(maxlen=64)
        self.lock = threading.Lock()
        self.n = 0
        self.asm = av.FrameAssembler()
        self.error = None

    def _put(self, f):
        pts, ok = av.points_array(f.payload())
        with self.lock:
            self.n += 1
            self.frames.append((self.n, f.t_last_ns / 1e9, pts[ok]))

    def run(self):
        try:
            if self.a.from_session:
                self._play(Path(self.a.from_session) / "altos" / "altos.pcap")
            else:
                self._live()
        except Exception as exc:                      # shown on the page
            self.error = str(exc)
            self.stop.set()

    def _live(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.a.group, self.a.port))
        try:
            s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                         socket.inet_aton(self.a.group) + socket.inet_aton(self.a.host_ip))
        except OSError as exc:
            raise RuntimeError(f"cannot join {self.a.group}:{self.a.port} on {self.a.host_ip} ({exc}): is "
                               f"{self.a.host_ip} on the Altos's USB-Ethernet adapter?")
        s.settimeout(0.2)
        log(f"altos: listening on {self.a.group}:{self.a.port} via {self.a.host_ip}")
        while not self.stop.is_set():
            try:
                pkt = s.recv(65535)
            except socket.timeout:
                continue
            for f in self.asm.feed(pkt, time.monotonic_ns()):
                self._put(f)
        s.close()

    def _play(self, pcap: Path):
        log(f"altos: playing {pcap}")
        while not self.stop.is_set():
            frames = list(av.iter_frames(pcap))
            if not frames:
                raise RuntimeError(f"no frames in {pcap}")
            t0_rec, t0 = frames[0].t_last_ns, time.monotonic_ns()
            for f in frames:
                due = t0 + (f.t_last_ns - t0_rec)
                while not self.stop.is_set() and time.monotonic_ns() < due:
                    self.stop.wait(min(0.05, (due - time.monotonic_ns()) / 1e9))
                if self.stop.is_set():
                    return
                f.t_last_ns = time.monotonic_ns()      # "received" now, as live
                self._put(f)

    def newest(self, k: int):
        with self.lock:
            return list(self.frames)[-k:]


def strongest_cells(p: np.ndarray, extent=None, near_m: float = 0.3) -> list:
    """[x, y, range, azimuth deg (+ left), SNR dB of the cell's strongest point, n points], strongest first."""
    if not len(p):
        return []
    x, y, r, snr = p[:, 5], p[:, 6], p[:, 0], p[:, 4]
    ok = r > near_m
    if extent is not None:
        x0, x1, y0, y1 = extent
        ok &= (x > x0) & (x < x1) & (y > y0) & (y < y1)
    if not ok.any():
        return []
    x, y, snr = x[ok], y[ok], snr[ok]
    cell = np.floor(x / 0.5).astype(np.int64) * 100000 + np.floor((y + 500) / 0.5).astype(np.int64)
    keys, inv = np.unique(cell, return_inverse=True)
    score = np.bincount(inv, weights=snr)
    out = []
    for k in np.argsort(-score):
        m = inv == k
        w = snr[m] / snr[m].sum()
        cx, cy = float(w @ x[m]), float(w @ y[m])
        if any(np.hypot(cx - o[0], cy - o[1]) < 1.5 for o in out):
            continue
        out.append([cx, cy, float(np.hypot(cx, cy)), float(np.degrees(np.arctan2(cy, cx))),
                    float(10 * np.log10(max(float(snr[m].max()), 1e-9))), int(m.sum())])
        if len(out) >= N_STRONG:
            break
    return out


def render(frames: list, extent) -> np.ndarray:
    """RGB top view: rows = forward (far at the top), columns = y (left at the left)."""
    x0, x1, y0, y1 = extent
    W = PX
    H = int(round(PX * (x1 - x0) / (y1 - y0)))
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (18, 20, 26)
    n = len(frames)
    for k, (_n, _t, p) in enumerate(frames):
        if not len(p):
            continue
        x, y, v = p[:, 5], p[:, 6], p[:, 1]
        col = np.tile(np.array([150, 155, 165], np.float32), (len(p), 1))
        col[v <= -MOVING_V] = (80, 150, 255)          # approaching
        col[v >= MOVING_V] = (255, 90, 70)            # receding
        fade = 0.35 + 0.65 * (k + 1) / n              # older frames fainter
        ci = ((y1 - y) / (y1 - y0) * W).astype(int)
        ri = ((x1 - x) / (x1 - x0) * H).astype(int)
        ok = (ci >= 1) & (ci < W - 1) & (ri >= 1) & (ri < H - 1)
        c8 = (col[ok] * fade).astype(np.uint8)
        for dr in (-1, 0, 1):                          # 3x3 pixel dots
            for dc in (-1, 0, 1):
                img[ri[ok] + dr, ci[ok] + dc] = c8
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description="Altos V4 live view (no saving)")
    add_args(ap)
    ap.add_argument("--radar-ip", default=av.DEFAULT_RADAR_IP)
    ap.add_argument("--host-ip", default=av.DEFAULT_HOST_IP)
    ap.add_argument("--group", default=av.DEFAULT_GROUP)
    ap.add_argument("--port", type=int, default=av.DEFAULT_PORT)
    a = ap.parse_args()
    link = Link("altos", a.max_secs, {"range_m": a.range_m})
    src = Source(a, link.stop)
    src.start()
    cap = Capture("altos", CAPTURE_FRAMES)
    saved_bg = {}

    def background_cells(spec):
        f = Path(spec["dir"]).parent / "background" / "altos" / "frames.npz"
        if not f.is_file():
            return None
        if str(f) not in saved_bg:
            q = np.load(f)["points"]
            saved_bg[str(f)] = set(map(tuple, np.floor(q[:, 5:7] / BG_CELL).astype(np.int64).tolist()))
        return saved_bg[str(f)]

    def finish_capture(items, spec):
        pts = [p for _, _, p in items]
        allp = np.concatenate(pts) if pts else np.zeros((0, 8), np.float32)
        meta = {"frames": len(items), "frame": "the Altos's own frame (x forward, y left, z up); mounting not applied",
                "columns": "range doppler azimuth elevation snr x y z", "kind": spec.get("kind"),
                "source": str(a.from_session) if a.from_session else "live",
                "time_local": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        result = {}
        if spec.get("kind") == "position":
            lo, hi = float(spec.get("r_lo", 1.0)), float(spec.get("r_hi", 15.0))
            gate = allp[(allp[:, 0] >= lo) & (allp[:, 0] <= hi)] if len(allp) else allp
            bg = background_cells(spec)
            where = "saved" if bg is not None else None
            if bg is not None and len(gate):
                cells = np.floor(gate[:, 5:7] / BG_CELL).astype(np.int64)
                new = np.array([tuple(c) not in bg for c in cells.tolist()], bool)
                gate = gate[new]
            cand = strongest_cells(gate)
            refl = {"found": bool(cand), "background": where}
            if cand:
                c = cand[0]
                near = np.hypot(gate[:, 5] - c[0], gate[:, 6] - c[1]) <= 0.5
                refl.update(x=c[0], y=c[1], z=float(np.median(gate[near, 7])), range=c[2], az_deg=c[3],
                            snr_db=c[4], n_per_frame=float(near.sum()) / max(1, len(items)))
            result["reflector"] = refl
            meta["reflector"] = refl

        def write():
            d = cap.out_dir()
            np.savez_compressed(d / "frames.npz", points=allp.astype(np.float32),
                                counts=np.array([len(p) for p in pts]), t=np.array([t for _, t, _ in items]))
            if "reflector" in meta:
                json_dump(d / "reflector.json", meta["reflector"])
            json_dump(d / "meta.json", meta)          # last: the capture is complete
            cap.saved(**result)
        cap.in_thread(write)

    rate, pacer = Rate(), Pacer(a.hz, link.stop)
    fps = Rate(3.0)
    seen, last_new, t_start = 0, time.monotonic(), time.monotonic()
    try:
        while link.running():
            if src.error:
                link.error(src.error)
                return 1
            frames = src.newest(TRAIL_FRAMES)
            newest_n = frames[-1][0] if frames else 0
            if newest_n == seen:
                if time.monotonic() - last_new > 3 and time.monotonic() - t_start > 3:
                    link.send({"waiting": True, "frames": src.n})
                    last_new = time.monotonic()
                link.stop.wait(0.02)
                continue
            new = [f for f in src.newest(64) if f[0] > seen]
            seen, last_new = newest_n, time.monotonic()
            for f in new:
                fps.tick()
            cap.follow(link.control.get("capture"))
            if cap.collecting:
                for f in new:
                    done = cap.add(f)
                    if done is not None:
                        finish_capture(done, cap.spec)
                        break
            tic = time.perf_counter()
            R = float(link.control.get("range_m", 20.0))
            ext = bev_extent(R)
            img = render(frames, ext)
            p = frames[-1][2]
            rate.tick()
            link.send({"t": frames[-1][1], "extent": ext, "ms": round((time.perf_counter() - tic) * 1e3, 1),
                       "hz": round(rate.hz, 1), "fps": round(fps.hz, 1), "frames": src.n, "points": int(len(p)),
                       "moving": int((np.abs(p[:, 1]) >= MOVING_V).sum()) if len(p) else 0,
                       "lost": src.asm.missing_ids, "incomplete": src.asm.incomplete, "restarts": src.asm.restarts,
                       # [x, y, range, azimuth (deg, + left), SNR dB, n points], strongest first
                       "strong": rounded(strongest_cells(p, ext)), "capture": cap.state(),
                       "frame": "own (mounting not applied)"},
                      jpeg(img, 80))
            pacer.wait()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        link.error(str(exc))
        return 1
    finally:
        link.stop.set()
        log("altos: live view stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
