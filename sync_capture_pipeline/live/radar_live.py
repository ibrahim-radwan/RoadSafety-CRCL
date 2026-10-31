#!/usr/bin/env python3
"""
live/radar_live.py - the radar's live view: newest frame -> our detector -> a range-azimuth fan
image plus the detections, a few times a second. Nothing is written to disk.

INPUT
    live            the DCA1000, armed with EXACTLY the command sequence and packet delay of
                    dca_capture_radar_sync.py (its own send_recv, constants and Receiver are
                    imported; nothing about the radar's network or DCA settings differs). The
                    receiver's packet loop is unchanged; only its disk writer is replaced by a
                    one-frame slot that the newest frame overwrites, so the receive thread never
                    waits on processing. STOP_RECORD is sent on every exit path.
    --from-session  a recorded session's frame files, at their recorded times (looping).

PROCESSING (perception/radar.py RadarDetector, the flashed firmware's .cfg, in-situ RX phase)
    moving detections (CFAR, TDM Doppler compensation, multipath ghosts dropped), and the
    stationary-reflector map and detections (zero Doppler, no clutter removal): a corner reflector
    is stationary, so the "strongest reflector" for extrinsic alignment comes from these.

The page asks for the map shown ({"radar_map": "static" | "moving"}) and the range
({"range_m": R}) over stdin.
"""
from __future__ import annotations

import argparse
import csv
import socket
import threading
import time
from pathlib import Path

import numpy as np

from common import (PIPELINE, Link, Pacer, Rate, add_args, bev_extent, jpeg, log, rounded)

import dca_capture_radar_sync as dca            # noqa: E402  (the recorder's receiver, unchanged)
import radar_firmware                           # noqa: E402

W, H = 600, 400                                  # image size (ASPECT 1.5)
N_STATIC = 40                                    # strongest static detections sent to the page


class LatestFrame:
    """Stands in for the receiver's FrameWriter (same start / submit / finish): keeps only the
    newest frame. submit() runs on the receive thread and never blocks."""

    def __init__(self):
        self.cv = threading.Condition()
        self.frame, self.n, self.t = None, 0, 0.0

    def start(self):
        pass

    def submit(self, fn, data) -> bool:
        with self.cv:
            self.frame, self.t = data, time.monotonic()
            self.n += 1
            self.cv.notify_all()
        return True

    def finish(self):
        pass

    def newest(self, after: int, timeout: float):
        """(n, frame bytes, t) of a frame newer than `after`, or None after timeout."""
        with self.cv:
            if self.n <= after:
                self.cv.wait(timeout)
            return (self.n, self.frame, self.t) if self.n > after else None


class LiveReceiver(dca.Receiver):
    """dca_capture_radar_sync.Receiver with its frames going to a LatestFrame instead of disk."""

    def __init__(self, sink: LatestFrame):
        super().__init__(outdir="<memory>")      # not None: the receiver assembles frames
        self.writer = sink                       # ... and hands them here instead of to a FrameWriter


class DcaSource:
    """The radar live, through the DCA1000 (see the module docstring)."""

    def __init__(self, link: Link):
        self.link, self.sink, self.rx, self.sock = link, LatestFrame(), None, None
        self.armed = False
        self.pkt_s = 0

    def start(self) -> str:
        """Arms the DCA as dca_capture_radar_sync.main() does. Returns an error text or ''."""
        s = self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        s.bind(("", dca.DCA_PORT))
        # Same sequence, waits and packet delay as dca_capture_radar_sync.main() - keep them in step.
        dca.send_recv(s, "RESET_FPGA", dca.CMD_RESET_FPGA, no_ack_ok=True)
        time.sleep(3.0)
        if not dca.send_recv(s, "CONFIG_FPGA", dca.CMD_CONFIG_FPGA, dca.FPGA_CFG):
            return "radar capture board (DCA1000) not responding - check the radar's 12 V power and cable"
        time.sleep(1.0)
        dca.send_recv(s, "SYSTEM_CONNECT", dca.CMD_SYSTEM_CONNECT)
        time.sleep(0.5)
        if not dca.send_recv(s, "CONFIG_PACKET_DATA", dca.CMD_CONFIG_PACKET, dca.pkt_cfg(dca.DEFAULT_PKT_DELAY_US)):
            return "CONFIG_PACKET_DATA failed"
        time.sleep(0.5)
        dca.send_recv(s, "SYSTEM_CONNECT (pre-record)", dca.CMD_SYSTEM_CONNECT)
        time.sleep(0.5)
        self.rx = LiveReceiver(self.sink)
        self.rx.start()
        time.sleep(0.3)
        log(f"  SO_RCVBUF granted: {self.rx.granted:,} B")
        if not dca.send_recv(s, "START_RECORD", dca.CMD_START_RECORD):
            return "START_RECORD failed"
        self.armed = True
        threading.Thread(target=self._count, name="radar-rate", daemon=True).start()
        return ""

    def _count(self):
        last, i = 0, 0
        while not self.link.stop.wait(1.0):
            i += 1
            n = self.rx.n
            self.pkt_s, last = n - last, n
            log(f"    t+{i:>3}s  {self.pkt_s:>7,} pkt/s")

    def newest(self, after, timeout):
        return self.sink.newest(after, timeout)

    def stop(self):
        if self.sock is not None:
            if self.armed:
                dca.send_recv(self.sock, "STOP_RECORD", dca.CMD_STOP_RECORD)
            if self.rx is not None:
                self.rx.stop.set()
                self.rx.join(timeout=5.0)
            self.sock.close()

    def status(self) -> dict:
        return {"pkt_s": self.pkt_s, "frames": self.sink.n}


class SessionSource:
    """A recorded session's radar frames at their recorded times, looping (no sensor needed)."""

    def __init__(self, link: Link, session: Path):
        self.link, self.dir = link, Path(session) / "radar"
        with open(self.dir / "frames.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.files = [r["filename"] for r in rows]
        self.t = np.array([float(r["t_burst_monotonic"]) for r in rows])
        self.t -= self.t[0]
        self.sink = LatestFrame()

    def start(self) -> str:
        threading.Thread(target=self._play, name="radar-play", daemon=True).start()
        return ""

    def _play(self):
        while not self.link.stop.is_set():
            t0 = time.monotonic()
            k_last = -1
            while not self.link.stop.is_set():
                k = int(np.searchsorted(self.t, time.monotonic() - t0, side="right")) - 1
                if k >= len(self.t) - 1 and k == k_last:
                    break                                   # end: start over
                if k != k_last:                             # only the newest due frame is read
                    self.sink.submit(None, (self.dir / self.files[k]).read_bytes())
                    k_last = k
                self.link.stop.wait(0.01)

    def newest(self, after, timeout):
        return self.sink.newest(after, timeout)

    def stop(self):
        pass

    def status(self) -> dict:
        return {"frames": self.sink.n}


class Fan:
    """Bird's-eye image of a [range, azimuth] dB map: forward up, vehicle-left on the left, over the
    shared extent (common.bev_extent)."""

    def __init__(self, range_res: float, n_range: int, az_world_deg: np.ndarray, range_m: float,
                 fov_deg: float = 70.0):
        x0, x1, y0, y1 = self.extent = bev_extent(range_m)
        x = x1 - (np.arange(H) + 0.5) * (x1 - x0) / H
        y = y1 - (np.arange(W) + 0.5) * (y1 - y0) / W
        X, Y = np.meshgrid(x, y, indexing="ij")
        az = np.degrees(np.arctan2(Y, X))
        ri = np.round(np.hypot(X, Y) / range_res).astype(int)
        order = np.argsort(az_world_deg)
        pos = np.clip(np.searchsorted(az_world_deg[order], az), 1, len(order) - 1)
        nearer = np.where(np.abs(az_world_deg[order][pos - 1] - az) < np.abs(az_world_deg[order][pos] - az), pos - 1, pos)
        self.ai = order[nearer]
        self.valid = (X > 0) & (ri < n_range) & (np.abs(az) <= min(fov_deg, np.abs(az_world_deg).max()))
        self.ri = np.where(self.valid, ri, 0)
        from sources import _lut                  # the replay's colour tables (radar: inferno)
        self.lut = _lut("inferno")

    def render(self, m: np.ndarray) -> np.ndarray:
        v = m[self.ri, self.ai][self.valid]
        # colour as the replay's radar BEV: from 3 dB above the noise floor, at most 30 dB below the peak
        hi = float(np.percentile(v, 99.9))
        lo = max(float(np.median(v)) + 3, hi - 30)
        hi = max(hi, lo + 15)
        img = np.full((H, W, 3), 28, np.uint8)
        img[self.valid] = self.lut[np.clip((v - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)]
        return img


def main() -> int:
    ap = argparse.ArgumentParser(description="Radar live view (no saving)")
    add_args(ap)
    a = ap.parse_args()
    link = Link("radar", a.max_secs, {"range_m": a.range_m, "radar_map": "static"})
    if a.from_session:
        variant, how = radar_firmware.session_variant(a.from_session)
        src = SessionSource(link, a.from_session)
        log(f"radar: playing {a.from_session} (firmware {variant}, {how})")
    else:
        fw = radar_firmware.read_flashed()
        variant = (fw or {}).get("variant") or radar_firmware.LEGACY_VARIANT
        if fw is None:
            log(f"radar: WARNING: no {radar_firmware.FLASHED_RECORD} - assuming {variant}")
        src = DcaSource(link)
    from perception.radar import C, RadarDetector, rx_phase_for
    rx = rx_phase_for(variant)
    det = RadarDetector(radar_firmware.cfg_for_variant(variant), rx_phase_deg=rx)
    log(f"radar: firmware {variant}, in-situ RX phase correction {'on' if rx is not None else 'off'}, "
        f"device {det.dev}")
    fans = {}
    rate, pacer = Rate(), Pacer(a.hz, link.stop)
    try:
        err = src.start()
        if err:
            link.error(err)
            return 1
        seen = 0
        while link.running():
            got = src.newest(seen, 1.0)
            if got is None:
                if seen == 0:
                    link.send({"waiting": True, **src.status()})
                continue
            seen, raw, t_frame = got
            raw = np.frombuffer(raw, dtype=np.int16)
            tic = time.perf_counter()
            out = det.process(raw, moving_only=True, static=True)
            kind, R = link.control.get("radar_map", "static"), float(link.control.get("range_m", 20.0))
            if kind == "moving":
                m, az_w = out["ra"].T.cpu().numpy(), -det.az_axis    # [range, az], world azimuth (+ left)
            else:
                kind = "static"
                m, az_w = out["static_map"].cpu().numpy(), det.static_az_deg
            key = (kind, R)
            if key not in fans:
                fans[key] = Fan(det.range_res, det.n_range, np.asarray(az_w, float), R)
            img = fans[key].render(m)
            mv = out["detections"]
            mv = mv[mv[:, C["ghost"]] == 0]
            st = out["static_detections"]
            st = st[st[:, C["ghost"]] == 0]
            st = st[np.argsort(-st[:, C["power_db"]])][:N_STATIC]
            ms = (time.perf_counter() - tic) * 1e3
            rate.tick()
            cols = [C["x"], C["y"], C["range"], C["az_deg"]]
            link.send({"t": t_frame, "extent": fans[key].extent, "map": kind, "ms": round(ms, 1),
                       "hz": round(rate.hz, 1), "noise_db": round(float(out["noise_db"]), 1),
                       # [x, y, range, azimuth (deg, + left), v_r (m/s, + receding), SNR dB]
                       "moving": rounded(mv[:, cols + [C["v_r"], C["snr_db"]]]),
                       # [x, y, range, azimuth, power dB, SNR dB], strongest first
                       "static": rounded(st[:, cols + [C["power_db"], C["snr_db"]]]),
                       **src.status()}, jpeg(img))
            pacer.wait()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        link.error(str(exc))
        return 1
    finally:
        src.stop()
        log("radar: live view stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
