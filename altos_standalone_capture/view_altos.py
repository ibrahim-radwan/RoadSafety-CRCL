#!/usr/bin/env python3
"""
view_altos.py — 3D viewer for the Altos V4 radar's point cloud, live or from
a recording.

    python view_altos.py                                   # live from the radar
    python view_altos.py --session dataset/altos_session_<stamp>

LIVE
    Joins the radar's multicast group (default 224.1.2.4:4040 on
    192.168.3.1) and shows each frame as it arrives. It can run while
    record_altos_raw.py records: both sockets receive every packet.

REPLAY
    Plays altos.pcap at the recorded pace (host receipt times), with
    play/pause, a time slider, speed and single-frame steps.
    Keys: space play/pause, left/right one frame.

VIEW
    Axes are vehicle-style: x forward (boresight), y left, z up, origin at
    the radar. Mouse: left drag orbits, wheel zooms, ctrl/middle drag pans.
    Drawn: a ground grid at --height below the radar, range rings every
    10 m, the 150 deg x 30 deg field of view (radar blue).

    Colour by:
        doppler  blue = approaching, red = receding (|v| up to the scale)
        snr      in dB
        height   above the ground grid
        range
    Filters: minimum SNR (dB), minimum |doppler| (to see only moving
    things), maximum range. Trail overlays the last N frames, older ones
    fainter. Points with NaN fields are never drawn.

    The doppler sign (negative = approaching) follows the vendor driver; the
    elevation sign (z up) follows the vendor guide. Neither is checked
    against a target yet.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

# `conda activate` runs qt-main's activation script, which sets
# QT_XCB_GL_INTEGRATION=none: Qt then has neither GLX nor EGL and the 3D view
# fails ("Cannot create platform OpenGL context"). Re-enable Qt's default GL
# integration for this process, before Qt loads (as replay_session.py does).
if os.environ.get("QT_XCB_GL_INTEGRATION") == "none":
    del os.environ["QT_XCB_GL_INTEGRATION"]

import numpy as np
from matplotlib import cm          # before PyQt5: loads the environment's libstdc++ first

import altos_v4 as av

RADAR_RGB = (0x2A / 255, 0x78 / 255, 0xD6 / 255)
COLOUR_MODES = ("doppler", "snr", "height", "range")
HALF_HFOV, HALF_VFOV = 75.0, 15.0


def doppler_colours(v, scale):
    """Blue (approaching, v < 0) - grey (still) - red (receding), RGBA."""
    t = np.clip(v / scale, -1, 1)[:, None]
    grey = np.array([0.62, 0.64, 0.68], np.float32)
    blue = np.array([0.20, 0.55, 1.00], np.float32)
    red = np.array([1.00, 0.30, 0.22], np.float32)
    rgb = np.where(t < 0, grey + (blue - grey) * (-t), grey + (red - grey) * t)
    return np.concatenate([rgb, np.ones((len(v), 1), np.float32)], axis=1)


# ---------------------------------------------------------------- sources

class DecodedFrame:
    __slots__ = ("frame_id", "t_ns", "radar_time", "pts", "n_sent", "n_invalid", "complete")

    def __init__(self, f: av.Frame):
        pts, ok = av.points_array(f.payload())
        self.frame_id = f.frame_id
        self.t_ns = f.t_last_ns
        self.radar_time = f.radar_time
        self.pts = pts[ok]
        self.n_sent = f.n_points
        self.n_invalid = int((~ok).sum())
        self.complete = f.complete


class LiveSource:
    """Receives the multicast stream in a thread; the GUI takes new frames."""

    def __init__(self, group: str, port: int, host_ip: str):
        self.group, self.port, self.host_ip = group, port, host_ip
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        try:
            s.bind((group, port))
            s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                         socket.inet_aton(group) + socket.inet_aton(host_ip))
        except OSError as exc:
            raise SystemExit("Cannot join %s:%d on %s (%s). Is %s on the radar's port (ip -br addr)?"
                             % (group, port, host_ip, exc, host_ip))
        s.settimeout(0.2)
        self.sock = s
        self.asm = av.FrameAssembler()
        self.frames = collections.deque(maxlen=64)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.packets = 0
        self.arrivals = collections.deque(maxlen=31)
        self.thread = threading.Thread(target=self._run, name="altos-live", daemon=True)
        self.thread.start()

    def _run(self):
        buf = bytearray(65536)
        while not self.stop.is_set():
            try:
                n, _addr = self.sock.recvfrom_into(buf)
            except socket.timeout:
                continue
            except OSError:
                return
            self.packets += 1
            for f in self.asm.feed(bytes(buf[:n]), time.monotonic_ns()):
                d = DecodedFrame(f)
                with self.lock:
                    self.frames.append(d)
                    self.arrivals.append(d.t_ns)

    def take(self):
        with self.lock:
            out = list(self.frames)
            self.frames.clear()
        return out

    def fps(self) -> float:
        a = list(self.arrivals)
        return (len(a) - 1) / ((a[-1] - a[0]) * 1e-9) if len(a) > 2 and a[-1] > a[0] else 0.0

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1.0)
        self.sock.close()


class SessionSource:
    """All frames of a recording, decoded up front."""

    def __init__(self, session: Path):
        pcap = session / "altos.pcap"
        if not pcap.exists():
            raise SystemExit("No altos.pcap in %s" % session)
        self.asm = av.FrameAssembler()
        frames = []
        for t_ns, pkt in av.read_pcap(pcap):
            frames.extend(self.asm.feed(pkt, t_ns))
        frames.extend(self.asm.flush())
        if not frames:
            raise SystemExit("No frames in %s" % pcap)
        self.frames = [DecodedFrame(f) for f in frames]
        self.t = np.array([f.t_ns for f in self.frames], dtype=np.int64)
        self.t0 = int(self.t[0])
        meta = session / "metadata.json"
        self.meta = json.loads(meta.read_text()) if meta.exists() else {}


def radar_install(radar_ip: str):
    try:
        with urllib.request.urlopen("http://%s:%d/install" % (radar_ip, av.WEB_PORT), timeout=1.5) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data[0] if isinstance(data, list) and data else {}
    except Exception:
        return {}


# ---------------------------------------------------------------- view

class Viewer:
    def __init__(self, a, source, title: str):
        import pyqtgraph.opengl as gl
        from PyQt5 import QtCore, QtWidgets

        self.a, self.src, self.gl = a, source, gl
        self.live = isinstance(source, LiveSource)
        self.history = collections.deque(maxlen=15)     # newest last, for the trail
        self.index = 0                                  # replay position
        self.playing = not self.live
        self.speed = 1.0
        self.play_t0 = None                             # (wall ns, session ns) when play started
        self.last_frame_wall = time.monotonic()

        self.win = QtWidgets.QWidget()
        self.win.setWindowTitle(title)
        self.win.resize(1400, 900)
        root = QtWidgets.QVBoxLayout(self.win)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # controls
        bar = QtWidgets.QHBoxLayout()
        root.addLayout(bar)

        def spin(label, lo, hi, val, step, dec, tip):
            bar.addWidget(QtWidgets.QLabel(label))
            w = QtWidgets.QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setDecimals(dec)
            w.setSingleStep(step)
            w.setValue(val)
            w.setToolTip(tip)
            w.valueChanged.connect(lambda _v: self.redraw())
            bar.addWidget(w)
            return w

        bar.addWidget(QtWidgets.QLabel("Colour"))
        self.colour = QtWidgets.QComboBox()
        self.colour.addItems(COLOUR_MODES)
        self.colour.setCurrentText(a.colour)
        self.colour.currentIndexChanged.connect(lambda _i: self.redraw())
        bar.addWidget(self.colour)
        self.vscale = spin("±v", 0.5, 60, 5.0, 0.5, 1, "doppler colour scale, m/s")
        self.min_snr = spin("SNR ≥", 0, 70, a.min_snr_db, 1, 0, "hide points below this SNR (dB)")
        self.min_v = spin("|v| ≥", 0, 50, 0.0, 0.1, 1, "hide points slower than this (m/s); 0 = all")
        self.max_r = spin("range ≤", 1, 160, a.range, 5, 0, "hide points beyond this range (m)")
        self.trail = QtWidgets.QSpinBox()
        self.trail.setRange(1, 15)
        self.trail.setValue(a.trail)
        self.trail.setToolTip("frames overlaid (15 = 1 s); older ones fainter")
        self.trail.valueChanged.connect(lambda _v: self.redraw())
        bar.addWidget(QtWidgets.QLabel("Trail"))
        bar.addWidget(self.trail)
        self.size = spin("Size", 1, 20, a.point_size, 1, 0, "point size, pixels")
        bar.addStretch(1)
        for name, fn in (("Behind", self.cam_behind), ("Top", self.cam_top), ("Side", self.cam_side)):
            b = QtWidgets.QPushButton(name)
            b.clicked.connect(fn)
            bar.addWidget(b)

        # 3D view
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((14, 16, 20))
        root.addWidget(self.view, stretch=1)
        self._build_scene()

        # status and (replay) transport
        self.info = QtWidgets.QLabel()
        self.info.setStyleSheet("font-family: monospace;")
        root.addWidget(self.info)
        self.legend = QtWidgets.QLabel()
        root.addWidget(self.legend)
        if not self.live:
            tb = QtWidgets.QHBoxLayout()
            root.addLayout(tb)
            self.b_play = QtWidgets.QPushButton("Pause")
            self.b_play.clicked.connect(self.toggle_play)
            tb.addWidget(self.b_play)
            for txt, step in (("◀", -1), ("▶", 1)):
                b = QtWidgets.QPushButton(txt)
                b.setFixedWidth(36)
                b.clicked.connect(lambda _c, s=step: self.step(s))
                tb.addWidget(b)
            self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.slider.setRange(0, len(source.frames) - 1)
            self.slider.valueChanged.connect(self.seek)
            tb.addWidget(self.slider, stretch=1)
            self.speed_box = QtWidgets.QComboBox()
            self.speed_box.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
            self.speed_box.setCurrentText("1x")
            self.speed_box.currentTextChanged.connect(self.set_speed)
            tb.addWidget(self.speed_box)
            self.time_label = QtWidgets.QLabel()
            tb.addWidget(self.time_label)
            sc = QtWidgets.QShortcut
            from PyQt5.QtGui import QKeySequence
            sc(QKeySequence("Space"), self.win, activated=self.toggle_play)
            sc(QKeySequence("Left"), self.win, activated=lambda: self.step(-1))
            sc(QKeySequence("Right"), self.win, activated=lambda: self.step(1))

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(10)
        {"behind": self.cam_behind, "top": self.cam_top, "side": self.cam_side}[a.camera]()
        if not self.live:
            self.show_index(0)
        self.redraw()

    # ── scene ───────────────────────────────────────────────────────────
    def _build_scene(self):
        gl = self.gl
        R, g = self.a.range, -self.a.height
        grid = gl.GLGridItem()
        grid.setSize(2 * R, 2 * R)
        grid.setSpacing(5, 5)
        grid.translate(R / 2, 0, g)
        grid.setColor((44, 48, 56, 255))
        self.view.addItem(grid)

        rgba = RADAR_RGB + (0.9,)
        segs = []
        # range rings on the ground, every 10 m, within the field of view
        th = np.radians(np.linspace(-HALF_HFOV, HALF_HFOV, 61))
        for r in range(10, int(R) + 1, 10):
            ring = np.stack([r * np.cos(th), r * np.sin(th), np.full_like(th, g)], axis=1)
            segs.append(np.repeat(ring, 2, axis=0)[1:-1])
        rings = gl.GLLinePlotItem(pos=np.concatenate(segs).astype(np.float32), mode="lines",
                                  color=(0.30, 0.36, 0.46, 0.8), width=1.0, antialias=True)
        self.view.addItem(rings)

        # field of view: ground wedge edges, and the frustum edges to 20 m
        edges = []
        for s in (-1, 1):
            a = np.radians(s * HALF_HFOV)
            edges += [[0, 0, g], [R * np.cos(a), R * np.sin(a), g]]
        L = min(20.0, R)
        for sa in (-1, 1):
            for se in (-1, 1):
                a, e = np.radians(sa * HALF_HFOV), np.radians(se * HALF_VFOV)
                edges += [[0, 0, 0], [L * np.cos(e) * np.cos(a), L * np.cos(e) * np.sin(a), L * np.sin(e)]]
        self.view.addItem(gl.GLLinePlotItem(pos=np.array(edges, np.float32), mode="lines",
                                            color=rgba, width=1.5, antialias=True))
        # radar at the origin, and its mast down to the ground
        self.view.addItem(gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), size=12,
                                               color=RADAR_RGB + (1.0,), pxMode=True))
        self.view.addItem(gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, g]], np.float32),
                                            color=rgba, width=2.0))
        # axes at the radar, 2 m: x red, y green, z blue
        axes = np.array([[0, 0, 0], [2, 0, 0], [0, 0, 0], [0, 2, 0], [0, 0, 0], [0, 0, 2]], np.float32)
        cols = np.array([[1, .25, .25, 1]] * 2 + [[.25, .9, .35, 1]] * 2 + [[.3, .5, 1, 1]] * 2, np.float32)
        self.view.addItem(gl.GLLinePlotItem(pos=axes, color=cols, mode="lines", width=3.0))
        self.cloud = gl.GLScatterPlotItem(pos=np.zeros((0, 3), np.float32), pxMode=True)
        self.cloud.setGLOptions("translucent")     # the default ("additive") turns overlapping points white
        self.view.addItem(self.cloud)

    def _camera(self, distance, elevation, azimuth, centre_x):
        from pyqtgraph import Vector
        self.view.opts["center"] = Vector(centre_x, 0, -self.a.height)
        self.view.setCameraPosition(distance=distance, elevation=elevation, azimuth=azimuth)

    def cam_behind(self):
        # azimuth 180: camera behind the radar looking along +x, vehicle-left on screen-left
        self._camera(self.a.range * 0.75, 28, 180, self.a.range * 0.35)

    def cam_top(self):
        self._camera(self.a.range * 1.25, 90, 180, self.a.range * 0.38)

    def cam_side(self):
        # from the vehicle's right (-y), forward to the right of the screen
        self._camera(self.a.range * 0.8, 6, 270, self.a.range * 0.4)

    # ── replay transport ────────────────────────────────────────────────
    def toggle_play(self):
        self.playing = not self.playing
        self.play_t0 = None
        self.b_play.setText("Pause" if self.playing else "Play")

    def set_speed(self, text):
        self.speed = float(text.rstrip("x"))
        self.play_t0 = None

    def step(self, k):
        if self.playing:
            self.toggle_play()
        self.slider.setValue(int(np.clip(self.index + k, 0, len(self.src.frames) - 1)))

    def seek(self, i):
        if i != self.index:
            self.show_index(i)
            self.play_t0 = None

    def show_index(self, i):
        fr = self.src.frames
        self.index = i
        self.history.clear()
        for k in range(max(0, i - 14), i + 1):
            self.history.append(fr[k])
        self.redraw()
        if self.slider.value() != i:
            self.slider.blockSignals(True)
            self.slider.setValue(i)
            self.slider.blockSignals(False)

    # ── loop ────────────────────────────────────────────────────────────
    def tick(self):
        if self.live:
            new = self.src.take()
            if new:
                self.history.extend(new)
                self.last_frame_wall = time.monotonic()
                self.redraw()
            elif time.monotonic() - self.last_frame_wall > 2.0:
                self.info.setText("waiting for packets on %s:%d via %s ... (%d packets so far)" % (
                    self.src.group, self.src.port, self.src.host_ip, self.src.packets))
            return
        if not self.playing:
            return
        now = time.monotonic_ns()
        if self.play_t0 is None:
            self.play_t0 = (now, int(self.src.t[self.index]))
        target = self.play_t0[1] + (now - self.play_t0[0]) * self.speed
        i = int(np.searchsorted(self.src.t, target, side="right")) - 1
        if i >= len(self.src.frames) - 1:
            i = len(self.src.frames) - 1
            self.toggle_play()
        if i != self.index:
            self.show_index(i)

    def redraw(self):
        frames = list(self.history)[-self.trail.value():]
        mode = self.colour.currentText()
        if not frames:
            self.cloud.setData(pos=np.zeros((0, 3), np.float32))
            self._legend(mode)
            return
        n = len(frames)
        parts, alphas = [], []
        for k, f in enumerate(frames):
            parts.append(f.pts)
            alphas.append(np.full(len(f.pts), 0.25 + 0.75 * (k + 1) / n if n > 1 else 1.0, np.float32))
        p = np.concatenate(parts)
        alpha = np.concatenate(alphas)
        with np.errstate(divide="ignore"):
            snr_db = 10 * np.log10(np.maximum(p[:, 4], 1e-6))
        keep = ((snr_db >= self.min_snr.value()) & (np.abs(p[:, 1]) >= self.min_v.value())
                & (p[:, 0] <= self.max_r.value()))
        p, alpha, snr_db = p[keep], alpha[keep], snr_db[keep]

        if mode == "doppler":
            rgba = doppler_colours(p[:, 1], self.vscale.value())
        elif mode == "snr":
            rgba = cm.viridis(np.clip(snr_db / 50.0, 0, 1))
        elif mode == "height":
            rgba = cm.turbo(np.clip((p[:, 7] + self.a.height) / 3.0, 0, 1))
        else:
            rgba = cm.viridis(1 - np.clip(p[:, 0] / max(1.0, self.max_r.value()), 0, 1))
        rgba = rgba.astype(np.float32)
        rgba[:, 3] = alpha
        self.cloud.setData(pos=np.ascontiguousarray(p[:, 5:8], dtype=np.float32),
                           color=np.ascontiguousarray(rgba), size=float(self.size.value()))

        f = frames[-1]
        shown_now = int(keep[-len(f.pts):].sum()) if len(f.pts) else 0
        txt = "frame %d | %d points sent, %d valid, %d shown | trail %d frames, %d points drawn" % (
            f.frame_id, f.n_sent, len(f.pts), shown_now, n, len(p))
        if f.n_invalid:
            txt += " | %d NaN" % f.n_invalid
        if not f.complete:
            txt += " | INCOMPLETE"
        if self.live:
            txt = "LIVE %.1f fps | lost %d | incomplete %d | " % (
                self.src.fps(), self.src.asm.missing_ids, self.src.asm.incomplete) + txt
        else:
            t = (f.t_ns - self.src.t0) * 1e-9
            self.time_label.setText("%6.2f / %.2f s" % (t, (self.src.t[-1] - self.src.t0) * 1e-9))
        self.info.setText(txt)
        self._legend(mode)

    def _legend(self, mode):
        txt = {
            "doppler": "doppler: <span style='color:#338cff'>■ approaching</span> · "
                       "<span style='color:#9ea3ad'>■ still</span> · "
                       "<span style='color:#ff4d38'>■ receding</span>  (full colour at ±%.1f m/s)"
                       % self.vscale.value(),
            "snr": "snr: dark = 0 dB … yellow = 50 dB",
            "height": "height above the grid: blue = 0 m … red = 3 m",
            "range": "range: yellow = near … dark = far",
        }[mode]
        self.legend.setText(txt + "   |   rings every 10 m, grid 5 m, %.1f m below the radar   |   "
                                  "axes 2 m: x forward red, y left green, z up blue   |   "
                                  "drag: orbit · wheel: zoom · ctrl+drag: pan" % self.a.height)


def main() -> int:
    ap = argparse.ArgumentParser(description="3D viewer for the Altos V4 radar point cloud.")
    ap.add_argument("--session", type=Path, help="replay this recording; without it, show the radar live")
    ap.add_argument("--radar-ip", default=av.DEFAULT_RADAR_IP)
    ap.add_argument("--host-ip", default=av.DEFAULT_HOST_IP)
    ap.add_argument("--group", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--height", type=float, default=None,
                    help="radar height above the ground, m (default: the radar's install_height, else 1.0)")
    ap.add_argument("--range", type=float, default=40.0, help="view extent, m (filter adjustable in the window)")
    ap.add_argument("--colour", choices=COLOUR_MODES, default="doppler")
    ap.add_argument("--min-snr-db", type=float, default=0.0)
    ap.add_argument("--trail", type=int, default=3)
    ap.add_argument("--point-size", type=float, default=5.0)
    ap.add_argument("--camera", choices=("behind", "top", "side"), default="behind", help="starting view")
    ap.add_argument("--screenshot", type=Path, help="save the window to this PNG and exit")
    ap.add_argument("--screenshot-after", type=float, default=5.0, help="seconds before the screenshot")
    a = ap.parse_args()

    if a.session:
        src = SessionSource(a.session)
        inst = (src.meta.get("radar_settings", {}).get("install") or [{}])[0]
        title = "Altos V4 replay — %s" % a.session.name
    else:
        inst = radar_install(a.radar_ip)
        group = a.group or inst.get("multicast_addr") or av.DEFAULT_GROUP
        port = a.port or inst.get("multicast_port") or av.DEFAULT_PORT
        src = LiveSource(group, port, a.host_ip)
        title = "Altos V4 live — %s:%d" % (group, port)
    if a.height is None:
        a.height = float(inst.get("install_height") or 1.0)
    if inst:
        title += "  (profile %s, sync %s)" % (inst.get("profile"), inst.get("sync_mode"))

    # The QApplication must exist before pyqtgraph (and through it PyOpenGL)
    # is imported: otherwise the system's X11/xkb libraries load next to the
    # environment's and Qt crashes in xkbcommon (seen on ibrahim-desktop,
    # 2026-11-15).
    from PyQt5 import QtCore, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    import pyqtgraph as pg
    pg.setConfigOptions(antialias=True)
    v = Viewer(a, src, title)
    v.win.show()

    if a.screenshot:
        def shot():
            v.win.grab().save(str(a.screenshot))
            print("saved %s" % a.screenshot)
            app.quit()
        QtCore.QTimer.singleShot(int(a.screenshot_after * 1000), shot)

    rc = app.exec_()
    if isinstance(src, LiveSource):
        src.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
