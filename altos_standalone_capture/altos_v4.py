"""
altos_v4.py — the Altos V4 point-cloud packet format, frame reassembly and
PCAP reading. Shared by the recorder and the checker. Stdlib only, except
frames_to_arrays() (numpy).

SOURCE
    Altos "RCU + RF6 Interface Control Document" (it covers the V4) and the
    vendor header altos-pc-v2.h (github.com/Altos-Radar/altosRadarParser).
    Checked on our unit (serial V4-260804-0146, firmware 8e75cfeb) on
    2026-11-15. The V2 format (48-byte header, 44-byte points,
    github.com/Altos-Radar/altosRadarRosV2) does NOT apply to the V4.

PACKET (UDP, little-endian, at most 1472 bytes)
    header, 32 bytes:
        magic      u32   0x6F746C41 ("Alto" on the wire)
        nsec       u32   radar timestamp, nanoseconds part
        sec        u64   radar timestamp, Unix seconds (UTC; only correct with PTP)
        frame_id   u32   +1 per point cloud
        radar_id   u8    0 for the V4
        mode       u8    always 0 on the V4
        reserved   u16
        length     u32   bytes of point data in the whole frame (20 x points)
        offset     u32   where this packet's points start in the frame
    points, 20 bytes each, at most 72 per packet:
        range m, doppler m/s, azimuth rad, elevation rad, snr (linear); float32

    The radar timestamp is taken at the middle of the modulation (vendor
    guide). A frame of more than 72 points comes in several packets that
    share frame_id and differ in offset.

    The radar sometimes sends fields as 0xFFFFFFFF (NaN), all or some of a
    point's five; seen on 0.2% of points, always among the last ~16 points
    of a frame (2026-11-15). They are recorded as sent and marked invalid
    by frames_to_arrays().

    Without PTP ("Free running") the radar's date is wrong (2025-03 on
    2026-11-15) and its timestamps are exactly 66.7 ms apart, while the
    frames really arrive at a rate that wanders: fitted against host
    receipt, -505 ppm in one 20 s recording and +408 ppm in the next 30 s
    one (2026-11-15). Use host receipt time; the radar clock is only worth
    using with PTP.

RADAR FRAME
    The raw output is x forward, y LEFT, z DOWN (vendor guide): positive
    azimuth is to the left, as in our vehicle frame. Elevation is flipped
    when converting to xyz, so z is up.
"""
from __future__ import annotations

import struct
from typing import Dict, Iterator, List, Optional, Tuple

MAGIC = 0x6F746C41
HEADER = struct.Struct("<IIQIBBHII")
HEADER_BYTES = HEADER.size           # 32
POINT = struct.Struct("<5f")
POINT_BYTES = POINT.size             # 20
MAX_POINTS_PER_PACKET = 72
POINT_FIELDS = ("range", "doppler", "azimuth", "elevation", "snr")

DEFAULT_RADAR_IP = "192.168.3.10"
DEFAULT_HOST_IP = "192.168.3.1"
DEFAULT_GROUP = "224.1.2.4"
DEFAULT_PORT = 4040
WEB_PORT = 3000
# Read-only settings the radar's web server serves (GET). Changes go by POST,
# which nothing here sends.
WEB_READ_PATHS = ("/status", "/install", "/eth", "/time")


class Header:
    __slots__ = ("nsec", "sec", "frame_id", "radar_id", "mode", "length", "offset")

    def __init__(self, nsec, sec, frame_id, radar_id, mode, length, offset):
        self.nsec, self.sec, self.frame_id = nsec, sec, frame_id
        self.radar_id, self.mode, self.length, self.offset = radar_id, mode, length, offset

    @property
    def radar_time(self) -> float:
        return self.sec + self.nsec * 1e-9


def parse_header(pkt: bytes) -> Optional[Header]:
    """The packet's header, or None if it is not an Altos point-cloud packet."""
    if len(pkt) < HEADER_BYTES:
        return None
    magic, nsec, sec, fid, rid, mode, _res, length, offset = HEADER.unpack_from(pkt)
    if magic != MAGIC or (len(pkt) - HEADER_BYTES) % POINT_BYTES:
        return None
    return Header(nsec, sec, fid, rid, mode, length, offset)


class Frame:
    """One point cloud being put together from its packets."""

    __slots__ = ("frame_id", "radar_time", "length", "chunks", "received",
                 "t_first_ns", "t_last_ns", "packets", "overlap")

    def __init__(self, h: Header, t_ns: int):
        self.frame_id = h.frame_id
        self.radar_time = h.radar_time
        self.length = h.length
        self.chunks: Dict[int, bytes] = {}
        self.received = 0
        self.t_first_ns = t_ns
        self.t_last_ns = t_ns
        self.packets = 0
        self.overlap = False

    def add(self, h: Header, payload: bytes, t_ns: int) -> None:
        self.packets += 1
        self.t_last_ns = t_ns
        if h.offset in self.chunks or h.offset + len(payload) > self.length:
            self.overlap = True          # duplicate or inconsistent packet; keep the first
            return
        self.chunks[h.offset] = payload
        self.received += len(payload)

    @property
    def complete(self) -> bool:
        return self.received == self.length and not self.overlap

    @property
    def n_points(self) -> int:
        return self.length // POINT_BYTES

    def payload(self) -> bytes:
        """Point bytes in frame order; missing parts are left out."""
        return b"".join(self.chunks[o] for o in sorted(self.chunks))


class FrameAssembler:
    """Feeds packets in arrival order, yields finished frames.

    A frame is finished when all its bytes have arrived, or when a packet of
    a later frame arrives first (the frame is then incomplete). Counts frame
    IDs that never appeared (whole frames lost) and incomplete frames. A last
    frame cut off by the end of the recording is counted in truncated_end,
    not as incomplete."""

    def __init__(self):
        self.current: Optional[Frame] = None
        self.last_id: Optional[int] = None
        self.frames = 0
        self.incomplete = 0
        self.truncated_end = 0
        self.missing_ids = 0
        self.out_of_order = 0
        self.bad_packets = 0

    def feed(self, pkt: bytes, t_ns: int) -> List[Frame]:
        h = parse_header(pkt)
        if h is None:
            self.bad_packets += 1
            return []
        done: List[Frame] = []
        cur = self.current
        if cur is not None and h.frame_id != cur.frame_id:
            done.append(self._close(cur))
            cur = None
        if cur is None:
            if self.last_id is not None:
                step = (h.frame_id - self.last_id) & 0xFFFFFFFF
                if step == 0 or step > 0x7FFFFFFF:
                    self.out_of_order += 1   # late packet of a closed frame
                    return done
                self.missing_ids += step - 1
            cur = self.current = Frame(h, t_ns)
            self.last_id = h.frame_id
        cur.add(h, pkt[HEADER_BYTES:], t_ns)
        if cur.complete:
            done.append(self._close(cur))
        return done

    def flush(self) -> List[Frame]:
        """End of the recording: the frame still open, if any."""
        f = self.current
        if f is None:
            return []
        if not f.complete and not f.overlap:
            self.truncated_end += 1
            return [self._close(f, count_incomplete=False)]
        return [self._close(f)]

    def _close(self, f: Frame, count_incomplete: bool = True) -> Frame:
        self.current = None
        self.frames += 1
        if not f.complete and count_incomplete:
            self.incomplete += 1
        return f


# ---------------------------------------------------------------- PCAP

PCAP_MAGIC_NS = 0xA1B23C4D
PCAP_MAGIC_US = 0xA1B2C3D4
UDP_OFFSET = 42                      # Ethernet 14 + IPv4 20 + UDP 8, as the recorder writes


def read_pcap(path) -> Iterator[Tuple[int, bytes]]:
    """(host CLOCK_MONOTONIC ns at receipt, UDP payload) per packet of a
    recorder PCAP."""
    with open(path, "rb") as fh:
        gh = fh.read(24)
        if len(gh) < 24:
            return
        magic = struct.unpack_from("<I", gh)[0]
        if magic not in (PCAP_MAGIC_NS, PCAP_MAGIC_US):
            raise ValueError("%s: not a little-endian PCAP file" % path)
        scale = 1 if magic == PCAP_MAGIC_NS else 1000
        rec = struct.Struct("<IIII")
        while True:
            rh = fh.read(16)
            if len(rh) < 16:
                return
            sec, frac, incl, _orig = rec.unpack(rh)
            data = fh.read(incl)
            if len(data) < incl:
                return                   # truncated last record (recording cut off)
            yield sec * 1_000_000_000 + frac * scale, data[UDP_OFFSET:]


def iter_frames(path) -> Iterator[Frame]:
    """All frames of a recorder PCAP, complete or not, in order."""
    asm = FrameAssembler()
    for t_ns, pkt in read_pcap(path):
        for f in asm.feed(pkt, t_ns):
            yield f
    for f in asm.flush():
        yield f


def frame_points(f: Frame) -> List[Tuple[float, float, float, float, float]]:
    data = f.payload()
    return [POINT.unpack_from(data, k) for k in range(0, len(data) - POINT_BYTES + 1, POINT_BYTES)]


def frames_to_arrays(frames):
    """Frames -> numpy arrays for analysis.

    Returns a dict:
        points     (N, 8) float32: range, doppler, azimuth, elevation, snr, x, y, z
                   (vehicle-style axes: x forward, y left, z up; z = -raw)
        valid      (N,) bool: all five sent fields are finite (see NaN note above)
        frame_idx  (N,) int32: index into the per-frame arrays
        frame_id, t_host_first_ns, t_host_last_ns, radar_time, n_points, complete
    """
    import numpy as np

    rows, idx = [], []
    fid, t0, t1, rt, npts, comp = [], [], [], [], [], []
    for i, f in enumerate(frames):
        raw = np.frombuffer(f.payload(), dtype="<f4").reshape(-1, 5)
        rows.append(raw)
        idx.append(np.full(len(raw), i, dtype=np.int32))
        fid.append(f.frame_id)
        t0.append(f.t_first_ns)
        t1.append(f.t_last_ns)
        rt.append(f.radar_time)
        npts.append(len(raw))
        comp.append(f.complete)
    raw = np.concatenate(rows) if rows else np.zeros((0, 5), np.float32)
    r, az, el = raw[:, 0], raw[:, 2], raw[:, 3]
    with np.errstate(invalid="ignore"):
        xyz = np.stack([r * np.cos(el) * np.cos(az),
                        r * np.cos(el) * np.sin(az),
                        -r * np.sin(el)], axis=1)
    return {
        "points": np.concatenate([raw, xyz.astype(np.float32)], axis=1),
        "valid": np.isfinite(raw).all(axis=1),
        "frame_idx": np.concatenate(idx) if idx else np.zeros(0, np.int32),
        "frame_id": np.asarray(fid, dtype=np.uint32),
        "t_host_first_ns": np.asarray(t0, dtype=np.int64),
        "t_host_last_ns": np.asarray(t1, dtype=np.int64),
        "radar_time": np.asarray(rt, dtype=np.float64),
        "n_points": np.asarray(npts, dtype=np.int32),
        "complete": np.asarray(comp, dtype=bool),
    }
