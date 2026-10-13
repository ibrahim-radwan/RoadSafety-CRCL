#!/usr/bin/env python3
"""
record_lidar_raw.py — record the RoboSense RS-Fairy's raw UDP stream to PCAP.

WHY RAW PACKETS
    Decoded, a frame is 138,240 points x 5 float32 at 10 Hz: ~27.6 MB/s.
    The packets carrying it are ~4.7 MB/s and lose nothing; rs_driver decodes
    them offline (python_wrapper_2: Lidar.start(pcap_path=...)). Capture does
    no decoding, so it costs little CPU next to the radar receiver and camera.

TIMESTAMPS
    PCAP record time = host CLOCK_MONOTONIC at receipt (time.monotonic_ns),
    the clock the radar and camera capture scripts log. MSOP and IMU packets
    also carry the LiDAR's own timestamp, which rs_driver decodes.
    clock_map.csv pairs CLOCK_MONOTONIC with CLOCK_REALTIME once per second.

LOSS ACCOUNTING
    MSOP packets carry a counter that steps by one per packet (bytes 12-15,
    big-endian, wrapping 65535 -> 0 every ~18 s; verified on hardware). Its
    gaps count packets lost anywhere between the LiDAR and this process. The
    kernel's per-socket drop counter (/proc/net/udp) shows how many of those
    this host dropped.

OUTPUT  dataset/rsfairy_session_<stamp>/
    lidar.pcap        Ethernet/IPv4/UDP frames, nanosecond PCAP (Wireshark,
                      tcpdump and rs_driver read it)
    stats.csv         per-second packet counts, losses and queue depth
    clock_map.csv     CLOCK_MONOTONIC <-> CLOCK_REALTIME pairs
    config_used.yaml  copy of the rs_driver config used for decoding
    metadata.json     addresses, ports, socket buffers, totals, completion

USAGE
    python record_lidar_raw.py --duration 60
    python check_lidar_session.py dataset/rsfairy_session_<stamp>
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
import selectors
import shutil
import signal
import socket
import struct
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "rs_driver" / "config" / "config.yaml"

MSOP_ID = b"\x55\xaa\x05\x5a"
MSOP_CNT_OFFSET = 12            # pkt_cnt_bot2top: +1 per packet
MSOP_CNT_MOD = 1 << 16          # the counter wraps at 16 bits
MSOP_TS_OFFSET = 20             # RSTimestampUTC: 6 B seconds + 4 B microseconds, big-endian

PCAP_MAGIC_NS = 0xA1B23C4D
LINKTYPE_ETHERNET = 1


def lidar_time(pkt: bytes, offset: int) -> float:
    return (int.from_bytes(pkt[offset:offset + 6], "big")
            + int.from_bytes(pkt[offset + 6:offset + 10], "big") * 1e-6)


def ipv4_checksum(header: bytes) -> int:
    total = sum(struct.unpack("!10H", header))
    total = (total & 0xFFFF) + (total >> 16)
    total = (total & 0xFFFF) + (total >> 16)
    return ~total & 0xFFFF


class PcapWriter:
    """Wraps each UDP payload in Ethernet/IPv4/UDP headers, as rs_driver's
    PCAP input expects (fixed 42-byte header, filtered by 'udp dst port')."""

    def __init__(self, path: Path, host_ip: str):
        self.fh = path.open("wb", buffering=1024 * 1024)
        self.fh.write(struct.pack("<IHHiIII", PCAP_MAGIC_NS, 2, 4, 0, 0, 65535, LINKTYPE_ETHERNET))
        self.dst_ip = socket.inet_aton(host_ip)
        self.headers = {}
        self.bytes = 0

    def _headers(self, src_ip: str, sport: int, dport: int, n: int) -> bytes:
        key = (src_ip, sport, dport, n)
        if key not in self.headers:
            eth = b"\x00" * 12 + b"\x08\x00"
            ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 28 + n, 0, 0x4000, 64, 17, 0,
                             socket.inet_aton(src_ip), self.dst_ip)
            ip = ip[:10] + struct.pack("!H", ipv4_checksum(ip)) + ip[12:]
            udp = struct.pack("!HHHH", sport, dport, 8 + n, 0)   # IPv4 allows no UDP checksum
            self.headers[key] = eth + ip + udp
        return self.headers[key]

    def write(self, t_ns: int, src_ip: str, sport: int, dport: int, payload: bytes) -> None:
        frame_len = 42 + len(payload)
        self.fh.write(struct.pack("<IIII", t_ns // 1_000_000_000, t_ns % 1_000_000_000,
                                  frame_len, frame_len))
        self.fh.write(self._headers(src_ip, sport, dport, len(payload)))
        self.fh.write(payload)
        self.bytes += 16 + frame_len

    def close(self) -> None:
        self.fh.flush()
        self.fh.close()


class Receiver(threading.Thread):
    """Receives on all LiDAR ports, stamps each packet, and queues it for the
    writer. Never blocks on disk: if the queue is full the packet is counted
    in queue_drops instead."""

    def __init__(self, socks: dict, out: queue.Queue, stop: threading.Event):
        super().__init__(name="lidar-rx", daemon=True)
        self.socks, self.out, self.stop = socks, out, stop
        self.ports = {name: s.getsockname()[1] for name, s in socks.items()}
        self.counts = {name: 0 for name in socks}
        self.queue_drops = 0
        self.msop_lost = 0
        self.msop_dupes = 0
        self.last_cnt = None
        self.first_msop = None    # (host monotonic ns, lidar seconds)
        self.last_msop = None
        self.error = None

    def run(self) -> None:
        try:
            sel = selectors.DefaultSelector()
            for name, s in self.socks.items():
                sel.register(s, selectors.EVENT_READ, name)
            buf = bytearray(65536)
            while not self.stop.is_set():
                for key, _ in sel.select(0.2):
                    while True:
                        try:
                            n, (src_ip, sport) = key.fileobj.recvfrom_into(buf)
                        except BlockingIOError:
                            break
                        t_ns = time.monotonic_ns()
                        pkt = bytes(buf[:n])
                        name = key.data
                        self.counts[name] += 1
                        if name == "msop" and pkt[:4] == MSOP_ID:
                            self._track_msop(t_ns, pkt)
                        try:
                            self.out.put_nowait((t_ns, src_ip, sport, self.ports[name], pkt))
                        except queue.Full:
                            self.queue_drops += 1
        except Exception as exc:
            self.error = exc
            self.stop.set()

    def _track_msop(self, t_ns: int, pkt: bytes) -> None:
        cnt = struct.unpack_from(">I", pkt, MSOP_CNT_OFFSET)[0]
        if self.last_cnt is not None:
            step = (cnt - self.last_cnt) % MSOP_CNT_MOD
            if step == 0:
                self.msop_dupes += 1
            else:
                self.msop_lost += step - 1
        self.last_cnt = cnt
        sample = (t_ns, lidar_time(pkt, MSOP_TS_OFFSET))
        self.first_msop = self.first_msop or sample
        self.last_msop = sample


def writer_loop(pcap: PcapWriter, items: queue.Queue, state: dict) -> None:
    try:
        while True:
            item = items.get()
            if item is None:
                return
            pcap.write(*item)
    except Exception as exc:
        state["error"] = exc


def kernel_udp_drops(ports) -> dict:
    """Per-socket receive drops from /proc/net/udp, keyed by local port."""
    drops = {}
    with open("/proc/net/udp") as fh:
        next(fh)
        for line in fh:
            fields = line.split()
            port = int(fields[1].split(":")[1], 16)
            if port in ports:
                drops[port] = drops.get(port, 0) + int(fields[-1])
    return drops


def clock_pair() -> tuple:
    m1 = time.monotonic_ns()
    r = time.time_ns()
    m2 = time.monotonic_ns()
    return (m1 + m2) // 2, r


def save_json(path: Path, content: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(content, indent=2) + "\n")
    tmp.replace(path)


def open_socket(host_ip: str, port: int, rcvbuf: int) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
    try:
        s.bind((host_ip, port))
    except OSError as exc:
        s.close()
        raise SystemExit(f"Cannot bind {host_ip}:{port} ({exc}). Is another LiDAR program "
                         f"(rs_driver demo, test_lidar*.py, visualizer.py) running, or is the "
                         f"USB Ethernet interface not on {host_ip}?")
    s.setblocking(False)
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description="Record the RS-Fairy raw UDP stream to PCAP.")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out-root", type=Path, default=HERE / "dataset")
    ap.add_argument("--session-dir", type=Path,
                    help="record into exactly this new directory instead of <out-root>/rsfairy_session_<time>")
    ap.add_argument("--duration", type=float, help="stop after this many seconds (default: Ctrl+C)")
    ap.add_argument("--rcvbuf", type=int, default=32 * 1024 * 1024,
                    help="requested SO_RCVBUF bytes; the kernel caps it at 2 x net.core.rmem_max")
    ap.add_argument("--queue-packets", type=int, default=200_000,
                    help="packets buffered between receiver and disk writer (~50 s at full rate)")
    ap.add_argument("--min-free-gb", type=float, default=2.0)
    ap.add_argument("--first-packet-timeout", type=float, default=5.0)
    a = ap.parse_args()

    cfg = yaml.safe_load(a.config.read_text())
    host_ip, lidar_ip = cfg["network"]["host_ip"], cfg["network"]["lidar_ip"]
    ports = {"msop": cfg["lidar"]["msop_port"], "difop": cfg["lidar"]["difop_port"]}
    if cfg["lidar"].get("imu_port"):
        ports["imu"] = cfg["lidar"]["imu_port"]

    out_root = a.session_dir.parent if a.session_dir else a.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out_root).free < a.min_free_gb * 1024 ** 3:
        raise SystemExit(f"Less than {a.min_free_gb} GB free under {out_root}")

    socks = {name: open_socket(host_ip, port, a.rcvbuf) for name, port in ports.items()}
    granted = {name: s.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF) for name, s in socks.items()}
    drops_start = kernel_udp_drops(set(ports.values()))

    session = a.session_dir or out_root / f"rsfairy_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session.mkdir()
    shutil.copy2(a.config, session / "config_used.yaml")

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    pcap = PcapWriter(session / "lidar.pcap", host_ip)
    items: queue.Queue = queue.Queue(maxsize=a.queue_packets)
    wstate: dict = {}
    writer = threading.Thread(target=writer_loop, args=(pcap, items, wstate), name="lidar-pcap", daemon=True)
    rx = Receiver(socks, items, stop)

    mono0, real0 = clock_pair()
    meta = {
        "sensor": "RoboSense RS-Fairy", "lidar_ip": lidar_ip, "host_ip": host_ip, "ports": ports,
        "pcap_timestamp_clock": "CLOCK_MONOTONIC (time.monotonic_ns) at host receipt",
        "lidar_timestamp": "in MSOP/IMU payloads (LiDAR clock); decoded by rs_driver",
        "rcvbuf_requested": a.rcvbuf, "rcvbuf_granted": granted,
        "start_monotonic_ns": mono0, "start_realtime_ns": real0,
        "start_local": datetime.now().astimezone().isoformat(),
        "complete": False,
    }
    save_json(session / "metadata.json", meta)

    writer.start()
    rx.start()
    print(f"Recording RS-Fairy {lidar_ip} -> {host_ip} ports {ports}")
    print(f"Session: {session}")
    if min(granted.values()) < 4 * 1024 * 1024:
        print(f"  note: SO_RCVBUF granted {granted['msop']:,} B (requested {a.rcvbuf:,}); "
              f"raise net.core.rmem_max for more headroom")

    error = None
    t_start = time.monotonic()
    with (session / "stats.csv").open("w", newline="") as sf, \
         (session / "clock_map.csv").open("w", newline="") as cf:
        stats = csv.writer(sf)
        stats.writerow(["t_monotonic_ns", "msop", "imu", "difop", "msop_lost", "queue_drops",
                        "queue_depth", "pcap_mbit_s", "last_msop_monotonic_ns", "last_msop_lidar_s"])
        clocks = csv.writer(cf)
        clocks.writerow(["monotonic_ns", "realtime_ns"])
        clocks.writerow([mono0, real0])
        prev = dict(rx.counts, lost=0, qd=0, bytes=0)
        next_tick = t_start + 1.0
        while not stop.is_set():
            stop.wait(max(0.0, next_tick - time.monotonic()))
            now = time.monotonic()
            next_tick += 1.0
            clocks.writerow(clock_pair())
            c = dict(rx.counts)
            d = {k: c[k] - prev[k] for k in c}
            lost, qd, nbytes = rx.msop_lost - prev["lost"], rx.queue_drops - prev["qd"], pcap.bytes - prev["bytes"]
            last = rx.last_msop or (0, 0.0)
            stats.writerow([time.monotonic_ns(), d["msop"], d.get("imu", 0), d["difop"], lost, qd,
                            items.qsize(), round(nbytes * 8 / 1e6, 2), last[0], f"{last[1]:.6f}"])
            sf.flush()
            cf.flush()
            prev = dict(c, lost=rx.msop_lost, qd=rx.queue_drops, bytes=pcap.bytes)
            print(f"{now - t_start:6.0f}s | msop {d['msop']:5d}/s | imu {d.get('imu', 0):3d}/s | "
                  f"lost {rx.msop_lost} | queue {items.qsize()} | {pcap.bytes / 1e6:8.1f} MB", flush=True)

            if rx.error or wstate.get("error"):
                error = f"receiver/writer failed: {rx.error or wstate['error']}"
            elif c["msop"] == 0 and now - t_start > a.first_packet_timeout:
                error = (f"no MSOP packets on {host_ip}:{ports['msop']} after {a.first_packet_timeout:.0f} s; "
                         f"check the LiDAR destination IP and the USB Ethernet link")
            elif shutil.disk_usage(session).free < a.min_free_gb * 1024 ** 3:
                error = f"free space below {a.min_free_gb} GB; recording stopped"
            elif a.duration and now - t_start >= a.duration:
                break
            if error:
                break

    stop.set()
    rx.join()
    drops_end = kernel_udp_drops(set(ports.values()))   # before close: closed sockets leave /proc/net/udp
    mono1, real1 = clock_pair()
    items.put(None)
    writer.join()
    pcap.close()
    for s in socks.values():
        s.close()

    msop_total = rx.counts["msop"]
    expected = msop_total + rx.msop_lost
    meta.update({
        "complete": error is None and rx.error is None and "error" not in wstate,
        "error": error,
        "stop_monotonic_ns": mono1, "stop_realtime_ns": real1,
        "duration_s": round((mono1 - mono0) / 1e9, 3),
        "packets": rx.counts,
        "msop_lost": rx.msop_lost,
        "msop_loss_ratio": round(rx.msop_lost / expected, 6) if expected else None,
        "msop_duplicates": rx.msop_dupes,
        "queue_drops": rx.queue_drops,
        "kernel_udp_drops": {p: drops_end.get(p, 0) - drops_start.get(p, 0) for p in drops_end},
        "first_msop": {"monotonic_ns": rx.first_msop[0], "lidar_s": rx.first_msop[1]} if rx.first_msop else None,
        "last_msop": {"monotonic_ns": rx.last_msop[0], "lidar_s": rx.last_msop[1]} if rx.last_msop else None,
        "pcap_bytes": pcap.bytes,
    })
    save_json(session / "metadata.json", meta)

    print(f"\n{'Finalized' if meta['complete'] else 'INCOMPLETE'}: {session}")
    print(f"  packets {rx.counts} | MSOP lost {rx.msop_lost} ({meta['msop_loss_ratio']}) | "
          f"queue drops {rx.queue_drops} | kernel drops {meta['kernel_udp_drops']}")
    print(f"  {pcap.bytes / 1e6:.1f} MB in {meta['duration_s']} s")
    if error:
        print(f"  ERROR: {error}")
    return 0 if meta["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
