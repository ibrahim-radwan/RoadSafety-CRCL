#!/usr/bin/env python3
"""
record_altos_raw.py — record the Altos V4 radar's point-cloud UDP stream to PCAP.

WHAT THE RADAR SENDS
    A processed point cloud (no raw ADC): up to 15 frames/s, each frame one or
    more UDP packets of at most 72 points, sent to a multicast group
    (default 224.1.2.4:4040). Packet format: altos_v4.py.

WHY RAW PACKETS
    The packets are the radar's whole output, and small (tens of KB/s indoors,
    at most ~3.6 MB/s at the spec's 12,000 points x 15 fps). Recording them
    unchanged keeps everything for offline decoding (check_altos_session.py,
    altos_v4.py) and for Wireshark (the vendor's altos.lua dissector).

TIMESTAMPS
    PCAP record time = host CLOCK_MONOTONIC at receipt (time.monotonic_ns),
    the clock the other recorders log. Each packet also carries the radar's
    own timestamp; it is only UTC-correct when the radar is PTP-synced
    ("Sync" on its web page). clock_map.csv pairs CLOCK_MONOTONIC with
    CLOCK_REALTIME once per second.

LOSS ACCOUNTING
    frame_id steps by one per frame: gaps count whole frames lost. A frame
    whose bytes do not add up to its header's length lost packets. The
    kernel's per-socket drop counter (/proc/net/udp) shows how many were
    dropped by this host.

OUTPUT  dataset/altos_session_<stamp>/
    altos.pcap        Ethernet/IPv4/UDP frames, nanosecond PCAP
    frames.csv        one row per frame: id, host receipt times, radar time,
                      points, packets, complete
    stats.csv         per-second packets, frames, points, losses, queue depth
    clock_map.csv     CLOCK_MONOTONIC <-> CLOCK_REALTIME pairs
    metadata.json     addresses, the radar's own settings at the start
                      (profile, FPS limit, threshold, sync mode), totals

USAGE
    python record_altos_raw.py --duration 30
    python check_altos_session.py dataset/altos_session_<stamp>
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
import shutil
import signal
import socket
import struct
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import altos_v4 as av

HERE = Path(__file__).resolve().parent
LINKTYPE_ETHERNET = 1


def ipv4_checksum(header: bytes) -> int:
    total = sum(struct.unpack("!10H", header))
    total = (total & 0xFFFF) + (total >> 16)
    total = (total & 0xFFFF) + (total >> 16)
    return ~total & 0xFFFF


class PcapWriter:
    """Wraps each UDP payload in Ethernet/IPv4/UDP headers (42 bytes), so
    Wireshark and tcpdump read the file as a normal capture."""

    def __init__(self, path: Path, dst_ip: str):
        self.fh = path.open("wb", buffering=1024 * 1024)
        self.fh.write(struct.pack("<IHHiIII", av.PCAP_MAGIC_NS, 2, 4, 0, 0, 65535, LINKTYPE_ETHERNET))
        self.dst_ip = socket.inet_aton(dst_ip)
        self.headers = {}
        self.bytes = 0

    def _headers(self, src_ip: str, sport: int, dport: int, n: int) -> bytes:
        key = (src_ip, sport, dport, n)
        if key not in self.headers:
            eth = b"\x00" * 12 + b"\x08\x00"
            ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 28 + n, 0, 0x4000, 64, 17, 0,
                             socket.inet_aton(src_ip), self.dst_ip)
            ip = ip[:10] + struct.pack("!H", ipv4_checksum(ip)) + ip[12:]
            udp = struct.pack("!HHHH", sport, dport, 8 + n, 0)
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
    """Receives the radar's packets, stamps each one, tracks frames, and
    queues packets and finished frames for the writer. Never blocks on disk:
    if the queue is full the packet is counted in queue_drops instead."""

    def __init__(self, sock: socket.socket, radar_ip: str, port: int,
                 out: queue.Queue, stop: threading.Event):
        super().__init__(name="altos-rx", daemon=True)
        self.sock, self.radar_ip, self.port, self.out, self.stop = sock, radar_ip, port, out, stop
        self.asm = av.FrameAssembler()
        self.packets = 0
        self.points = 0
        self.foreign = 0              # packets from another sender on the group/port
        self.queue_drops = 0
        self.first = None             # (host monotonic ns, radar seconds) of the first frame
        self.last = None
        self.error = None

    def run(self) -> None:
        try:
            buf = bytearray(65536)
            while not self.stop.is_set():
                try:
                    n, (src_ip, sport) = self.sock.recvfrom_into(buf)
                except socket.timeout:
                    continue
                t_ns = time.monotonic_ns()
                pkt = bytes(buf[:n])
                if src_ip != self.radar_ip:
                    self.foreign += 1
                    continue
                self.packets += 1
                frames = self.asm.feed(pkt, t_ns)
                try:
                    self.out.put_nowait(("pkt", (t_ns, src_ip, sport, self.port, pkt)))
                except queue.Full:
                    self.queue_drops += 1
                self._frames(frames)
        except Exception as exc:
            self.error = exc
            self.stop.set()

    def finish(self) -> None:
        self._frames(self.asm.flush())

    def _frames(self, frames) -> None:
        for f in frames:
            self.points += f.n_points
            sample = (f.t_first_ns, f.radar_time)
            self.first = self.first or sample
            self.last = sample
            try:
                self.out.put_nowait(("frame", f))
            except queue.Full:
                self.queue_drops += 1


def writer_loop(pcap: PcapWriter, frames_csv, items: queue.Queue, state: dict) -> None:
    try:
        while True:
            item = items.get()
            if item is None:
                return
            kind, payload = item
            if kind == "pkt":
                pcap.write(*payload)
            else:
                f = payload
                frames_csv.writerow([f.frame_id, f.t_first_ns, f.t_last_ns, "%.9f" % f.radar_time,
                                     f.n_points, f.packets, int(f.complete)])
    except Exception as exc:
        state["error"] = exc


def kernel_udp_drops(port: int) -> int:
    """Receive drops of the sockets bound to this port, from /proc/net/udp."""
    drops = 0
    with open("/proc/net/udp") as fh:
        next(fh)
        for line in fh:
            fields = line.split()
            if int(fields[1].split(":")[1], 16) == port:
                drops += int(fields[-1])
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


def read_radar_settings(radar_ip: str, timeout: float = 2.0) -> dict:
    """The radar's own status and settings (GET only), or the error per page."""
    out = {}
    for path in av.WEB_READ_PATHS:
        url = "http://%s:%d%s" % (radar_ip, av.WEB_PORT, path)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                out[path.strip("/")] = json.loads(r.read().decode("utf-8"))
        except Exception as exc:
            out[path.strip("/")] = {"error": str(exc)}
    return out


def open_socket(group: str, port: int, host_ip: str, rcvbuf: int) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
    try:
        s.bind((group, port))      # bound to the group: other multicast traffic on this port is not delivered
        s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                     socket.inet_aton(group) + socket.inet_aton(host_ip))
    except OSError as exc:
        s.close()
        raise SystemExit("Cannot join %s:%d on %s (%s). Is %s configured on the radar's "
                         "Ethernet port (ip -br addr)?" % (group, port, host_ip, exc, host_ip))
    s.settimeout(0.2)
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description="Record the Altos V4 radar's point-cloud UDP stream to PCAP.")
    ap.add_argument("--radar-ip", default=av.DEFAULT_RADAR_IP)
    ap.add_argument("--host-ip", default=av.DEFAULT_HOST_IP,
                    help="this machine's address on the radar's subnet (joins the multicast group there)")
    ap.add_argument("--group", default=None,
                    help="multicast group (default: the radar's own setting, else %s)" % av.DEFAULT_GROUP)
    ap.add_argument("--port", type=int, default=None,
                    help="UDP port (default: the radar's own setting, else %d)" % av.DEFAULT_PORT)
    ap.add_argument("--out-root", type=Path, default=HERE / "dataset")
    ap.add_argument("--session-dir", type=Path,
                    help="record into exactly this new directory instead of <out-root>/altos_session_<time>")
    ap.add_argument("--duration", type=float, help="stop after this many seconds (default: Ctrl+C)")
    ap.add_argument("--rcvbuf", type=int, default=8 * 1024 * 1024,
                    help="requested SO_RCVBUF bytes; the kernel caps it at 2 x net.core.rmem_max")
    ap.add_argument("--queue-items", type=int, default=100_000)
    ap.add_argument("--min-free-gb", type=float, default=2.0)
    ap.add_argument("--first-packet-timeout", type=float, default=5.0)
    a = ap.parse_args()

    settings = read_radar_settings(a.radar_ip)
    install = settings.get("install")
    install = install[0] if isinstance(install, list) and install else {}
    group = a.group or install.get("multicast_addr") or av.DEFAULT_GROUP
    port = a.port or install.get("multicast_port") or av.DEFAULT_PORT

    out_root = a.session_dir.parent if a.session_dir else a.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out_root).free < a.min_free_gb * 1024 ** 3:
        raise SystemExit("Less than %s GB free under %s" % (a.min_free_gb, out_root))

    sock = open_socket(group, port, a.host_ip, a.rcvbuf)
    granted = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    drops_start = kernel_udp_drops(port)

    session = a.session_dir or out_root / ("altos_session_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    session.mkdir()

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    pcap = PcapWriter(session / "altos.pcap", group)
    ff = (session / "frames.csv").open("w", newline="")
    frames_csv = csv.writer(ff)
    frames_csv.writerow(["frame_id", "t_first_monotonic_ns", "t_last_monotonic_ns",
                         "radar_time_s", "points", "packets", "complete"])
    items: queue.Queue = queue.Queue(maxsize=a.queue_items)
    wstate: dict = {}
    writer = threading.Thread(target=writer_loop, args=(pcap, frames_csv, items, wstate),
                              name="altos-writer", daemon=True)
    rx = Receiver(sock, a.radar_ip, port, items, stop)

    mono0, real0 = clock_pair()
    status = settings.get("status", {})
    meta = {
        "sensor": status.get("machine_name", "Altos V4"),
        "serial": status.get("serial_num"), "firmware": status.get("firmware_ver"),
        "radar_ip": a.radar_ip, "host_ip": a.host_ip, "group": group, "port": port,
        "radar_settings": settings,
        "pcap_timestamp_clock": "CLOCK_MONOTONIC (time.monotonic_ns) at host receipt",
        "radar_timestamp": "in every packet header (radar clock, UTC only when PTP-synced)",
        "rcvbuf_requested": a.rcvbuf, "rcvbuf_granted": granted,
        "start_monotonic_ns": mono0, "start_realtime_ns": real0,
        "start_local": datetime.now().astimezone().isoformat(),
        "complete": False,
    }
    save_json(session / "metadata.json", meta)

    writer.start()
    rx.start()
    print("Recording %s %s (fw %s) %s -> %s:%d on %s" % (meta["sensor"], meta["serial"], meta["firmware"],
                                                         a.radar_ip, group, port, a.host_ip))
    print("  profile %s | fps limit %s | threshold %s | sync %s" % (
        install.get("profile"), install.get("fps_limit"),
        (install.get("rpvb_control_overrides") or {}).get("threshold_coeff"), install.get("sync_mode")))
    print("Session: %s" % session)

    error = None
    t_start = time.monotonic()
    with (session / "stats.csv").open("w", newline="") as sf, \
         (session / "clock_map.csv").open("w", newline="") as cf:
        stats = csv.writer(sf)
        stats.writerow(["t_monotonic_ns", "packets", "frames", "points", "frames_lost",
                        "frames_incomplete", "foreign", "queue_drops", "queue_depth"])
        clocks = csv.writer(cf)
        clocks.writerow(["monotonic_ns", "realtime_ns"])
        clocks.writerow([mono0, real0])
        prev = (0, 0, 0, 0, 0, 0, 0)
        next_tick = t_start + 1.0
        while not stop.is_set():
            stop.wait(max(0.0, next_tick - time.monotonic()))
            now = time.monotonic()
            next_tick += 1.0
            clocks.writerow(clock_pair())
            asm = rx.asm
            cur = (rx.packets, asm.frames, rx.points, asm.missing_ids, asm.incomplete,
                   rx.foreign, rx.queue_drops)
            d = [c - p for c, p in zip(cur, prev)]
            stats.writerow([time.monotonic_ns()] + d + [items.qsize()])
            sf.flush()
            cf.flush()
            prev = cur
            print("%6.0fs | %3d frames/s | %5d pkt/s | %5.0f pts/frame | lost %d | incomplete %d | %6.2f MB"
                  % (now - t_start, d[1], d[0], d[2] / d[1] if d[1] else 0.0,
                     asm.missing_ids, asm.incomplete, pcap.bytes / 1e6), flush=True)

            if rx.error or wstate.get("error"):
                error = "receiver/writer failed: %s" % (rx.error or wstate["error"])
            elif rx.packets == 0 and now - t_start > a.first_packet_timeout:
                error = ("no packets from %s on %s:%d after %.0f s; check the radar is on, the "
                         "link is up, and %s is on its port" % (a.radar_ip, group, port,
                                                                a.first_packet_timeout, a.host_ip))
            elif shutil.disk_usage(session).free < a.min_free_gb * 1024 ** 3:
                error = "free space below %s GB; recording stopped" % a.min_free_gb
            elif a.duration and now - t_start >= a.duration:
                break
            if error:
                break

    stop.set()
    rx.join()
    rx.finish()
    kdrops = kernel_udp_drops(port) - drops_start    # before close: closed sockets leave /proc/net/udp
    mono1, real1 = clock_pair()
    items.put(None)
    writer.join()
    pcap.close()
    ff.close()
    sock.close()

    asm = rx.asm
    meta.update({
        "complete": error is None and rx.error is None and "error" not in wstate,
        "error": error,
        "stop_monotonic_ns": mono1, "stop_realtime_ns": real1,
        "duration_s": round((mono1 - mono0) / 1e9, 3),
        "packets": rx.packets, "bad_packets": asm.bad_packets, "foreign_packets": rx.foreign,
        "frames": asm.frames, "frames_lost": asm.missing_ids, "frames_incomplete": asm.incomplete,
        "last_frame_cut_by_stop": asm.truncated_end,
        "out_of_order_packets": asm.out_of_order,
        "points": rx.points,
        "queue_drops": rx.queue_drops,
        "kernel_udp_drops": kdrops,
        "first_frame": {"monotonic_ns": rx.first[0], "radar_s": rx.first[1]} if rx.first else None,
        "last_frame": {"monotonic_ns": rx.last[0], "radar_s": rx.last[1]} if rx.last else None,
        "pcap_bytes": pcap.bytes,
    })
    save_json(session / "metadata.json", meta)

    print("\n%s: %s" % ("Finalized" if meta["complete"] else "INCOMPLETE", session))
    print("  %d frames, %d packets, %d points | frames lost %d | incomplete %d | last frame cut by stop %d | "
          "queue drops %d | kernel drops %d" % (asm.frames, rx.packets, rx.points, asm.missing_ids,
                                                asm.incomplete, asm.truncated_end, rx.queue_drops, kdrops))
    print("  %.2f MB in %s s" % (pcap.bytes / 1e6, meta["duration_s"]))
    if error:
        print("  ERROR: %s" % error)
    return 0 if meta["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
