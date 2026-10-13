"""
dca_capture_radar_sync.py — capture with EXACT loss measurement, and write frames.

WHY THIS EXISTS
    v3 proved the Linux path works: 24,701 packets, all 1466 B, header
    confirmed as seq(4) + byte_count(6) + 1456 B ADC.

    But it received 1,647 pkt/s where the Windows capture ran at 8,658 pkt/s
    - 19%. That is either packet LOSS or a genuinely lower frame rate, and
    v3 counted packets without checking sequence gaps, so it could not tell.

HOW THIS TELLS THE DIFFERENCE
    The DCA stamps every packet with a cumulative byte_count: the number of
    ADC bytes it had sent BEFORE that packet. That is the sender's own
    ledger, so:

        bytes the DCA SENT     = last_byte_count + 1456
        bytes we RECEIVED      = packets * 1456
        loss                   = 1 - received / sent

    Sequence numbers give the same answer independently (expected count =
    last_seq - first_seq + 1). Two independent measures of the same thing:
    if they agree, the number is trustworthy.

    If loss is ~0 and the rate is still low, the radar is simply running
    slower than the Windows session - nothing is wrong with the receiver.

USAGE
    python dca_capture_radar_sync.py --secs 15
    python dca_capture_radar_sync.py --secs 15 --out ./frames    # also write files
"""
from __future__ import annotations
import argparse, csv, os, queue, socket, struct, sys, threading, time

HDR, FTR = 0xA55A, 0xEEAA
CMD_RESET_FPGA, CMD_CONFIG_FPGA = 0x0001, 0x0003
CMD_START_RECORD, CMD_STOP_RECORD = 0x0005, 0x0006
CMD_SYSTEM_ERROR, CMD_SYSTEM_CONNECT, CMD_CONFIG_PACKET = 0x000A, 0x0009, 0x000B

DCA_IP, DCA_PORT, DATA_PORT = "192.168.33.180", 4096, 4098
HDR_BYTES, ADC_BYTES = 10, 1456          # proven on hardware, not assumed
FRAME_BYTES = 256 * 64 * 48 * 4          # 3,145,728

FPGA_CFG = struct.pack("<BBBBBB", 1, 1, 1, 2, 3, 0x1e)
PKT_BYTES = 1470
# CONFIG_PACKET_DATA delay is in 8 ns FPGA ticks (125 per us). Each packet
# costs delay + ~12 us on the wire, so a 2,160.5-packet frame takes
#   50 us (6250, the old value from a Windows capture): 134 ms -> max 7.44 fps
#   25 us (3125, TI default):                            ~80 ms -> fits 8 fps
# If the DCA cannot drain a frame within one frame period it streams
# continuously and drops LVDS data internally - invisible to seq/byte_count.
DEFAULT_PKT_DELAY_US = 25.0


def pkt_cfg(delay_us: float) -> bytes:
    return struct.pack("<HI", PKT_BYTES, int(round(delay_us * 125)))


def make_pkt(cmd, data=b""):
    return struct.pack("<HHH", HDR, cmd, len(data)) + data + struct.pack("<H", FTR)


def send_recv(sock, name, cmd, data=b"", timeout=3.0, no_ack_ok=False):
    sock.settimeout(timeout)
    sock.sendto(make_pkt(cmd, data), (DCA_IP, DCA_PORT))
    try:
        resp, _ = sock.recvfrom(2048)
    except socket.timeout:
        print(f"  {name}: {'no ACK (tolerated)' if no_ack_ok else 'TIMEOUT'}")
        return no_ack_ok
    _, rcmd, val = struct.unpack_from("<HHH", resp, 0)
    if rcmd == CMD_SYSTEM_ERROR:
        print(f"  {name}: DEVICE ERROR code {val}")
        return False
    if rcmd != cmd or val != 0:
        print(f"  {name}: FAIL (cmd 0x{rcmd:04X}, val {val})")
        return False
    print(f"  {name}: OK")
    return True


class FrameWriter(threading.Thread):
    """Writes frame files off the receive thread. A storage stall (eMMC
    stalls of ~200 ms were measured on this Jetson) would otherwise block
    recvfrom() for that long, and at 8 fps the socket buffer is the only
    thing absorbing it. The queue is bounded; if it ever fills, frames are
    counted as overflows rather than blocking the receiver."""

    def __init__(self, outdir, max_frames):
        super().__init__(name="radar-writer", daemon=True)
        self.outdir = outdir
        self.q: queue.Queue = queue.Queue(maxsize=max_frames)
        self.overflows = 0
        self.errors = 0
        self.high_water = 0

    def submit(self, fn, data) -> bool:
        try:
            self.q.put_nowait((fn, data))
        except queue.Full:
            self.overflows += 1
            return False
        self.high_water = max(self.high_water, self.q.qsize())
        return True

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            fn, data = item
            try:
                with open(os.path.join(self.outdir, fn), "wb") as f:
                    f.write(data)
            except OSError as exc:
                self.errors += 1
                print(f"  WRITE ERROR {fn}: {exc}", flush=True)

    def finish(self):
        self.q.put(None)
        self.join()


class Receiver(threading.Thread):
    def __init__(self, outdir=None, rcvbuf=32 * 1024 * 1024, write_queue_frames=160):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.outdir, self.rcvbuf = outdir, rcvbuf
        self.n = 0
        self.first_seq = self.last_seq = None
        self.last_bc = 0
        self.gaps = 0          # number of discontinuities
        self.missing = 0       # packets implied missing by seq jumps
        self.reorder = 0
        self.granted = 0
        self._buf = bytearray()
        self.frames = 0
        # Per-frame timing for camera pairing (Stage 1).
        # t_first: CLOCK_MONOTONIC of the first packet contributing to the
        #          frame. Closest host-observable proxy for acquisition.
        # t_last : the packet that completed it. A frame takes ~54 ms to
        #          transmit (2,161 packets x 25 us), so t_last - t_first is
        #          the transmission span, not jitter. Both are recorded
        #          because Stage 3 must MEASURE which better matches the
        #          camera rather than assume.
        self._t_first = None
        self._t_prev_pkt = None
        self._t_burst = None      # first packet of the CURRENT transmission burst
        # Burst that carried the CURRENT frame's first byte. Measured on
        # hardware (2026-10-09): each burst's first packet starts slightly
        # BEFORE a frame boundary (burst_offset -16..-1440 B), so the packet
        # that completes frame n is the first packet of burst n+1. Stamping a
        # frame with the burst current at completion - as the original
        # receiver did - made t_burst one whole frame period late.
        self._frame_t_burst = None
        self.frame_rows: list = []
        # BYTE PLACEMENT. Frames are sliced from the byte stream. Appending
        # payloads in arrival order meant ONE lost packet shifted every later
        # frame by 1,456 B, so all of them straddled two radar frames. The
        # DCA's byte_count says where each payload belongs: gaps are filled
        # with zeros (and counted per frame), late or duplicate packets are
        # dropped, and frame boundaries stay correct after any loss.
        self._base_bc = None      # byte_count of the first packet = stream origin
        self._next_bc = None      # byte_count the next in-order packet should carry
        self._fills: list = []    # zero-filled [start, end) ranges, stream-relative
        self.zero_filled = 0      # bytes
        self.late_dropped = 0     # packets
        # BURST ALIGNMENT. Each transmission burst should start at a frame
        # boundary. burst_offset is where the latest burst began relative to
        # the nearest boundary, in bytes; within one packet means aligned.
        self._burst_offset = 0
        self.misaligned_bursts = 0
        self.bursts = 0
        self.writer = FrameWriter(outdir, write_queue_frames) if outdir else None

    def _place(self, bc, payload):
        if self._base_bc is None:
            self._base_bc = self._next_bc = bc
        if bc < self._next_bc:
            self.late_dropped += 1
            return
        if bc > self._next_bc:
            gap = bc - self._next_bc
            self._buf += bytes(gap)
            self._fills.append((self._next_bc - self._base_bc, self._next_bc - self._base_bc + gap))
            self.zero_filled += gap
        self._buf += payload
        self._next_bc = bc + len(payload)

    def _lost_in(self, start, end) -> int:
        lost = sum(max(0, min(e, end) - max(s, start)) for s, e in self._fills)
        self._fills = [(s, e) for s, e in self._fills if e > end]
        return lost

    def run(self):
        if self.writer:
            self.writer.start()
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.rcvbuf)
        except OSError:
            pass
        self.granted = s.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        s.settimeout(0.5)
        s.bind(("", DATA_PORT))
        while not self.stop.is_set():
            try:
                p, _ = s.recvfrom(65536)
            except socket.timeout:
                continue
            if len(p) < HDR_BYTES:
                continue
            t_pkt = time.monotonic()      # same clock the camera logs
            seq = struct.unpack_from("<I", p, 0)[0]
            bc = int.from_bytes(p[4:10], "little")
            self.n += 1
            # BURST DETECTION. The DCA does not stream continuously: it sends
            # ~54 ms of packets per frame then idles until the next frame
            # period. Seeding a frame's start with the packet that completed
            # the PREVIOUS frame therefore lands that idle time too early - at
            # 4 fps ~196 ms, a 27 cm bias at 1.4 m/s, about 4 range bins. An
            # inter-packet gap far larger than the 25 us packet delay marks a
            # real burst start.
            new_burst = self._t_prev_pkt is None or (t_pkt - self._t_prev_pkt) > 0.010
            if new_burst:
                self._t_burst = t_pkt
            self._t_prev_pkt = t_pkt
            if self._t_first is None:
                self._t_first = t_pkt
            if self.first_seq is None:
                self.first_seq = seq
            elif seq != self.last_seq + 1:
                if seq > self.last_seq:
                    self.gaps += 1
                    self.missing += seq - self.last_seq - 1
                else:
                    self.reorder += 1
            self.last_seq, self.last_bc = seq, bc
            if self.outdir is None:
                continue
            self._place(bc, p[HDR_BYTES:])
            if self._frame_t_burst is None:
                self._frame_t_burst = self._t_burst
            if new_burst:
                r = (bc - self._base_bc) % FRAME_BYTES
                self._burst_offset = r if r <= FRAME_BYTES // 2 else r - FRAME_BYTES
                self.bursts += 1
                if abs(self._burst_offset) > ADC_BYTES:
                    self.misaligned_bursts += 1
            while len(self._buf) >= FRAME_BYTES:
                fn = f"frame_{self.frames:06d}.bin"
                start = self.frames * FRAME_BYTES
                lost = self._lost_in(start, start + FRAME_BYTES)
                self.writer.submit(fn, bytes(self._buf[:FRAME_BYTES]))
                del self._buf[:FRAME_BYTES]
                self.frame_rows.append(
                    (self.frames, fn, self._t_first, t_pkt, seq, self._frame_t_burst,
                     lost, self._burst_offset))
                self.frames += 1
                # Bytes left in the buffer belong to the next frame and came
                # in this packet, so its burst is the next frame's burst. If
                # the boundary fell exactly at the packet end, the next
                # packet sets it.
                self._frame_t_burst = self._t_burst if self._buf else None
                # A frame spans 2,160.5 packets, so boundaries fall
                # MID-PACKET: this packet also carries the start of the
                # next frame. Seeding t_first with it is correct, not an
                # off-by-one.
                self._t_first = t_pkt
        s.close()
        if self.writer:
            self.writer.finish()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=15.0,
                    help="recording length; 0 = until Ctrl+C / SIGINT")
    ap.add_argument("--out", default=None, help="write frame_*.bin here")
    ap.add_argument("--write-queue-frames", type=int, default=160,
                    help="frames buffered for the disk writer (~20 s at 8 fps, ~500 MB)")
    ap.add_argument("--arm-and-wait", action="store_true",
                    help="Arm the DCA (config + START_RECORD) and hold, so the "
                         "radar can be triggered AFTERWARDS from Windows. Use "
                         "this whenever radar start and capture start are on "
                         "different machines.")
    ap.add_argument("--packet-delay-us", type=float, default=DEFAULT_PKT_DELAY_US,
                    help="DCA inter-packet delay; 50 us caps the stream at 7.44 fps")
    a = ap.parse_args()

    if a.out:
        os.makedirs(a.out, exist_ok=True)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SO_REUSEPORT"):
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    s.bind(("", DCA_PORT))          # bind, not connect: DCA replies from :1024

    rx = None
    try:
        send_recv(s, "RESET_FPGA", CMD_RESET_FPGA, no_ack_ok=True)
        time.sleep(3.0)
        if not send_recv(s, "CONFIG_FPGA", CMD_CONFIG_FPGA, FPGA_CFG):
            return 1
        time.sleep(1.0)
        send_recv(s, "SYSTEM_CONNECT", CMD_SYSTEM_CONNECT)
        time.sleep(0.5)
        if not send_recv(s, "CONFIG_PACKET_DATA", CMD_CONFIG_PACKET, pkt_cfg(a.packet_delay_us)):
            return 1
        time.sleep(0.5)
        # Second status query. The Windows capture issues 0x09 three times
        # (frames 5, 9, 11); v3 issued two and streamed successfully. v4
        # originally dropped one while tidying - an unforced deviation from a
        # known-good sequence. Restored.
        send_recv(s, "SYSTEM_CONNECT (pre-record)", CMD_SYSTEM_CONNECT)
        time.sleep(0.5)

        rx = Receiver(a.out, write_queue_frames=a.write_queue_frames); rx.start(); time.sleep(0.3)
        print(f"  SO_RCVBUF granted: {rx.granted:,} B")
        if rx.granted < 8 * 1024 * 1024:
            print("  WARNING: small receive buffer. "
                  "sudo sysctl -w net.core.rmem_max=33554432")

        t0 = time.time()
        if not send_recv(s, "START_RECORD", CMD_START_RECORD):
            return 1

        if a.arm_and_wait:
            # The DCA forwards to Ethernet ONLY while recording, and the radar
            # emits a FINITE burst starting at firmware load. With radar start
            # and capture start on different machines, the human gap between
            # them (switch machines + 5.5 s of DCA setup) can exceed the burst
            # entirely - which is why a run that worked once then returned
            # zero. Arming first removes the race: the DCA sits recording and
            # catches the burst whenever it arrives.
            print("\n" + "=" * 72)
            print("  DCA IS ARMED AND RECORDING - now trigger the radar")
            print("=" * 72)
            print("  On the Windows machine, run:")
            print("      python test.py --load-only")
            print(f"\n  Listening up to {a.secs:.0f} s. Stops early once a burst")
            print("  has arrived and then gone quiet for 3 s.\n")

        print(f"  (per-second packet counts)")
        # Print per second. A FINITE frame burst shows as a few busy seconds
        # then zeros - which is invisible in a single total and was why the
        # v3 result looked like 81% loss rather than a short burst.
        last, quiet, seen, i = 0, 0, False, 0
        # Ctrl+C / SIGINT ends the recording CLEANLY: STOP_RECORD must still
        # reach the DCA (or it stays armed for the next session) and the
        # manifest must still be written. Sessions run until the operator
        # stops them, so this is the normal exit, not an error path.
        try:
            while a.secs <= 0 or i < int(a.secs):
                time.sleep(1.0)
                i += 1
                d = rx.n - last
                last = rx.n
                bar = "#" * min(40, d // 200)
                print(f"    t+{i:>3}s  {d:>7,} pkt/s  {bar}")
                if d > 0:
                    seen, quiet = True, 0
                elif seen:
                    quiet += 1
                    if quiet >= 3:
                        print("    burst complete (3 s quiet) - stopping early")
                        break
        except KeyboardInterrupt:
            print("    interrupted - stopping the recording")
        send_recv(s, "STOP_RECORD", CMD_STOP_RECORD)
        elapsed = time.time() - t0
    finally:
        if rx:
            # The receiver drains the writer queue before it exits.
            rx.stop.set(); rx.join(timeout=120.0)
        s.close()

    print("\n" + "=" * 72)
    print("  LOSS MEASUREMENT")
    print("=" * 72)
    if not rx or rx.n == 0:
        print("  no packets received")
        return 1

    recv_adc = rx.n * ADC_BYTES
    sent_adc = rx.last_bc + ADC_BYTES          # DCA's own ledger
    exp_pkts = rx.last_seq - rx.first_seq + 1  # sequence-number ledger

    print(f"  packets received   : {rx.n:,}")
    print(f"  seq range          : {rx.first_seq} .. {rx.last_seq}")
    print(f"  packets expected   : {exp_pkts:,}   (from sequence numbers)")
    print(f"  sequence gaps      : {rx.gaps}   ({rx.missing:,} packets implied missing)")
    print(f"  out-of-order       : {rx.reorder}")
    print()
    print(f"  ADC bytes received : {recv_adc:,}")
    print(f"  ADC bytes SENT     : {sent_adc:,}   (from the DCA's byte_count)")

    loss_bc = (1 - recv_adc / sent_adc) * 100 if sent_adc else 0
    loss_sq = (1 - rx.n / exp_pkts) * 100 if exp_pkts else 0
    print(f"\n  loss by byte_count : {loss_bc:6.2f}%")
    print(f"  loss by sequence   : {loss_sq:6.2f}%")
    agree = abs(loss_bc - loss_sq) < 0.5
    print(f"  the two agree      : {'yes' if agree else 'NO - investigate'}")

    print(f"\n  elapsed            : {elapsed:.1f} s")
    print(f"  received rate      : {recv_adc/elapsed/1e6:.2f} MB/s "
          f"= {recv_adc/FRAME_BYTES/elapsed:.2f} fps")
    print(f"  DCA send rate      : {sent_adc/elapsed/1e6:.2f} MB/s "
          f"= {sent_adc/FRAME_BYTES/elapsed:.2f} fps")
    write_problems = False
    if a.out:
        print(f"  frames written     : {rx.frames}  -> {a.out}")
        print(f"  zero-filled        : {rx.zero_filled:,} B   (lost packets, placed by byte_count)")
        print(f"  late/dup dropped   : {rx.late_dropped}")
        print(f"  writer             : queue peak {rx.writer.high_water} frames, "
              f"overflows {rx.writer.overflows}, errors {rx.writer.errors}")
        print(f"  burst alignment    : {rx.misaligned_bursts} of {rx.bursts} bursts "
              f"start > 1 packet from a frame boundary")
        if rx.frames > 4 and rx.bursts < rx.frames / 2:
            print(f"  WARNING: the DCA streamed with no idle gaps ({rx.bursts} bursts for {rx.frames} frames). "
                  f"It cannot send a frame within one frame period, so it is DROPPING radar data "
                  f"internally (not visible in seq/byte_count). Lower --packet-delay-us "
                  f"(now {a.packet_delay_us:g}) or the frame rate.")
        if rx.misaligned_bursts:
            print("  WARNING: frame boundaries do not match transmission bursts - "
                  "frames may straddle two radar frames. Check before using this data.")
        write_problems = bool(rx.writer.overflows or rx.writer.errors)
        # Sidecar manifest consumed by the Stage 4 association tool. Written
        # here rather than in the receive loop so file I/O never sits between
        # two recvfrom() calls.
        man = os.path.join(a.out, "frames.csv")
        with open(man, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            # t_burst is the RECOMMENDED pairing timestamp: the first packet
            # of the transmission burst carrying this frame, i.e. the
            # closest host-observable proxy for acquisition. t_first and
            # t_last are retained so Stage 3 can compare all three
            # against the camera rather than trusting this choice.
            # lost_bytes: zero-filled bytes inside this frame (0 = complete).
            # burst_offset_bytes: where the latest burst began relative to the
            # nearest frame boundary; |value| <= 1456 means aligned.
            w.writerow(["idx", "filename", "t_burst_monotonic",
                        "t_first_monotonic", "t_last_monotonic", "last_seq",
                        "lost_bytes", "burst_offset_bytes"])
            for idx, fn, t0, t1, sq, tb, lost, boff in rx.frame_rows:
                w.writerow([idx, fn, f"{(tb if tb else t0):.6f}",
                            f"{t0:.6f}", f"{t1:.6f}", sq, lost, boff])
        print(f"  manifest           : {man}")
        if rx.frame_rows:
            import statistics as _st
            bursts = [(tb if tb else t0) for _, _, t0, _, _, tb, _, _ in rx.frame_rows]
            spans = [t1 - (tb if tb else t0)
                     for _, _, t0, t1, _, tb, _, _ in rx.frame_rows]
            gaps = [b - a_ for a_, b in zip(bursts, bursts[1:])]
            # The completing packet opens the NEXT burst, so this is about one
            # frame period, not the transmission time.
            print(f"  frame span         : median {_st.median(spans)*1000:.1f} ms "
                  f"(own burst start -> completing packet, in the next burst)")
            if gaps and _st.median(gaps) > 0:
                print(f"  frame interval     : median {_st.median(gaps)*1000:.1f} ms "
                      f"(= {1/_st.median(gaps):.2f} fps)")

    print("\n" + "=" * 72)
    print("  INTERPRETATION")
    print("=" * 72)
    if loss_bc < 0.1:
        print(f"""  Loss is negligible. The receiver is keeping up.
  The DCA itself sent only {sent_adc/FRAME_BYTES/elapsed:.2f} fps worth of data, so the low
  throughput is the RADAR's frame rate, not a Linux problem. Compare the
  chirp/frame config against the Windows session that produced 4 fps.""")
    elif loss_bc < 5:
        print(f"""  Modest loss ({loss_bc:.2f}%). Receiver is mostly keeping up but not
  fully. Raise SO_RCVBUF, and consider moving frame writing off the
  receive thread.""")
    else:
        print(f"""  HEAVY loss ({loss_bc:.2f}%). The receive path cannot keep up.
  In order: raise net.core.rmem_max; drop file writing from the receive
  thread (queue to a writer thread); pin the receiver to a core; and if
  it persists, move the hot loop out of Python.""")
    if write_problems:
        print("\n  FRAME FILES MISSING: the disk writer overflowed or failed (see above).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
