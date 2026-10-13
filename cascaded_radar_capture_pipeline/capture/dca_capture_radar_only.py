"""
dca_capture_radar_only.py — capture with EXACT loss measurement, and write frames.

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
    python dca_capture_radar_only.py --secs 15
    python dca_capture_radar_only.py --secs 15 --out ./frames    # also write files
"""
from __future__ import annotations
import argparse, os, socket, struct, sys, threading, time

HDR, FTR = 0xA55A, 0xEEAA
CMD_RESET_FPGA, CMD_CONFIG_FPGA = 0x0001, 0x0003
CMD_START_RECORD, CMD_STOP_RECORD = 0x0005, 0x0006
CMD_SYSTEM_ERROR, CMD_SYSTEM_CONNECT, CMD_CONFIG_PACKET = 0x000A, 0x0009, 0x000B

DCA_IP, DCA_PORT, DATA_PORT = "192.168.33.180", 4096, 4098
HDR_BYTES, ADC_BYTES = 10, 1456          # proven on hardware, not assumed
FRAME_BYTES = 256 * 64 * 48 * 4          # 3,145,728

FPGA_CFG = struct.pack("<BBBBBB", 1, 1, 1, 2, 3, 0x1e)
PKT_CFG  = struct.pack("<HI", 1470, 6250)


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


class Receiver(threading.Thread):
    def __init__(self, outdir=None, rcvbuf=32 * 1024 * 1024):
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

    def run(self):
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
            seq = struct.unpack_from("<I", p, 0)[0]
            bc = int.from_bytes(p[4:10], "little")
            self.n += 1
            if self.first_seq is None:
                self.first_seq = seq
            elif seq != self.last_seq + 1:
                if seq > self.last_seq:
                    self.gaps += 1
                    self.missing += seq - self.last_seq - 1
                else:
                    self.reorder += 1
            self.last_seq, self.last_bc = seq, bc
            if self.outdir is not None:
                self._buf += p[HDR_BYTES:]
                while len(self._buf) >= FRAME_BYTES:
                    fn = os.path.join(self.outdir, f"frame_{self.frames:06d}.bin")
                    with open(fn, "wb") as f:
                        f.write(self._buf[:FRAME_BYTES])
                    del self._buf[:FRAME_BYTES]
                    self.frames += 1
        s.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=15.0)
    ap.add_argument("--out", default=None, help="write frame_*.bin here")
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
        if not send_recv(s, "CONFIG_PACKET_DATA", CMD_CONFIG_PACKET, PKT_CFG):
            return 1
        time.sleep(0.5)
        # Second status query. The Windows capture issues 0x09 three times
        # (frames 5, 9, 11); v3 issued two and streamed successfully. v4
        # originally dropped one while tidying - an unforced deviation from a
        # known-good sequence. Restored.
        send_recv(s, "SYSTEM_CONNECT (pre-record)", CMD_SYSTEM_CONNECT)
        time.sleep(0.5)

        rx = Receiver(a.out); rx.start(); time.sleep(0.3)
        print(f"  SO_RCVBUF granted: {rx.granted:,} B")
        if rx.granted < 8 * 1024 * 1024:
            print("  WARNING: small receive buffer. "
                  "sudo sysctl -w net.core.rmem_max=33554432")

        t0 = time.time()
        if not send_recv(s, "START_RECORD", CMD_START_RECORD):
            return 1
        print(f"\n  recording {a.secs:.0f} s ... (per-second packet counts)")
        # Print per second. A FINITE frame burst shows as a few busy seconds
        # then zeros - which is invisible in a single total and was why the
        # v3 result looked like 81% loss rather than a short burst.
        last = 0
        for i in range(int(a.secs)):
            time.sleep(1.0)
            d = rx.n - last
            last = rx.n
            bar = "#" * min(40, d // 200)
            print(f"    t+{i+1:>3}s  {d:>7,} pkt/s  {bar}")
        send_recv(s, "STOP_RECORD", CMD_STOP_RECORD)
        elapsed = time.time() - t0
    finally:
        if rx:
            rx.stop.set(); rx.join(timeout=5.0)
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
    if a.out:
        print(f"  frames written     : {rx.frames}  -> {a.out}")

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
