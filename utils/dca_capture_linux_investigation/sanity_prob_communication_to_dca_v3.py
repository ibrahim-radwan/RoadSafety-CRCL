"""
sanity_prob_communication_to_dca_v3.py — corrected against the Windows capture.

CHANGES FROM v2, each justified by radar_full_capture_2026-08-31.pcapng
-----------------------------------------------------------------------
1. CONFIG_FPGA_GEN payload: SIX single bytes, not '<BBBBH'.
       v2 sent      01 01 01 03 32 00   (5 fields, packetDelay as uint16)
       capture has  01 01 01 02 03 1e   (6 single bytes)
   v2 omitted dataCaptureMode entirely and shifted every later field.
   dataCaptureMode=2 is what puts data on ETHERNET instead of the SD card,
   so without it the board is never told to stream. This is the primary
   reason nothing appeared on port 4098.

2. CONFIG_PACKET_DATA payload: '<HI' (packetSize, packetDelay), not '<IH'.
       v2 sent      00 00 00 00 00 04   -> packetSize 0
       capture has  be 05 6a 18 00 00   -> packetSize 1470, delay 6250
   0x0b configures UDP packet GEOMETRY, not record-file limits. A zero
   packet size cannot produce a stream.

3. parse_ack now verifies the reply command matches the request, and
   treats 0x000A as an error notification.
       v2 reported "OK (short payload, treating as success)" for
       5aa5 0a00 0300 aaee - an ERROR message with code 3 - because it
       never compared the reply command against the request.

4. Optional --listen binds UDP :4098 during the record window so the
   result is observable without a second terminal.

USAGE
    python sanity_prob_communication_to_dca_v3.py --listen
    python sanity_prob_communication_to_dca_v3.py --listen --secs 5
"""
from __future__ import annotations
import argparse, socket, struct, threading, time

HDR, FTR = 0xA55A, 0xEEAA

CMD_RESET_FPGA     = 0x0001
CMD_CONFIG_FPGA    = 0x0003
CMD_START_RECORD   = 0x0005
CMD_STOP_RECORD    = 0x0006
CMD_SYSTEM_ERROR   = 0x000A      # asynchronous error notification, never an ACK
CMD_SYSTEM_CONNECT = 0x0009
CMD_CONFIG_PACKET  = 0x000B

DCA_IP, DCA_PORT, DATA_PORT = "192.168.33.180", 4096, 4098

# ── Payloads, byte-for-byte from the Windows capture ─────────────────────────
# CONFIG_FPGA_GEN: logging=raw(1), lvds=4-lane(1), transfer=LVDSCapture(1),
#                  capture=ETHERNET(2), format=16-bit(3), timeout=30 s(0x1e)
FPGA_CFG = struct.pack("<BBBBBB", 1, 1, 1, 2, 3, 0x1e)
# CONFIG_PACKET_DATA: packetSize=1470 (uint16), packetDelay=6250 (uint32)
PKT_CFG  = struct.pack("<HI", 1470, 6250)


def make_pkt(cmd: int, data: bytes = b"") -> bytes:
    return struct.pack("<HHH", HDR, cmd, len(data)) + data + struct.pack("<H", FTR)


def parse_ack(resp: bytes, req_cmd: int, name: str) -> bool:
    if len(resp) < 8:
        print(f"  {name}: ERROR - response too short ({len(resp)} B)")
        return False
    hdr, cmd, val = struct.unpack_from("<HHH", resp, 0)
    if hdr != HDR:
        print(f"  {name}: ERROR - bad header 0x{hdr:04X}")
        return False
    if cmd == CMD_SYSTEM_ERROR:
        print(f"  {name}: DEVICE ERROR - cmd 0x000A, code {val}")
        print(f"           (v2 reported this as OK; it is an error notification)")
        return False
    if cmd != req_cmd:
        print(f"  {name}: ERROR - reply cmd 0x{cmd:04X} != request 0x{req_cmd:04X}")
        return False
    if val != 0:
        print(f"  {name}: ERROR - status {val}")
        return False
    print(f"  {name}: OK")
    return True


def send_recv(sock, name, cmd, data=b"", timeout=3.0, no_ack_ok=False) -> bool:
    pkt = make_pkt(cmd, data)
    print(f"\n[{name}]  tx: {pkt.hex()}")
    sock.settimeout(timeout)
    sock.sendto(pkt, (DCA_IP, DCA_PORT))
    try:
        resp, addr = sock.recvfrom(2048)
        print(f"  rx from {addr}: {resp.hex()}")
        return parse_ack(resp, cmd, name)
    except socket.timeout:
        if no_ack_ok:
            print(f"  {name}: no ACK (tolerated)")
            return True
        print(f"  {name}: TIMEOUT")
        return False


class Listener(threading.Thread):
    """Bind :4098 for the record window so the outcome is directly observable."""
    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.n = self.nbytes = 0
        self.first = None
        self.sizes: set[int] = set()

    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 32 MB: at 8 fps the stream is ~25 MB/s, and the default (~200 KB)
        # overflows in milliseconds if the reader stalls at all.
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32 * 1024 * 1024)
        except OSError:
            pass
        s.settimeout(0.5)
        s.bind(("", DATA_PORT))
        while not self.stop.is_set():
            try:
                d, _ = s.recvfrom(65536)
            except socket.timeout:
                continue
            self.n += 1
            self.nbytes += len(d)
            self.sizes.add(len(d))
            if self.first is None:
                self.first = d[:16]
        s.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", action="store_true",
                    help="bind UDP :4098 during the record window")
    ap.add_argument("--secs", type=float, default=10.0)
    a = ap.parse_args()

    print(f"FPGA config  ({len(FPGA_CFG)} B): {FPGA_CFG.hex()}   "
          f"(capture: 01010102031e)")
    print(f"Packet config({len(PKT_CFG)} B): {PKT_CFG.hex()}   "
          f"(capture: be056a180000)")

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SO_REUSEPORT"):
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    # bind, not connect: the DCA replies from port 1024, so a connected
    # socket would silently never see the responses.
    s.bind(("", DCA_PORT))

    lis = None
    try:
        send_recv(s, "RESET_FPGA", CMD_RESET_FPGA, no_ack_ok=True)
        print("  sleeping 3 s for FPGA re-init ...")
        time.sleep(3.0)

        if not send_recv(s, "CONFIG_FPGA", CMD_CONFIG_FPGA, FPGA_CFG):
            print("\nFATAL: CONFIG_FPGA rejected.")
            return 1
        time.sleep(1.0)
        send_recv(s, "SYSTEM_CONNECT (post-config)", CMD_SYSTEM_CONNECT)
        time.sleep(0.5)
        if not send_recv(s, "CONFIG_PACKET_DATA", CMD_CONFIG_PACKET, PKT_CFG):
            print("\nFATAL: CONFIG_PACKET_DATA rejected.")
            return 1
        time.sleep(0.5)
        send_recv(s, "SYSTEM_CONNECT (pre-record)", CMD_SYSTEM_CONNECT)
        time.sleep(0.5)

        if a.listen:
            lis = Listener(); lis.start(); time.sleep(0.3)
            print(f"\n  listening on UDP :{DATA_PORT} ...")

        if not send_recv(s, "START_RECORD", CMD_START_RECORD):
            print("\nFATAL: START_RECORD rejected.")
            return 1

        print(f"\n  recording for {a.secs:.0f} s ...")
        time.sleep(a.secs)

        ok_stop = send_recv(s, "STOP_RECORD", CMD_STOP_RECORD)
    finally:
        if lis:
            lis.stop.set(); lis.join(timeout=3.0)
        s.close()

    if lis:
        print("\n" + "=" * 66)
        print("  DATA PLANE")
        print("=" * 66)
        print(f"  packets   : {lis.n:,}")
        print(f"  bytes     : {lis.nbytes:,}")
        print(f"  sizes seen: {sorted(lis.sizes) or 'none'}")
        if lis.first:
            seq = struct.unpack_from("<I", lis.first, 0)[0]
            bc = int.from_bytes(lis.first[4:10], "little")
            print(f"  first 16 B: {lis.first.hex()}")
            print(f"    seq       = {seq}   (expect a small number, then +1 per packet)")
            print(f"    byte_count= {bc}   (48-bit, 6 bytes)")
        if lis.n:
            print(f"\n  Expect payload 1466 B = 10 header + 1456 ADC.")
            print(f"  If sizes show 1466, the packet geometry is confirmed.")
        else:
            print("\n  NO DATA. Next checks, in order:")
            print("    1. Is the Jetson's IP 192.168.33.30?  ip addr show")
            print("       The DCA streams to the address in its EEPROM.")
            print("    2. Firewall: sudo iptables -I INPUT -p udp --dport 4098 -j ACCEPT")
            print("    3. Is the radar actually streaming LVDS? (sensorStart accepted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
