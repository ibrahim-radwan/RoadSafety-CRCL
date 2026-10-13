import socket, struct, time

HDR = 0xA55A
FTR = 0xEEAA

CMD_RESET_FPGA     = 0x0001
CMD_CONFIG_FPGA    = 0x0003
CMD_START_RECORD   = 0x0005
CMD_STOP_RECORD    = 0x0006
CMD_SYSTEM_CONNECT = 0x0009
CMD_CONFIG_PACKET  = 0x000B

DCA_IP   = "192.168.33.180"
DCA_PORT = 4096

def make_pkt(cmd, data=b""):
    return struct.pack("<HHH", HDR, cmd, len(data)) + data + struct.pack("<H", FTR)

def parse_ack(resp, cmd_name):
    if len(resp) < 6:
        print(f"  {cmd_name}: ERROR — response too short ({len(resp)} B)")
        return False
    hdr  = struct.unpack_from("<H", resp, 0)[0]
    cmd  = struct.unpack_from("<H", resp, 2)[0]
    dlen = struct.unpack_from("<H", resp, 4)[0]
    if hdr != HDR:
        print(f"  {cmd_name}: ERROR — bad header 0x{hdr:04X}")
        return False
    if dlen == 0:
        print(f"  {cmd_name}: OK (empty-echo)")
        return True
    if dlen >= 2 and len(resp) >= 6 + dlen:
        status = struct.unpack_from("<H", resp, 6)[0]
        ok = (status == 0)
        print(f"  {cmd_name}: {'OK' if ok else 'ERROR'} (status=0x{status:04X})")
        return ok
    print(f"  {cmd_name}: OK (short payload, treating as success)")
    return True

def send_recv(sock, cmd_name, cmd_code, data=b"", timeout=3.0, no_ack_ok=False):
    pkt = make_pkt(cmd_code, data)
    print(f"\n[{cmd_name}]  tx: {pkt.hex()}")
    sock.settimeout(timeout)
    sock.sendto(pkt, (DCA_IP, DCA_PORT))
    try:
        resp, addr = sock.recvfrom(2048)
        print(f"  rx from {addr}: {resp.hex()}")
        return parse_ack(resp, cmd_name)
    except socket.timeout:
        if no_ack_ok:
            print(f"  {cmd_name}: no ACK (expected after reset — continuing)")
            return True
        print(f"  {cmd_name}: TIMEOUT")
        return False

# ── FPGA config payload (6 bytes per TI SPRUIJ4) ─────────────────────────────
# dataLoggingMode=raw(1), lvdsMode=1, dataTransferMode=LVDSCapture(1),
# dataFormatMode=3 (16-bit), packetDelay_us=50
fpga_cfg = struct.pack("<BBBBH", 1, 1, 1, 3, 50)
print(f"FPGA config bytes ({len(fpga_cfg)} B): {fpga_cfg.hex()}")

# ── Record config payload (6 bytes per TI SPRUIJ4) ───────────────────────────
# bytesToCapture=0 (infinite), maxRecFileSize_MB=1024
rec_cfg = struct.pack("<IH", 0, 1024)
print(f"Record config bytes ({len(rec_cfg)} B): {rec_cfg.hex()}")

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
if hasattr(socket, "SO_REUSEPORT"):
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
s.bind(("", DCA_PORT))

try:
    # 1. Reset FPGA — ACK may not arrive (FPGA resets mid-send)
    send_recv(s, "RESET_FPGA", CMD_RESET_FPGA, no_ack_ok=True)
    print("  Sleeping 3s for FPGA re-init...")
    time.sleep(3.0)

    # 2. Configure FPGA
    ok = send_recv(s, "CONFIG_FPGA", CMD_CONFIG_FPGA, fpga_cfg)
    if not ok:
        print("\nFATAL: CONFIG_FPGA failed — check lvdsMode / cable / power")
        raise SystemExit(1)
    time.sleep(1.0)

    # 3. Query status
    send_recv(s, "SYSTEM_CONNECT (post-config)", CMD_SYSTEM_CONNECT)
    time.sleep(0.5)

    # 4. Configure record
    send_recv(s, "CONFIG_PACKET_DATA", CMD_CONFIG_PACKET, rec_cfg)
    time.sleep(0.5)

    # 5. Query status again
    send_recv(s, "SYSTEM_CONNECT (pre-record)", CMD_SYSTEM_CONNECT)
    time.sleep(0.5)

    # 6. Start record
    ok = send_recv(s, "START_RECORD", CMD_START_RECORD)
    if not ok:
        print("\nFATAL: START_RECORD failed")
        raise SystemExit(1)

    print("\n=== DCA is now recording — check tcpdump on port 4098 ===")
    print("Run in another terminal:")
    print("  sudo tcpdump -i eth1 udp port 4098 -c 20 -v")
    print("\nWaiting 10s then sending STOP_RECORD...")
    time.sleep(10.0)

    # 7. Stop record
    send_recv(s, "STOP_RECORD", CMD_STOP_RECORD, no_ack_ok=True)
    print("\nDone.")

finally:
    s.close()
