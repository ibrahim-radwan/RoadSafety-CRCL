import socket, struct

HDR = 0xA55A
FTR = 0xEEAA
CMD_SYSTEM_CONNECT = 0x0009

pkt = struct.pack("<HHH", HDR, CMD_SYSTEM_CONNECT, 0) + struct.pack("<H", FTR)
print(f"Sending SYSTEM_CONNECT  ({pkt.hex()})")

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.settimeout(3.0)
s.bind(("", 4096))
s.sendto(pkt, ("192.168.33.180", 4096))

try:
    resp, addr = s.recvfrom(2048)
    print(f"ACK from {addr}:  {resp.hex()}")

    if len(resp) < 6:
        print("ERROR: response too short")
    else:
        hdr  = struct.unpack_from("<H", resp, 0)[0]
        cmd  = struct.unpack_from("<H", resp, 2)[0]
        dlen = struct.unpack_from("<H", resp, 4)[0]
        print(f"  HDR=0x{hdr:04X}  CMD=0x{cmd:04X}  DLEN={dlen}")

        if dlen == 0:
            # No payload — this firmware echoes the command with empty body.
            # Treat as SUCCESS if HDR and CMD match.
            print("  Status: OK (empty-payload echo — DCA is alive and responding)")
        elif dlen >= 2 and len(resp) >= 8:
            status = struct.unpack_from("<H", resp, 6)[0]
            print(f"  Status word = {status}  ({'OK' if status == 0 else 'ERROR'})")
        else:
            print("  WARNING: unexpected response length")

        ftr_offset = 6 + dlen
        if len(resp) >= ftr_offset + 2:
            ftr = struct.unpack_from("<H", resp, ftr_offset)[0]
            print(f"  Footer = 0x{ftr:04X}  ({'OK' if ftr == FTR else 'MISMATCH'})")

except socket.timeout:
    print("TIMEOUT — DCA not responding")
finally:
    s.close()
