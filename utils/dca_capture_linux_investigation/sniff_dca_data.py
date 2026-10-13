"""
sniff_dca_data.py — Passive UDP listener to find where DCA1000 is streaming data.

This script does NOT need root/sudo. It binds a UDP socket to 0.0.0.0:4098
(all interfaces) and waits for DCA1000 data packets. It also tries ports
4096 and 4099 in case the destination port is misconfigured.

Run this WHILE a capture is in progress (start_record has been sent):
    # Terminal 1: start capture (it will show "Waiting for firmware...")
    python capture/test.py --frames 10 --uart /dev/ttyACM0 --skip-load

    # Terminal 2: run this sniffer simultaneously
    python helper_utils/sniff_dca_data.py

If data appears on a port or from a source you don't expect, that tells
you what the DCA EEPROM destination is currently set to.
"""
from __future__ import annotations

import socket
import struct
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capture.dca1000_udp import (
    DCA1000UDP,
    CMD_RESET_FPGA,
    CMD_START_RECORD,
    CMD_STOP_RECORD,
    load_dca_network_params,
    _fpga_config_bytes,
    _record_config_bytes,
)
from capture.config import DCA_CAPTURE_JSON
import json

LISTEN_PORTS = [4098, 4096, 4099]
LISTEN_SECS  = 30


def listen_on_port(port: int, results: dict, stop: threading.Event) -> None:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.settimeout(1.0)
        sock.bind(("", port))
        results[port] = {"bound": True, "packets": 0, "sources": set(), "bytes": 0}
        print(f"  Listening on UDP 0.0.0.0:{port} ...")
        while not stop.is_set():
            try:
                data, addr = sock.recvfrom(65536)
                results[port]["packets"] += 1
                results[port]["bytes"]   += len(data)
                results[port]["sources"].add(addr[0])
                if results[port]["packets"] <= 3:
                    print(f"  [port {port}] PKT from {addr[0]}:{addr[1]}  len={len(data)}  "
                          f"first8={data[:8].hex()}", flush=True)
            except socket.timeout:
                continue
    except OSError as exc:
        results[port] = {"bound": False, "error": str(exc)}
        print(f"  [port {port}] Could not bind: {exc}", file=sys.stderr)
    finally:
        try:
            sock.close()
        except Exception:
            pass


def main() -> int:
    params = load_dca_network_params(DCA_CAPTURE_JSON)
    raw    = json.loads(DCA_CAPTURE_JSON.read_text(encoding="utf-8"))
    cfg    = raw.get("DCA1000Config", {})

    print("=== DCA1000 Data Plane Sniffer ===")
    print(f"  DCA1000 IP  : {params['dca_ip']}")
    print(f"  Host IP     : {params['local_ip']}")
    print(f"  Listening for {LISTEN_SECS}s on ports: {LISTEN_PORTS}")
    print()
    print("  Starting DCA record sequence now so data flows ...")

    # Fire start_record via UDP
    udp = DCA1000UDP(params["dca_ip"], params["config_port"], params["local_ip"],
                     timeout=5.0)
    try:
        udp.open()
        udp.reset_fpga()
        time.sleep(3.0)
        udp.config_fpga(_fpga_config_bytes(cfg))
        time.sleep(1.0)
        udp.system_connect()
        time.sleep(0.5)
        cap = cfg.get("captureConfig", {})
        udp.config_record(_record_config_bytes(cap))
        time.sleep(0.5)
        ok = udp.start_record()
        if not ok:
            print("  WARNING: start_record FAILED — data may not flow", file=sys.stderr)
    except Exception as exc:
        print(f"  DCA control error: {exc}", file=sys.stderr)
    finally:
        udp.close()

    print(f"\n  Now listening for {LISTEN_SECS}s — look for incoming packets ...\n")

    stop  = threading.Event()
    results: dict = {}
    threads = []
    for port in LISTEN_PORTS:
        t = threading.Thread(target=listen_on_port, args=(port, results, stop), daemon=True)
        t.start()
        threads.append(t)

    time.sleep(LISTEN_SECS)
    stop.set()
    for t in threads:
        t.join(timeout=3.0)

    # Stop the record
    try:
        udp2 = DCA1000UDP(params["dca_ip"], params["config_port"], params["local_ip"],
                          timeout=3.0)
        with udp2:
            udp2.stop_record()
    except Exception:
        pass

    print("\n=== Sniffer Results ===")
    data_found = False
    for port, info in results.items():
        if not info.get("bound"):
            print(f"  Port {port}: could not bind ({info.get('error', '?')})")
            continue
        pkts = info["packets"]
        src  = info["sources"]
        byt  = info["bytes"]
        if pkts > 0:
            data_found = True
            print(f"  Port {port}: {pkts} packets  {byt:,} B  from {src}")
        else:
            print(f"  Port {port}: 0 packets")

    print()
    if not data_found:
        print("  NO data received on any port.")
        print("  Possible causes:")
        print("    1. DCA1000 EEPROM destination IP is a different machine")
        print("       → Run: python helper_utils/provision_dca_eeprom.py")
        print("          then power-cycle DCA1000")
        print("    2. Radar firmware (AWR2243) not streaming LVDS")
        print("       → Check sensorStart was accepted without Error")
        print("    3. Firewall blocking UDP 4098 inbound")
        print("       → Run: sudo iptables -I INPUT -p udp --dport 4098 -j ACCEPT")
    else:
        print("  Data received! The capture pipeline should work.")
        print("  If capture/test.py still shows 0 packets, check the socket binding in DCA1000DataReceiver.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
