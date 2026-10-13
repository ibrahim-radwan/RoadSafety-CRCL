"""
provision_dca_eeprom.py — One-time DCA1000 EEPROM provisioning tool.

Run this ONCE whenever the DCA1000 needs to be reprogrammed to stream
data to a new host IP (e.g. moving from Windows to Linux, or changing
the host Ethernet adapter).

After this script succeeds you MUST power-cycle the DCA1000 for the
new EEPROM values to take effect.

Usage
-----
    python helper_utils/provision_dca_eeprom.py
    # Script will print the EEPROM payload and ask for confirmation.
    # After it reports OK, power-cycle the DCA1000, wait 15 s, then
    # run the normal capture:
    #   python capture/test.py --frames 10 --uart /dev/ttyACM0

The script reads IP/port settings from:
    config/AM273X_Capture_fixed.json
"""
from __future__ import annotations

import socket
import struct
import sys
import time
from pathlib import Path

# ── project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capture.dca1000_udp import (
    DCA1000UDP,
    CMD_RESET_FPGA,
    CMD_CONFIG_EEPROM,
    load_dca_network_params,
    _eeprom_config_bytes,
    _get_iface_mac,
    _get_dca_mac_from_arp,
)
from capture.config import DCA_CAPTURE_JSON

import json


def main() -> int:
    json_path = DCA_CAPTURE_JSON
    if not json_path.exists():
        print(f"ERROR: capture JSON not found: {json_path}", file=sys.stderr)
        return 1

    params = load_dca_network_params(json_path)
    dca_ip    = params["dca_ip"]
    cfg_port  = params["config_port"]
    local_ip  = params["local_ip"]

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    eu  = raw.get("DCA1000Config", {}).get("ethernetConfigUpdate", {})

    payload = _eeprom_config_bytes(eu, dca_ip=dca_ip, local_ip=local_ip)

    dca_mac_bytes = payload[4:10]
    pc_mac_bytes  = payload[14:20]

    print("=== DCA1000 EEPROM Provisioning ===")
    print(f"  DCA1000 IP  : {dca_ip}")
    print(f"  DCA1000 MAC : {':'.join(f'{b:02x}' for b in dca_mac_bytes)}")
    print(f"  Host IP     : {local_ip}  ← data will stream HERE after power-cycle")
    print(f"  Host MAC    : {':'.join(f'{b:02x}' for b in pc_mac_bytes)}")
    print(f"  Config port : {cfg_port}")
    print(f"  Data port   : {params['data_port']}")
    print(f"  Payload     : {payload.hex()}")

    if dca_mac_bytes == b"\x00" * 6:
        print(
            "\nWARNING: DCA1000 MAC is all-zeros. ARP entry not found.",
            "\n  Make sure the DCA1000 is reachable (UDP 4096) and try again.",
            file=sys.stderr,
        )
        return 1

    if pc_mac_bytes == b"\x00" * 6:
        print(
            "\nWARNING: Host MAC is all-zeros. Check that the interface with IP",
            f"{local_ip} is up.", file=sys.stderr,
        )

    print("\nThis will update the DCA1000 EEPROM permanently.")
    print("You must POWER-CYCLE the DCA1000 after this step for changes to take effect.")
    try:
        answer = input("Proceed? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        return 1
    if answer not in ("y", "yes"):
        print("Aborted.")
        return 1

    print(f"\nOpening UDP control socket → {dca_ip}:{cfg_port} ...")
    udp = DCA1000UDP(dca_ip, cfg_port, local_ip, timeout=5.0)
    try:
        udp.open()

        # Step 1: reset FPGA so it enters config-ready state
        print("  Sending reset_fpga ...")
        ok, _ = udp._send_recv(CMD_RESET_FPGA, retries=1)
        print(f"  reset_fpga: {'OK (or no-ack — normal)' if True else 'FAILED'}")
        print("  Waiting 4 s for FPGA re-init ...")
        time.sleep(4.0)

        # Step 2: write EEPROM
        print("  Sending config_eeprom ...")
        ok, extra = udp._send_recv(CMD_CONFIG_EEPROM, payload, retries=3)
        if ok:
            print("  config_eeprom: OK")
        else:
            print("  config_eeprom: FAILED (no ACK from DCA1000)", file=sys.stderr)
            print(
                "\n  Possible causes:",
                "\n    - DCA1000 not reachable (check Ethernet / IP)",
                "\n    - FPGA firmware does not support CMD_CONFIG_EEPROM (0x000C)",
                "\n    - DCA1000 needs a longer boot delay after reset",
                file=sys.stderr,
            )
            return 1

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        udp.close()

    print(
        "\n=== EEPROM write complete ===",
        "\nNow POWER-CYCLE the DCA1000 (toggle the power switch OFF → ON).",
        "\nWait 15 s, then run the capture:",
        "\n  python capture/test.py --frames 10 --uart /dev/ttyACM0",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
