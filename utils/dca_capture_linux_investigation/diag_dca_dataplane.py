"""
diag_dca_dataplane.py — why is the control plane healthy but no data arriving?

Two causes produce IDENTICAL symptoms (all commands ACK, zero packets on
:4098), and they need opposite fixes:

  A. The radar is not streaming LVDS.
     The DCA1000 is a PASSIVE forwarder. With no LVDS input it forwards
     nothing and reports no error, because nothing is wrong with the DCA.
     v2/v3 never start the radar - no UART, no firmware load, no
     sensorStart - so this is expected for those scripts.

  B. The DCA's EEPROM data destination is not this host.
     The destination IP/MAC lives in EEPROM, not in any command. The
     Windows capture shows it targets 192.168.33.30. Control traffic works
     regardless, because the DCA replies to whatever source address the
     request came from - so a healthy control plane says NOTHING about the
     data destination.

This script checks everything reachable without root, then tells you the
one privileged command that discriminates the remaining cases.

USAGE
    python diag_dca_dataplane.py
"""
from __future__ import annotations
import os, re, socket, subprocess, sys

DCA_IP      = "192.168.33.180"
EXPECT_HOST = "192.168.33.30"      # from the Windows capture
DATA_PORT   = 4098


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=10).stdout
    except Exception as exc:                                  # noqa: BLE001
        return f"<failed: {exc}>"


def main() -> int:
    print("=" * 74)
    print("  DCA1000 DATA-PLANE DIAGNOSTIC")
    print("=" * 74)

    # ── 1. Host addressing ───────────────────────────────────────────────────
    addrs = sh("ip -4 -o addr show")
    print("\n[1] Host IPv4 addresses")
    found_expected = False
    subnet_iface = None
    for line in addrs.splitlines():
        m = re.search(r"^\d+:\s+(\S+)\s+inet\s+([\d.]+)/(\d+)", line)
        if not m:
            continue
        iface, ip, _pfx = m.groups()
        mark = ""
        if ip == EXPECT_HOST:
            found_expected = True
            subnet_iface = iface
            mark = "   <-- matches the DCA's EEPROM destination"
        elif ip.startswith("192.168.33."):
            subnet_iface = iface
            mark = "   <-- on the radar subnet, but NOT .30"
        print(f"    {iface:<10} {ip}{mark}")

    print()
    if found_expected:
        print(f"    OK: this host holds {EXPECT_HOST}. Cause B is ruled out.")
    else:
        print(f"    PROBLEM: {EXPECT_HOST} is NOT assigned to this host.")
        print(f"    The DCA will stream to {EXPECT_HOST} regardless of what")
        print(f"    address your control packets came from, so the data goes")
        print(f"    to a host that is not there.")
        print()
        print(f"    Cheapest fix - reversible, no EEPROM write, no power cycle:")
        ifn = subnet_iface or "<iface>"
        print(f"        sudo ip addr add {EXPECT_HOST}/24 dev {ifn}")
        print(f"    (an interface may hold several addresses; the existing one")
        print(f"     keeps working, so this does not disturb the control plane)")

    # ── 2. Reachability and ARP ──────────────────────────────────────────────
    print("\n[2] DCA reachability")
    arp = sh(f"ip neigh show {DCA_IP}")
    print(f"    arp entry : {arp.strip() or '<none>'}")
    if "lladdr" in arp:
        print("    OK: MAC resolved, layer 2 is up.")
    else:
        print("    No MAC yet - send any control command first, then re-check.")

    # ── 3. Can we bind the data port? ────────────────────────────────────────
    print(f"\n[3] UDP :{DATA_PORT} bindability")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", DATA_PORT))
        cur = s.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        print(f"    bind OK.  SO_RCVBUF default = {cur:,} B")
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32 * 1024 * 1024)
            got = s.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            print(f"    after requesting 32 MB -> {got:,} B")
            if got < 8 * 1024 * 1024:
                print("    Capped by the kernel. For sustained 8 fps (~25 MB/s):")
                print("        sudo sysctl -w net.core.rmem_max=33554432")
        except OSError as exc:
            print(f"    could not raise SO_RCVBUF: {exc}")
        s.close()
    except OSError as exc:
        print(f"    CANNOT BIND: {exc}")
        print("    Another process holds the port - stop it before capturing.")

    # ── 4. Firewall ──────────────────────────────────────────────────────────
    print("\n[4] Firewall")
    ipt = sh("sudo -n iptables -L INPUT -n 2>/dev/null")
    if not ipt.strip():
        print("    (needs sudo to inspect; skipped)")
    elif "DROP" in ipt or "REJECT" in ipt:
        print("    INPUT chain contains DROP/REJECT rules - verify UDP 4098:")
        print(f"        sudo iptables -I INPUT -p udp --dport {DATA_PORT} -j ACCEPT")
    else:
        print("    INPUT chain looks permissive.")

    # ── 5. The discriminating test ───────────────────────────────────────────
    print("\n" + "=" * 74)
    print("  THE TEST THAT SEPARATES CAUSE A FROM CAUSE B")
    print("=" * 74)
    ifn = subnet_iface or "<iface>"
    print(f"""
  Capture ALL traffic from the DCA - no port filter - while a record
  window is open. In one terminal:

      sudo tcpdump -i {ifn} -n host {DCA_IP}

  In another:

      python sanity_prob_communication_to_dca_v3.py --listen --secs 15

  Read the tcpdump output:

    only UDP on port 4096, nothing else
        -> CAUSE A. The DCA has no LVDS input. The radar is not
           streaming. Expected for v3, which never starts the radar.

    ARP 'who-has {EXPECT_HOST}' from {DCA_IP}, repeated
        -> CAUSE B. The DCA has data and is trying to deliver it to a
           host that is not answering. Assign {EXPECT_HOST} to this box.

    UDP {DCA_IP} > {EXPECT_HOST}.{DATA_PORT}
        -> data IS flowing but not reaching your socket. Firewall, or
           the address is not local so the kernel drops it.

  ARP is the discriminator: a DCA with data to send must resolve its
  destination first, and you will see it try.
""")
    print("=" * 74)
    print("  IF IT IS CAUSE A")
    print("=" * 74)
    print("""
  The radar plane still has to be started. Two options:

    Fastest proof of concept - start the radar from the Windows host
    (proven to work), leave it streaming, then move only the Ethernet
    cable to the Jetson and run v3. LVDS keeps flowing to the DCA
    independently of which machine holds the Ethernet link. This
    separates 'can Linux capture?' from 'can Linux configure the radar?'
    and answers the capture question today.

    Full port - bring up radar.py on Linux (UART /dev/ttyACM*, firmware
    load, sensorStart). This is the remaining work item, and it is
    genuinely separate from the DCA path you have just proven.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
