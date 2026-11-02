#!/usr/bin/env bash
# One-time Jetson setup for field capture. Run with:  sudo bash setup_jetson_capture.sh [--hotspot]
#   1. kernel settings, permanently: net.core.rmem_max = 32 MB (radar + LiDAR UDP
#      receive buffers) and smooth writeback for the SD card (vm.dirty_*)
#   2. the capture web control as a boot-time service (port 8080)
#   3. only with --hotspot: the Jetson's Wi-Fi becomes its own network (access point),
#      so the iPad reaches the page at http://192.168.50.1:8080 without any venue
#      network, proxy or firewall in between. See docs/knowledge/02_hardware_and_network.md.
#      The Wi-Fi then no longer joins other networks (no internet over Wi-Fi) until
#      the hotspot is switched off:  nmcli con down jetson-hotspot
#      Password: HOTSPOT_PASSWORD=... in the environment, or asked for (8+ characters;
#      kept if the profile already exists and none is given). Name: --hotspot-ssid=NAME.
# Safe to run again. Does not touch the radar's or the LiDAR's network settings.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HOTSPOT=0
HOTSPOT_CON=jetson-hotspot
HOTSPOT_SSID=JetsonHotSpot
HOTSPOT_ADDR=192.168.50.1/24          # not 192.168.33.x (radar) or 192.168.34.x (LiDAR)
for arg in "$@"; do
    case "$arg" in
        --hotspot) HOTSPOT=1 ;;
        --hotspot-ssid=*) HOTSPOT=1; HOTSPOT_SSID="${arg#*=}" ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg (see --help)"; exit 1 ;;
    esac
done
if [ "$(id -u)" -ne 0 ]; then
    echo "Run with sudo:  sudo bash $0 [--hotspot]"; exit 1
fi

echo "== 1. Kernel settings (UDP receive buffers, SD-card writeback)"
# This file is owned by this script: keep ALL settings here (it is rewritten).
cat > /etc/sysctl.d/90-sensor-capture.conf <<'CONF'
net.core.rmem_max=33554432
vm.dirty_background_bytes=67108864
vm.dirty_bytes=1073741824
CONF
sysctl -p /etc/sysctl.d/90-sensor-capture.conf

echo "== 2. Web control service"
# A hand-started server would hold port 8080; stop it (a running recording is
# not affected - it runs in its own session and the service re-adopts it).
pkill -f "web_control/server.py --host 0.0.0.0 --port 8080" && sleep 1 || true
cp "$HERE/web_control/capture-web.service" /etc/systemd/system/capture-web.service
systemctl daemon-reload
systemctl enable --now capture-web
sleep 3

HOTSPOT_STARTED=0
if [ "$HOTSPOT" -eq 1 ]; then
    echo "== 3. Wi-Fi hotspot '$HOTSPOT_SSID' ($HOTSPOT_ADDR, profile $HOTSPOT_CON)"
    PSK="${HOTSPOT_PASSWORD:-}"
    if [ -z "$PSK" ] && ! nmcli -t -f NAME con show | grep -qx "$HOTSPOT_CON"; then
        read -r -s -p "Hotspot password (8+ characters): " PSK; echo
    fi
    if [ -n "$PSK" ] && [ "${#PSK}" -lt 8 ]; then
        echo "The password needs at least 8 characters."; exit 1
    fi
    # 2.4 GHz: reaches farther through an enclosure, and an access point on 5 GHz
    # can fail to start on radar-protected (DFS) channels. Live view needs ~3 Mbit/s.
    SETTINGS=(802-11-wireless.ssid "$HOTSPOT_SSID" 802-11-wireless.mode ap 802-11-wireless.band bg
              ipv4.method shared ipv4.addresses "$HOTSPOT_ADDR" ipv6.method ignore
              wifi-sec.key-mgmt wpa-psk
              connection.autoconnect yes connection.autoconnect-priority 100)
    [ -n "$PSK" ] && SETTINGS+=(wifi-sec.psk "$PSK")
    if nmcli -t -f NAME con show | grep -qx "$HOTSPOT_CON"; then
        nmcli con modify "$HOTSPOT_CON" connection.interface-name wlan0 "${SETTINGS[@]}"
    else
        nmcli con add type wifi ifname wlan0 con-name "$HOTSPOT_CON" "${SETTINGS[@]}"
    fi
    # The firewall, if on, must let the iPad get an address (DHCP), resolve (DNS) and open the page.
    if command -v ufw > /dev/null && ufw status | grep -q "Status: active"; then
        ufw allow in on wlan0 to any port 67 proto udp comment "hotspot DHCP"
        ufw allow in on wlan0 to any port 53 comment "hotspot DNS"
        ufw allow in on wlan0 to any port 8080 proto tcp comment "capture web control"
    fi
    # Switching wlan0 to the hotspot drops its current Wi-Fi connection: never do that
    # to an SSH session that came in over that Wi-Fi.
    SSH_CONN="${SSH_CONNECTION:-}"
    SSH_IP="${SSH_CONN%% *}"
    if [ -n "$SSH_IP" ] && ip route get "$SSH_IP" 2>/dev/null | grep -q "dev wlan0"; then
        echo "You are connected over this Wi-Fi: the hotspot is set up but NOT started now"
        echo "(it would cut this session). It starts at the next boot, or run on the Jetson itself:"
        echo "    sudo nmcli con up $HOTSPOT_CON"
    else
        nmcli con up "$HOTSPOT_CON" && HOTSPOT_STARTED=1
        sleep 2
    fi
fi

echo "== Check"
echo "rmem_max      : $(sysctl -n net.core.rmem_max)"
echo "dirty bytes   : background $(sysctl -n vm.dirty_background_bytes), max $(sysctl -n vm.dirty_bytes)"
echo "service       : $(systemctl is-active capture-web) / $(systemctl is-enabled capture-web) at boot"
if [ "$HOTSPOT" -eq 1 ]; then
    echo "hotspot       : $([ "$HOTSPOT_STARTED" -eq 1 ] && echo on || echo "set up, not started") - Wi-Fi '$HOTSPOT_SSID'"
fi
if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/ | grep -q 200; then
    if [ "$HOTSPOT_STARTED" -eq 1 ]; then
        echo "web page      : OK  ->  join Wi-Fi '$HOTSPOT_SSID' on the iPad, open http://${HOTSPOT_ADDR%/*}:8080"
    else
        echo "web page      : OK  ->  http://$(hostname).local:8080  or  http://$(ip -4 -o addr show wlan0 | awk '{print $4}' | cut -d/ -f1):8080 (Wi-Fi)"
    fi
else
    echo "web page      : NOT responding - see: journalctl -u capture-web -n 50"
fi
