#!/usr/bin/env bash
# One-time Jetson setup for field capture. Run with:  sudo bash setup_jetson_capture.sh [--hotspot]
#   1. kernel settings, permanently: net.core.rmem_max = 32 MB (radar + LiDAR UDP
#      receive buffers) and smooth writeback for the SD card (vm.dirty_*)
#   2. the capture web control as a boot-time service (port 8080), written for THIS board:
#      it runs as the user who ran sudo, from this checkout, with that user's conda Python
#      (~/miniconda3/envs/rel_1_py_3_8, or CAPTURE_PYTHON=/path/to/python)
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
# Which version runs, so that an output shows at once whether the checkout is up to date.
# git runs as the owner of the checkout: as root it may refuse ("dubious ownership").
VERSION="$(sudo -u "${SUDO_USER:-root}" git -C "$HERE" log -1 --format='%h (%cd)' --date=short 2>/dev/null || echo unknown)"
if [ -n "$(sudo -u "${SUDO_USER:-root}" git -C "$HERE" status --porcelain 2>/dev/null)" ]; then
    VERSION="$VERSION + local changes"
fi
echo "setup_jetson_capture.sh in $HERE, commit $VERSION"

echo "== 1. Kernel settings (UDP receive buffers, SD-card writeback)"
# This file is owned by this script: keep ALL settings here (it is rewritten).
cat > /etc/sysctl.d/90-sensor-capture.conf <<'CONF'
net.core.rmem_max=33554432
vm.dirty_background_bytes=67108864
vm.dirty_bytes=1073741824
CONF
sysctl -p /etc/sysctl.d/90-sensor-capture.conf

echo "== 2. Web control service"
# The service runs as the user who ran sudo (not root), from this checkout, with that user's
# conda Python. web_control/capture-web.service is a template; until 2026-11-04 it was copied
# as is and named user jetson and /home/jetson/dev/CascadedRadarRoadSafetyOnTwoWheelers, so on
# any other board or checkout folder the service failed in a restart loop.
RUN_USER="${SUDO_USER:-}"
if [ -z "$RUN_USER" ] || [ "$RUN_USER" = root ]; then
    echo "Run this with sudo from the account that should run the web service (e.g. jetson),"
    echo "not as root:  sudo bash $0"; exit 1
fi
RUN_GROUP="$(id -gn "$RUN_USER")"
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"
PY="${CAPTURE_PYTHON:-$RUN_HOME/miniconda3/envs/rel_1_py_3_8/bin/python}"
if [ ! -x "$PY" ]; then
    echo "No Python at $PY - create the rel_1_py_3_8 conda environment (docs/NEW_BOARD_SETUP.md)"
    echo "or name its python:  sudo CAPTURE_PYTHON=/path/to/python bash $0"; exit 1
fi
# The same clean environment as systemd gives the service (no PYTHONPATH, e.g. from ROS).
if ! sudo -u "$RUN_USER" env -i HOME="$RUN_HOME" PATH=/usr/bin:/bin "$PY" -c "import flask, psutil" 2> /dev/null; then
    echo "$PY cannot import flask and psutil, which the web server needs (docs/NEW_BOARD_SETUP.md)"; exit 1
fi
echo "runs as $RUN_USER, from $HERE, with $PY"
# A hand-started server would hold port 8080; stop it (a running recording is
# not affected - it runs in its own session and the service re-adopts it).
pkill -f "web_control/server.py --host 0.0.0.0 --port 8080" && sleep 1 || true
sed -e "s|@USER@|$RUN_USER|" -e "s|@GROUP@|$RUN_GROUP|" -e "s|@PIPELINE@|$HERE|" \
    -e "s|@PYTHON@|$PY|" -e "s|@HOME@|$RUN_HOME|" \
    "$HERE/web_control/capture-web.service" > /etc/systemd/system/capture-web.service
systemctl daemon-reload
systemctl enable capture-web
systemctl restart capture-web             # picks up a changed unit when run again

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
echo "commit        : $VERSION"
echo "rmem_max      : $(sysctl -n net.core.rmem_max)"
echo "dirty bytes   : background $(sysctl -n vm.dirty_background_bytes), max $(sysctl -n vm.dirty_bytes)"
if [ "$HOTSPOT" -eq 1 ]; then
    echo "hotspot       : $([ "$HOTSPOT_STARTED" -eq 1 ] && echo on || echo "set up, not started") - Wi-Fi '$HOTSPOT_SSID'"
elif nmcli -t -f NAME con show 2> /dev/null | grep -qx "$HOTSPOT_CON"; then
    echo "hotspot       : profile $HOTSPOT_CON exists, not changed (add --hotspot to update it)"
else
    echo "hotspot       : not set up - add --hotspot so the iPad can join the Jetson's own Wi-Fi"
fi
PAGE_OK=0
for _ in $(seq 20); do                    # the server needs a few seconds to start
    if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/ | grep -q 200; then PAGE_OK=1; break; fi
    sleep 1
done
echo "service       : $(systemctl is-active capture-web) / $(systemctl is-enabled capture-web) at boot"
if [ "$PAGE_OK" -eq 1 ]; then
    if [ "$HOTSPOT_STARTED" -eq 1 ]; then
        echo "web page      : OK  ->  join Wi-Fi '$HOTSPOT_SSID' on the iPad, open http://${HOTSPOT_ADDR%/*}:8080"
    else
        echo "web page      : OK  ->  http://$(hostname).local:8080  or  http://$(ip -4 -o addr show wlan0 | awk '{print $4}' | cut -d/ -f1):8080 (Wi-Fi)"
    fi
else
    echo "web page      : NOT responding"
    # Why: systemd's exit status names the common causes; then the server's own last errors.
    CODE="$(systemctl show capture-web -p ExecMainStatus --value)"
    case "$CODE" in
        217) echo "  cause: the service's user does not exist (status 217/USER): $RUN_USER?" ;;
        200) echo "  cause: the service's folder does not exist (status 200/CHDIR): $HERE?" ;;
        203) echo "  cause: the service's Python cannot be started (status 203/EXEC) - check $PY" ;;
        0|"") ;;
        *) echo "  the server exited with status $CODE" ;;
    esac
    journalctl -u capture-web -n 40 --no-pager -o cat \
        | grep -iE "error|failed|traceback|address already in use" | tail -3 | sed 's/^/  log: /' || true
    echo "  full log: journalctl -u capture-web -n 50"
fi
