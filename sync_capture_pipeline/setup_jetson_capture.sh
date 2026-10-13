#!/usr/bin/env bash
# One-time Jetson setup for field capture. Run with:  sudo bash setup_jetson_capture.sh
#   1. kernel settings, permanently: net.core.rmem_max = 32 MB (radar + LiDAR UDP
#      receive buffers) and smooth writeback for the SD card (vm.dirty_*)
#   2. the capture web control as a boot-time service (port 8080)
# Safe to run again. Does not touch the radar's network settings.
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then
    echo "Run with sudo:  sudo bash $0"; exit 1
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

echo "== Check"
echo "rmem_max      : $(sysctl -n net.core.rmem_max)"
echo "dirty bytes   : background $(sysctl -n vm.dirty_background_bytes), max $(sysctl -n vm.dirty_bytes)"
echo "service       : $(systemctl is-active capture-web) / $(systemctl is-enabled capture-web) at boot"
if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/ | grep -q 200; then
    echo "web page     : OK  ->  http://jetson-desktop.local:8080  or  http://$(ip -4 -o addr show wlan0 | awk '{print $4}' | cut -d/ -f1):8080 (Wi-Fi)"
else
    echo "web page     : NOT responding - see: journalctl -u capture-web -n 50"
fi
