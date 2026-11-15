#!/usr/bin/env python3
"""
altos_settings.py — show or change the Altos V4 radar's operating settings
(profile, detection threshold, FPS limit) from the command line.

    python altos_settings.py                           # show status and settings
    python altos_settings.py --profile 150m_0          # change, then wait until it streams again
    python altos_settings.py --threshold 250000 --fps-limit 10
    python altos_settings.py --threshold default --fps-limit off
    python altos_settings.py --profile 150m_1 --dry-run   # print the request only
    python altos_settings.py --restore settings_backups/<file>.json

HOW
    Sends the request the radar's own web page sends on "Apply" in
    Installation Configuration (POST http://<radar>:3000/install, read from
    the page's code on firmware 8e75cfeb): a one-element list with
    multicast_addr, multicast_port, profile, fps_limit (null = no limit),
    sync_mode and rpvb_control_overrides (threshold_coeff; left out = the
    firmware default, 250000). Values not given on the command line are
    kept as the radar reports them.

    The page sends ground_flip and install_height only for another device
    type, not for "Altos V4": on the V4 they are not radar settings, and
    this script does not send them either. (The flip below ground is done
    by the vendor's driver on the host; apply it offline if wanted.)

    Before every change the current settings are saved to
    settings_backups/<time>.json, so --restore can put them back. After the
    change the script waits until frames arrive again (the radar stops for
    a few seconds) and reads the settings back to confirm them.

    Network settings (/eth) and time sync (/time) are never touched.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import altos_v4 as av

HERE = Path(__file__).resolve().parent
BACKUPS = HERE / "settings_backups"
THRESHOLD_KEY = "threshold_coeff"


def url(radar_ip: str, path: str) -> str:
    return "http://%s:%d%s" % (radar_ip, av.WEB_PORT, path)


def get_json(radar_ip: str, path: str, timeout: float = 3.0):
    with urllib.request.urlopen(url(radar_ip, path), timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def post_install(radar_ip: str, body, timeout: float = 20.0):
    req = urllib.request.Request(url(radar_ip, "/install"), data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def summary(inst: dict) -> str:
    thr = (inst.get("rpvb_control_overrides") or {}).get(THRESHOLD_KEY)
    thr = "default (250000)" if thr is None else thr.get("Float", [thr])[0]
    fps = inst.get("fps_limit")
    return "profile %s | threshold %s | fps limit %s | sync %s | multicast %s:%s" % (
        inst.get("profile"), thr, "off (15 fps)" if fps is None else fps, inst.get("sync_mode"),
        inst.get("multicast_addr"), inst.get("multicast_port"))


def request_from(inst: dict) -> dict:
    """The fields the web page sends for a V4, from a GET /install entry."""
    return {
        "multicast_addr": inst["multicast_addr"],
        "multicast_port": inst["multicast_port"],
        "profile": inst["profile"],
        "fps_limit": inst.get("fps_limit"),
        "sync_mode": inst.get("sync_mode", "FreeRunning"),
        "rpvb_control_overrides": dict(inst.get("rpvb_control_overrides") or {}),
    }


class RestartWatch:
    """Listens to the stream from before the change is sent. The radar
    applies a change by restarting (seen 2026-11-15 for a threshold change:
    the POST returns, ~1 s later the stream stops for ~1.5 s, then frame
    ids start again from 0). wait() returns once that restart has happened
    and `need` complete frames followed it."""

    def __init__(self, group: str, port: int, host_ip: str):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((group, port))
        s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                     socket.inet_aton(group) + socket.inet_aton(host_ip))
        s.settimeout(0.1)
        self.sock = s
        self.asm = av.FrameAssembler()

    def wait(self, timeout: float, need: int = 15, gap_s: float = 0.8):
        """(seconds from the call, how the restart was seen) or (-1, None)."""
        t0 = last = time.monotonic()
        seen, got = None, 0
        try:
            while time.monotonic() - t0 < timeout:
                try:
                    pkt = self.sock.recv(65535)
                except socket.timeout:
                    if seen is None and time.monotonic() - last >= gap_s:
                        seen, got = "stream gap", 0
                    continue
                last = time.monotonic()
                restarts = self.asm.restarts
                frames = self.asm.feed(pkt, time.monotonic_ns())
                if self.asm.restarts > restarts and seen != "frame ids from 0":
                    seen, got = "frame ids from 0", 0
                if seen:
                    got += sum(1 for f in frames if f.complete)
                    if got >= need:
                        return last - t0, seen
        finally:
            self.sock.close()
        return -1.0, seen


def main() -> int:
    ap = argparse.ArgumentParser(description="Show or change the Altos V4 radar's operating settings.")
    ap.add_argument("--radar-ip", default=av.DEFAULT_RADAR_IP)
    ap.add_argument("--host-ip", default=av.DEFAULT_HOST_IP, help="for checking frames arrive after a change")
    ap.add_argument("--profile", help="one of the profiles the radar lists (status 'profiles')")
    ap.add_argument("--threshold", help="threshold coefficient (number, 0 = most detections) or 'default'")
    ap.add_argument("--fps-limit", help="frames per second (1-15) or 'off' (15 fps)")
    ap.add_argument("--restore", type=Path, help="apply a settings file saved in settings_backups/")
    ap.add_argument("--dry-run", action="store_true", help="print the request, do not send it")
    ap.add_argument("--wait", type=float, default=60.0, help="seconds to wait for frames after a change")
    a = ap.parse_args()

    status = get_json(a.radar_ip, "/status")
    installs = get_json(a.radar_ip, "/install")
    if not isinstance(installs, list) or len(installs) != 1:
        raise SystemExit("Expected one radar in /install (a V4), got: %r" % installs)
    cur = installs[0]
    print("%s %s, firmware %s" % (status.get("machine_name"), status.get("serial_num"), status.get("firmware_ver")))
    print("now:  " + summary(cur))

    if a.restore:
        body = json.loads(a.restore.read_text())
        if isinstance(body, list):
            body = body[0]
        req = request_from(body)
    elif a.profile or a.threshold is not None or a.fps_limit is not None:
        req = request_from(cur)
        if a.profile:
            if a.profile not in status.get("profiles", []):
                raise SystemExit("Unknown profile %r; the radar lists %s" % (a.profile, status.get("profiles")))
            req["profile"] = a.profile
        if a.threshold is not None:
            if a.threshold == "default":
                req["rpvb_control_overrides"].pop(THRESHOLD_KEY, None)
            else:
                v = float(a.threshold)
                lim = status.get("control_overrides", {}).get(THRESHOLD_KEY, {})
                lo, hi = lim.get("min", 0.0), lim.get("max", 1e12)
                if not lo <= v < hi:
                    raise SystemExit("threshold must be in [%g, %g)" % (lo, hi))
                req["rpvb_control_overrides"][THRESHOLD_KEY] = {"Float": [v]}
        if a.fps_limit is not None:
            if a.fps_limit == "off":
                req["fps_limit"] = None
            else:
                v = float(a.fps_limit)
                if not 1.0 <= v <= 15.0:
                    raise SystemExit("fps limit must be between 1 and 15 (15 is the radar's maximum)")
                req["fps_limit"] = v
    else:
        print("\n(no change requested; see --help)")
        return 0

    body = [req]
    print("new:  " + summary(req))
    print("request: POST %s\n  %s" % (url(a.radar_ip, "/install"), json.dumps(body)))
    if a.dry_run:
        print("dry run: not sent")
        return 0
    if request_from(cur) == req:
        print("no difference from the current settings: not sent")
        return 0

    BACKUPS.mkdir(exist_ok=True)
    backup = BACKUPS / ("install_%s.json" % datetime.now().strftime("%Y%m%d_%H%M%S"))
    backup.write_text(json.dumps(installs, indent=2) + "\n")
    print("saved the previous settings to %s" % backup)

    # Listen on the stream the radar sends now (the change keeps the multicast settings).
    watch = RestartWatch(cur["multicast_addr"], int(cur["multicast_port"]), a.host_ip)
    t_post = time.monotonic()
    code, text = post_install(a.radar_ip, body)
    print("radar answered %s after %.1f s: %s" % (code, time.monotonic() - t_post, text.strip()[:300]))
    if code // 100 != 2:
        watch.sock.close()
        return 1

    took, seen = watch.wait(a.wait)
    if took >= 0:
        print("restarted (%s); streaming again, %.1f s after the answer" % (seen, took))
    else:
        print("NO restart and fresh frames within %.0f s (%s); check the web page" % (a.wait, seen or "nothing seen"))
    after = get_json(a.radar_ip, "/install")[0]
    print("read back: " + summary(after))
    ok = request_from(after) == req
    print("settings confirmed" if ok else "WARNING: the radar reports different settings than requested")
    return 0 if ok and took >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
