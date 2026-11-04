# Hardware and network

## Platform

- **NVIDIA Jetson AGX Orin dev kit:** L4T R35.4.1 (JetPack 5.1.2),
  Ubuntu 20.04, CUDA 11.4. Power mode MAXN.
  - `jetson_clocks` is **not** used in the field (heat).
  - `ibrahim-desktop` (development board, 2026-10-26, owner's approval):
    `sudo jetson_clocks --store` then `sudo jetson_clocks` (CPUs 2.2 GHz, GPU
    1.3 GHz, fan at full speed; lost at reboot; undo with
    `sudo jetson_clocks --restore`), and the `whoopsie` crash-report daemon
    disabled (`sudo systemctl disable --now whoopsie`; it held one core at
    ~94% when idle; undo with `enable --now`).
- All USB-A ports share one internal hub; the USB-C ports are separate.
- Wi-Fi `wlan0` is for the iPad web control: either the Jetson's own
  **hotspot** (`http://192.168.50.1:8080`, see "Wi-Fi for the iPad" below)
  or a shared Wi-Fi (`http://jetson-desktop.local:8080`, mDNS). The sensor
  subnets must not use the Wi-Fi subnet (192.168.1.x, or 192.168.50.x with
  the hotspot).

## Sensors

| Sensor | Connection | Notes |
|---|---|---|
| Cascade radar: 2 × AWR2243 + AM273x MSS, DCA1000 capture board | DCA1000 Ethernet → **onboard** Ethernet | Needs a **full 12 V supply**: on a weak supply it accepts commands but sends no data. Firmware in QSPI streams at power-on. |
| RoboSense RS-Fairy LiDAR | Ethernet → **USB-Ethernet adapter** (ASIX AX88179, MAC 00:0E:C6:BC:78:54) | 10 Hz, 100 Mbit link |
| Stereolabs ZED 2i | USB | 30 fps, SVO2 lossless |

## Network (NetworkManager)

| Link | Profile | Host IP | Device |
|---|---|---|---|
| Radar | "Wired connection 2" | 192.168.33.30/24 | DCA1000 192.168.33.180 (config UDP 4096, data UDP 4098) |
| LiDAR | "lidar-usb-eth" | 192.168.34.30/24, never-default | LiDAR 192.168.34.200: MSOP 6699, DIFOP 7788, IMU 6688; web UI http://192.168.34.200 |

- **Both profiles are bound by MAC address, not interface name.** `eth0` and
  `eth1` swap depending on whether the USB adapter is present at boot. A
  profile bound to a name silently fails to start.
- Radar profile: onboard MAC 3C:6D:66:17:8F:E4. **These MACs belong to this
  board**; a new board needs its own (see `docs/NEW_BOARD_SETUP.md`).
- "Wired connection 1" (an automatic DHCP profile) has autoconnect **off**, so
  it cannot grab the LiDAR adapter.
- **Rule from the owner:** do not change the radar's network configuration
  (addresses and DCA settings) without being asked.
- History: both NICs once had 192.168.33.30, and radar traffic was routed out
  of the USB adapter. Hence the separate subnets.

## Wi-Fi for the iPad: the Jetson's own hotspot (added 2026-11-03)

**Why:** on managed networks (the university's Wi-Fi), devices often cannot
reach each other, and proxies or firewalls block the page. The team could
not open the web page by the Jetson's name or IP there. With a hotspot,
the Jetson's Wi-Fi *is* the network: the iPad joins it and talks to the
Jetson directly. The same works in the field, where there is no other
Wi-Fi.

| | |
|---|---|
| Network name (SSID) | `JetsonHotSpot` (change with `--hotspot-ssid=NAME`) |
| Password | chosen at setup (WPA2, 8+ characters) |
| Jetson address | **192.168.50.1**, page at **http://192.168.50.1:8080** |
| iPad address | given out by the Jetson (192.168.50.x, DHCP) |
| Band | 2.4 GHz: reaches farther through an enclosure; an access point on 5 GHz can fail to start on radar-protected (DFS) channels |
| NetworkManager profile | `jetson-hotspot`: `ipv4.method shared`, autoconnect with priority 100, so it starts at every boot |

**Set it up** (once per board, on the board's own screen or over Ethernet,
**not over SSH through its Wi-Fi**: switching the Wi-Fi to hotspot ends
that connection, so the script then sets it up without starting it):

```bash
cd ~/dev/CascadedRadarRoadSafetyOnTwoWheelers/sync_capture_pipeline
sudo bash setup_jetson_capture.sh --hotspot              # asks for the password
# or: sudo HOTSPOT_PASSWORD='...' bash setup_jetson_capture.sh --hotspot-ssid=RigHotSpot
```

**First, make sure the checkout is up to date.** Run `git pull` in the same
folder you run the script from. The script's first line, and the "commit"
line under "== Check", name the commit it runs: it must be the latest one.
An older script ignores `--hotspot` without a word: its output then has
no "== 3. Wi-Fi hotspot" section (seen on the rig, 2026-11-04).

A successful run ends with:
```
== Check
commit        : <latest commit>
...
hotspot       : on - Wi-Fi 'JetsonHotSpot'
service       : active / enabled at boot
web page      : OK  ->  join Wi-Fi 'JetsonHotSpot' on the iPad, open http://192.168.50.1:8080
```

What the script does, besides its usual steps:
- creates or updates the `jetson-hotspot` profile. An existing profile
  keeps its password unless a new one is given;
- if the `ufw` firewall is on, opens DHCP (UDP 67), DNS (53) and the page
  (TCP 8080) on `wlan0`. Without the first two, the iPad joins but gets no
  address;
- starts the hotspot, and prints the address to open.

It does not touch the radar and LiDAR profiles (192.168.33.x / 34.x, bound
by MAC).

**Use it:**
1. On the iPad, join the Wi-Fi `JetsonHotSpot`.
2. Open **http://192.168.50.1:8080**.

iOS may say "No Internet Connection" and drift back to a known network.
Stay on the hotspot, and turn off Auto-Join for other networks while
working.

**What you give up while the hotspot is on:**
- **No internet over Wi-Fi.** The card can only be either a client or the
  hotspot reliably; both Ethernet ports are taken by the sensors. For a
  `git pull` or updates, switch over for a moment:
  ```bash
  sudo nmcli con up "<your Wi-Fi profile>"   # internet (the hotspot stops)
  sudo nmcli con up jetson-hotspot           # back to the hotspot
  ```
  A USB phone tether also works.
- **No network time sync.** Sensor alignment uses CLOCK_MONOTONIC and is
  unaffected. Only the wall-clock time in `session.json` (`started_local`)
  can drift. Session names come from the iPad's clock.

**If the phone or iPad joins the hotspot but the page stays blank:** the
hotspot works, and the web service is the problem.

1. Look at the end of the script's output, under "== Check". Since
   2026-11-04 it waits up to 20 s for the page and, if it does not answer,
   prints the cause and the server's last errors.
2. Or ask the Jetson directly:
   ```bash
   systemctl status capture-web            # should be "active (running)"
   journalctl -u capture-web -n 30 --no-pager
   ```
   - `status=217/USER` or `status=200/CHDIR`: the service was installed
     for another user or folder. Typically this is the old service file,
     which was copied as is and named user `jetson` and
     `/home/jetson/dev/CascadedRadarRoadSafetyOnTwoWheelers`. Run
     `sudo bash setup_jetson_capture.sh --hotspot` again from the current
     checkout, as the normal user; it rewrites the service for this board.
   - `203/EXEC` or a Python traceback: the conda environment or a package
     is missing (`docs/NEW_BOARD_SETUP.md`).
3. Test the page on the Jetson itself: `curl -s -o /dev/null -w "%{http_code}\n"
   http://127.0.0.1:8080/` should print `200`. If it does, but the iPad
   still gets nothing, check the iPad's address (192.168.50.x) and the
   firewall (`sudo ufw status`; the script opens the ports when it is on).

Found on `ibrahim-desktop` (2026-11-04): with the old service file, the
hotspot came up, and a phone joined and got an address, but the page was
blank. The service failed with 217/USER every 3 s.

**Turn it off:**
- `sudo nmcli con down jetson-hotspot` turns it off until the next boot.
- `sudo nmcli con modify jetson-hotspot connection.autoconnect no` keeps it
  off at boot.
- `sudo nmcli con delete jetson-hotspot` removes it.

**Bandwidth:** live view needs ~0.3 MB/s (≈2.6 Mbit/s); recording status
needs far less.

**Interference:** none with the sensors (radar at 77 GHz, LiDAR optical).

**History:** `ibrahim-desktop` got such a profile by hand (`JetsonHotSpot`,
192.168.50.1/24, created by the end user). The script reproduces it.

## Kernel settings (`/etc/sysctl.d/90-sensor-capture.conf`)

```
net.core.rmem_max = 33554432        # 32 MB UDP receive buffers (radar + LiDAR)
vm.dirty_background_bytes = 67108864 # smooth writeback (SD card)
vm.dirty_bytes = 1073741824
```

`sync_capture_pipeline/setup_jetson_capture.sh` writes all three and installs
the web service; with `--hotspot` it also sets up the Wi-Fi hotspot (above).
