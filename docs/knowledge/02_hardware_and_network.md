# Hardware and network

## Platform

- **NVIDIA Jetson AGX Orin dev kit:** L4T R35.4.1 (JetPack 5.1.2),
  Ubuntu 20.04, CUDA 11.4. Power mode MAXN.
  - `jetson_clocks` is **not** used in the field (heat).
- All USB-A ports share one internal hub; the USB-C ports are separate.
- Wi-Fi `wlan0` is for the iPad web control
  (`http://jetson-desktop.local:8080`, mDNS). The sensor subnets must not use
  the Wi-Fi subnet (192.168.1.x).

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

## Kernel settings (`/etc/sysctl.d/90-sensor-capture.conf`)

```
net.core.rmem_max = 33554432        # 32 MB UDP receive buffers (radar + LiDAR)
vm.dirty_background_bytes = 67108864 # smooth writeback (SD card)
vm.dirty_bytes = 1073741824
```

`sync_capture_pipeline/setup_jetson_capture.sh` writes all three and installs
the web service.
