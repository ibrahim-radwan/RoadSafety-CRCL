# Setting up a new Jetson board

Everything that lives **outside** the code and had to be set up by hand on
the original board. Follow it in order on a fresh Jetson AGX Orin. Steps that
need `sudo` are marked.

The reference board ran L4T **R35.4.1** (JetPack 5.1.2), Ubuntu 20.04,
CUDA 11.4. Use the **same JetPack**: the ZED SDK and the NVIDIA PyTorch wheel
are tied to it.

---

## 1. User, paths and the repository

Several files contain absolute paths:
- the flash configs: `cascaded_radar_capture_pipeline/firmware/*/flash_*.cfg`;
- the web service: `sync_capture_pipeline/web_control/capture-web.service`;
- `capture_session.py` and `web_control/server.py`.

They all assume user **`jetson`** and this exact location:

```bash
mkdir -p ~/dev
git clone https://github.com/ibrahim-radwan/RoadSafety-CRCL.git ~/dev/CascadedRadarRoadSafetyOnTwoWheelers
```

Clone **into that folder name**, not the default `RoadSafety-CRCL`. Using a
different user or path means updating those files (and regenerating the flash
configs with `cascaded_radar_capture_pipeline/firmware/make_flash_cfgs.py`).

## 2. System packages (sudo)

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git curl libpcap-dev libyaml-cpp-dev zstd
```

`libpcap-dev` and `libyaml-cpp-dev` are needed by the LiDAR wrapper.

## 3. Miniconda and the Python environment

```bash
# Miniconda for aarch64 (see docs.conda.io), installed to ~/miniconda3
conda env create -f ~/dev/CascadedRadarRoadSafetyOnTwoWheelers/environment/rel_1_py_3_8.yml
conda activate rel_1_py_3_8
```

Then install the three vendor wheels the environment file can't include.

**3a. PyTorch for JetPack 5 (CUDA).** NVIDIA's wheel, the one used on the
original board:

```bash
# from NVIDIA's JetPack 5 PyTorch index (jp/v512) - check the exact URL on NVIDIA's
# "PyTorch for Jetson" page for your JetPack version
pip install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
python -c "import torch; print(torch.cuda.is_available())"     # must print True
```

**3b. torchvision 0.16.0 (optional).** It was built from source against that
torch on the original board, but **no code in this repository imports it**.
Skip it unless other work needs it.

**3c. ZED SDK + pyzed.** See section 4.

## 4. ZED SDK 4.1.4 (sudo for the installer)

1. Download **ZED SDK 4.1.4 for L4T 35.4 (JetPack 5.1.2)** from Stereolabs'
   release archive (`stereolabs.com/developers/release/4.1`).
2. Install:
   ```bash
   chmod +x ZED_SDK_*.run        # e.g. ZED_SDK_Tegra_L4T35.4_v4.1.4.zstd.run (check the exact name)
   ./ZED_SDK_*.run               # accept defaults; the Python API is installed in step 3
   ```
3. Install the Python API **inside the conda environment**:
   ```bash
   conda activate rel_1_py_3_8
   python /usr/local/zed/get_python_api.py
   python -c "import pyzed.sl as sl; print(sl.Camera().get_sdk_version())"   # 4.1.4
   ```

Recording and offline extraction must use the same SDK version. The pyzed
wheel is not kept in git.

## 5. Build the LiDAR Python wrapper

```bash
conda activate rel_1_py_3_8
pip install pybind11            # if not already in the environment
cd ~/dev/CascadedRadarRoadSafetyOnTwoWheelers/lidar_standalone_capture/rs_driver/python_wrapper_2
mkdir -p build && cd build
cmake -DPython3_EXECUTABLE=$(which python) ..
make -j8
cp robosense_lidar.cpython-38-aarch64-linux-gnu.so ..
python -c "import sys; sys.path.insert(0, '..'); import lidar; print('wrapper OK')"
```

## 6. Network (sudo): bind profiles by MAC

Find this board's MACs (`ip -br link`): the onboard port and the USB-Ethernet
adapter (the ASIX AX88179 used so far has MAC 00:0E:C6:BC:78:54).

```bash
# Radar: onboard port -> DCA1000 (192.168.33.180)
sudo nmcli con add type ethernet con-name "Wired connection 2" ifname "" \
     802-3-ethernet.mac-address <ONBOARD_MAC> ipv4.method manual ipv4.addresses 192.168.33.30/24 \
     ipv4.never-default yes ipv6.method ignore
# LiDAR: USB-Ethernet adapter -> RS-Fairy (192.168.34.200)
sudo nmcli con add type ethernet con-name "lidar-usb-eth" ifname "" \
     802-3-ethernet.mac-address <USB_ADAPTER_MAC> ipv4.method manual ipv4.addresses 192.168.34.30/24 \
     ipv4.never-default yes ipv6.method ignore
# Any automatic DHCP profile on these ports must not autoconnect:
nmcli -t -f NAME,DEVICE con show            # then: sudo nmcli con modify "<name>" connection.autoconnect no
```

**Bind by MAC, not by interface name.** `eth0`/`eth1` swap depending on
whether the USB adapter is present at boot.

Check: `ip -br addr` shows .33.30 and .34.30. The LiDAR web page opens at
http://192.168.34.200.

## 7. SD card (sudo)

```bash
# only if the card is new or must be reformatted - this ERASES it:
#   sudo mkfs.ext4 -F -L sensor_data -m 0 -E lazy_itable_init=0,lazy_journal_init=0 /dev/mmcblk1p1
sudo mkdir -p /mnt/sdcard
echo "UUID=$(sudo blkid -s UUID -o value /dev/mmcblk1p1)  /mnt/sdcard  ext4  defaults,noatime,data=writeback,nofail,x-systemd.automount,x-systemd.device-timeout=10  0  2" | sudo tee -a /etc/fstab
sudo systemctl daemon-reload && sudo systemctl start mnt-sdcard.automount
ls /mnt/sdcard && sudo chown jetson:jetson /mnt/sdcard
mkdir -p /mnt/sdcard/sessions ~/camera_stage
```

Moving the existing 1 TB card to the new board needs only the fstab line,
since the UUID travels with the card.

## 8. Kernel settings and the web service (sudo)

```bash
sudo bash ~/dev/CascadedRadarRoadSafetyOnTwoWheelers/sync_capture_pipeline/setup_jetson_capture.sh
```

This writes `/etc/sysctl.d/90-sensor-capture.conf`:
- `net.core.rmem_max` = 32 MB;
- `vm.dirty_background_bytes` = 64 MB;
- `vm.dirty_bytes` = 1 GB.

It also installs and starts the `capture-web` service on port 8080. The
iPad reaches it at `http://<hostname>.local:8080` (mDNS; the original board
was `jetson-desktop`).

## 9. Radar firmware record

The radar keeps its firmware in its own flash, but the record of what is in
it (`~/.radar_flashed`) lives on the Jetson. On a new board, either re-flash
with `flashing/boot/flash_variant.sh` (which writes the record), or recreate
it for the image already in the radar (currently `r075m_v2_8fps`):

```bash
FW=~/dev/CascadedRadarRoadSafetyOnTwoWheelers/cascaded_radar_capture_pipeline/firmware; V=r075m_v2_8fps
cat > ~/.radar_flashed <<EOF
variant=$V
appimage=$FW/$V/am273x_cascade_$V.appimage
appimage_md5=$(md5sum $FW/$V/am273x_cascade_$V.appimage | cut -d' ' -f1)
processing_cfg=$FW/$V/cascade_$V.cfg
slope_mhz_us=19.9878
max_range_m=74.99
hpf_floor_m=2.62
flashed_at=<date of the original flash>
note=recreated on a new board; image flashed earlier
EOF
```

Without it, the launcher warns that the firmware is unknown, and processing
assumes the old `r075m`.

## 10. Verify

```bash
conda activate rel_1_py_3_8
cd ~/dev/CascadedRadarRoadSafetyOnTwoWheelers/sync_capture_pipeline
python -m pip check
python -c "import pyzed.sl, torch, serial, xmodem, pyqtgraph.opengl, OpenGL; print('imports OK', torch.cuda.is_available())"
python -m pytest -q tests          # tests skip if the reference sessions are absent
```

With the sensors connected:
1. **Radar only, 30 s:**
   `python3 dca_capture_radar_sync.py --secs 30 --out /tmp/radar_check/radar`,
   then `python check_radar_frames.py /tmp/radar_check`. Expect ~17,300
   packets/s, 0 lost, and 0.00% identical-in-every-frame.
2. **A 20 s three-sensor session** from the iPad page. It should show
   **PASS**.
3. `python make_report.py --session /mnt/sdcard/sessions/<name>`, then open
   the HTML.

## Not needed / do not do

- **`jetson_clocks`:** not used in the field (heat).
- **The `cascaded-radar` conda environment:** the research team's (Python 3.10,
  with CuPy). This pipeline doesn't need it.
- **Changing the radar's addresses or the DCA1000 settings:** the receiver's
  defaults (packet delay 25 µs) are part of the timing calibration.
