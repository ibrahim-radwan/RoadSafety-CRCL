# Altos V4 radar: standalone capture

The team is evaluating an **Altos V4** 4D imaging radar (since 2026-11-15),
either alongside or in place of the cascaded TI radar.

The sensor:
- 16 TX × 16 RX;
- 150° horizontal and 30° vertical field of view;
- up to 15 frames/s;
- outputs a **processed point cloud**, not raw ADC data.

This folder records it on its own. It is not yet part of
`sync_capture_pipeline/`.

Our unit: serial V4-260804-0146, firmware 8e75cfeb.

**Use the V4 materials, not the V2 ones.** The V2 driver
(`Altos-Radar/altosRadarRosV2`) has a different packet format and does not
decode this sensor. The V4 materials are:
- the [V4 user guide](https://forum.altosradar.com/t/altos-v4-and-rcu-rf6-user-guide/72);
- the V4/RCU Interface Control Document, linked from that guide;
- `altos-pc-v2.h` and the Wireshark dissector `altos.lua` in
  [Altos-Radar/altosRadarParser](https://github.com/Altos-Radar/altosRadarParser).

## Hardware and network

- **Power:** 9–36 V. The guide asks for a supply of more than 25 W; the spec
  sheet says the radar draws 18 W. We use the 15 V / 4 A adaptor; 1.5 A is not
  enough. The barrel plug is 5.5 × 2.5 mm. Boot takes 10–20 s.
- **Ethernet:** 100BASE-T1, through the harness's media converter to RJ45.
- **Addresses:** radar 192.168.3.10/24, host 192.168.3.1/24.
- **Web pages:** configuration at `http://192.168.3.10:3000`; firmware update
  at port 8080.
- **Data:** UDP multicast to 224.1.2.4:4040.

On `ibrahim-desktop` the radar shares `eth0` with the DCA1000 profile. Add
the address for a session, without changing the saved profiles:

```bash
sudo ip addr add 192.168.3.1/24 dev eth0     # undo: sudo ip addr del 192.168.3.1/24 dev eth0
curl -s http://192.168.3.10:3000/status      # model, serial, firmware
curl -s http://192.168.3.10:3000/install     # profile, FPS limit, threshold, sync mode
```

### Radar settings in use (web page, "Installation Configuration")

| Setting | Value | Why |
|---|---|---|
| Profile | `150m_1` | The finer-resolution variant of the 150 m profile; enough range for two-wheeler scenes. |
| FPS limit | off | The radar runs at its maximum, 15 fps. Setting 30 is reset to 15 (checked 2026-11-15). |
| Threshold coefficient | 0 | The most detections the firmware allows. Every point has its SNR, so a stricter threshold can be applied afterwards. |
| Sync | Free running | No PTP yet: use host receipt time (see below). |

## Recording and checking

Run with the project environment. On `ibrahim-desktop`, unset `PYTHONPATH`
first (ROS sets it):

```bash
cd altos_standalone_capture
python record_altos_raw.py --duration 30          # Ctrl+C to stop early
python check_altos_session.py dataset/altos_session_<stamp> [--export]
```

The recorder uses only the standard library. The checker needs numpy.

A session folder (`dataset/` is git-ignored) contains:

| File | Content |
|---|---|
| `altos.pcap` | Every packet as received, stamped with host CLOCK_MONOTONIC at receipt. Opens in Wireshark (use `altos.lua` to decode it). |
| `frames.csv` | One row per frame: id, first and last packet receipt time, radar time, points, packets, complete. |
| `stats.csv` | Per second: packets, frames, points, lost and incomplete frames, queue depth. |
| `clock_map.csv` | CLOCK_MONOTONIC and CLOCK_REALTIME pairs, once per second. |
| `metadata.json` | Addresses, the radar's own settings at the start (`/status`, `/install`, `/eth`, `/time`), and totals. |
| `points.npz` | Written by `--export`: points (range, doppler, azimuth, elevation, snr, x, y, z), a `valid` mask, and per-frame times. |

What the checker reports:
- frame-id gaps (whole frames lost);
- incomplete frames;
- frame spacing on both clocks;
- the radar clock fitted against host time;
- point statistics.

It exits 0 when no frames were lost or incomplete. The last frame, cut off
by the stop, is reported but does not fail the check.

## Data format

Packet layout is in `altos_v4.py`. Each packet has a 32-byte header (magic
"Alto", radar time, frame_id, length, offset), then up to 72 points of 20
bytes: range, doppler, azimuth, elevation and SNR as float32.

- **Multi-packet frames:** a frame of more than 72 points comes in several
  packets, which are put back together by `offset`.
- **Axes:**
  - raw: x forward, y **left**, z **down**;
  - positive azimuth is left, as in our vehicle frame;
  - `frames_to_arrays()` flips the sign of z so that z is up.

## Measured on our unit (2026-11-15, indoors, ibrahim-desktop)

| Item | Result |
|---|---|
| Frame rate | 15.0 frames/s; no lost or incomplete frames over 30 s |
| Points per frame | 72–284 indoors (frames over 72 points come as 2–4 packets) |
| Packet spread within a frame | about 0.5 ms from first to last packet |
| Azimuth spread of points | −72° to +73° |
| Elevation spread of points | −15° to +15° |
| NaN points | Some points come with fields set to `0xFFFFFFFF` (NaN): 0.2% in one recording, 3 points in another, always among a frame's last ~16 points. Kept as sent, marked invalid when decoded. |

**Radar clock:** free running, it reads March 2025. Its timestamps are
exactly 66.7 ms apart, but the real frame rate wanders against host time:
−505 ppm in one recording, +408 ppm in the next. Use host receipt time, as
for the other sensors. The radar's timestamps are only worth using with PTP.

## Not yet verified

- **Axis signs** of azimuth and elevation, against a target at a known
  position, such as the corner reflector.
- **Range accuracy.**
- **Recording on the rig**, next to the other sensors (CPU, network ports).
- **PTP and the second-offset trigger.**
