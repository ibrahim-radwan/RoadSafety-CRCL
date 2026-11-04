# Capture web control

Start and stop three-sensor recording sessions from a browser, so you
don't need a screen, keyboard or SSH on the scooter.

## Use it

1. Connect the iPad to the Jetson's network:
   - **Recommended: the Jetson's own hotspot.** Join the Wi-Fi
     `JetsonHotSpot` and open **`http://192.168.50.1:8080`**. Set it up once
     with `sudo bash setup_jetson_capture.sh --hotspot`; see
     `docs/knowledge/02_hardware_and_network.md`, "Wi-Fi for the iPad". This
     avoids networks whose proxies or firewalls block the page.
   - **Or the same Wi-Fi as the Jetson.**
2. Without the hotspot, open `http://<jetson-wifi-ip>:8080`. To find the
   IP on the Jetson, run `ip -4 -br addr show wlan0`.
3. Enter a session name and a note, choose the sensors, and press
   **Start recording**.
   - Duration `0` means the session runs until you press **Stop**.
   - The countdown gives you time to get clear of the scene before
     recording begins.
4. **Live status** shows each sensor. A red tag means it has not updated for
   more than 3 s, or has a problem:
   - radar: packets/s
   - camera: FPS and dropped frames
   - LiDAR: packets/s and losses
5. **Board (Jetson)** card: the live junction temperature, CPU and GPU
   temperatures, and fan speed, shown while idle and while recording. The tag
   reads **warm** at 80 °C and above, and **HOT** at 90 °C and above. The Orin
   throttles near 99 °C. During a session, the launcher also logs these values
   every 5 s into `session.log` and records the peak in `session.json`.
   `check_session.py` warns if the peak reached 80 °C. A session will not start
   if the board is already at 90 °C or above.
6. **Stop recording** is the same as Ctrl+C in a terminal. The radar stops
   first, then camera and LiDAR trail and finalize. The camera's SVO2 check
   takes about half the recording length.
7. `check_session.py` then runs automatically. **PASS** means the session is
   complete. Its full report is saved as `<session>/check_session.txt` and is
   linked from the session list. When the log shows a known failure, the result
   starts with a **HINT** naming what to check:
   - radar board not responding: its 12 V power and cable;
   - radar data rate too low (DCA1000 throttled): the launcher stops the
     session by itself after 5 s below half the expected rate; power-cycle the
     radar + DCA1000;
   - SD card not found;
   - no LiDAR data;
   - camera not detected;
   - disk full;
   - overheating.

Sessions are written to the **SD card**: `/mnt/sdcard/sessions/<name>/` by
default (`--sessions-root`). The camera records to `/home/jetson/camera_stage`
on the eMMC during the session, and the launcher moves it into the session
afterwards (`--camera-stage`; `none` disables it). Sharing the SD card with the
radar's 25 MB/s stream made the ZED drop 2–5% of frames; staging brings it
to about 0.2%.

If the card is missing, the launcher refuses to start and the page shows a hint.
Sessions recorded before 2026-10-10 are still in
`sync_capture_pipeline/sessions/` and do not appear on the page.

**No authentication:** anyone on the same Wi-Fi can start and stop recordings.

## Timing and beeps

Every session behaves the same:

- **Duration:** 0 = record until **Stop** is pressed. Any other value stops
  the session by itself after that many seconds of radar recording; Stop
  still works earlier.
- **Countdown:** delays the start, so the operator can get into position.
- **Beeps:**
  - one long beep when recording starts;
  - three short beeps when it ends (duration reached, Stop pressed, or a
    sensor failure).
  - Sound on the iPad works only after a tap on the page, which the Start
    button provides.
- **Banner:** shows elapsed time, and the set duration when there is one.

**Stage 3 calibration guide (opt-in).**

- Tick *"Stage 3 calibration guide"* before Start to get the plate routine:
  1. walk 4 ↔ 14 m;
  2. side-steps 2–3 m each way at about 8 m;
  3. flips.
- There is a double beep at each phase change. The phases scale with the
  duration: 90 s → 0 / 45 / 75 s.
- The tick clears itself after Start, so the next session is a plain
  recording again.
- The guide is never triggered by the session name. **Leave it off for data
  collection.**

## Live view (for aligning the sensors)

The **Live view** card shows the three sensors live, for the
radar–LiDAR–camera extrinsic alignment. The panels are the camera's left
image, the LiDAR top view (optionally with the radar detections drawn on
it), and the radar range–azimuth map with our detections. A table shows
the strongest reflector per sensor (range, azimuth) inside a range window
you set, and the radar − LiDAR difference.

- **Nothing is saved.**
- **Never at the same time as a recording:**
  - Start recording is off while live view is on.
  - Start live view is off while recording.
- **It switches itself off after 10 minutes.** Press Start again for more.
- **Sensors** are ticked as for recording; all three by default.

Details, the design and what is still to verify on the rig:
[../live/README.md](../live/README.md).

## How it works

| What | How |
|---|---|
| Start | Runs `capture_session.py` exactly as a terminal would, in its own session |
| Stop | Sends SIGINT (Ctrl+C) to the launcher |
| Status | The server follows `<session>/session.log`, which the launcher writes line by line |
| Check | `check_session.py` runs after the launcher exits |

**A recording survives the web server.** The launcher never writes to a pipe
owned by the server. Its output goes to `sessions/.launcher/<name>.txt` and
`session.log`. If the server crashes or restarts, the recording continues.
When the server comes back, it finds the running `capture_session.py`,
re-reads its log so live status returns, and Stop still works. This was tested
by killing the server with SIGKILL during a three-sensor recording: the
session completed, and the check passed.

## When the page shows an error

Under the title, the page says what failed. Since 2026-11-02 it no longer
shows a bare "connection lost":

- **no connection:** the Jetson did not answer. Check the Wi-Fi, and check
  that the service runs (`systemctl status capture-web`).
- **server error 500 on /api/…: <error>:** the server hit a bug. It answers
  in JSON with the error, and the full traceback is in its log:
  `journalctl -u capture-web --since "1 hour ago"`. Send both.
- **unreadable reply:** the answer was not valid JSON.

Errors in the live view appear on its card.

The server never writes NaN or Infinity in its JSON; it writes `null`
instead. Browsers refuse to parse NaN, and one such value would leave the
page blank.

## Run it by hand

```bash
/home/jetson/miniconda3/envs/rel_1_py_3_8/bin/python web_control/server.py --port 8080
```

Run this from `sync_capture_pipeline/`. The live view's workers use the
server's interpreter (`--live-python`), which must be the conda
environment (torch, pyzed, the LiDAR wrapper). Without sensors, add
`--live-from-session <session>`, and live view plays that recording
instead.

## Install as a service (starts at boot)

Install it with the setup script, as the user who should run it
(normally `jetson`), from `sync_capture_pipeline/`. Add `--hotspot` for
the Wi-Fi hotspot:

```bash
sudo bash setup_jetson_capture.sh
systemctl status capture-web          # should be "active (running)"
journalctl -u capture-web -f          # server log
```

`capture-web.service` in this folder is a **template**. The script fills
in the user who ran sudo, this checkout's folder and that user's conda
Python (`~/miniconda3/envs/rel_1_py_3_8`, or `CAPTURE_PYTHON=...`). It
checks that this Python can import flask and psutil, then writes
`/etc/systemd/system/capture-web.service`.

**Do not copy the template by hand.** Until 2026-11-04 the file named user
`jetson` and `/home/jetson/dev/CascadedRadarRoadSafetyOnTwoWheelers`, and a
copy failed on any other board or checkout folder, restarting every 3 s.
If the page does not load, the script's "== Check" output names the cause;
see also `docs/knowledge/02_hardware_and_network.md`, "Wi-Fi for the iPad".

The unit uses `KillMode=process`. Restarting or stopping the service stops
only the web server, never a recording in progress. **Do not remove it.**

### Recommended alongside: persistent system settings

The service runs as `jetson` and cannot change kernel settings. These were set
by hand and are lost on reboot:

```bash
# UDP receive buffers for the radar and LiDAR sockets (32 MB)
echo 'net.core.rmem_max=33554432' | sudo tee /etc/sysctl.d/90-sensor-capture.conf
sudo sysctl --system
```

The launcher's preflight warns if `rmem_max` is too small, and the warning
appears in the web page's session log.
