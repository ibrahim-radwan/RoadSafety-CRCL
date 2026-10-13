# Capture web control

Start and stop three-sensor recording sessions from a browser, so you
don't need a screen, keyboard or SSH on the scooter.

## Use it

1. Connect the iPad to the **same Wi-Fi** as the Jetson.
2. Open `http://<jetson-wifi-ip>:8080`. To find the IP on the Jetson, run
   `ip -4 -br addr show wlan0`.
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

## Run it by hand

```bash
/home/jetson/miniconda3/envs/rel_1_py_3_8/bin/python web_control/server.py --port 8080
```

Run this from `sync_capture_pipeline/`.

## Install as a service (starts at boot)

```bash
sudo cp web_control/capture-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now capture-web
systemctl status capture-web          # should be "active (running)"
journalctl -u capture-web -f          # server log
```

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
