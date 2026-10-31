# Live view

Watch the three sensors live on the iPad page (`web_control/`) while you
align them: the radar–LiDAR–camera extrinsic calibration. **Nothing is
saved.** Added 2026-10-31.

## What the page shows

| Panel | Content | Rate | Size |
|---|---|---|---|
| Camera | ZED 2i left image, 640 × 360 (live: HD720 at 15 fps, no depth) | 5 /s | ~28 kB |
| LiDAR | The replay's top view (`replay/sources.py` `render_lidar`): forward up, vehicle-left on the left, coloured by height above our local ground. Optional overlay of the radar detections | 4 /s | ~16 kB |
| Radar | Range–azimuth fan with our detections (`perception/radar.py`): the **stationary** map (zero Doppler, no clutter removal; shows a corner reflector) or the **moving-target** map | 4 /s | ~28 kB |

That is about 0.3 MB/s (~2.6 Mbit/s) over the Wi-Fi, measured on
`freewalk_ref_02`. The LiDAR and radar views share one extent
(forward 0..R, lateral ±0.75 R), so radar marks drawn over the LiDAR image
are in the same frame. The radar is assumed to sit where the LiDAR is,
because there are no extrinsics yet; finding that offset is what the view
is for.

**The strongest reflector** (the table above the panels), inside the
range window you set on the page:

- **Radar:** the strongest stationary detection, by power. Multipath ghosts
  are left out.
- **LiDAR:** the brightest spot. Returns at or above the sweep's 99.9th
  intensity percentile (at least 100 of 255) are binned in 0.5 m cells, and
  the cell with the highest summed intensity wins. Corner reflectors are by
  far the brightest LiDAR returns.
- **Radar − LiDAR:** the range and azimuth difference. If the two points
  are more than 1.5 m apart, the row says they are probably not the same
  object.

Azimuth is positive to the **left**, as everywhere in the pipeline
(`perception/core/conventions.py`). LiDAR-over-camera needs the team's
camera calibration and is not done yet.

## Live view and recording never run together

- **Both start buttons go through one lock** in `web_control/server.py`.
  - A live view is refused while a recording runs, including one started
    from a terminal: the server looks for `capture_session.py` and the
    recorder processes.
  - A recording is refused while any live-view worker is still running,
    including while one is stopping.
- **Live view switches itself off after 10 minutes.** The server stops it
  at 600 s. Each worker also has its own deadline (615 s), in case the
  server does not.
- **The workers cannot outlive the server.** They are its children, linked
  by pipes. When the server stops or crashes, their stdin closes and they
  stop. Tested: SIGKILL of the server stopped all three workers within 2 s.
  At start-up the server also ends any leftover workers.
- **What the server-side lock does not cover:** `capture_session.py` started
  from a terminal while live view is on. The two radar receivers would then
  share the DCA1000's data port. Don't do that; the page is the way to run
  both.

## How it works

```
web_control/server.py ── liveview.LiveView ──┬─ live/radar_live.py   (conda env, GPU)
   /api/live/start|stop|control|state|log    ├─ live/lidar_live.py   (conda env, GPU)
   /api/live/frame/<sensor>?after=<seq>      └─ live/camera_live.py  (conda env, pyzed)
```

- **One process per sensor,** as for recording. `pyzed.grab()` holds the GIL,
  and the radar's receive thread must not wait on anything.
- **Worker → server** (`live/common.py`): stdout carries length-prefixed
  messages, each a JSON header (numbers, detections, extent) plus a JPEG.
  stdin carries control lines from the server (`{"range_m": 20}`,
  `{"radar_map": "static"}`). All printing, including from C++ (rs_driver,
  the ZED SDK), goes to stderr and appears in the page's "Live view log".
- **The server keeps only the newest message** per sensor, in memory.
- **The page pulls images.** Each panel asks for the next image only after
  the previous one has arrived, so a slow Wi-Fi lowers the rate instead of
  building a backlog. The numbers come from `/api/live/state` four times a
  second, and the page draws them on canvases.

### The recorders are not modified

- **Radar** (`radar_live.py`):
  - It imports `dca_capture_radar_sync.py`'s `send_recv`, constants and
    `Receiver`, and arms the DCA1000 with the same command sequence, waits
    and 25 µs packet delay as its `main()`. **Keep the two sequences in
    step.**
  - The receiver's packet loop is unchanged. Only its disk writer is replaced,
    by a one-frame slot that the newest frame overwrites. The receive thread
    never waits on processing; frames in between are dropped.
  - STOP_RECORD is sent on every exit path.
  - Processing: `RadarDetector(..., rx_phase_deg=<in-situ>)`
    `.process(raw, moving_only=True, static=True)`, about 8 ms of GPU per
    frame, 30–60 ms with the rendering.
- **LiDAR** (`lidar_live.py`): our rs_driver wrapper on the live sockets,
  with the recorder's `rs_driver/config/config.yaml` (`split_angle: 180`).
  Then `LidarDetector` for the ground height, and `render_lidar`.
- **Camera** (`camera_live.py`): opens the ZED without `enable_recording`.

`dca_capture_radar_sync.py`, `recorder.py`, `record_lidar_raw.py` and
`capture_session.py` are unchanged. The `tool_sha256` hashes in
`session.json` stay the same.

## Without the sensors: a recorded session

```bash
cd sync_capture_pipeline
python web_control/server.py --port 8099 --sessions-root /tmp/sessions --camera-stage none \
    --live-from-session /media/.../freewalk_ref_02
```

The workers then play the session's radar frames, `lidar.pcap` and
`camera.svo2` in real time, looping. The page says so under the panels.
This is how live view was developed and tested on `ibrahim-desktop`, which
has no sensors. A single worker can be run the same way:
`python live/radar_live.py --from-session <session>`. Its messages are
binary on stdout.

Tests: `tests/test_live_view.py`:
- the lockout both ways (stand-ins, no data needed);
- on a reference session: images and numbers from all three workers, the
  time limit, clean exits, no file in the session changed, and a worker
  stopping when its server goes away.

## Not yet verified on the rig (2026-10-31)

Live view was built on `ibrahim-desktop`, which has no sensors. Still to do
on the rig:

1. Live DCA1000 control in live view, including STOP_RECORD at the end.
2. The live LiDAR sockets.
3. Opening the live ZED at HD720/15.
4. The page on the iPad over the rig's Wi-Fi.
5. A normal recording straight after a live view, which must PASS
   `check_session`.
6. The radar's packet rate in live view. The receiver shares its process
   with the GPU processing, so the "pkt/s" on the radar panel should stay
   near ~17,300 at 8 fps. Lost packets are zero-filled and frame boundaries
   stay correct, so live view tolerates some loss; recording never shares
   that process.
