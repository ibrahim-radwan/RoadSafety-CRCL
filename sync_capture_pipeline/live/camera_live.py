#!/usr/bin/env python3
"""
live/camera_live.py - the camera's live view: the ZED 2i's left image, scaled down, ~5 times a
second. Nothing is written to disk (recording is never enabled).

INPUT
    live            the ZED 2i at HD720 / 15 fps, no depth (the recorder's 1080p / 30 fps is not
                    needed for a preview and costs USB and GPU time).
    --from-session  a recorded session's camera.svo2 in real time (looping).

Runs in its own process like the recorder and the replay's camera: pyzed's grab() holds the GIL.
"""
from __future__ import annotations

import argparse
import time

from common import Link, Rate, add_args, jpeg, log

WIDTH, HEIGHT = 640, 360


def main() -> int:
    ap = argparse.ArgumentParser(description="Camera live view (no saving)")
    add_args(ap)
    ap.set_defaults(hz=5.0)
    a = ap.parse_args()
    link = Link("camera", a.max_secs)
    cam = None
    try:
        import pyzed.sl as sl
        init = sl.InitParameters()
        init.depth_mode = sl.DEPTH_MODE.NONE
        init.sdk_verbose = 0
        if a.from_session:
            init.set_from_svo_file(str(a.from_session / "camera" / "camera.svo2"))
            init.svo_real_time_mode = True
            init.optional_settings_path = str(a.from_session / "camera" / "calibration")
            log(f"camera: playing {a.from_session / 'camera' / 'camera.svo2'}")
        else:
            if len(sl.Camera.get_device_list()) == 0:
                link.error("camera not detected - unplug and replug the ZED USB cable")
                return 1
            init.camera_resolution = sl.RESOLUTION.HD720
            init.camera_fps = 15
        cam = sl.Camera()
        code = cam.open(init)
        if code != sl.ERROR_CODE.SUCCESS:
            link.error(f"cannot open the camera: {code}")
            cam = None
            return 1
        rt, mat, res = sl.RuntimeParameters(), sl.Mat(), sl.Resolution(WIDTH, HEIGHT)
        rate, period, next_send, fails = Rate(), 1.0 / max(0.2, a.hz), time.monotonic(), 0
        while link.running():
            code = cam.grab(rt)
            if code == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                cam.set_svo_position(0)
                continue
            if code != sl.ERROR_CODE.SUCCESS:
                fails += 1
                if fails > 200:
                    link.error(f"camera grab failing: {code}")
                    return 1
                link.stop.wait(0.01)
                continue
            fails = 0
            now = time.monotonic()
            if now < next_send:                     # grab every frame, send hz of them
                continue
            next_send = max(next_send + period, now)
            cam.retrieve_image(mat, sl.VIEW.LEFT, sl.MEM.CPU, res)
            rgb = mat.get_data()[..., 2::-1]          # BGRA -> RGB
            rate.tick()
            link.send({"t": now, "w": WIDTH, "h": HEIGHT, "hz": round(rate.hz, 1),
                       "fps": round(float(cam.get_current_fps()), 1)}, jpeg(rgb, 70))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        link.error(str(exc))
        return 1
    finally:
        if cam is not None:
            cam.close()
        log("camera: live view stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
