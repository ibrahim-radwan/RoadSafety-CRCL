#!/usr/bin/env python3
"""Verify a recorded SVO2 offline: frame count, then every frame decoded and its timestamp checked.

This is the check the recorder used to run at stop (it took ~0.6x the recording
time and blocked the next session). Since 2026-11-23 the recorder checks only the
frame count at stop (config.yaml recording.verify_after_stop: false), and this
tool does the rest when the rig is free, as part of the full session check.

It reads the camera folder and writes ONE new file, camera/svo_verification.json,
which check_session.py reads. It never changes the recording. It also works on a
session whose camera stopped early (e.g. a full disk): it then tells whether the
frames saved up to the stop are readable.

Needs the ZED SDK and pyzed (the conda env), the same SDK version that recorded.

USAGE
    python verify_svo.py --session /mnt/sdcard/sessions/run_20261123_1619
    python verify_svo.py --camera-dir <folder with camera.svo2, capture/, calibration/>
"""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

from dataset import load_config, read_svo_frame, require_success, save_json, sdk_init, sdk_runtime

RESULT = 'svo_verification.json'


def open_svo(sl, config, cam_dir):
    """The camera folder's SVO2, opened for reading without depth."""
    camera = sl.Camera()
    parameters = sdk_init(sl, config, Path(cam_dir) / 'camera.svo2', Path(cam_dir) / 'calibration')
    parameters.depth_mode = sl.DEPTH_MODE.NONE
    parameters.depth_stabilization = 0
    require_success(sl, camera.open(parameters), 'Open SVO2')
    return camera


def decode_all(sl, config, camera, frames_csv, progress=None):
    """Decode every frame listed in frames.csv and check its timestamp; returns
    (frames decoded, decoder recoveries). Raises on the first frame that cannot be read."""
    runtime = sdk_runtime(sl, config)
    runtime.enable_depth = False
    done = recoveries = 0
    with Path(frames_csv).open() as handle:
        for entry in csv.DictReader(handle):
            _, recovered = read_svo_frame(sl, camera, runtime, int(entry['frame_index']), int(entry['timestamp_ns']))
            recoveries += int(recovered)
            done += 1
            if progress:
                progress(done)
    return done, recoveries


def verify(cam_dir, config, sl=None, quiet=False):
    """Verify cam_dir and write cam_dir/svo_verification.json; returns the result."""
    if sl is None:
        import pyzed.sl as sl
    cam_dir = Path(cam_dir)
    status_path = cam_dir / 'capture_status.json'
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    frames_csv = cam_dir / 'capture' / 'frames.csv'
    with frames_csv.open() as handle:
        listed = sum(1 for _ in csv.DictReader(handle))
    expected = status.get('recorded_frames', listed)
    t0 = time.monotonic()
    result = {'ok': False, 'error': None, 'frames_expected': expected, 'frames_in_frames_csv': listed,
              'frames_in_svo': None, 'frames_decoded': 0, 'decode_recoveries': 0,
              'recorder_complete': status.get('complete'), 'recorder_error': status.get('error'),
              'sdk_version': sl.Camera.get_sdk_version(), 'tool': 'verify_svo.py'}
    camera = None
    next_print = [t0 + 10.0]

    def progress(n):
        result['frames_decoded'] = n          # kept if a later frame fails
        now = time.monotonic()
        if not quiet and now >= next_print[0]:
            next_print[0] = now + 10.0
            print(f'  {n}/{listed} frames decoded ({n / (now - t0):.0f} frames/s)', flush=True)

    try:
        if listed != expected:
            raise RuntimeError(f'frames.csv lists {listed} frames, the recorder counted {expected}')
        camera = open_svo(sl, config, cam_dir)
        result['frames_in_svo'] = camera.get_svo_number_of_frames()
        if result['frames_in_svo'] != expected:
            raise RuntimeError(f'SVO2 holds {result["frames_in_svo"]} frames, the recorder counted {expected}')
        result['frames_decoded'], result['decode_recoveries'] = decode_all(sl, config, camera, frames_csv, progress)
        result['ok'] = True
    except Exception as exc:
        result['error'] = str(exc)
    finally:
        if camera is not None:
            camera.close()
    result['seconds'] = round(time.monotonic() - t0, 1)
    result['verified_local'] = datetime.now().astimezone().isoformat()
    save_json(cam_dir / RESULT, result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument('--session', type=Path, help='a capture_session.py session (uses <session>/camera)')
    where.add_argument('--camera-dir', type=Path, help='a recorder output folder')
    ap.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    a = ap.parse_args()
    cam_dir = a.camera_dir or a.session / 'camera'
    if not (cam_dir / 'camera.svo2').is_file():
        print(f'No camera.svo2 in {cam_dir}')
        return 1
    print(f'Verifying {cam_dir / "camera.svo2"} ...', flush=True)
    r = verify(cam_dir, load_config(a.config))
    if r['ok']:
        print(f'OK: {r["frames_decoded"]} frames decoded, timestamps match frames.csv '
              f'({r["decode_recoveries"]} decoder recoveries) in {r["seconds"]} s')
    else:
        print(f'FAILED: {r["error"]}  ({r["frames_decoded"]} frames decoded before it)')
    print(f'Result: {cam_dir / RESULT}')
    return 0 if r['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
