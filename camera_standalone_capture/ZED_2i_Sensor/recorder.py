#!/usr/bin/env python3
"""Save ZED 2i stereo images, independent IMU/sensors and an SVO2 archive, without ROS."""
import argparse
import math
import shutil
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml
from images import ImageSink
from sensors import collect_sensors

from dataset import (CsvSink, calibration_data, capture_space_headroom, check_size, clock_check, dataset_root,
                     file_hash, load_config, pose_record,
                     read_svo_frame, require_success, save_json, sdk_init, sdk_runtime, image_views)
from dataset import validate_config

def capture(config, config_path, duration=None, sl=None, stop=None, session_dir=None):
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError('Duration must be a finite positive number')
    if sl is None:
        import pyzed.sl as sl
    version = sl.Camera.get_sdk_version()
    if tuple(int(v) for v in version.split('.')[:2]) < (4, 1):
        raise RuntimeError('ZED SDK 4.1 or newer is required for SVO2')
    camera = sl.Camera()
    stop = stop or threading.Event()
    sensor_thread = None
    sink = None
    image_sink = None
    image_mats = {}
    session = None
    recording = False
    error = None
    state = {'imu_ready': threading.Event()}
    count, valid_poses = 0, 0
    unconfirmed_encoder_frames = 0
    svo_decode_recoveries = 0
    startup = time.monotonic()
    first_stamp = last_stamp = None
    gap_estimate = 0
    started = None
    try:
        if config['camera']['serial_number'] is None:
            devices = sl.Camera.get_device_list()
            if len(devices) > 1:
                raise RuntimeError('Multiple cameras found; configure camera.serial_number to select the ZED 2i')
        parameters = sdk_init(sl, config)
        if not config['tracking']['enabled']:
            parameters.depth_mode = sl.DEPTH_MODE.NONE
            parameters.depth_stabilization = 0
        require_success(sl, camera.open(parameters), 'Open camera')
        info = camera.get_camera_information()
        if info.camera_model != sl.MODEL.ZED2i:
            raise RuntimeError(f'Expected ZED 2i, detected {info.camera_model}')
        size = info.camera_configuration.resolution
        check_size(size.width, size.height, config)
        if abs(info.camera_configuration.fps - config['camera']['fps']) > .01:
            raise RuntimeError('SDK changed requested camera FPS; recording aborted')
        if config['tracking']['enabled']:
            tracking = sl.PositionalTrackingParameters()
            tracking.enable_imu_fusion = config['tracking']['imu_fusion']
            tracking.enable_area_memory = config['tracking']['area_memory']
            tracking.enable_pose_smoothing = config['tracking']['pose_smoothing']
            require_success(sl, camera.enable_positional_tracking(tracking), 'Enable tracking')
        runtime = sdk_runtime(sl, config)
        runtime.enable_depth = config['tracking']['enabled']
        warm_until = time.monotonic() + config['runtime']['warmup_sec']
        last_success = time.monotonic()
        while not stop.is_set():
            code = camera.grab(runtime)
            if int(code.value) <= 0:
                last_success = time.monotonic()
                if time.monotonic() >= warm_until:
                    break
            elif time.monotonic() - last_success > config['runtime']['grab_failure_timeout_sec']:
                raise RuntimeError(f'No usable camera frames: {code}')
            else:
                stop.wait(.005)
        if stop.is_set():
            raise RuntimeError('Stopped before camera validation completed')
        image_stamp = int(camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
        image_offset = clock_check(image_stamp, config['synchronization']['timestamp_tolerance_sec'])
        # Validate native image dimensions; depth is never computed in minimal capture.
        mat = sl.Mat()
        for view in (sl.VIEW.LEFT, sl.VIEW.RIGHT, sl.VIEW.LEFT_UNRECTIFIED, sl.VIEW.RIGHT_UNRECTIFIED):
            require_success(sl, camera.retrieve_image(mat, view), f'Validate {view}')
            check_size(mat.get_width(), mat.get_height(), config)
        if config['tracking']['enabled']:
            require_success(sl, camera.retrieve_measure(mat, sl.MEASURE.DEPTH), 'Validate depth')
            check_size(mat.get_width(), mat.get_height(), config)
        mat.free()
        # An explicit session directory lets a multi-sensor launcher place
        # this recording inside its own session folder.
        root = Path(session_dir).parent if session_dir else dataset_root(config, config_path)
        root.mkdir(parents=True, exist_ok=True)
        minimum_free = config['recording']['min_free_space_gb'] * 1024**3
        space_threshold = minimum_free + capture_space_headroom(config)
        if shutil.disk_usage(root).free < space_threshold:
            raise RuntimeError('Insufficient free recording space')
        session = Path(session_dir) if session_dir else \
            root / (config['recording']['session_prefix'] + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        session.mkdir()
        (session / 'config_used.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        calibration = calibration_data(camera.get_camera_information(), config)
        save_json(session / 'calibration/camera.json', calibration)
        factory_name = f'SN{info.serial_number}.conf'
        factory = Path('/usr/local/zed/settings') / factory_name
        if not factory.is_file():
            raise RuntimeError(f'Cannot preserve SDK factory calibration: {factory}')
        shutil.copy2(factory, session / 'calibration' / factory_name)
        metadata = {'schema_version': 1, 'sdk_version': version, 'camera_model': str(info.camera_model),
                    'serial_number': int(info.serial_number), 'width': size.width, 'height': size.height,
                    'requested_fps': config['camera']['fps'], 'clock_domain': 'SYSTEM_CLOCK',
                    'image_clock_offset_sec': image_offset, 'compression': config['recording']['compression'],
                    'stereo_pixel_lossless': config['recording']['compression'] in ('LOSSLESS', 'H264_LOSSLESS', 'H265_LOSSLESS'),
                    'factory_calibration_file': factory_name, 'factory_calibration_sha256': file_hash(factory),
                    'derived_outputs': 'Depth/disparity/confidence available through optional offline SVO2 export',
                    'live_outputs': 'Direct stereo images when enabled; SVO2 stereo; original calibrated/raw IMU and physical sensors; optional SDK poses',
                    'direct_image_format': config['images']['format'] if config['images']['enabled'] else None,
                    'direct_image_pixel_lossless': config['images']['format'] != 'JPEG' if config['images']['enabled'] else None,
                    'live_depth_enabled': config['tracking']['enabled'],
                    'live_tracking_enabled': config['tracking']['enabled'],
                    'direct_image_streams': config['images']['streams'] if config['images']['enabled'] else [],
                    'capture_write_headroom_bytes': capture_space_headroom(config),
                    'recording_started_local': datetime.now().astimezone().isoformat()}
        save_json(session / 'metadata.json', metadata)
        save_json(session / 'capture_status.json', {'complete': False, 'phase': 'capturing',
                  'error': 'Capture has not been finalized'})
        save_json(session / 'dataset_schema.json', {
            'clock_domain': 'SYSTEM_CLOCK', 'timestamp_units': 'integer nanoseconds',
            'alignment_timestamp': 'Original SDK acquisition timestamps, not host_receipt_ns or file write time',
            'monotonic_conversion': 'timestamp_ns - (host_receipt_ns - host_receipt_mono_ns) gives CLOCK_MONOTONIC; '
                                    'both host stamps are taken back-to-back',
            'coordinate_system': config['camera']['coordinate_system'],
            'orientation_quaternion_order': 'x,y,z,w; original SDK IMU reference',
            'angles': 'roll_rad,pitch_rad,yaw_rad; intrinsic ZYX convention, SDK reference; yaw is not absolute heading; yaw=0 at gimbal lock, quaternion is authoritative',
            'gyroscope': 'rad/s; angular velocity about sensor axes, not Euler angle derivatives',
            'accelerometer': 'm/s^2; includes gravity',
            'imu_csv_sampling': 'Latest CURRENT samples observed by Python; SVO2 also retains native SDK sensor archive',
            'camera_calibration': 'calibration/camera.json and factory SN*.conf; raw/rectified parameters must match chosen images',
            'image_files': ('capture/images/<stream>/<original_timestamp_ns>_<frame_index>.' +
                           ('jpg' if config['images']['format'] == 'JPEG' else config['images']['format'].lower()))
                           if config['images']['enabled'] else 'export/<stream>/<original_timestamp_ns>_<frame_index>.png',
            'world_pose': 'Available only with optional live tracking enabled; otherwise derive offline'})
        sink = CsvSink(session / 'capture', config['recording']['csv_queue_records'])
        if config['images']['enabled']:
            image_sink = ImageSink(session / 'capture', config, sink)
            views = dict(image_views(sl, config['images']['streams']))
            image_mats = {name: sl.Mat() for name in config['images']['streams']}
            for name, image_mat in image_mats.items():
                # Allocate GPU/CPU retrieval buffers before the recording loop.
                require_success(sl, camera.retrieve_image(image_mat, views[name]), f'Warm live {name}')
        sensor_thread = threading.Thread(target=collect_sensors,
            args=(sl, camera, config, sink, stop, state), name='zed-sensors')
        sensor_thread.start()
        if not state['imu_ready'].wait(timeout=5) or state.get('error'):
            raise RuntimeError('IMU startup failed: ' + str(state.get('error', 'no samples')))
        params = sl.RecordingParameters(str(session / 'camera.svo2'),
                    getattr(sl.SVO_COMPRESSION_MODE, config['recording']['compression']))
        require_success(sl, camera.enable_recording(params), 'Enable SVO2 recording (try LOSSLESS if the hardware encoder is unavailable)')
        recording = True
        pose = sl.Pose() if config['tracking']['enabled'] else None
        started = time.monotonic()
        metadata['startup_sec'] = started - startup
        metadata['imu_clock_offset_sec'] = state['imu_clock_offset_sec']
        save_json(session / 'metadata.json', metadata)
        next_status = next_space = started
        last_success = started
        print(f'Recording {size.width}x{size.height} at requested {config["camera"]["fps"]} FPS; SYSTEM_CLOCK', flush=True)
        print('Session:', session, flush=True)
        while not stop.is_set() and (duration is None or time.monotonic() - started < duration):
            if image_sink is not None:
                image_sink.prepare()
            if sink.error or state.get('error'):
                raise RuntimeError('Sensor/CSV capture failed: ' + str(state.get('error') or sink.error))
            code = camera.grab(runtime)
            if int(code.value) > 0:
                if time.monotonic() - last_success > config['runtime']['grab_failure_timeout_sec']:
                    raise RuntimeError(f'Camera grab stalled: {code}')
                stop.wait(.005)
                continue
            last_success = time.monotonic()
            timestamp = int(camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
            host = time.time_ns()
            # Same instant on CLOCK_MONOTONIC: the clock the radar and LiDAR
            # recorders log. host - host_mono converts timestamp_ns per frame.
            host_mono = time.monotonic_ns()
            status = camera.get_recording_status()
            # Hardware encoding is asynchronous: startup status can lag the grab.
            # The finalized SVO frame count/timestamps are the authoritative check.
            if not status.status:
                unconfirmed_encoder_frames += 1
            frame_interval_ns = 0 if last_stamp is None else timestamp-last_stamp
            missing_before = 0
            if last_stamp is not None:
                if timestamp <= last_stamp:
                    raise RuntimeError('Camera timestamp repeated or moved backwards')
                missing_before = max(0, round(frame_interval_ns*config['camera']['fps']/1e9) - 1)
                gap_estimate += missing_before
                if missing_before:
                    sink.put('timing/camera_gaps.csv', {'frame_index': count,
                        'timestamp_ns': timestamp, 'previous_timestamp_ns': last_stamp,
                        'frame_interval_ns': frame_interval_ns, 'estimated_missing_frames': missing_before})
            first_stamp = timestamp if first_stamp is None else first_stamp
            last_stamp = timestamp
            frame_row = {'frame_index': count, 'timestamp_ns': timestamp, 'host_receipt_ns': host,
                       'host_receipt_mono_ns': host_mono,
                       'frame_interval_ns': frame_interval_ns, 'estimated_missing_frames_before': missing_before,
                       'encoder_reported_status': bool(status.status), 'encoder_reported_recording': bool(status.is_recording),
                       'compression_time_ms': float(status.current_compression_time),
                       'compression_ratio_percent': float(status.current_compression_ratio)}
            if image_sink is not None:
                arrays = {}
                for name, image_mat in image_mats.items():
                    require_success(sl, camera.retrieve_image(image_mat, views[name]), f'Retrieve live {name}')
                    view = image_mat.get_data()
                    arrays[name] = view.copy() if image_sink.backend == 'thread' else view
                image_sink.put(count, timestamp, host, arrays)
                frame_row.update({name + '_file': filename for name, filename in image_sink.filenames(count, timestamp).items()})
                frame_row.update({'width': size.width, 'height': size.height})
            sink.put('frames.csv', frame_row)
            if pose is not None:
                tracking_state = camera.get_position(pose, sl.REFERENCE_FRAME.WORLD)
                row = pose_record(pose, tracking_state, timestamp, host, count)
                sink.put('pose/pose.csv', row)
                if row['valid'] and tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                    valid_poses += 1
            count += 1
            now = time.monotonic()
            if now >= next_space:
                clock_check(timestamp, config['synchronization']['timestamp_tolerance_sec'], host)
                if shutil.disk_usage(session).free < space_threshold:
                    raise RuntimeError('Minimum free space reached; recording stopped')
                next_space = now + 2
            if now >= next_status:
                observed = (count-1)*1e9/(last_stamp-first_stamp) if count > 1 else 0.
                queue_status = f' | image queue {len(image_sink.pending)}/{image_sink.limit}' if image_sink else ''
                print(f'{count} frames | observed {observed:.2f} FPS | SDK {camera.get_current_fps():.2f} FPS{queue_status} | gaps {gap_estimate}', flush=True)
                next_status = now + config['runtime']['status_interval_sec']
            # SDK 4.1's blocking Python grab can hold the GIL while awaiting the
            # next image. Yield part of the frame interval to independent sensor
            # polling instead of spending that interval inside grab().
            stop.wait(max(0., config['sensors']['poll_window_fraction'] / config['camera']['fps']
                          - (time.monotonic() - last_success)))
        if count == 0 or (config['tracking']['enabled'] and valid_poses == 0):
            raise RuntimeError('Recording lacks frames or valid tracked poses')
    except Exception as exc:
        error = exc
    finally:
        finish_started = time.monotonic()
        stop.set()
        if sensor_thread is not None:
            sensor_thread.join()
        if state.get('error') and error is None:
            error = RuntimeError(state['error'])
        if recording:
            try:
                camera.disable_recording()
            except Exception as exc:
                error = error or exc
        try:
            camera.close()
        except Exception as exc:
            error = error or exc
        for image_mat in image_mats.values():
            try:
                image_mat.free()
            except Exception as exc:
                error = error or exc
        if image_sink is not None:
            try:
                image_sink.close()
                if any(image_sink.counts[name] != count for name in image_sink.streams):
                    raise RuntimeError('Direct image counts differ from successful frame ledger')
            except Exception as exc:
                error = error or exc
        if sink is not None:
            try:
                sink.close()
            except Exception as exc:
                error = error or exc
        if error is None and session is not None:
            verification = sl.Camera()
            try:
                parameters = sdk_init(sl, config, session / 'camera.svo2', session / 'calibration')
                parameters.depth_mode = sl.DEPTH_MODE.NONE
                parameters.depth_stabilization = 0
                require_success(sl, verification.open(parameters), 'Verify finalized SVO2')
                if verification.get_svo_number_of_frames() != count:
                    raise RuntimeError(f'Finalized SVO2 frame count differs from {count} successful grabs')
                import csv
                verification_runtime = sdk_runtime(sl, config)
                verification_runtime.enable_depth = False
                with (session / 'capture/frames.csv').open() as handle:
                    for entry in csv.DictReader(handle):
                        _, recovered = read_svo_frame(sl, verification, verification_runtime,
                            int(entry['frame_index']), int(entry['timestamp_ns']))
                        svo_decode_recoveries += int(recovered)
            except Exception as exc:
                error = exc
            finally:
                try:
                    verification.close()
                except Exception as exc:
                    error = error or exc
        if session is not None:
            result = {'complete': error is None, 'error': str(error) if error else None,
                      'phase': 'complete' if error is None else 'incomplete',
                      'recording_wall_sec': finish_started-started if started is not None else 0.,
                      'finalization_sec': time.monotonic()-finish_started,
                      'recorded_frames': count, 'valid_pose_frames': valid_poses,
                      'first_timestamp_ns': first_stamp, 'last_timestamp_ns': last_stamp,
                      'observed_fps': (count-1)*1e9/(last_stamp-first_stamp) if count > 1 else None,
                      'estimated_missing_camera_frames': gap_estimate,
                      'encoder_status_unconfirmed_frames': unconfirmed_encoder_frames,
                      'finalized_svo_frame_count_verified': error is None,
                      'finalized_svo_timestamps_verified': error is None,
                      'svo_decode_recoveries_during_verification': svo_decode_recoveries,
                      'direct_images': image_sink.stats() if image_sink else None,
                      'streams': sink.stats() if sink else {}, 'stopped_local': datetime.now().astimezone().isoformat()}
            save_json(session / 'capture_status.json', result)
    if error:
        raise RuntimeError(f'Recording failed: {error}. Session: {session}') from error
    print(f'Finalized {count} frames in {session}', flush=True)
    return session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    parser.add_argument('--check-config', action='store_true')
    parser.add_argument('--duration', type=float, help='Stop automatically after this many recording seconds.')
    parser.add_argument('--no-export', action='store_true', help='Keep SVO2/sidecars and export later.')
    parser.add_argument('--direct-images', action='store_true', help='Write configured image files during capture; can reduce FPS.')
    parser.add_argument('--image-format', choices=('BMP', 'PNG', 'JPEG'),
                        help='Override direct image format; JPEG is lossy and SVO2 archival quality is unchanged.')
    parser.add_argument('--jpeg-quality', type=int, help='Explicit JPEG quality 1..100 (default config: 95, 4:4:4).')
    parser.add_argument('--session-dir', type=Path,
                        help='Record into exactly this new directory instead of dataset_root/<prefix>_<time>.')
    args = parser.parse_args()
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        config = load_config(args.config)
        if args.direct_images:
            config['images']['enabled'] = True
        if args.image_format is not None:
            config['images']['format'] = args.image_format
            config['images']['enabled'] = True
        if args.jpeg_quality is not None:
            config['images']['jpeg_quality'] = args.jpeg_quality
        validate_config(config)
        if args.check_config:
            print('Valid standalone SDK config: ZED 2i, 1920x1080, SYSTEM_CLOCK; image format:', config['images']['format'],
                  '; save during capture:', config['images']['enabled'], '; auto export:', config['recording']['auto_export'])
            return 0
        session = capture(config, args.config, args.duration, stop=stop, session_dir=args.session_dir)
        if config['recording']['auto_export'] and not args.no_export:
            # Restore Ctrl+C for cancellation of the offline export after capture is finalized.
            signal.signal(signal.SIGINT, signal.default_int_handler)
            from export import export_session
            export_session(session)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print('Error:', exc)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
