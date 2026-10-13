#!/usr/bin/env python3
"""Export every SVO2 frame and the original live sensor/pose sidecars, without ROS."""
import argparse
import csv
import json
import shutil
import tempfile
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import (calibration_data, check_size, file_hash, load_config, save_json,
                     read_svo_frame, require_success, sdk_init, sdk_runtime, image_views)


STREAMS = ('left', 'right', 'left_raw', 'right_raw', 'depth', 'disparity', 'confidence')


def write_frame(root, index, stamp, sdk_stamp, arrays, config, decode_recovered=False):
    rows = []
    for name, array in arrays.items():
        check_size(array.shape[1], array.shape[0], config)
        color = name in ('left', 'right', 'left_raw', 'right_raw')
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (f'{stamp}_{index:08d}' + ('.png' if color else '.npy'))
        if color:
            # Native conversion avoids costly strided NumPy channel shuffles.
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 4:
                raise ValueError(f'{name} is not a BGRA8 SDK image')
            Image.frombytes('RGB',(array.shape[1],array.shape[0]),array,'raw','BGRX',0,1).save(
                path, compress_level=config['export']['png_compression'])
        else:
            if name != 'point_cloud' and (array.dtype != np.float32 or array.ndim != 2):
                raise ValueError(f'{name} must retain its native single-channel float32 data')
            np.save(path, array, allow_pickle=False)
        rows.append({'frame_index': index, 'timestamp_ns': stamp, 'sdk_replay_timestamp_ns': sdk_stamp,
                     'sdk_decode_recovered_by_seek': decode_recovered,
                     'stream': name, 'file': str(path.relative_to(root)),
                     'width': array.shape[1], 'height': array.shape[0],
                     'encoding': 'RGB8' if color else ('XYZRGBA_float32_packed_color' if name == 'point_cloud' else '32FC1'),
                     'units': {'depth': 'meters', 'disparity': 'pixels', 'confidence': 'SDK confidence 0(best)-100(worst)',
                               'point_cloud': 'XYZ meters; fourth float preserves packed RGBA bits'}.get(name, ''),
                     'source': 'recorded_stereo' if name.endswith('_raw') else 'SDK_recomputed_from_recorded_stereo'})
    return rows


def quaternion_inverse(q):
    return q * np.array([-1., -1., -1., 1.]) / np.dot(q, q)


def quaternion_multiply(a, b):
    av, aw, bv, bw = a[:3], a[3], b[:3], b[3]
    return np.r_[aw*bv + bw*av + np.cross(av, bv), aw*bw - np.dot(av, bv)]


def quaternion_matrix(q):
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def derived_trajectory(source, destination):
    """Keep live poses; express trajectory relative to first valid pose, label derived velocities."""
    writers, handles = {}, []
    origin = previous = None
    valid = 0

    def row(filename, value):
        if filename not in writers:
            path = destination / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open('w', newline='')
            handles.append(handle)
            writers[filename] = csv.DictWriter(handle, fieldnames=list(value))
            writers[filename].writeheader()
        writers[filename].writerow(value)

    try:
        with source.open() as handle:
            for item in csv.DictReader(handle):
                if item['valid'] != 'True' or item['tracking_state'] != 'OK':
                    previous = None
                    continue
                stamp = int(item['timestamp_ns'])
                p = np.array([float(item['position_m_' + a]) for a in 'xyz'])
                q = np.array([float(item['orientation_' + a]) for a in ['x', 'y', 'z', 'w']])
                if not np.all(np.isfinite(np.r_[p, q])) or np.linalg.norm(q) < .5:
                    raise ValueError('Invalid SDK pose marked valid')
                q /= np.linalg.norm(q)
                if origin is None:
                    origin = (p.copy(), q.copy())
                op, oq = origin
                relative_p = quaternion_matrix(oq).T @ (p - op)
                relative_q = quaternion_multiply(quaternion_inverse(oq), q)
                value = {'frame_index': item['frame_index'], 'timestamp_ns': stamp,
                         'reference_frame': 'first_valid_SDK_world_pose', 'source': 'derived_from_live_SDK_world_poses',
                         'velocity_available': False}
                value.update({'position_m_'+a: v for a, v in zip('xyz', relative_p)})
                value.update({'orientation_'+a: v for a, v in zip(['x','y','z','w'], relative_q)})
                for a in 'xyz':
                    value['linear_velocity_origin_m_s_' + a] = ''
                    value['angular_velocity_previous_body_rad_s_' + a] = ''
                if previous is not None:
                    previous_stamp, pp, pq = previous
                    dt = (stamp - previous_stamp) / 1e9
                    if dt <= 0:
                        raise ValueError('Pose timestamps are not increasing')
                    linear = (relative_p-pp)/dt
                    dq = quaternion_multiply(quaternion_inverse(pq), relative_q)
                    if dq[3] < 0:
                        dq = -dq
                    length = np.linalg.norm(dq[:3])
                    angular = dq[:3] * (2*np.arctan2(length, dq[3])/length/dt) if length > 1e-12 else np.zeros(3)
                    value['velocity_available'] = True
                    value.update({'linear_velocity_origin_m_s_'+a: v for a, v in zip('xyz', linear)})
                    value.update({'angular_velocity_previous_body_rad_s_'+a: v for a, v in zip('xyz', angular)})
                row('pose/odom.csv', value)
                transform = {'frame_index': item['frame_index'], 'timestamp_ns': stamp,
                             'parent_frame': 'zed_sdk_world', 'child_frame': 'zed_left_camera'}
                transform.update({'translation_m_'+a: v for a, v in zip('xyz', p)})
                transform.update({'rotation_'+a: v for a, v in zip(['x','y','z','w'], q)})
                row('tf/dynamic.csv', transform)
                previous = (stamp, relative_p.copy(), relative_q.copy())
                valid += 1
    finally:
        for handle in handles:
            handle.close()
    return valid


def copy_sidecars(session, output, status):
    counts = {}
    for name, stats in status['streams'].items():
        source = session / 'capture' / name
        if not source.is_file():
            raise ValueError(f'Missing live sidecar: {source}')
        with source.open() as handle:
            count = sum(1 for _ in csv.DictReader(handle))
        if count != stats['count']:
            raise ValueError(f'Sidecar row count mismatch: {name}')
        counts[name] = count
        if name not in ('frames.csv', 'images/frames.csv'):
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    needed_streams = ['imu/imu.csv', 'imu/imu_raw.csv']
    if load_config(session / 'config_used.yaml')['tracking']['enabled']:
        needed_streams.append('pose/pose.csv')
    for needed in needed_streams:
        if not counts.get(needed):
            raise ValueError(f'Missing essential sensor stream: {needed}')
    shutil.copytree(session / 'calibration', output / 'calibration')
    return counts


def export_data(session, output, sl, allow_sdk_version_change=False, include_depth=None):
    config = load_config(session / 'config_used.yaml')
    include_depth = config['export']['include_depth'] if include_depth is None else include_depth
    metadata = json.loads((session / 'metadata.json').read_text())
    status = json.loads((session / 'capture_status.json').read_text())
    if not status['complete']:
        raise ValueError('Capture is incomplete: ' + str(status.get('error')))
    if not allow_sdk_version_change and sl.Camera.get_sdk_version() != metadata['sdk_version']:
        raise ValueError('SDK version changed; use the recording SDK or explicitly --allow-sdk-version-change')
    factory = session / 'calibration' / metadata['factory_calibration_file']
    if file_hash(factory) != metadata['factory_calibration_sha256']:
        raise ValueError('Saved calibration file changed')
    sidecars = copy_sidecars(session, output, status)
    camera = sl.Camera()
    counts = Counter()
    mats = []
    try:
        parameters = sdk_init(sl, config, session / 'camera.svo2', session / 'calibration')
        if not include_depth:
            parameters.depth_mode = sl.DEPTH_MODE.NONE
            parameters.depth_stabilization = 0
        require_success(sl, camera.open(parameters), 'Open recorded SVO2')
        check_size(camera.get_camera_information().camera_configuration.resolution.width,
                   camera.get_camera_information().camera_configuration.resolution.height, config)
        frames_total = camera.get_svo_number_of_frames()
        if frames_total != status['recorded_frames'] or sidecars['frames.csv'] != frames_total:
            raise ValueError(f'SVO2/ledger frame count mismatch: SVO={frames_total}, capture={status["recorded_frames"]}')
        calibration = calibration_data(camera.get_camera_information(), config)
        recorded_calibration = json.loads((session / 'calibration/camera.json').read_text())
        if config['camera']['disable_self_calibration']:
            for mode in ('raw', 'rectified'):
                for eye in ('left', 'right'):
                    for field in ('k', 'distortion'):
                        if not np.allclose(calibration[mode][eye][field], recorded_calibration[mode][eye][field], rtol=1e-5, atol=1e-5):
                            raise ValueError('Replay calibration differs from the saved recording calibration')
        save_json(output / 'calibration/export_camera.json', calibration)
        streams = STREAMS[:4] if include_depth else config['images']['streams']
        specs = [(n, 'image', v) for n,v in image_views(sl, streams)]
        if include_depth:
            specs += [(n, 'measure', v) for n, v in [('depth', sl.MEASURE.DEPTH),
                      ('disparity', sl.MEASURE.DISPARITY), ('confidence', sl.MEASURE.CONFIDENCE)]]
        if include_depth and config['export']['point_cloud']:
            specs.append(('point_cloud', 'measure', sl.MEASURE.XYZRGBA))
        mats = [sl.Mat() for _ in specs]
        runtime = sdk_runtime(sl, config)
        runtime.enable_depth = include_depth
        expected_streams = [name for name,_,_ in specs]
        frame_bytes = config['camera']['width'] * config['camera']['height'] * sum(
            16 if name == 'point_cloud' else 4 for name in expected_streams)
        count = 0
        decode_recoveries = 0
        pending = deque()
        with ThreadPoolExecutor(max_workers=2) as pool, (session / 'capture/frames.csv').open() as ledger_handle, (output / 'frames.csv').open('w', newline='') as handle:
            ledger = iter(csv.DictReader(ledger_handle))
            csv_writer = None

            def flush_frame():
                nonlocal csv_writer
                rows = pending.popleft().result()
                if csv_writer is None:
                    csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    csv_writer.writeheader()
                for row in rows:
                    csv_writer.writerow(row)
                    counts[row['stream']] += 1

            while count < frames_total:
                # Protect the source session's disk reserve even for a long,
                # much larger offline export. Include the bounded in-flight frames.
                reserve = config['recording']['min_free_space_gb'] * 1024**3
                if shutil.disk_usage(output).free < reserve + 3 * frame_bytes * 1.1:
                    raise RuntimeError('Insufficient export space to preserve the configured disk reserve')
                entry = next(ledger, None)
                if entry is None or int(entry['frame_index']) != count:
                    raise ValueError('Frame ledger is truncated or out of order')
                original_stamp = int(entry['timestamp_ns'])
                timestamp, recovered = read_svo_frame(sl, camera, runtime, count, original_stamp)
                decode_recoveries += int(recovered)
                arrays = {}
                for (name, kind, value), mat in zip(specs, mats):
                    method = camera.retrieve_image if kind == 'image' else camera.retrieve_measure
                    require_success(sl, method(mat, value), f'Retrieve {name}')
                    check_size(mat.get_width(), mat.get_height(), config)
                    # Own buffers before the next grab/retrieve can reuse SDK memory.
                    arrays[name] = mat.get_data().copy()
                pending.append(pool.submit(write_frame, output, count, original_stamp, timestamp, arrays, config, recovered))
                if len(pending) >= 2:
                    flush_frame()
                count += 1
                if count % 30 == 0:
                    print(f'Exporting frame {count}/{frames_total}', flush=True)
            while pending:
                flush_frame()
            if next(ledger, None) is not None or count != frames_total:
                raise ValueError('Export did not read every recorded frame')
        pose_source = output / 'pose/pose.csv'
        valid_poses = derived_trajectory(pose_source, output) if pose_source.is_file() else 0
        if valid_poses != status['valid_pose_frames'] or (config['tracking']['enabled'] and not valid_poses):
            raise ValueError('Valid live pose count does not match capture status')
        static = {'coordinate_system': config['camera']['coordinate_system'], 'units': 'meters',
                  'T_left_camera_imu': calibration['camera_imu_transform_sdk'],
                  'T_imu_magnetometer': calibration['imu_magnetometer_transform_sdk'],
                  'stereo_transform_sdk': calibration['raw']['stereo_transform_sdk']}
        save_json(output / 'tf/static.json', static)
        for stream in expected_streams:
            if counts[stream] != frames_total:
                raise ValueError(f'Incomplete frame output: {stream}')
        summary = {'schema_version': 1, 'complete': True, 'recorded_frames': frames_total,
                   'streams': dict(counts), 'sidecar_rows': sidecars, 'valid_live_poses': valid_poses,
                   'clock_domain': 'SYSTEM_CLOCK', 'timestamps_preserved': True,
                   'timestamp_source': 'Original nanosecond capture ledger; SDK SVO replay timestamps may be truncated to microseconds',
                   'width': config['camera']['width'], 'height': config['camera']['height'],
                   'recording_sdk_version': metadata['sdk_version'], 'export_sdk_version': sl.Camera.get_sdk_version(),
                   'capture_observed_fps': status['observed_fps'], 'estimated_missing_camera_frames': status['estimated_missing_camera_frames'],
                   'sdk_decode_recoveries_by_seek': decode_recoveries,
                   'stereo_pixel_lossless': metadata['stereo_pixel_lossless'],
                   'image_encoding': 'RGB8 PNG', 'depth_enabled': include_depth,
                   'image_views': {name: 'unrectified' if name.endswith('_raw') else 'rectified'
                                   for name in streams},
                   'depth_disparity_confidence_source': 'SDK recomputed offline, using saved config/calibration; not archived live maps'
                                                        if include_depth else 'Disabled; images-only export',
                   'odom_source': 'Live SDK world poses relative to first valid pose; finite-difference velocities, no independent map/odom estimator' if valid_poses
                                  else 'Unavailable: live tracking was disabled; IMU orientation remains in imu.csv'}
        save_json(output / 'export_summary.json', summary)
        return summary
    finally:
        for mat in mats:
            mat.free()
        camera.close()


def export_session(session, output=None, overwrite=False, sl=None, allow_sdk_version_change=False, include_depth=None):
    session = Path(session).resolve()
    output = Path(output) if output is not None else session / 'export'
    if output.is_symlink():
        raise ValueError('Export destination cannot be a symlink')
    output = output.resolve()
    if output == session or output in session.parents or output == session / 'capture' or session / 'capture' in output.parents or output == session / 'calibration' or session / 'calibration' in output.parents:
        raise ValueError('Export destination overlaps recording sources')
    if output.exists() and not overwrite:
        raise FileExistsError('Export exists; use --overwrite to rebuild generated files')
    if output.exists() and not (output / 'export_summary.json').is_file():
        raise ValueError('Refusing to replace a directory that is not a generated export')
    if sl is None:
        import pyzed.sl as sl
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'-pending-', dir=output.parent))
    backup = None
    try:
        summary = export_data(session, staging, sl, allow_sdk_version_change, include_depth)
        if output.exists():
            backup = output.with_name(output.name + '-previous-' + staging.name.rsplit('-', 1)[-1])
            output.rename(backup)
        try:
            staging.rename(output)
        except BaseException:
            if backup is not None:
                backup.rename(output)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        print('Complete dataset export:', output, flush=True)
        return summary
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--allow-sdk-version-change', action='store_true', help='Explicitly permit recomputation using a different SDK version.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--images-only', action='store_true', help='Export configured stereo PNGs without depth computation.')
    mode.add_argument('--with-depth', action='store_true', help='Also compute depth/disparity/confidence offline.')
    args = parser.parse_args()
    try:
        include_depth = True if args.with_depth else False if args.images_only else None
        export_session(args.session, args.output, args.overwrite, allow_sdk_version_change=args.allow_sdk_version_change,
                       include_depth=include_depth)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print('Export failed:', exc)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
