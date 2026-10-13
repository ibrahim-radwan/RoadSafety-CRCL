"""Configuration, calibrated sensor records and bounded streaming I/O (no ROS)."""
import csv
import hashlib
import json
import math
import os
import queue
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f'Duplicate configuration key: {key}')
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


SCHEMA = {
    'camera': {'model', 'serial_number', 'resolution', 'fps', 'width', 'height', 'coordinate_system', 'units', 'disable_self_calibration'},
    'synchronization': {'clock_domain', 'timestamp_tolerance_sec'},
    'depth': {'mode', 'confidence_threshold', 'texture_confidence_threshold', 'stabilization', 'min_distance_m', 'max_distance_m'},
    'tracking': {'enabled', 'imu_fusion', 'area_memory', 'pose_smoothing'},
    'sensors': {'poll_rate_hz', 'poll_window_fraction', 'magnetometer', 'barometer', 'temperature'},
    'recording': {'dataset_root', 'session_prefix', 'compression', 'auto_export', 'min_free_space_gb', 'csv_queue_records'},
    'export': {'point_cloud', 'png_compression', 'include_depth'},
    'runtime': {'open_timeout_sec', 'warmup_sec', 'status_interval_sec', 'grab_failure_timeout_sec'},
    'images': {'enabled', 'streams', 'format', 'workers', 'queue_frames', 'png_compression', 'jpeg_quality', 'writer_backend'},
}


def validate_config(config):
    # Sessions recorded before direct image capture remain replayable.
    if isinstance(config, dict):
        if isinstance(config.get('export'), dict):
            config['export'].setdefault('include_depth', True)
        config.setdefault('images', {'enabled': False, 'streams': ['left', 'right'], 'format': 'PNG',
                                    'workers': 4, 'queue_frames': 8, 'png_compression': 1})
        if isinstance(config.get('images'), dict):
            config['images'].setdefault('jpeg_quality', 95)
            config['images'].setdefault('writer_backend', 'auto')
    if not isinstance(config, dict) or set(config) != set(SCHEMA):
        raise ValueError('Configuration must contain exactly: ' + ', '.join(SCHEMA))
    for section, fields in SCHEMA.items():
        if not isinstance(config[section], dict) or set(config[section]) != fields:
            raise ValueError(f'Invalid or missing {section} fields; expected {sorted(fields)}')

    def require(section, field, kind):
        value = config[section][field]
        numeric = kind is float
        if (type(value) not in (float, int) if numeric else type(value) is not kind):
            raise ValueError(f'{section}.{field} must be {kind.__name__}')
        if numeric and not math.isfinite(value):
            raise ValueError(f'{section}.{field} must be finite')
        return value

    for section, fields in {
        'camera': ['disable_self_calibration'], 'tracking': ['enabled', 'imu_fusion', 'area_memory', 'pose_smoothing'],
        'sensors': ['magnetometer', 'barometer', 'temperature'], 'recording': ['auto_export'],
        'export': ['point_cloud', 'include_depth'], 'images': ['enabled']
    }.items():
        for field in fields:
            require(section, field, bool)
    expected = [('camera', 'model', 'zed2i'), ('camera', 'resolution', 'HD1080'),
                ('camera', 'coordinate_system', 'RIGHT_HANDED_Z_UP_X_FWD'), ('camera', 'units', 'METER'),
                ('synchronization', 'clock_domain', 'SYSTEM_CLOCK')]
    for section, field, value in expected:
        if config[section][field] != value:
            raise ValueError(f'{section}.{field} must be {value}')
    for field, expected_value in [('width', 1920), ('height', 1080)]:
        if require('camera', field, int) != expected_value:
            raise ValueError('Native image size must be width 1920, height 1080')
    if require('camera', 'fps', int) not in (15, 30):
        raise ValueError('HD1080 supports 15 or 30 FPS on ZED 2i')
    serial = config['camera']['serial_number']
    if serial is not None and (type(serial) is not int or serial <= 0):
        raise ValueError('camera.serial_number must be null or a positive integer')
    if config['depth']['mode'] not in ('PERFORMANCE', 'QUALITY', 'ULTRA', 'NEURAL', 'NEURAL_PLUS'):
        raise ValueError('Unsupported SDK 4.1 depth mode')
    if config['tracking']['enabled'] and not config['tracking']['imu_fusion']:
        raise ValueError('When tracking is enabled, IMU fusion must be enabled')
    streams = config['images']['streams']
    if (type(streams) is not list or not streams or any(type(s) is not str for s in streams)
            or len(set(streams)) != len(streams)
            or not set(streams) <= {'left', 'right', 'left_raw', 'right_raw'}):
        raise ValueError('images.streams must be unique stereo stream names')
    if config['images']['enabled'] and not {'left', 'right'} <= set(streams):
        raise ValueError('Direct capture requires both left and right images')
    if config['images']['format'] not in ('BMP', 'PNG', 'JPEG'):
        raise ValueError('images.format must be BMP, PNG or JPEG')
    if config['images']['writer_backend'] not in ('auto', 'thread', 'process'):
        raise ValueError('images.writer_backend must be auto, thread or process')
    if config['export']['point_cloud'] and not config['export']['include_depth']:
        raise ValueError('Point cloud export requires export.include_depth=true')
    for field, low, high in [('workers', 1, 8), ('queue_frames', 1, 32), ('png_compression', 0, 9), ('jpeg_quality', 1, 100)]:
        if not low <= require('images', field, int) <= high:
            raise ValueError(f'images.{field} must be between {low} and {high}')
    for field in ('confidence_threshold', 'texture_confidence_threshold'):
        if not 0 <= require('depth', field, int) <= 100:
            raise ValueError(f'depth.{field} must be between 0 and 100')
    if not 0 <= require('depth', 'stabilization', int) <= 100:
        raise ValueError('depth.stabilization must be between 0 and 100')
    low, high = require('depth', 'min_distance_m', float), require('depth', 'max_distance_m', float)
    if not 0 <= low < high:
        raise ValueError('Invalid depth distance range')
    if not 200 <= require('sensors', 'poll_rate_hz', float) <= 1000:
        raise ValueError('Sensor poll rate must be between 200 and 1000 Hz')
    if not 0 <= require('sensors', 'poll_window_fraction', float) <= .8:
        raise ValueError('sensors.poll_window_fraction must be between 0 and 0.8')
    if config['recording']['compression'] not in ('LOSSLESS', 'H264', 'H265', 'H264_LOSSLESS', 'H265_LOSSLESS'):
        raise ValueError('Invalid SVO compression mode')
    for field in ('dataset_root', 'session_prefix'):
        value = require('recording', field, str)
        if not value.strip() or value != value.strip():
            raise ValueError(f'recording.{field} must be a nonempty string')
    import re
    if re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_-]*', config['recording']['session_prefix']) is None:
        raise ValueError('session_prefix must be a safe folder name')
    if require('recording', 'min_free_space_gb', float) <= 0:
        raise ValueError('min_free_space_gb must be positive')
    if not 100 <= require('recording', 'csv_queue_records', int) <= 100000:
        raise ValueError('csv_queue_records must be between 100 and 100000')
    if not 0 <= require('export', 'png_compression', int) <= 9:
        raise ValueError('png_compression must be between 0 and 9')
    for section, fields in [('runtime', SCHEMA['runtime']), ('synchronization', ['timestamp_tolerance_sec'])]:
        for field in fields:
            value = require(section, field, float)
            if value < 0 or (field != 'warmup_sec' and value == 0):
                raise ValueError(f'{section}.{field} must be positive (warmup may be zero)')
    return config


def load_config(path):
    return validate_config(yaml.load(Path(path).read_text(), Loader=UniqueLoader))


def dataset_root(config, config_path):
    path = Path(os.path.expandvars(os.path.expanduser(config['recording']['dataset_root'])))
    return path if path.is_absolute() else Path(config_path).resolve().parent / path


def capture_space_headroom(config, check_interval_sec=2.):
    """Budget queued images and worst-case stereo writes until the next check.

    Compression savings are not assumed. Other sensor recorders must budget
    their own writes separately when sharing this filesystem.
    """
    streams = len(config['images']['streams']) if config['images']['enabled'] else 0
    queued = config['images']['queue_frames'] if streams else 0
    frame_bytes = config['camera']['width'] * config['camera']['height'] * (8 + 4*streams)
    frames = queued + math.ceil(config['camera']['fps'] * check_interval_sec) + 1
    return math.ceil(frame_bytes * frames * 1.1)


def numeric(value):
    if hasattr(value, 'get') and callable(value.get):
        value = value.get()
    elif hasattr(value, 'r'):
        value = value.r
    elif hasattr(value, 'm'):
        value = value.m
    return np.asarray(value)


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('+Inf' if value > 0 else '-Inf')
    return value


def save_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(json_value(content), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def clock_check(timestamp_ns, tolerance_sec, host_ns=None):
    offset = ((time.time_ns() if host_ns is None else host_ns) - timestamp_ns) / 1e9
    if timestamp_ns <= 0 or abs(offset) > tolerance_sec:
        raise RuntimeError(f'SDK timestamp does not match SYSTEM_CLOCK: offset={offset:.3f}s')
    return offset


def vector(row, prefix, value, names):
    array = numeric(value).reshape(-1)
    if len(array) != len(names):
        raise ValueError(f'{prefix} expected {len(names)} values; got {len(array)}')
    row.update({prefix + '_' + key: float(item) for key, item in zip(names, array)})


def matrix(row, prefix, value, count):
    vector(row, prefix, value, [str(i) for i in range(count)])


def imu_records(imu, host_ns, host_mono_ns=None):
    base = {'timestamp_ns': int(imu.timestamp.get_nanoseconds()), 'host_receipt_ns': host_ns,
            'host_receipt_mono_ns': host_mono_ns, 'effective_rate_hz': float(imu.effective_rate)}
    calibrated, raw = dict(base), dict(base)
    vector(calibrated, 'orientation', imu.get_pose().get_orientation(), ['x', 'y', 'z', 'w'])
    q = [calibrated['orientation_' + a] for a in 'xyzw']
    angles = quaternion_angles(q)
    calibrated.update(zip(('roll_rad', 'pitch_rad', 'yaw_rad'), angles))
    vector(calibrated, 'linear_acc_m_s2', imu.get_linear_acceleration(), 'xyz')
    vector(raw, 'linear_acc_m_s2', imu.get_linear_acceleration_uncalibrated(), 'xyz')
    # SDK angular velocity and covariance are in degrees; export SI radians.
    vector(calibrated, 'angular_vel_rad_s', numeric(imu.get_angular_velocity()) * math.pi / 180, 'xyz')
    vector(raw, 'angular_vel_rad_s', numeric(imu.get_angular_velocity_uncalibrated()) * math.pi / 180, 'xyz')
    matrix(calibrated, 'orientation_covariance_sdk', imu.get_pose_covariance(), 9)
    matrix(calibrated, 'linear_acc_covariance_m2_s4', imu.get_linear_acceleration_covariance(), 9)
    matrix(calibrated, 'angular_vel_covariance_rad2_s2', numeric(imu.get_angular_velocity_covariance()) * (math.pi / 180)**2, 9)
    # Raw channels have no SDK-provided raw orientation or covariance; do not invent them.
    return calibrated, raw


def quaternion_angles(quaternion):
    """Intrinsic ZYX yaw/pitch/roll, returned roll/pitch/yaw in radians.

    The SDK orientation reference is preserved; yaw is not an absolute heading.
    An invalid SDK quaternion yields unavailable angles, never invented zeros.
    """
    q = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.all(np.isfinite(q)) or norm < 1e-12:
        return (float('nan'),) * 3
    x, y, z, w = q / norm
    sin_pitch = max(-1., min(1., 2*(w*y-z*x)))
    if abs(sin_pitch) >= 1 - 1e-12:
        # Euler angles are nonunique at gimbal lock; choose yaw=0.
        return (math.atan2(2*(w*x-y*z), 1-2*(x*x+z*z)),
                math.copysign(math.pi/2, sin_pitch), 0.)
    return (math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
            math.asin(sin_pitch),
            math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


def pose_record(pose, state, timestamp_ns, host_ns, frame_index):
    row = {'frame_index': frame_index, 'timestamp_ns': timestamp_ns, 'pose_timestamp_ns': int(pose.timestamp.get_nanoseconds()),
           'host_receipt_ns': host_ns, 'tracking_state': str(state), 'valid': bool(pose.valid),
           'pose_confidence': int(pose.pose_confidence)}
    vector(row, 'position_m', pose.get_translation(), 'xyz')
    vector(row, 'orientation', pose.get_orientation(), ['x', 'y', 'z', 'w'])
    matrix(row, 'pose_covariance_sdk', pose.pose_covariance, 36)
    vector(row, 'twist_sdk', pose.twist, ['vx', 'vy', 'vz', 'wx', 'wy', 'wz'])
    matrix(row, 'twist_covariance_sdk', pose.twist_covariance, 36)
    return row


class CsvSink:
    """Bounded background CSV writes; failures propagate and close drains all accepted rows."""
    def __init__(self, root, max_records=10000, flush_interval_sec=1.):
        if not math.isfinite(flush_interval_sec) or flush_interval_sec <= 0:
            raise ValueError('CSV flush interval must be finite and positive')
        self.root = Path(root)
        self.queue = queue.Queue(maxsize=max_records)
        self.error = None
        self.counts = Counter()
        self.first, self.last = {}, {}
        self.flush_interval_sec = flush_interval_sec
        self._state_lock = threading.Lock()
        self._closed = False
        self.thread = threading.Thread(target=self._run, name='dataset-csv-writer')
        self.thread.start()

    def put(self, filename, row):
        while True:
            try:
                with self._state_lock:
                    if self.error:
                        raise RuntimeError('CSV storage failed') from self.error
                    if self._closed or not self.thread.is_alive():
                        raise RuntimeError('CSV writer is closed')
                    self.queue.put((filename, row), timeout=.1)
                return
            except queue.Full:
                if not self.thread.is_alive():
                    raise RuntimeError('CSV writer stopped')

    def _run(self):
        handles, writers = {}, {}
        dirty = set()
        deadline = time.monotonic() + self.flush_interval_sec
        def flush():
            nonlocal deadline
            for name in dirty:
                handles[name].flush()
            dirty.clear()
            deadline = time.monotonic() + self.flush_interval_sec
        try:
            while True:
                try:
                    item = self.queue.get(timeout=max(0., deadline-time.monotonic()))
                except queue.Empty:
                    flush()
                    continue
                if item is None:
                    break
                filename, row = item
                if filename not in writers:
                    path = self.root / filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    handles[filename] = path.open('w', newline='', buffering=1024*1024)
                    writers[filename] = csv.DictWriter(handles[filename], fieldnames=list(row))
                    writers[filename].writeheader()
                writers[filename].writerow(row)
                dirty.add(filename)
                self.counts[filename] += 1
                stamp = row.get('timestamp_ns')
                if stamp is not None:
                    self.first[filename] = min(stamp, self.first.get(filename, stamp))
                    self.last[filename] = max(stamp, self.last.get(filename, stamp))
                if time.monotonic() >= deadline:
                    flush()
        except BaseException as exc:
            self.error = exc
        finally:
            for handle in handles.values():
                try:
                    handle.close()
                except Exception as exc:
                    self.error = self.error or exc

    def close(self):
        with self._state_lock:
            already_closed = self._closed
            self._closed = True
        if already_closed:
            self.thread.join()
            if self.error:
                raise RuntimeError('CSV recording failed') from self.error
            return
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=.1)
                break
            except queue.Full:
                continue
        self.thread.join()
        if self.error:
            raise RuntimeError('CSV recording failed') from self.error

    def stats(self):
        return {name: {'count': count, 'first_timestamp_ns': self.first.get(name),
                'last_timestamp_ns': self.last.get(name),
                'observed_rate_hz': (count-1)*1e9/(self.last[name]-self.first[name])
                    if name in self.first and self.last[name] > self.first[name] else None}
                for name, count in self.counts.items()}


def sdk_init(sl, config, svo=None, settings=None):
    p = sl.InitParameters()
    p.camera_resolution = getattr(sl.RESOLUTION, config['camera']['resolution'])
    p.camera_fps = config['camera']['fps']
    p.camera_image_flip = sl.FLIP_MODE.OFF
    p.camera_disable_self_calib = config['camera']['disable_self_calibration']
    p.coordinate_system = getattr(sl.COORDINATE_SYSTEM, config['camera']['coordinate_system'])
    p.coordinate_units = sl.UNIT.METER
    p.depth_mode = getattr(sl.DEPTH_MODE, config['depth']['mode'])
    p.depth_stabilization = config['depth']['stabilization']
    p.depth_minimum_distance = float(config['depth']['min_distance_m'])
    p.depth_maximum_distance = float(config['depth']['max_distance_m'])
    p.open_timeout_sec = float(config['runtime']['open_timeout_sec'])
    p.sensors_required = True
    p.sdk_verbose = 1
    if svo is not None:
        p.set_from_svo_file(str(svo))
        p.svo_real_time_mode = False  # Process every recorded frame, regardless of export speed.
    elif config['camera']['serial_number'] is not None:
        p.set_from_serial_number(config['camera']['serial_number'])
    if settings is not None:
        p.optional_settings_path = str(settings)
    return p


def require_success(sl, status, action):
    # Negative SDK codes are warnings with a usable frame; positive codes are errors.
    if int(status.value) > int(sl.ERROR_CODE.SUCCESS.value):
        raise RuntimeError(f'{action}: {status}')


def image_views(sl, streams):
    """Single source for stereo eye and raw/rectified SDK view selection."""
    names = {'left': 'LEFT', 'right': 'RIGHT',
             'left_raw': 'LEFT_UNRECTIFIED', 'right_raw': 'RIGHT_UNRECTIFIED'}
    return [(stream, getattr(sl.VIEW, names[stream])) for stream in streams]


def sdk_runtime(sl, config):
    p = sl.RuntimeParameters()
    p.confidence_threshold = config['depth']['confidence_threshold']
    p.texture_confidence_threshold = config['depth']['texture_confidence_threshold']
    return p


def check_size(width, height, config):
    expected = (config['camera']['width'], config['camera']['height'])
    if (width, height) != expected:
        raise RuntimeError(f'Expected {expected[0]}x{expected[1]}, received {width}x{height}')


def read_svo_frame(sl, camera, runtime, index, original_stamp, max_attempts=3):
    """Require the requested archived frame; recover SDK 4.1 decoder skips by seeking."""
    code = None
    actual_index = actual_stamp = None
    for attempt in range(max_attempts):
        if attempt:
            camera.set_svo_position(index)
        code = camera.grab(runtime)
        if int(code.value) <= int(sl.ERROR_CODE.SUCCESS.value):
            actual_index = camera.get_svo_position()
            actual_stamp = int(camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
            # SVO 4.1 stores microseconds; the sidecar retains the source nanoseconds.
            if actual_index == index and actual_stamp // 1000 == original_stamp // 1000:
                return actual_stamp, attempt > 0
    raise ValueError(f'Cannot decode original SVO frame {index}: code={code}, '
                     f'position={actual_index}, timestamp={actual_stamp}, expected={original_stamp}')


def calibration_data(info, config):
    c = info.camera_configuration

    def camera(p):
        return {'width': int(p.image_size.width), 'height': int(p.image_size.height),
                'fx': float(p.fx), 'fy': float(p.fy), 'cx': float(p.cx), 'cy': float(p.cy),
                'k': [[float(p.fx), 0, float(p.cx)], [0, float(p.fy), float(p.cy)], [0, 0, 1]],
                'distortion': numeric(p.disto).tolist(), 'h_fov_deg': float(p.h_fov), 'v_fov_deg': float(p.v_fov)}

    def stereo(p):
        return {'left': camera(p.left_cam), 'right': camera(p.right_cam),
                'stereo_transform_sdk': numeric(p.stereo_transform).tolist(),
                'baseline_m': float(p.get_camera_baseline())}

    sensors = info.sensors_configuration
    return {'serial_number': int(info.serial_number), 'camera_model': str(info.camera_model),
            'camera_firmware': int(c.firmware_version), 'sensor_firmware': int(sensors.firmware_version),
            'coordinate_system': config['camera']['coordinate_system'], 'coordinate_units': 'meters',
            'rectified': stereo(c.calibration_parameters), 'raw': stereo(c.calibration_parameters_raw),
            'camera_imu_transform_sdk': numeric(sensors.camera_imu_transform).tolist(),
            'imu_magnetometer_transform_sdk': numeric(sensors.imu_magnetometer_transform).tolist()}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()
