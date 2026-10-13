"""Independent ZED physical-sensor acquisition; no image/depth processing."""
import json
import time

from dataset import clock_check, imu_records, json_value, vector


def collect_sensors(sl, camera, config, sink, stop, state):
    data = sl.SensorsData()
    seen = {}
    period = 1 / config['sensors']['poll_rate_hz']
    deadline = time.monotonic()
    temperature_deadline = 0.
    try:
        while not stop.is_set():
            code = camera.get_sensors_data(data, sl.TIME_REFERENCE.CURRENT)
            if code == sl.ERROR_CODE.SUCCESS:
                host = time.time_ns()
                host_mono = time.monotonic_ns()
                imu = data.get_imu_data()
                if imu.is_available:
                    stamp = int(imu.timestamp.get_nanoseconds())
                    if stamp < seen.get('imu', 0):
                        raise RuntimeError('IMU acquisition timestamp moved backwards')
                    if stamp > seen.get('imu', 0):
                        if 'imu' not in seen:
                            state['imu_clock_offset_sec'] = clock_check(stamp, config['synchronization']['timestamp_tolerance_sec'], host)
                        seen['imu'] = stamp
                        calibrated, raw = imu_records(imu, host, host_mono)
                        calibrated['camera_moving_state'] = str(data.camera_moving_state)
                        calibrated['image_sync_trigger'] = int(data.image_sync_trigger)
                        sink.put('imu/imu.csv', calibrated)
                        sink.put('imu/imu_raw.csv', raw)
                        state['imu_ready'].set()
                mag = data.get_magnetometer_data()
                if config['sensors']['magnetometer'] and mag.is_available:
                    stamp = int(mag.timestamp.get_nanoseconds())
                    if stamp > seen.get('mag', 0):
                        seen['mag'] = stamp
                        row = {'timestamp_ns': stamp, 'host_receipt_ns': host, 'host_receipt_mono_ns': host_mono,
                               'heading_deg': float(mag.magnetic_heading), 'heading_accuracy': float(mag.magnetic_heading_accuracy),
                               'heading_state': str(mag.magnetic_heading_state), 'effective_rate_hz': float(mag.effective_rate)}
                        vector(row, 'magnetic_field_calibrated_uT', mag.get_magnetic_field_calibrated(), 'xyz')
                        vector(row, 'magnetic_field_uncalibrated_uT', mag.get_magnetic_field_uncalibrated(), 'xyz')
                        sink.put('imu/magnetometer.csv', row)
                baro = data.get_barometer_data()
                if config['sensors']['barometer'] and baro.is_available:
                    stamp = int(baro.timestamp.get_nanoseconds())
                    if stamp > seen.get('baro', 0):
                        seen['baro'] = stamp
                        sink.put('sensors/barometer.csv', {'timestamp_ns': stamp, 'host_receipt_ns': host,
                            'host_receipt_mono_ns': host_mono,
                            'pressure_hPa': float(baro.pressure), 'relative_altitude_m': float(baro.relative_altitude),
                            'effective_rate_hz': float(baro.effective_rate)})
                if config['sensors']['temperature'] and time.monotonic() >= temperature_deadline:
                    temperatures = {}
                    for location in sl.SENSOR_LOCATION:
                        if getattr(location, 'name', '') == 'LAST':
                            continue
                        temperatures[str(location)] = float(data.get_temperature_data().get(location))
                    sink.put('sensors/temperature.csv', {'timestamp_ns': host, 'timestamp_source': 'host_poll',
                             'temperatures_celsius_json': json.dumps(json_value(temperatures), allow_nan=False)})
                    temperature_deadline = time.monotonic() + 1.
            deadline += period
            if deadline < time.monotonic() - period:
                deadline = time.monotonic()
            stop.wait(max(0., deadline - time.monotonic()))
    except Exception as exc:
        state['error'] = str(exc)
        stop.set()

