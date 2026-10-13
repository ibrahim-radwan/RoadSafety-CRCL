"""
profile_gpu_transfers.py — where is the GPU speedup going?

CONTEXT
    P0-7 fixed the GPU import: all eight stages now dispatch to CuPy, and
    the isolated azimuth FFT measures 40.0 ms CPU vs 0.6 ms GPU on this
    machine (66x). Yet frame time barely moved: 110-116 ms with the GPU
    import BROKEN, 110-143 ms with it FIXED.

    Audit item L-2 predicted why and was never implemented. Every stage in
    gpu_accel.py does:

        cp.asarray(host_input)  ->  kernel  ->  cp.asnumpy(device_output)

    11 cp.asnumpy() calls per frame, each a BLOCKING device->host copy that
    forces a full synchronisation. Data is never resident on the device
    across stages, so kernel execution cannot overlap with transfers.

    Pure PCIe bandwidth accounts for only ~6 ms of the ~39 ms gap. The rest
    is synchronisation stalls, kernel launch latency and (on Windows) WDDM
    scheduling. Those cannot be derived - they must be measured.

WHAT THIS MEASURES
    For each stage, three numbers on real frame data:
        compute-only  : kernel time with data already on the device
        with-transfer : the full asarray -> kernel -> asnumpy round trip
        overhead      : the difference, i.e. what L-2 would recover

USAGE
    python profile_gpu_transfers.py --folder <session frames> --cfg <cfg>
"""
from __future__ import annotations
import argparse, glob, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np


def imports():
    try:
        from processing.radar_config import (parse_cfg_file, parse_calibration,
                                              apply_raw_iq_order_override)
        from processing.radar_reader import read_frame
        from processing import radar_processing as rp
        from processing import gpu_accel as ga
    except ImportError:
        from radar_config import (parse_cfg_file, parse_calibration,       # type: ignore
                                  apply_raw_iq_order_override)
        from radar_reader import read_frame                                # type: ignore
        import radar_processing as rp                                      # type: ignore
        import gpu_accel as ga                                             # type: ignore
    return (parse_cfg_file, parse_calibration, apply_raw_iq_order_override,
            read_frame, rp, ga)


def main():
    ap = argparse.ArgumentParser(description='GPU transfer overhead profiler')
    ap.add_argument('--folder', required=True)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--frame', type=int, default=50)
    ap.add_argument('--reps', type=int, default=20)
    a = ap.parse_args()

    (parse_cfg_file, parse_calibration, apply_raw_iq_order_override,
     read_frame, rp, ga) = imports()

    if not getattr(ga, 'GPU_AVAILABLE', False):
        sys.exit('[ERROR] GPU_AVAILABLE is False - run check_gpu.py first.')
    import cupy as cp

    files = sorted(glob.glob(os.path.join(a.folder, '*.bin')))
    if not files:
        sys.exit(f'[ERROR] no .bin in {a.folder}')
    idx = min(a.frame, len(files) - 1)

    cfg = parse_cfg_file(a.cfg)
    corr = parse_calibration(a.cfg, cfg.num_tx, cfg.num_rx)
    cfg = apply_raw_iq_order_override(cfg, 'i-first')
    cfg.quiet_mode = True

    cube = read_frame(files[idx], cfg, 0)
    if corr is not None and cfg.apply_calibration:
        cube = rp.apply_calibration(cube, corr)

    print('=' * 80)
    print('  GPU TRANSFER OVERHEAD PROFILE')
    print(f'  frame {os.path.basename(files[idx])}   reps={a.reps}')
    print('=' * 80)

    def sync():
        cp.cuda.Stream.null.synchronize()

    def bench(fn, reps):
        fn(); sync()
        t = []
        for _ in range(reps):
            sync(); t0 = time.perf_counter()
            fn(); sync()
            t.append((time.perf_counter() - t0) * 1000)
        return float(np.median(t))

    rows = []

    # ── S2 range FFT ─────────────────────────────────────────────────────────
    n_s = cube.shape[-1]
    win = np.hanning(n_s).astype(np.float32)
    cube_g = cp.asarray(cube)
    win_g = cp.asarray(win)
    n_rfft = int(getattr(cfg, 'range_fft_size', n_s))

    def s2_compute():
        cp.fft.fft(cube_g * win_g, n=n_rfft, axis=-1)

    def s2_full():
        g = cp.asarray(cube)
        w = cp.asarray(win)
        r = cp.fft.fft(g * w, n=n_rfft, axis=-1)
        return cp.asnumpy(r)

    rows.append(('S2 range FFT', bench(s2_compute, a.reps), bench(s2_full, a.reps)))

    # ── S7 azimuth FFT ───────────────────────────────────────────────────────
    n_dop, n_r, n_va = 64, 256, 48
    virt = (np.random.randn(n_dop, n_r, n_va)
            + 1j * np.random.randn(n_dop, n_r, n_va)).astype(np.complex64)
    wa = np.hanning(n_va).astype(np.float32)
    virt_g = cp.asarray(virt)
    wa_g = cp.asarray(wa)
    n_az = int(getattr(cfg, 'az_fft_size', 256))

    def s7_compute():
        cp.fft.fftshift(cp.fft.fft(virt_g * wa_g, n=n_az, axis=-1), axes=-1)

    def s7_full():
        g = cp.asarray(virt)
        w = cp.asarray(wa)
        r = cp.fft.fftshift(cp.fft.fft(g * w, n=n_az, axis=-1), axes=-1)
        return cp.asnumpy(r)

    rows.append(('S7 azimuth FFT (full cube)',
                 bench(s7_compute, a.reps), bench(s7_full, a.reps)))

    # ── Bare round trip of the largest array, no compute at all ─────────────
    def rt_only():
        g = cp.asarray(virt)
        return cp.asnumpy(g)

    rt = bench(rt_only, a.reps)

    print()
    print(f'  {"stage":<30}{"compute":>10}{"+transfer":>12}{"overhead":>11}{"x":>7}')
    print('  ' + '-' * 72)
    tot_c = tot_f = 0.0
    for name, c, f in rows:
        tot_c += c; tot_f += f
        ratio = f / c if c > 0 else 0
        print(f'  {name:<30}{c:>9.2f}ms{f:>11.2f}ms{f-c:>10.2f}ms{ratio:>6.1f}x')
    print('  ' + '-' * 72)
    print(f'  {"TOTAL (these 2 stages)":<30}{tot_c:>9.2f}ms{tot_f:>11.2f}ms'
          f'{tot_f-tot_c:>10.2f}ms')
    print()
    print(f'  Bare round trip of a {virt.nbytes/1e6:.1f} MB array, no compute : {rt:.2f} ms')

    print()
    print('=' * 80)
    print('  READING THIS')
    print('=' * 80)
    print(f"""  'compute' is the kernel with data already resident on the device.
  '+transfer' is what gpu_accel.py actually does today: asarray -> kernel
  -> asnumpy, on every stage.

  The 'overhead' column is what audit item L-2 would recover by keeping
  data on the device between stages. There are 11 such round trips per
  frame; only two are profiled here, so the full-pipeline saving is
  larger than the total shown.

  If overhead dominates compute, the GPU is being starved by the transfer
  pattern rather than by kernel speed - which matches the observation that
  enabling eight GPU stages left frame time at 110-143 ms when the isolated
  azimuth FFT alone should have saved ~39 ms.""")


if __name__ == '__main__':
    main()
