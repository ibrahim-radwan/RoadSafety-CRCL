"""
check_gpu.py — is the pipeline actually using the GPU, and if not, WHY?

WHY THIS EXISTS
    Stage 2.2 saved 63 ms/frame from a 5.6x reduction in the azimuth FFT,
    implying the full-cube stage cost ~77 ms. A GPU FFT of that shape takes
    ~0.7 ms. If the GPU path were live there would have been almost nothing
    to save — so the pipeline is probably running that stage on CPU despite
    a working CUDA device.

    radar_processing.py selects its implementations with:

        try:
            from .gpu_accel import (GPU_AVAILABLE, range_fft_gpu, ...)   # 8 names
        except ImportError:
            ... CPU fallbacks ...

    The except clause catches ONLY ImportError and covers EIGHT names. If any
    single one is missing or fails to import, the WHOLE block silently falls
    back to CPU with no warning. This script reproduces that exact import and
    reports the real exception.

NOTE
    An earlier version of this script inferred the bindings by grepping source
    lines. That was invalid: the GPU lines read `az_fft_gpu as _az_fft_fn`, so
    a prefix match on `_az_fft_fn` only ever hit the CPU fallback branch and
    reported CPU unconditionally. This version imports and inspects instead.

USAGE
    python check_gpu.py
"""
from __future__ import annotations
import importlib, os, sys, time, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print('=' * 76)
print('  GPU ACCELERATION CHECK')
print('=' * 76)

# ── 1. CuPy and device ───────────────────────────────────────────────────────
cupy_ok = False
try:
    import cupy as cp
    cupy_ok = True
    print(f'  cupy               : {cp.__version__}')
    n = cp.cuda.runtime.getDeviceCount()
    for i in range(n):
        nm = cp.cuda.runtime.getDeviceProperties(i)['name']
        print(f'  device {i}           : {nm.decode() if isinstance(nm, bytes) else nm}')
except Exception as e:                                       # noqa: BLE001
    print(f'  cupy               : UNAVAILABLE ({e})')

# ── 2. Reproduce the pipeline's exact import ─────────────────────────────────
print()
print('  Reproducing radar_processing\'s GPU import')
print('  ' + '-' * 72)

NAMES = ['GPU_AVAILABLE', 'range_fft_gpu', 'doppler_fft_gpu', 'az_fft_gpu',
         'power_db_reduce_gpu', 'elevation_fft_gpu', 'cfar_2d_gpu',
         'ra_focus_mask_gpu', 'collapse_ra_from_fft_gpu']

gpu_mod = None
for modname in ('processing.gpu_accel', 'gpu_accel'):
    try:
        gpu_mod = importlib.import_module(modname)
        print(f'  import {modname:<24}: OK')
        break
    except Exception as e:                                   # noqa: BLE001
        print(f'  import {modname:<24}: FAILED  {type(e).__name__}: {e}')

if gpu_mod is None:
    print()
    print('  >> gpu_accel is not importable at all. Every stage runs on CPU.')
    print('     Full traceback of the last attempt:')
    traceback.print_exc()
else:
    missing = [nm for nm in NAMES if not hasattr(gpu_mod, nm)]
    print(f'  module file        : {getattr(gpu_mod, "__file__", "?")}')
    print(f'  GPU_AVAILABLE      : {getattr(gpu_mod, "GPU_AVAILABLE", "ABSENT")}')
    if missing:
        print()
        print(f'  >> MISSING NAMES ({len(missing)}): {", ".join(missing)}')
        print('     The pipeline imports all of these in ONE statement, so a single')
        print('     missing name sends EVERY stage to the CPU fallback silently.')
    else:
        print('  all 8 required names present: YES')

# ── 3. What did radar_processing actually bind? (runtime, not grep) ──────────
print()
print('  Bindings actually selected at runtime')
print('  ' + '-' * 72)
rp = None
for modname in ('processing.radar_processing', 'radar_processing'):
    try:
        rp = importlib.import_module(modname)
        break
    except Exception:
        continue

if rp is None:
    print('  could not import radar_processing')
else:
    # The bindings are local to process_frame, so inspect its constants
    # for the names of the functions it will call.
    import dis
    try:
        code = rp.process_frame.__code__
        used = set()
        for inst in dis.get_instructions(code):
            if inst.opname in ('IMPORT_FROM', 'LOAD_GLOBAL', 'STORE_FAST') and inst.argval:
                used.add(str(inst.argval))
        gpu_names = sorted(n for n in used if n.endswith('_gpu'))
        cpu_names = sorted(n for n in used if n in
                           ('_az_fft', 'range_fft', 'doppler_fft', 'cfar_2d',
                            '_ra_focus_mask_from_rd', '_collapse_ra_from_fft'))
        print(f'  GPU symbols referenced : {", ".join(gpu_names) if gpu_names else "none"}')
        print(f'  CPU symbols referenced : {", ".join(cpu_names) if cpu_names else "none"}')
        print('  (both appear because both branches are compiled in; which one RUNS')
        print('   depends on whether the import above succeeded)')
    except Exception as e:                                   # noqa: BLE001
        print(f'  inspection failed: {e}')

# ── 4. Timing ────────────────────────────────────────────────────────────────
print()
print('  Azimuth FFT timing, [64, 256, 48] -> 256 bins')
print('  ' + '-' * 72)
import numpy as np
x = (np.random.randn(64, 256, 48) + 1j * np.random.randn(64, 256, 48)).astype(np.complex64)
w = np.hanning(48).astype(np.float32)

t = []
for _ in range(3):
    t0 = time.perf_counter()
    np.fft.fftshift(np.fft.fft(x * w[None, None, :], n=256, axis=-1), axes=-1)
    t.append((time.perf_counter() - t0) * 1000)
cpu_ms = min(t)
print(f'    numpy (CPU)      : {cpu_ms:7.1f} ms')

gpu_ms = None
if cupy_ok:
    try:
        xg, wg = cp.asarray(x), cp.asarray(w)
        g = []
        for _ in range(5):
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            cp.fft.fftshift(cp.fft.fft(xg * wg[None, None, :], n=256, axis=-1), axes=-1)
            cp.cuda.Stream.null.synchronize()
            g.append((time.perf_counter() - t0) * 1000)
        gpu_ms = min(g)
        print(f'    cupy  (GPU)      : {gpu_ms:7.1f} ms      {cpu_ms/gpu_ms:.0f}x faster')
    except Exception as e:                                   # noqa: BLE001
        print(f'    cupy  (GPU)      : FAILED {e}')

# ── 5. Verdict ───────────────────────────────────────────────────────────────
print()
print('=' * 76)
print('  VERDICT')
print('=' * 76)

gpu_live = bool(gpu_mod is not None and getattr(gpu_mod, 'GPU_AVAILABLE', False))

if gpu_live:
    print('  GPU_AVAILABLE = True and gpu_accel imported cleanly.')
    print('  The accelerated stages WILL be used:')
    print('    range FFT, doppler FFT, azimuth FFT, power reduce,')
    print('    elevation FFT, CFAR, focus mask, RA collapse')
    if gpu_ms:
        print()
        print(f'  Azimuth FFT on this machine: {cpu_ms:.1f} ms CPU -> {gpu_ms:.1f} ms GPU')
    print()
    print('  NOTE: any S7 figure derived from a PRE-FIX run is now obsolete.')
    print('  Re-measure before quoting timings. In particular Stage 2.2 will')
    print('  save far less in absolute terms once the FFT costs ~1 ms instead')
    print('  of ~40 ms - it is still correct and worth keeping, but its')
    print('  headline saving was largely a symptom of the CPU fallback.')
    print()
    print('  NEXT: GPU/CPU numerical parity has never been tested. Every')
    print('  result in this project so far was produced on the CPU path.')
    print('  Re-run verify_stage22.ps1 and compare against preserved CSVs.')
else:
    print('  GPU_AVAILABLE = False -> every stage runs on CPU.')
    if gpu_mod is None:
        print('  Cause: gpu_accel could not be imported at all (see above).')
    else:
        print('  Cause: gpu_accel imported but its capability probe failed.')
        print('  Look for the "[GPU] CuPy not available (...)" line above -')
        print('  it now prints the exception TYPE, which identifies the step.')
    if gpu_ms:
        print()
        print(f'  Available headroom: {cpu_ms:.1f} ms -> {gpu_ms:.1f} ms on the')
        print(f'  azimuth FFT alone (~{cpu_ms-gpu_ms:.0f} ms/frame), no algorithm change.')
