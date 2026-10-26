"""The faster paths give the same results as the plain ones they replace:

  radar_dsp.doppler_band        = doppler_fft(...)[mid - b:mid + b + 1] (the static path)
  radar_dsp.unique_aperture     cached layout = the per-position mean
  replay render (_paint)        = the plain sort-by-value version
  radar_io.iter_frames          = the frames in order
  RadarDetector (CUDA graph)    = RadarDetector(cuda_graph=False), every configuration (session test)
  lidar_io.decode_parallel      = decode_sweeps (one driver), sweep for sweep, with detection
                                  (session test; the shortest session in $REPORT_TEST_SESSIONS)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

PIPELINE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PIPELINE), str(PIPELINE / "replay")]
torch = pytest.importorskip("torch")

DEFAULT = "/mnt/sdcard/sessions/freewalk_ref_02"
SESSIONS = [Path(s) for s in os.environ.get("REPORT_TEST_SESSIONS", DEFAULT).split(",") if s]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_doppler_band_equals_fft_slice():
    from perception import radar_dsp as dsp
    g = torch.Generator().manual_seed(0)
    x = torch.complex(torch.randn(64, 50, 38, generator=g), torch.randn(64, 50, 38, generator=g)).to(DEV)
    win = dsp.window("hann", 64, device=DEV)
    for b in (0, 1, 2):
        ref = dsp.doppler_fft(x, win, 64, remove_clutter=False)[32 - b:32 + b + 1]
        out = dsp.doppler_band(x, win, 64, b)
        assert out.shape == ref.shape
        assert float((out - ref).abs().max()) < 1e-3 * float(ref.abs().max())


def test_unique_aperture_cached_mean():
    from perception import radar_dsp as dsp
    pos = np.array([0, 1, 1, 2, 3, 3, 3, 5], float)
    x = torch.arange(2 * 8, dtype=torch.float32, device=DEV).view(2, 8).to(torch.complex64)
    for _ in range(2):                                    # second call: from the cache
        out, uniq = dsp.unique_aperture(x, pos)
        ref = np.stack([x.cpu().numpy()[:, pos == u].mean(axis=1) for u in np.unique(pos)], axis=1)
        assert np.array_equal(uniq, np.unique(pos))
        assert np.allclose(out.cpu().numpy(), ref)


def _paint_plain(cells, values, n, keep, lut, lo=None, hi=None):
    """The original render: sort the values, last write wins, colour every occupied cell."""
    order = np.argsort(values if keep == "max" else -values, kind="stable")
    img = np.full(n, np.nan, dtype=np.float32)
    img[cells[order]] = values[order]
    have = np.isfinite(img)
    if lo is None:
        lo, hi = np.percentile(values, [2, 98]) if len(values) else (0.0, 1.0)
    idx = np.clip((img[have] - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    rgb = np.zeros((n, 3), dtype=np.uint8)
    rgb[have] = lut[idx]
    return rgb


@pytest.mark.parametrize("keep", ["max", "min"])
@pytest.mark.parametrize("fixed", [False, True])
def test_render_paint_equals_plain(keep, fixed):
    import sources
    rng = np.random.default_rng(1)
    n = 5000
    cells = rng.integers(0, n, 40000)
    values = rng.normal(0.0, 1.5, 40000)
    values[::7] = np.round(values[::7], 1)                # ties
    lut = (rng.random((256, 3)) * 254 + 1).astype(np.uint8)   # no black entry: occupied = non-black
    lo, hi = (-1.0, 2.1) if fixed else (None, None)
    rgb, occ = sources._paint(cells, values, n, keep, lut, lo, hi)
    assert np.array_equal(rgb, _paint_plain(cells, values, n, keep, lut, lo, hi))
    assert np.array_equal(occ.astype(bool), np.isin(np.arange(n), cells))


def test_iter_frames_in_order():
    from perception.core.radar_io import iter_frames

    class Timeline:
        def __len__(self):
            return 37

        def raw(self, k):
            return np.full(4, k, np.int16)
    got = list(iter_frames(Timeline(), ahead=3))
    assert [k for k, _ in got] == list(range(37))
    assert all(int(r[0]) == k for k, r in got)
    it = iter_frames(Timeline(), ahead=3)                 # stopping early ends the reader
    next(it)
    it.close()


LIDAR = sorted([s for s in SESSIONS if (s / "lidar" / "lidar.pcap").is_file()
                and (s / "sync" / "lidar_pcap_index.npz").is_file()],
               key=lambda s: (s / "lidar" / "lidar.pcap").stat().st_size)


@pytest.mark.skipif(not LIDAR, reason="no session with lidar.pcap and sync/lidar_pcap_index.npz")
def test_decode_parallel_equals_one_driver():
    from perception.core.lidar_io import decode_parallel, decode_sweeps
    from perception.lidar import detect_sweep
    S = LIDAR[0]
    one = [(t, detect_sweep(p)) for t, p in decode_sweeps(S, 3.0)]
    par = list(decode_parallel(S, detect_sweep, workers=3, chunk_s=25.0))   # several chunks, two rounds
    assert len(par) == len(one)
    for (ta, (oa, ga, _)), (tb, (ob, gb, _)) in zip(one, par):
        assert ta == tb
        assert ga == gb or (np.isnan(ga) and np.isnan(gb))
        assert oa.shape == ob.shape and np.array_equal(oa, ob)


RADAR = [s for s in SESSIONS if (s / "sync" / "radar_timeline.csv").is_file()]


@pytest.mark.skipif(not RADAR or DEV != "cuda", reason="needs a session with radar frames and CUDA")
def test_radar_cuda_graph_equals_eager():
    """The CUDA-graph front end (RadarDetector default) and the eager one give identical products and
    detections, in every configuration and when configurations alternate (each is its own graph)."""
    from perception.core.session import Session
    from perception.radar import RadarDetector, rx_phase_for
    S = Session(RADAR[0])
    rx = rx_phase_for(S.variant)
    gr, eg = RadarDetector(S.cfg_path, rx_phase_deg=rx), RadarDetector(S.cfg_path, rx_phase_deg=rx, cuda_graph=False)
    assert gr.cuda_graph and not eg.cuda_graph
    k0 = len(S.radar) // 3
    for i, k in enumerate(range(k0, k0 + 12)):
        raw = S.radar.raw(k)
        kw = dict(moving_only=i % 3 != 2, detect=i % 4 != 3, static=i % 2 == 1)
        a, b = gr.process(raw, **kw), eg.process(raw, **kw)
        for key in ("rd", "ra", "p_az"):
            assert torch.equal(a[key], b[key]), key
        for key in ("detections", "static_detections", "noise_db"):
            if key in b:
                assert np.array_equal(a[key], b[key]), key
    assert len(gr._graphs) >= 4
