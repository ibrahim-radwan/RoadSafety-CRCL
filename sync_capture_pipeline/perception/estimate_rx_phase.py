#!/usr/bin/env python3
"""
estimate_rx_phase.py - in-situ radar phase correction (per virtual channel or
per RX), with the LiDAR as the angle reference.

The .cfg antenna calibration was measured with the old HPF setting; on the v2
images it leaves stable phase errors, which split and widen the azimuth beam
(a single person shows as an arc with a second peak ~13 deg away). This tool
measures the residual on sessions with a single strong target and a LiDAR
target track (sync/timing_lidar_track.npz, from make_report.py):

    for each radar frame with the LiDAR target at 2.5-15 m:
        the strongest radar detection within 0.5 m of the LiDAR range, SNR >=
        --min-snr -> its 40-channel snapshot (MTI, Doppler, TDM-compensated)
        steer it to the LiDAR azimuth, remove the snapshot's common phase
    phase per channel = circular mean over the snapshots
    the part linear in the aperture position is removed: it only steers the
    beam, i.e. it is the radar-LiDAR yaw of that mounting (extrinsic)
    --per-rx: average the 5 azimuth TX of each RX instead (8 values)

Best source: the Stage 3 plate sessions (a near-point reflector): measured
on stage3_02 + stage3_03, it cut the azimuth spread on freewalk_ref_02 (not
used to estimate it) from 5.95 to 2.43 deg. It prints the two halves of the
data separately (they must agree) and the azimuth spread before and after on
each half when trained on the other.

    python perception/estimate_rx_phase.py --session <s> [<s2> ...] [--per-rx] [--write]

--write stores it in calibration/radar_rx_phase.json under the sessions'
firmware variant. PROVISIONAL: the team owns the radar calibration; replace
this with their recalibration (a reflector over a wide angle span) when it
arrives.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from perception.core.conventions import processing_azimuth       # noqa: E402
from perception.core.session import Session                       # noqa: E402
from perception.radar import RX_PHASE_FILE, C, RadarDetector      # noqa: E402


def snapshots(S: Path, min_snr: float = 30.0, step: int = 1):
    """[(40-channel complex snapshot, processing azimuth of the LiDAR target)], detector."""
    sess = Session(S)
    det = RadarDetector(sess.cfg_path, rx_phase_deg=None)
    torch = det.torch
    tl = sess.radar
    out = []
    for k in range(0, len(tl), step):
        tg = sess.target_at(float(tl.t[k]), 0.05)
        if tg is None or not 2.5 <= tg[0] <= 15.0:
            continue
        raw = tl.raw(k)
        d = det.process(raw)["detections"]
        near = np.abs(d[:, C["range"]] - tg[0]) < 0.5 if len(d) else np.zeros(0, bool)
        if not near.any():
            continue
        q = d[near][np.argmax(d[near, C["snr_db"]])]
        if q[C["snr_db"]] < min_snr:
            continue
        virt = det.range_cube(raw)[det.ch_tx, :, det.ch_rx, :]
        virt = virt - virt.mean(dim=1, keepdim=True)
        dch = torch.fft.fftshift(torch.fft.fft(virt * det.win_d[None, :, None], dim=1), dim=1)
        dch = dch * det.tdm_comp[det.ch_tx][:, :, None]
        ri = int(round(q[C["range"]] / det.range_res))
        di = int(np.argmin(np.abs(det.dop_axis - q[C["v_r"]])))
        out.append((dch[:, di, ri].cpu().numpy(), float(processing_azimuth(tg[1]))))   # + = right
    return det, out


def _geometry(det):
    ch_az, _, _ = det.cfg.build_channel_table()
    return (np.array([c["az"] for c in ch_az], float), np.array([c["tx"] for c in ch_az]),
            np.array([c["rx"] for c in ch_az]))


def estimate(det, snaps, per_rx: bool = False):
    """Phase correction (deg): [tx, rx] (6 x 8; the elevation TX row stays 0) or [rx] (8);
    plus coherence and the beam shift removed (deg)."""
    pos, txi, rxi = _geometry(det)
    acc = np.zeros(len(pos), complex)
    for x, az in snaps:
        z = x * np.exp(-1j * np.pi * pos * np.sin(np.radians(az)))
        z = z * np.conj(z.sum()) / max(abs(z.sum()), 1e-12)          # remove the common phase
        acc += z / max(np.abs(z).mean(), 1e-12)
    ph = np.unwrap(np.angle(acc))
    b = np.polyfit(pos, ph, 1)[0]                                     # beam steering = yaw (extrinsic)
    ph = ph - b * pos
    ch = np.zeros((det.n_tx, det.n_rx))
    for t, r, v in zip(txi, rxi, ph):
        ch[t, r] = v
    if per_rx:
        az_tx = sorted(set(txi.tolist()))
        rx = np.array([np.angle(np.mean(np.exp(1j * ch[az_tx, r]))) for r in range(det.n_rx)])
        out = rx - rx[0]
    else:
        out = ch - ch[txi[0], rxi[0]]
        out[[t for t in range(det.n_tx) if t not in set(txi.tolist())]] = 0.0
    out = np.degrees(np.angle(np.exp(1j * out)))
    shift = float(np.degrees(np.arcsin(np.clip(b / np.pi, -1, 1))))
    return out, float(np.abs(acc).mean() / max(len(snaps), 1)), shift


def az_spread(det, snaps, phase_deg):
    """Median |error - bias| (deg) and bias of a windowed FFT beamformer on the snapshots."""
    pos, txi, rxi = _geometry(det)
    ph = np.asarray(phase_deg, float)
    corr = np.exp(-1j * np.radians(ph[rxi] if ph.ndim == 1 else ph[txi, rxi]))
    win = np.hanning(int(pos.max()) + 1)[pos.astype(int)]
    s = np.linspace(-1, 1, 4096, endpoint=False)
    m = np.abs(s) < np.sin(np.radians(70))
    A = np.exp(-1j * np.pi * np.outer(s[m], pos))
    e = np.array([np.degrees(np.arcsin(s[m][np.argmax(np.abs(A @ (x * corr * win)))])) - az for x, az in snaps])
    bias = float(np.median(e))
    return float(np.median(np.abs(e - bias))), bias


def main() -> int:
    ap = argparse.ArgumentParser(description="In-situ radar phase correction (LiDAR as reference)")
    ap.add_argument("--session", type=Path, nargs="+", required=True)
    ap.add_argument("--per-rx", action="store_true", help="8 per-RX phases instead of per channel")
    ap.add_argument("--min-snr", type=float, default=30.0)
    ap.add_argument("--write", action="store_true", help="store in calibration/radar_rx_phase.json")
    a = ap.parse_args()
    t0 = time.monotonic()
    snaps, det, variants = [], None, set()
    for S in a.session:
        det, sn = snapshots(S, a.min_snr)
        variants.add(Session(S).variant)
        print(f"{S.name}: {len(sn)} snapshots")
        snaps += sn
    if len(variants) != 1:
        raise SystemExit(f"sessions from different firmware variants: {sorted(variants)}")
    print(f"{len(snaps)} snapshots ({time.monotonic() - t0:.0f} s)")
    if len(snaps) < 100:
        raise SystemExit("too few snapshots (need a single strong target at 2.5-15 m and a LiDAR target track)")
    halves = (snaps[0::2], snaps[1::2])
    ests = [estimate(det, x, a.per_rx) for x in halves]
    for name, (ph, coh, shift) in zip(("half A", "half B"), ests):
        print(f"  {name}: coherence {coh:.2f}, beam shift removed {shift:+.2f} deg (radar-LiDAR yaw, extrinsic)")
    diff = np.degrees(np.abs(np.angle(np.exp(1j * np.radians(ests[0][0] - ests[1][0])))))
    print(f"  halves agree within {np.median(diff):.1f} deg median, {diff.max():.1f} deg max")
    zeros = np.zeros_like(ests[0][0])
    for tr_, te_, name in ((0, 1, "train A, test B"), (1, 0, "train B, test A")):
        s0, b0 = az_spread(det, halves[te_], zeros)
        s1, b1 = az_spread(det, halves[te_], ests[tr_][0])
        print(f"  {name}: azimuth spread {s0:.2f} -> {s1:.2f} deg (bias {b0:+.2f} -> {b1:+.2f})")
    ph, coh, shift = estimate(det, snaps, a.per_rx)
    if a.write:
        variant = variants.pop()
        doc = json.loads(RX_PHASE_FILE.read_text()) if RX_PHASE_FILE.is_file() else {"variants": {}}
        doc["note"] = ("In-situ radar phase correction (deg), applied on top of the .cfg antenna calibration by "
                       "perception/radar.py: per RX (8 values) or per channel [tx][rx] (6 x 8). PROVISIONAL: "
                       "measured against the LiDAR target azimuth by perception/estimate_rx_phase.py; replace "
                       "with the team's recalibration.")
        doc["variants"][variant] = {"rx_phase_deg": np.round(ph, 2).tolist(),
                                    "kind": "per RX" if a.per_rx else "per channel [tx][rx]",
                                    "sessions": [S.name for S in a.session], "snapshots": len(snaps),
                                    "min_snr_db": a.min_snr, "coherence": round(coh, 3),
                                    "beam_shift_removed_deg": round(shift, 2),
                                    "halves_agree_deg_median": round(float(np.median(diff)), 1),
                                    "measured": time.strftime("%Y-%m-%d")}
        RX_PHASE_FILE.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"  wrote {RX_PHASE_FILE} [{variant}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
