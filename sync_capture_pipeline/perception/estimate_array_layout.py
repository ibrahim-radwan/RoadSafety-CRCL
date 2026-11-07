#!/usr/bin/env python3
"""
estimate_array_layout.py - measure the radar's virtual-array layout and the phase between its two
chips from recorded sessions, with the LiDAR target azimuth as the angle reference (2026-11-07).

Why: the team's table (perception/core/radar_config.py _TX_AZ / _TX_EL / _RX_AZ) does not match the
board. With it, the beam was split and the azimuth error depended on the target's angle, which no
per-channel calibration can fix (docs/knowledge/03_radar_firmware_and_dca.md, "Off-axis azimuth").

Method, on sessions with one strong target and a LiDAR target track (the Stage 3 plate sessions are
best; sync/timing_lidar_track.npz from make_report.py):

    per radar frame with the LiDAR target at 2.5-15 m: the strongest detection within 0.5 m of the
    LiDAR range -> the target's range-Doppler cell -> its 48 channels (6 chirps x 8 RX),
    UNCALIBRATED (no .cfg calibration, no in-situ phase), not TDM-compensated

    positions  per chip, each channel's phase relative to that chip's chirp 0 / first RX, fitted as
               pi * p * sin(az) + (pi v / (6 v_max)) * slot: p = the channel's position (lambda/2)
               and slot = when its chirp fires; the coherence says how well it fits (an elevated TX
               fits badly: its phase also follows the target's elevation)
    inter-chip the slave's phase against the master's, both beamformed with the layout:
               c + k * range (+ pi * D * sin az, D = chip_offset): k is the slave's sampling delay
               (~0.4 us on r075m_v2_8fps)

    python perception/estimate_array_layout.py --session <s> [<s> ...] [--write] [--test <s> ...]

--write stores k and c for the sessions' firmware variant in calibration/radar_array.json (the
layout itself is edited by hand from the printed positions). --test scores the detector's azimuth
against the LiDAR on other sessions with the team layout (as it was in use) and the measured one.
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
from perception.core.calibration import ARRAY_FILE, array_layout, rx_phase_for   # noqa: E402
from perception.core.conventions import processing_azimuth                       # noqa: E402
from perception.core.session import Session                                       # noqa: E402
from perception.radar import C, RadarDetector                                     # noqa: E402

P_GRID = np.arange(-40, 45, 0.1)
T_GRID = np.arange(-12, 12.01, 0.25)


def snapshots(S: Path, min_snr: float = 20.0, step: int = 1) -> dict:
    """The target cell's 48 uncalibrated channels per frame: x [n, tx, rx], LiDAR processing azimuth
    (deg, + right), v_r, LiDAR range."""
    sess = Session(S)
    det = RadarDetector(sess.cfg_path, inter_chip=False)               # finds the target cell only
    raw_det = RadarDetector(sess.cfg_path, layout="team", cfg_calibration=False, cuda_graph=False)
    torch = det.torch
    tl = sess.radar
    X, az, v, r = [], [], [], []
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
        with torch.no_grad():
            cube = raw_det.range_cube(raw)                              # [tx, loop, rx, range]
            cube = cube - cube.mean(dim=1, keepdim=True)
            dop = torch.fft.fftshift(torch.fft.fft(cube * raw_det.win_d[None, :, None, None], dim=1), dim=1)
        ri = int(round(q[C["range"]] / det.range_res))
        di = int(np.argmin(np.abs(det.dop_axis - q[C["v_r"]])))
        X.append(dop[:, di, :, ri].cpu().numpy())
        az.append(float(processing_azimuth(tg[1])))
        v.append(float(q[C["v_r"]]))
        r.append(float(tg[0]))
    return {"x": np.array(X), "az": np.array(az), "v": np.array(v), "r": np.array(r),
            "vmax": float(np.abs(det.dop_axis).max()), "variant": sess.variant, "name": S.name}


def _merge(sn: list) -> dict:
    out = {k: np.concatenate([s[k] for s in sn]) for k in ("x", "az", "v", "r")}
    out["vmax"] = sn[0]["vmax"]
    return out


def fit_channel(sn: dict, tx: int, rx: int, ref: tuple) -> tuple:
    """(coherence, position, slot) of channel (tx, rx) relative to channel ref."""
    s = np.sin(np.radians(sn["az"]))
    z = sn["x"][:, tx, rx] * np.conj(sn["x"][:, ref[0], ref[1]])
    z = z / np.maximum(np.abs(z), 1e-12)
    Ep = np.exp(-1j * np.pi * np.outer(P_GRID, s))
    w = np.pi * sn["v"] / (6 * sn["vmax"])
    best = (0.0, 0.0, 0.0)
    for t in T_GRID:
        sc = np.abs(Ep @ (z * np.exp(-1j * w * t))) / len(s)
        i = int(np.argmax(sc))
        if sc[i] > best[0]:
            best = (float(sc[i]), float(P_GRID[i]), float(t))
    return best


def chip_beams(sn: dict, lay: dict) -> tuple:
    """Each chip's channels steered to the LiDAR azimuth with the layout and summed (TDM-compensated):
    (master, slave) complex per snapshot."""
    s = np.sin(np.radians(sn["az"]))
    w = np.pi * sn["v"] / (6 * sn["vmax"])
    out = [0j, 0j]
    for t, p in enumerate(lay["chirp_tx_pos"]):
        if p is None:
            continue
        for r in range(len(lay["rx_chip"])):
            ch = lay["rx_chip"][r]
            out[ch] = out[ch] + sn["x"][:, t, r] * np.exp(-1j * np.pi * (p + lay["rx_pos"][r]) * s) * np.exp(-1j * w * t)
    return out[0], out[1]


def fit_inter_chip(sn: dict, lay: dict) -> dict:
    """k (rad/m), c (deg) of slave x conj(master) = exp(j (c + k r + pi D s)); D fitted as a check."""
    s = np.sin(np.radians(sn["az"]))
    m, sl = chip_beams(sn, lay)
    z = sl * np.conj(m)
    z = z / np.maximum(np.abs(z), 1e-12)
    D = float(lay["chip_offset"])
    zD = z * np.exp(-1j * np.pi * D * s)
    ks = np.arange(-3, 3.0001, 0.002)
    sc = np.abs(np.exp(-1j * np.outer(ks, sn["r"])) @ zD) / len(z)
    k = float(ks[np.argmax(sc)])
    zk = zD * np.exp(-1j * k * sn["r"])
    c = float(np.degrees(np.angle(zk.mean())))
    Q = np.arange(-40, 60, 0.1)
    zq = z * np.exp(-1j * k * sn["r"])
    q = float(Q[np.argmax(np.abs(np.exp(-1j * np.pi * np.outer(Q, s)) @ zq))])
    return {"slave_phase_rad_per_m": round(k, 3), "slave_phase_deg": round(c, 1), "coherence": round(float(sc.max()), 3),
            "chip_offset_fitted": round(q, 1), "snapshots": int(len(z))}


def score(S: Path, **kw) -> dict:
    """The detector's azimuth (strongest detection within 0.5 m of the LiDAR range) against the LiDAR:
    bias, spread (median |e - bias|), share beyond 5 deg."""
    sess = Session(S)
    det = RadarDetector(sess.cfg_path, **kw)
    tl = sess.radar
    e = []
    for k in range(0, len(tl), 2):
        tg = sess.target_at(float(tl.t[k]), 0.05)
        if tg is None or not 2.5 <= tg[0] <= 15.0:
            continue
        d = det.process(tl.raw(k))["detections"]
        near = np.abs(d[:, C["range"]] - tg[0]) < 0.5 if len(d) else np.zeros(0, bool)
        if not near.any():
            continue
        q = d[near][np.argmax(d[near, C["snr_db"]])]
        if q[C["snr_db"]] >= 15:
            e.append(float(np.degrees(np.arctan2(q[C["y"]], q[C["x"]]))) - tg[1])
    e = np.array(e)
    if len(e) < 10:
        return {"n": len(e)}
    b = float(np.median(e))
    return {"n": len(e), "bias": b, "spread": float(np.median(np.abs(e - b))), "gt5": float(np.mean(np.abs(e - b) > 5) * 100)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure the radar virtual-array layout and the inter-chip phase")
    ap.add_argument("--session", type=Path, nargs="+", required=True)
    ap.add_argument("--test", type=Path, nargs="*", default=[], help="score the detector on these sessions")
    ap.add_argument("--write", action="store_true", help="store k, c for the variant in calibration/radar_array.json")
    ap.add_argument("--min-snr", type=float, default=20.0)
    a = ap.parse_args()
    lay = array_layout()
    if lay is None:
        raise SystemExit(f"no layout in {ARRAY_FILE}")
    t0 = time.monotonic()
    sns = []
    for S in a.session:
        sns.append(snapshots(S, a.min_snr, 1 if S.name.startswith("stage3") else 2))
        print(f"{S.name}: {len(sns[-1]['az'])} snapshots", flush=True)
    variants = {s["variant"] for s in sns}
    if len(variants) != 1:
        raise SystemExit(f"sessions from different firmware variants: {sorted(variants)}")
    variant = variants.pop()
    sn = _merge(sns)
    print(f"{len(sn['az'])} snapshots ({time.monotonic() - t0:.0f} s); sin(az) "
          f"{np.sin(np.radians(np.percentile(sn['az'], 2))):+.2f}..{np.sin(np.radians(np.percentile(sn['az'], 98))):+.2f}")
    print("\npositions (lambda/2) per chirp, relative to the chip's chirp 0 / first RX: fitted (coherence, slot) "
          "| layout file")
    for chip in sorted(set(lay["rx_chip"])):
        rxs = [r for r, c in enumerate(lay["rx_chip"]) if c == chip]
        ref = (0, rxs[0])
        print(f"  chip {chip} (RX {rxs[0]}-{rxs[-1]}):")
        for t in range(sn["x"].shape[1]):
            fits = [fit_channel(sn, t, r, ref) for r in rxs]
            tp = lay["chirp_tx_pos"][t]
            exp = "elevated" if tp is None else " ".join(f"{tp - lay['chirp_tx_pos'][0] + lay['rx_pos'][r]:+5.1f}"
                                                         for r in rxs)
            print(f"    chirp {t}: " + " ".join(f"{p:+5.1f}" for _, p, _ in fits)
                  + f"  (coh {np.mean([c for c, _, _ in fits]):.2f}, slot {np.mean([s for _, _, s in fits]):+.1f})"
                  + f" | {exp}")
    ic = fit_inter_chip(sn, lay)
    print(f"\ninter-chip: k {ic['slave_phase_rad_per_m']:+.3f} rad/m, c {ic['slave_phase_deg']:+.1f} deg, "
          f"coherence {ic['coherence']:.2f}; chip offset fitted {ic['chip_offset_fitted']:+.1f} "
          f"(layout {lay['chip_offset']})")
    if a.write:
        doc = json.loads(ARRAY_FILE.read_text())
        doc.setdefault("variants", {})[variant] = dict(
            ic, sessions=[S.name for S in a.session], measured=time.strftime("%Y-%m-%d"),
            provisional="measured in situ (LiDAR azimuth); a corner-reflector measurement can replace it")
        ARRAY_FILE.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {ARRAY_FILE} [{variant}]")
    for S in a.test:
        old = score(S, layout="team", rx_phase_deg=rx_phase_for(variant, layout="team"))
        new = score(S)
        fmt = lambda d: (f"bias {d['bias']:+5.1f} spread {d['spread']:4.1f} >5deg {d['gt5']:4.1f}% (n {d['n']})"
                         if "bias" in d else f"n {d['n']}")
        print(f"{S.name:18s} team layout + .cfg + in-situ: {fmt(old)}   measured: {fmt(new)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
