"""Replay synchronized camera/radar sessions using the recorded pair table.

Each display update consumes exactly one row from ``pairs.csv``.  The camera
image and radar result therefore remain associated even when the two devices
were captured at different frame rates.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SyncPair:
    radar_path: Path
    camera_path: Path
    radar_idx: int
    camera_idx: int
    error_ms: float


def load_pairs(pairs_path: Path, session_dir: Path) -> list[SyncPair]:
    """Load and validate the recorded camera/radar associations."""
    if not pairs_path.is_file():
        raise FileNotFoundError(f"pairs CSV not found: {pairs_path}")

    pairs: list[SyncPair] = []
    with pairs_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"radar_idx", "radar_file", "camera_idx", "camera_file", "error_ms"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{pairs_path} is missing required columns: {', '.join(sorted(missing))}"
            )
        for line_no, row in enumerate(reader, start=2):
            try:
                radar_path = session_dir / "radar" / row["radar_file"]
                camera_path = session_dir / "camera" / row["camera_file"]
                pair = SyncPair(
                    radar_path=radar_path,
                    camera_path=camera_path,
                    radar_idx=int(row["radar_idx"]),
                    camera_idx=int(row["camera_idx"]),
                    error_ms=float(row["error_ms"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid pair row {line_no}: {exc}") from exc
            if not pair.radar_path.is_file():
                raise FileNotFoundError(f"Radar file from row {line_no} not found: {pair.radar_path}")
            if not pair.camera_path.is_file():
                raise FileNotFoundError(f"Camera file from row {line_no} not found: {pair.camera_path}")
            pairs.append(pair)

    if not pairs:
        raise ValueError(f"No synchronization pairs found in {pairs_path}")
    return pairs


def _finite_limits(values: np.ndarray, span: float = 30.0) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    high = float(np.percentile(finite, 99.0))
    return high - span, high


class SyncReplay:
    def __init__(self, pairs: Iterable[SyncPair], cfg, correction: np.ndarray,
                 display: str, volume_percentile: float = 99.0):
        import matplotlib.pyplot as plt

        self.pairs = list(pairs)
        self.cfg = cfg
        self.correction = correction
        self.display = display
        self.volume_percentile = volume_percentile
        self.volume_type = str(getattr(cfg, "volume_type", "RDE")).upper()
        self.volume_range_max_m = 15.0
        self._plt = plt
        self._cloud_colorbar = None
        self.figure = plt.figure(figsize=(15, 7), constrained_layout=True)
        self.axes = self._build_axes()
        self._last_result: ProcessingResult | None = None

    def _build_axes(self):
        if self.display in ("pointcloud", "volume"):
            camera_ax = self.figure.add_subplot(1, 2, 1)
            volume_ax = self.figure.add_subplot(1, 2, 2, projection="3d")
            if self.display == "pointcloud":
                return {"camera": camera_ax, "cloud": volume_ax}
            return {"camera": camera_ax, "volume": volume_ax}
        camera_ax = self.figure.add_subplot(1, 3, 1)
        rd_ax = self.figure.add_subplot(1, 3, 2)
        ra_ax = self.figure.add_subplot(1, 3, 3)
        return {"camera": camera_ax, "rd": rd_ax, "ra": ra_ax}

    def _show_camera(self, pair: SyncPair) -> None:
        ax = self.axes["camera"]
        ax.clear()
        ax.imshow(self._plt.imread(pair.camera_path))
        ax.set_title(f"Camera {pair.camera_idx:06d}")
        ax.axis("off")

    def _show_heatmaps(self, result: ProcessingResult) -> None:
        rd_ax = self.axes["rd"]
        ra_ax = self.axes["ra"]
        rd_ax.clear()
        ra_ax.clear()

        if result.RD_mti is not None:
            rd_vmin, rd_vmax = _finite_limits(result.RD_mti)
            rd_ax.imshow(
                result.RD_mti.T,
                origin="lower",
                aspect="auto",
                extent=(result.doppler_axis[0], result.doppler_axis[-1],
                        result.range_axis[0], result.range_axis[-1]),
                vmin=rd_vmin,
                vmax=rd_vmax,
                cmap="magma",
            )
        rd_ax.set_title("Radar range-Doppler")
        rd_ax.set_xlabel("Doppler (m/s)")
        rd_ax.set_ylabel("Range (m)")

        if result.RA is not None:
            ra_vmin, ra_vmax = _finite_limits(result.RA)
            ra_ax.imshow(
                result.RA,
                origin="lower",
                aspect="auto",
                extent=(result.az_axis[0], result.az_axis[-1],
                        result.range_axis[0], result.range_axis[-1]),
                vmin=ra_vmin,
                vmax=ra_vmax,
                cmap="magma",
            )
            if result.ra_det is not None and result.ra_det.any():
                range_idx, az_idx = np.where(result.ra_det)
                ra_ax.scatter(
                    result.az_axis[az_idx], result.range_axis[range_idx],
                    marker="+", color="cyan", s=35, linewidths=0.8,
                )
        ra_ax.set_title("Radar range-azimuth")
        ra_ax.set_xlabel("Azimuth (deg)")
        ra_ax.set_ylabel("Range (m)")

    def _show_pointcloud(self, result: ProcessingResult) -> None:
        ax = self.axes["cloud"]
        # Preserve the 3D camera across frames - see _show_volume.
        view_elev, view_azim = ax.elev, ax.azim
        ax.clear()
        ax.view_init(elev=view_elev, azim=view_azim)
        cloud = result.xyz_cloud
        if cloud is not None and len(cloud):
            doppler = cloud[:, 4]
            scatter = ax.scatter(
                cloud[:, 0], cloud[:, 1], cloud[:, 2],
                c=doppler, cmap="coolwarm", vmin=-2.5, vmax=2.5,
                s=18, depthshade=False,
            )
            if self._cloud_colorbar is None:
                self._cloud_colorbar = self.figure.colorbar(
                    scatter, ax=ax, shrink=0.65, pad=0.08,
                    label="Doppler (m/s)"
                )
        else:
            ax.text2D(0.5, 0.5, "No radar detections", transform=ax.transAxes,
                      ha="center", va="center")
        # Fixed limits. These were derived from each frame's own min/max, so
        # the axes rescaled on every step: a subject appearing to move was
        # often just the axes moving under it, and with 1-3 points per frame
        # the scaling was extremely volatile. Physical bounds instead.
        ax.set_xlim(-4.0, 4.0)
        ax.set_ylim(0.0, min(self.volume_range_max_m,
                             float(self.cfg.range_axis[-1])))
        ax.set_zlim(-2.0, 2.0)
        ax.set_title("Radar point cloud")
        ax.set_xlabel("X lateral (m)")
        ax.set_ylabel("Y range (m)")
        ax.set_zlabel("Z elevation (m)")

    def _volume_source(self, result: ProcessingResult):
        """Pick the volume and describe how its dimensions map to the display.

        The two volumes have DIFFERENT dimension orders:
            RDE  (doppler, range, elevation)   -> range is dim 1
            RAE  (range,   azimuth, elevation) -> range is dim 0
        Confusing them silently plots range on the Doppler axis, which looks
        entirely plausible and is wrong. Everything downstream therefore reads
        the axis roles from here rather than assuming a fixed order.

        Returns (data, y_values, y_label, range_dim, title) or None.
        """
        if self.volume_type == "RAE":
            vol = getattr(result, "RAE", None)
            if vol is None:
                return None
            return (vol, self.cfg.az_axis_deg, "Azimuth (deg)", 0,
                    "RAE volume (FUSED RA x RE)")
        vol = getattr(result, "RDE", None)
        if vol is None:
            return None
        return (vol, self.cfg.doppler_axis, "Doppler (m/s)", 1, "RDE volume")

    def _show_volume(self, result: ProcessingResult) -> None:
        ax = self.axes["volume"]
        # ax.clear() resets the 3D camera. Without saving it, any rotation the
        # operator makes snaps back on the very next frame, which makes the
        # view unusable while stepping through pairs.
        view_elev, view_azim = ax.elev, ax.azim
        ax.clear()
        ax.view_init(elev=view_elev, azim=view_azim)

        range_min_m = float(self.cfg.range_axis[0])
        range_max_m = min(self.volume_range_max_m,
                          float(self.cfg.range_axis[-1]))

        src = self._volume_source(result)
        if src is None:
            ax.set_title(f"{self.volume_type} unavailable "
                         f"(cfg.volume_en / volume_type)")
            return
        volume, y_axis_vals, y_label, range_dim, title = src

        # Fixed limits so the axes do not jump between frames - otherwise a
        # stationary subject appears to move because the AXES are rescaling.
        ax.set_xlim(range_min_m, range_max_m)
        ax.set_ylim(float(y_axis_vals[0]), float(y_axis_vals[-1]))
        ax.set_zlim(float(self.cfg.el_axis_deg[0]),
                    float(self.cfg.el_axis_deg[-1]))

        finite = volume[np.isfinite(volume)]
        if finite.size == 0:
            ax.set_title(f"{title} - empty")
            return

        threshold = float(np.percentile(finite, self.volume_percentile))
        i0, i1, i2 = np.nonzero(volume > threshold)

        # Range lives in a different dimension per mode - use range_dim.
        range_idx = i0 if range_dim == 0 else i1
        in_range = self.cfg.range_axis[range_idx] <= range_max_m
        i0, i1, i2 = i0[in_range], i1[in_range], i2[in_range]
        if i0.size == 0:
            ax.set_title(f"{title} - no voxels above "
                         f"p{self.volume_percentile:g} in "
                         f"{range_min_m:g}-{range_max_m:g} m")
            return

        # Cap the point count; matplotlib 3D degrades badly past ~20k.
        max_points = 20000
        if i0.size > max_points:
            values = volume[i0, i1, i2]
            keep = np.argpartition(values, -max_points)[-max_points:]
            i0, i1, i2 = i0[keep], i1[keep], i2[keep]

        if range_dim == 0:                      # RAE: (range, az, el)
            x_vals = self.cfg.range_axis[i0]
            y_vals = y_axis_vals[i1]
        else:                                   # RDE: (doppler, range, el)
            x_vals = self.cfg.range_axis[i1]
            y_vals = y_axis_vals[i0]
        z_vals = self.cfg.el_axis_deg[i2]
        intensity = volume[i0, i1, i2]

        # Pin the colour scale. Without vmin/vmax matplotlib renormalises per
        # frame, so a weak frame renders as "hot" as a strong one and apparent
        # brightness carries no meaning across frames.
        c_hi = float(np.percentile(finite, 99.99))
        ax.scatter(
            x_vals, y_vals, z_vals,
            c=intensity, cmap="turbo",
            vmin=threshold, vmax=max(c_hi, threshold + 1e-3),
            s=4, alpha=0.5, linewidths=0,
        )
        ax.set_xlabel("Range (m)")
        ax.set_ylabel(y_label)
        ax.set_zlabel("Elevation angle estimate (deg)")
        ax.set_title(
            f"{title}  |  {i0.size} voxels > p{self.volume_percentile:g}  |  "
            f"range {range_min_m:g}-{range_max_m:g} m"
        )

    def show_pair(self, pair: SyncPair) -> None:
        from processing.radar_processing import process_frame
        from processing.radar_reader import read_frame

        self._show_camera(pair)
        cube = read_frame(str(pair.radar_path), self.cfg)
        result = process_frame(cube, self.cfg, self.correction, verbose=False)
        self._last_result = result
        if self.display == "volume":
            self._show_volume(result)
        elif self.display == "pointcloud":
            self._show_pointcloud(result)
        else:
            self._show_heatmaps(result)
        # Name the volume type in the caption so a screenshot is
        # self-identifying - RDE and RAE look similar but mean different
        # things, and RAE is fused rather than measured.
        _vtag = ""
        if self.display == "volume":
            _vtag = (f"  |  {self.volume_type}"
                     + ("  [FUSED]" if self.volume_type == "RAE" else ""))
        self.figure.suptitle(
            f"Synchronized replay  |  radar {pair.radar_idx:06d}  ↔  "
            f"camera {pair.camera_idx:06d}  |  pairing error "
            f"{pair.error_ms:+.2f} ms{_vtag}",
            fontsize=13,
        )
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def run(self, fps: float, loop: bool, max_frames: int) -> None:
        shown = 0
        self._plt.ion()
        while self._plt.fignum_exists(self.figure.number):
            for pair in self.pairs:
                if not self._plt.fignum_exists(self.figure.number):
                    return
                self.show_pair(pair)
                shown += 1
                if max_frames and shown >= max_frames:
                    return
                self._plt.pause(1.0 / fps if fps > 0 else 0.001)
            if not loop:
                return


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay paired camera and radar frames for sync verification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("session", type=Path, help="Session directory containing camera/, radar/, and pairs.csv")
    parser.add_argument("--cfg", type=Path,
                        default=Path(__file__).resolve().parent / "firmware" / "r017m" / "cascade_r017m.cfg")
    parser.add_argument("--pairs", type=Path, default=None, help="Pair CSV (default: SESSION/pairs.csv)")
    parser.add_argument(
        "--display", choices=("heatmap", "pointcloud", "volume"),
        default="heatmap",
    )
    parser.add_argument(
        "--volume", dest="volume_type", choices=("RDE", "RAE"), default="RDE",
        help="volume type for --display volume. RDE = Range x Doppler x "
             "Elevation (measured). RAE = Range x Azimuth x Elevation "
             "(FUSED - ghosts possible within a range bin).",
    )
    parser.add_argument(
        "--volume-percentile", type=float, default=99.0,
        help="keep voxels above this percentile of the frame",
    )
    parser.add_argument("--fps", type=float, default=0.0, help="Playback rate; 0 displays each pair as soon as it is processed")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many paired rows; 0 means all rows")
    parser.add_argument("--loop", action="store_true", help="Loop the paired session")
    parser.add_argument("--raw-iq-order", choices=("i-first", "q-first", "cfg"), default="i-first")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    session = args.session.resolve()
    pairs_path = (args.pairs or session / "pairs.csv").resolve()
    cfg_path = args.cfg.resolve()
    if not session.is_dir():
        raise SystemExit(f"Session directory not found: {session}")
    if not cfg_path.is_file():
        raise SystemExit(f"Radar config not found: {cfg_path}")

    pairs = load_pairs(pairs_path, session)
    from processing.radar_config import (
        apply_raw_iq_order_override,
        parse_calibration,
        parse_cfg_file,
    )

    cfg = parse_cfg_file(str(cfg_path))
    cfg.volume_en = args.display == "volume"
    cfg.volume_type = args.volume_type
    if cfg.volume_en:
        _note = ("FUSED RA x RE - ghosts possible within a range bin"
                 if args.volume_type == "RAE" else "measured")
        print(f"[SYNC] volume type: {args.volume_type}  ({_note})")
    if args.raw_iq_order != "cfg":
        cfg = apply_raw_iq_order_override(cfg, args.raw_iq_order)
    correction = parse_calibration(str(cfg_path), cfg.num_tx, cfg.num_rx)

    print(f"[SYNC_REPLAY] pairs={len(pairs)} display={args.display} session={session}")
    SyncReplay(
        pairs, cfg, correction, args.display, args.volume_percentile
    ).run(
        fps=max(args.fps, 0.0), loop=args.loop, max_frames=max(args.max_frames, 0)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())