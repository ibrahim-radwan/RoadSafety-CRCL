# Cascaded radar: firmware

This folder holds the radar firmware for the AWR2243 ×2 + AM273x cascade:
20 images, their `.cfg` files, flash configurations and `firmware_map.json`.
See [firmware/FIRMWARE_STATUS.md](firmware/FIRMWARE_STATUS.md) for which
images are in use and hardware-proven.

| Need | Where |
|---|---|
| Which firmware recorded a session, and its `.cfg` | `../sync_capture_pipeline/radar_firmware.py` |
| Flashing | `../flashing/boot/flash_variant.sh`, `../docs/knowledge/03_radar_firmware_and_dca.md` |
| Capture | `../sync_capture_pipeline/` (`capture_session.py`, `dca_capture_radar_sync.py`) |
| Processing, detection and tracking | `../sync_capture_pipeline/perception/` |
| Findings for the research team | `../sync_capture_pipeline/NOTES_FOR_TEAM.md` |

**History.** This folder came from the research team's Windows development
repository. It used to hold their processing chain (`processing/`), their
live viewer and capture tools (`live_capture.py`, `sync_replay.py`,
`capture/`), and their README.

At the team's request, on 2026-10-25 that chain was merged into
`../sync_capture_pipeline/perception/`:

- the general stages were ported to PyTorch and tested against the old code;
- the single-person stages were retired;
- the old code was deleted.

All of it, and the original README, is in git history (for example
`git show 2e0f49e:cascaded_radar_capture_pipeline/README.md`).
