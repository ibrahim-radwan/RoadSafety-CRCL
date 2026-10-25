# Working practices (for people and AI agents)

## Agreed with the project owner

- **Ask before anything hard to undo or outside the repository:**
  - deleting recordings;
  - flashing firmware;
  - changing system configuration (network, fstab, sysctl, services);
  - installing packages into the shared conda environment;
  - pushing to GitHub.
- **The radar processing is ours to improve (team approval, 2026-10-25).**
  - The research team asked for their early processing chain
    (`cascaded_radar_capture_pipeline/processing`) to be improved and brought
    up to date. Modify, restructure or replace it as needed.
  - The team no longer runs the tools from that folder: `main.py`,
    `live_capture.py`, the streaming viewer and `capture/` were part of their
    initial Windows work.
  - This repository is the up-to-date version. The team pulls it, so renamed
    paths do not affect them.
  - Keep what they rely on documented: output formats, conventions, and the
    findings in `NOTES_FOR_TEAM.md`.
  - Measure every change of an algorithm against the reference sessions
    (`run_perception.py` scores) before replacing the old one.
- **Calibration stays the team's.** Radar antenna/intrinsic calibration and
  the three-sensor extrinsics are theirs. Corrections we derive in situ (for
  example `calibration/radar_rx_phase.json`) are provisional until theirs
  arrive.
- **Recordings are not version-controlled.** Never commit sessions or capture
  binaries.
- **Commits:** no AI/assistant attribution of any kind (no "Co-Authored-By"
  lines or tool references), and author them as the human committer.
- **Verify on real data before reporting success.** Render the figure, run the
  replay, run the tests, and say plainly what was not verified.
- **Keep the docs current.** When behaviour changes, update the relevant
  README, the design doc and `docs/knowledge/` in the same piece of work.

## Briefs for other agents (for example the Windows firmware-build machine)

When the output will come back to this repository:

- **Give the exact output folder**, relative to the handover root, and the
  exact file layout.
- **Provide a baseline copy** of the current tree, with an md5 `MANIFEST.txt`,
  and say which differences are expected.
- **Ask for a complete mirror in a new subfolder**, never edits to existing
  ones.
- **On return:**
  1. verify the manifests;
  2. diff against the baseline;
  3. back up;
  4. copy;
  5. re-run the checker on the Jetson.
- **Include facts the other agent can't see:** absolute Jetson paths, the
  Python environment, conventions.

Example briefs: `sync_capture_pipeline/handover/`.

## Environments

| Environment | Used for |
|---|---|
| `rel_1_py_3_8` (conda, Python 3.8) | Camera (pyzed), LiDAR wrapper, replay (PyQt5, pyqtgraph, PyOpenGL), report, Stage 3, tests, flashing (pyserial, tqdm, xmodem), our detection (`perception/`, PyTorch), the team's `process_frame` GPU path (CuPy 12.3, added 2026-10-24 on `ibrahim-desktop`) |
| `/usr/bin/python3` (3.8) | The capture launcher and radar receiver (stdlib only), the web server's launcher |
| `cascaded-radar` (conda, Python 3.10, NumPy 2, **CuPy 13 for CUDA 11**) | The research team's tools. Not used by this pipeline. Their `process_frame` GPU path also runs in `rel_1_py_3_8` now (CuPy 12.3, the last series for Python 3.8). |
