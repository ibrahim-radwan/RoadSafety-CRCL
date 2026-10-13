# Working practices (for people and AI agents)

## Agreed with the project owner

- **Ask before anything hard to undo or outside the repository:**
  - deleting recordings;
  - flashing firmware;
  - changing system configuration (network, fstab, sysctl, services);
  - installing packages into the shared conda environment;
  - pushing to GitHub.
- **Do not modify the research team's processing chain**
  (`cascaded_radar_capture_pipeline/processing`) without their approval.
  Convert conventions on our side.
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
| `rel_1_py_3_8` (conda, Python 3.8) | Camera (pyzed), LiDAR wrapper, replay (PyQt5, pyqtgraph, PyOpenGL), report, Stage 3, tests, flashing (pyserial, tqdm, xmodem) |
| `/usr/bin/python3` (3.8) | The capture launcher and radar receiver (stdlib only), the web server's launcher |
| `cascaded-radar` (conda, Python 3.10, NumPy 2, **CuPy 13 for CUDA 11**) | The research team's tools. Not used by this pipeline. Their `process_frame` has a CuPy GPU path that this environment might enable (untested). |
