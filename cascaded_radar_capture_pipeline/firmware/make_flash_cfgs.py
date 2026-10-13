#!/usr/bin/env python3
"""
make_flash_cfgs.py — generate the uniflash configs from firmware_map.json.

WHY GENERATE RATHER THAN COPY-PASTE
    Seven configs, each naming its variant twice, differing by one line. That
    is the same shape as the three identically-named .xer5f files and the
    per-profile processing .cfg files: a class of error this project has
    already been bitten by twice. Deriving them from the map makes the variant
    name single-source, so a wrong pairing cannot be written in the first
    place. fill_firmware_map.py then still checks them, because a generator
    that is never run is worth nothing.

WHAT IT WRITES
    <root>/flash_provision.cfg          SBL at 0x0      — run ONCE
    <root>/<variant>/flash_<v>.cfg      appimage 0xA0000 — every variant switch

    Splitting them keeps the bootloader out of the path that runs repeatedly.
    A corrupted write to 0x0 leaves the board unbootable; a corrupted write to
    0xA0000 leaves the SBL intact and the board still recoverable over the
    same UART. There is no reason to re-write the SBL to change a chirp sweep.

USAGE
    python3 make_flash_cfgs.py                  # dry run, prints the files
    python3 make_flash_cfgs.py --write
    python3 make_flash_cfgs.py --write --root <dir> --boot <sbl_prebuilt dir>

    Then verify:
    python3 fill_firmware_map.py --root <dir>
"""
from __future__ import annotations
import argparse, json, os, sys

HOME = os.path.expanduser('~')
REPO = os.path.join(HOME, 'dev', 'CascadedRadarRoadSafetyOnTwoWheelers')
DEF_ROOT = os.path.join(REPO, 'cascaded_radar_capture_pipeline', 'firmware')
DEF_BOOT = os.path.join(REPO, 'flashing', 'boot', 'sbl_prebuilt', 'am273x-evm')

WRITER = 'sbl_uart_uniflash.release.tiimage'   # UART bootstrap, not a flash write
QSPI_SBL = 'sbl_qspi.release.tiimage'          # goes to 0x0, once
APP_SLOT = '0xA0000'                           # MCU+ am273x application slot


def main() -> int:
    ap = argparse.ArgumentParser(description='Generate uniflash configs')
    ap.add_argument('--root', default=DEF_ROOT, help=f'firmware tree (default {DEF_ROOT})')
    ap.add_argument('--boot', default=DEF_BOOT, help=f'sbl_prebuilt dir (default {DEF_BOOT})')
    ap.add_argument('--write', action='store_true', help='actually write the files')
    a = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(a.root))
    boot = os.path.abspath(os.path.expanduser(a.boot))
    mpath = os.path.join(root, 'firmware_map.json')

    if not os.path.exists(mpath):
        sys.exit(f'[ERROR] not found: {mpath}\n        pass --root <dir>')

    writer = os.path.join(boot, WRITER)
    sbl = os.path.join(boot, QSPI_SBL)
    warn = [f'[WARN] missing: {p}' for p in (writer, sbl) if not os.path.exists(p)]

    with open(mpath, encoding='utf-8') as f:
        m = json.load(f)

    planned: list[tuple[str, str]] = []

    # one-time provision config: bootloader only
    planned.append((os.path.join(root, 'flash_provision.cfg'),
                    f'--flash-writer={writer}\n'
                    f'--file={sbl} --operation=flash --flash-offset=0x0\n'))

    # per-variant: application slot only
    for name, v in m['variants'].items():
        vdir = os.path.join(root, name)
        img = os.path.join(vdir, v['appimage'])
        if not os.path.isdir(vdir):
            warn.append(f'[WARN] no directory for {name}: {vdir}')
            continue
        if not os.path.exists(img):
            warn.append(f'[WARN] {name}: {v["appimage"]} not present yet')
        planned.append((os.path.join(vdir, f'flash_{name}.cfg'),
                        f'--flash-writer={writer}\n'
                        f'--file={img} --operation=flash --flash-offset={APP_SLOT}\n'))

    print('=' * 78)
    print(f'  {"WRITING" if a.write else "DRY RUN"}  ({len(planned)} files)')
    print(f'  firmware root : {root}')
    print(f'  sbl_prebuilt  : {boot}')
    print('=' * 78)

    for path, body in planned:
        rel = os.path.relpath(path, root)
        print(f'\n--- {rel} ' + '-' * max(0, 70 - len(rel)))
        print(body, end='')
        if a.write:
            with open(path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(body)

    if warn:
        print('\n' + '=' * 78)
        for w in warn:
            print('  ' + w)

    print('\n' + '=' * 78)
    if a.write:
        print(f'  wrote {len(planned)} files.')
        print(f'  verify: python3 fill_firmware_map.py --root {root}')
    else:
        print('  dry run - nothing written. Re-run with --write.')
    print('=' * 78)
    return 0


if __name__ == '__main__':
    sys.exit(main())
