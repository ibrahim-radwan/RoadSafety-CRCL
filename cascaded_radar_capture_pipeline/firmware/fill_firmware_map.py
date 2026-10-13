#!/usr/bin/env python3
"""
fill_firmware_map.py — populate appimage_md5 in firmware_map.json, and verify
the firmware tree is complete and self-consistent.

Run on the Jetson AFTER transferring the firmware tree. Doing it here rather
than on Windows means it doubles as a transfer integrity check: a file that
failed to copy shows up as a missing hash rather than being discovered later
when a flash silently writes the wrong image.

EXPECTED LAYOUT
    <root>/firmware_map.json
    <root>/<variant>/am273x_cascade_<variant>.appimage
    <root>/<variant>/cascade_<variant>.cfg
    <root>/<variant>/flash_<variant>.cfg          (optional at this stage)
    <root>/<variant>/am273x_cascade_<variant>.rprc   (or _build_record/, see below)

WHAT IT CHECKS
  1. Every variant in the map has its .appimage present -> fills appimage_md5
  2. No two variants share an .appimage hash. Two matching means two
     "different" packages hold the same firmware, which is the failure the
     .rprc gate exists to catch (see build-determinism-finding.md §2)
  3. The processing .cfg is present, and its profileCfg slope matches the
     slope recorded in the map. Catches a .cfg paired with the wrong variant
  4. If a .rprc is present, its hash matches the recorded rprc_md5 — the
     deterministic fingerprint, so a mismatch means the build drifted
  5. If flash_<variant>.cfg is present: it writes exactly one file to the
     application slot 0xA0000, that file is THIS variant's .appimage, it
     resolves inside this variant's own directory, and it hashes to the
     recorded appimage_md5. Seven near-identical configs each naming a
     variant twice is the same copy-paste exposure that produced three
     identically-named .xer5f files; a config pointing at a sibling's image
     would flash the wrong sweep with no other symptom. Also warns if the
     config writes the SBL at 0x0 — that belongs in a one-time provision
     config, not in something run on every variant switch

USAGE
    python3 fill_firmware_map.py                 # dry run, reports only
    python3 fill_firmware_map.py --write         # also writes the hashes back

    # the Jetson tree is NOT at the ~/radar/firmware default:
    python3 fill_firmware_map.py --write --root \\
      ~/dev/CascadedRadarRoadSafetyOnTwoWheelers/cascaded_radar_capture_pipeline/firmware
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


C = 299792458.0
LSB = 3.6e6 * 900 / 2 ** 26 / 1e3          # 0.0482797623 MHz/us — NOT 48 kHz/us
HPF1_HZ = {0: 175e3, 1: 235e3, 2: 350e3, 3: 700e3}   # TI AWR2243 HPF1 corners (corrected 2026-10-10)
HPF2_HZ = {0: 350e3, 1: 700e3, 2: 1400e3, 3: 2800e3}

# profileCfg token index (after the command word) -> map field name.
# Token 7 (the slope) is NOT unique: r017m and r017m_calib share 44.6588, so
# checking the slope alone cannot tell those two apart. idle_time_us can.
PROF = {1: 'start_freq_ghz', 2: 'idle_time_us', 3: 'adc_start_time_us',
        4: 'ramp_end_time_us', 7: 'slope_mhz_us', 9: 'num_adc_samples',
        10: 'sample_rate_ksps', 11: 'hpf1_reg', 12: 'hpf2_reg', 13: 'rx_gain_db'}


def cfg_profile(path: str):
    """Return {map_field: float} from the .cfg's profileCfg line, or None."""
    try:
        with open(path, encoding='utf-8-sig') as f:
            for raw in f:
                line = raw.split('%')[0].strip()
                if line.split()[:1] == ['profileCfg']:
                    t = line.split()[1:]
                    return {fld: float(t[i]) for i, fld in PROF.items()}
    except (OSError, IndexError, ValueError):
        pass
    return None


def derive(v: dict) -> dict:
    """Recompute every derived field from the registers alone.

    This is what makes the map self-proving rather than a table of asserted
    numbers. Note v_max/v_res need start_freq, idle and ramp — if a schema-1
    map is passed (which lacks them) the velocity fields cannot be checked and
    are omitted from the result.
    """
    sl = v['freq_slope_const'] * LSB * 1e12        # Hz/s
    fs = v['sample_rate_ksps'] * 1e3
    maxR = C * fs / (2 * sl)
    d = {'slope_mhz_us': round(v['freq_slope_const'] * LSB, 4),
         'max_range_m': round(maxR, 2),
         'range_res_m': round(maxR / v['num_adc_samples'], 4)}
    if 'hpf2_reg' in v:
        d['hpf_floor_m'] = round(HPF2_HZ[v['hpf2_reg']] * C / (2 * sl), 2)
        if 'hpf1_floor_m' in v:
            d['hpf1_floor_m'] = round(HPF1_HZ[v['hpf1_reg']] * C / (2 * sl), 2)
    if {'start_freq_ghz', 'idle_time_us', 'ramp_end_time_us'} <= set(v):
        tl = v['num_chirps'] * (v['idle_time_us'] + v['ramp_end_time_us']) * 1e-6
        vmax = (C / (v['start_freq_ghz'] * 1e9)) / (4 * tl)
        d['v_max_mps'] = round(vmax, 4)
        d['v_res_mps'] = round(vmax / (v['num_loops'] / 2), 4)
    if 'frame_periodicity_ms' in v and 'fps' in v:
        d['fps'] = round(1000 / v['frame_periodicity_ms'], 2)
    d['frame_bytes'] = v['num_adc_samples'] * v['num_chirps'] * v['num_loops'] * 8 * 4
    return d


APP_SLOT = 0xA0000          # MCU+ SDK am273x application offset, proven by the
                            # board's working QSPI boot
SBL_SLOT = 0x0


def flash_ops(path: str) -> list[tuple[int, str]]:
    """Parse a uart_uniflash config into [(flash_offset, file_path), ...].

    Offsets are compared NUMERICALLY, so 0xA0000 / 0xa0000 / 0x000A0000 all
    match. --flash-writer is not a flash write and is deliberately ignored.
    """
    ops: list[tuple[int, str]] = []
    try:
        with open(path, encoding='utf-8-sig') as f:
            for raw in f:
                line = raw.split('#')[0].strip()
                if not line or '--file=' not in line:
                    continue
                fn = off = None
                for tok in line.split():
                    if tok.startswith('--file='):
                        fn = tok[len('--file='):].strip('"\'')
                    elif tok.startswith('--flash-offset='):
                        try:
                            off = int(tok[len('--flash-offset='):].strip('"\''), 0)
                        except ValueError:
                            off = None
                if fn is not None and off is not None:
                    ops.append((off, fn))
    except OSError:
        pass
    return ops


def main() -> int:
    ap = argparse.ArgumentParser(description='Fill and verify firmware_map.json')
    ap.add_argument('--root', default=os.path.expanduser('~/radar/firmware'))
    ap.add_argument('--write', action='store_true',
                    help='write appimage_md5 back into firmware_map.json')
    a = ap.parse_args()

    root = os.path.expanduser(a.root)
    mpath = os.path.join(root, 'firmware_map.json')
    if not os.path.exists(mpath):
        sys.exit(f'[ERROR] not found: {mpath}\n'
                 f'        pass --root <dir> if the tree is elsewhere')

    with open(mpath, encoding='utf-8') as f:
        m = json.load(f)

    print('=' * 92)
    print(f'  FIRMWARE TREE: {root}')
    print('=' * 92)
    print(f"  {'variant':<13}{'derived':>8}{'appimage':>10}{'cfg':>8}"
          f"{'rprc':>6}{'flash':>8}  appimage_md5")
    print('  ' + '-' * 88)

    problems: list[str] = []
    hashes: dict[str, list[str]] = {}
    rprc_checked = 0
    rprc_absent: list[str] = []

    for name, v in m['variants'].items():
        vdir = os.path.join(root, name)
        img = os.path.join(vdir, v['appimage'])
        cfg = os.path.join(vdir, v['processing_cfg'])
        # The .rprc may sit beside the appimage (named like it) or in a
        # separate build record. Look in every plausible place: a path miss
        # here shows up as '-', which reads like "fine" but means "the only
        # deterministic fingerprint was never checked".
        rpc = next((p for p in (
            os.path.join(vdir, v['appimage'].replace('.appimage', '.rprc')),
            os.path.join(vdir, f'{name}.rprc'),
            os.path.join(root, '_build_record', name, f'{name}.rprc'),
            os.path.join(root, '_build_record', f'{name}.rprc'),
        ) if os.path.exists(p)), None) or ''

        # 1. appimage — fill if unrecorded, VERIFY if already recorded.
        #    The verify branch is what makes a second run on another machine a
        #    real transfer check: fill on the USB, copy, re-run on the Jetson.
        if os.path.exists(img):
            h = md5(img)
            prev = (v.get('appimage_md5') or '').lower()
            if prev and prev != h:
                img_s, shown = 'MISMATCH', h
                problems.append(
                    f'{name}: {v["appimage"]} hashes {h} but the map records {prev} '
                    f'-- the file changed or did not copy intact')
            else:
                img_s, shown = ('ok' if prev else 'filled'), h
                v['appimage_md5'] = h
            hashes.setdefault(h, []).append(name)
        else:
            img_s, shown = 'MISSING', '-'
            problems.append(f'{name}: {v["appimage"]} not found in {vdir}')

        # 0. do the DERIVED fields follow from the registers?
        #    Without this the map is a table of asserted numbers; with it, the
        #    file proves itself and a hand-edit cannot pass unnoticed.
        try:
            want = derive(v)
        except (KeyError, TypeError, ValueError) as e:
            want, der_s = {}, 'ERR'
            problems.append(f'{name}: cannot recompute derived fields ({e})')
        else:
            off = {k: (v[k], w) for k, w in want.items()
                   if k in v and abs(v[k] - w) > 10 ** -4 / 2}
            der_s = 'ok' if not off else 'DRIFT'
            for k, (got, w) in off.items():
                problems.append(
                    f'{name}.{k}: map says {got}, registers give {w} '
                    f'-- a derived value was hand-edited or a register changed')
        # the two forced identities, as an independent cross-check
        if abs(v['max_range_m'] / v['range_res_m'] - v['num_adc_samples']) > 0.5:
            problems.append(f'{name}: max_range/range_res != num_adc_samples '
                            f'-- forced FFT identity violated')
        if abs(v['v_max_mps'] / v['v_res_mps'] - v['num_loops'] / 2) > 0.05:
            problems.append(f'{name}: v_max/v_res != num_loops/2 '
                            f'-- forced FFT identity violated')

        # 3. processing cfg: do ALL its profileCfg fields match the map?
        #    The slope alone is not enough — r017m and r017m_calib share it.
        if os.path.exists(cfg):
            p = cfg_profile(cfg)
            if p is None:
                cfg_s = 'NOPROF'
                problems.append(f'{name}: {v["processing_cfg"]} has no readable profileCfg')
            else:
                mism = [(f, p[f], v[f]) for f in p
                        if f in v and abs(p[f] - v[f]) > 5e-4]
                skipped = [f for f in p if f not in v]
                if mism:
                    cfg_s = 'WRONG'
                    for f, c, mv in mism:
                        problems.append(
                            f'{name}: {v["processing_cfg"]} {f}={c:g} but map says {mv:g} '
                            f'-- this .cfg belongs to a different variant')
                else:
                    cfg_s = f'ok/{len(p)-len(skipped)}'
        else:
            cfg_s = 'MISSING'
            problems.append(f'{name}: {v["processing_cfg"]} not found in {vdir}')

        # 4. rprc fingerprint — the ONLY deterministic build identity. The
        #    .xer5f cannot serve: it differs between two builds of the same
        #    source. This is where the W2 gate was relocated to.
        if rpc:
            want = (v.get('rprc_md5') or '').lower()
            got = md5(rpc)
            rpc_s = 'ok' if got == want else 'DRIFT'
            rprc_checked += 1
            if rpc_s == 'DRIFT':
                problems.append(
                    f'{name}: {os.path.basename(rpc)} hashes {got} but the map records '
                    f'{want} -- this is NOT the build the map describes')
        else:
            rpc_s = 'none'
            rprc_absent.append(name)

        # 5. uniflash config: does it point at THIS variant's appimage?
        #    Seven near-identical files naming a variant twice each. If one points
        #    at a sibling's image you flash the wrong sweep with no other symptom.
        fl = os.path.join(vdir, f'flash_{name}.cfg')
        if os.path.exists(fl):
            ops = flash_ops(fl)
            app = [p for off, p in ops if off == APP_SLOT]
            sbl = [p for off, p in ops if off == SBL_SLOT]

            if not app:
                fl_s = 'NOAPP'
                problems.append(
                    f'{name}: flash_{name}.cfg has no --file at offset 0x{APP_SLOT:X} '
                    f'-- it would flash no application')
            elif len(app) > 1:
                fl_s = 'DUPAPP'
                problems.append(
                    f'{name}: flash_{name}.cfg writes {len(app)} files to 0x{APP_SLOT:X}; '
                    f'the last one wins, which is almost certainly not intended')
            else:
                tgt = app[0]
                if os.path.basename(tgt) != v['appimage']:
                    fl_s = 'WRONG'
                    problems.append(
                        f'{name}: flash_{name}.cfg flashes {os.path.basename(tgt)} '
                        f'but this variant is {v["appimage"]} -- copy-paste slip, this '
                        f'would put the wrong sweep on the board silently')
                elif not os.path.exists(tgt):
                    fl_s = 'NOFILE'
                    problems.append(
                        f'{name}: flash_{name}.cfg points at {tgt}, which does not exist')
                elif os.path.realpath(os.path.dirname(tgt)) != os.path.realpath(vdir):
                    fl_s = 'STRAY'
                    problems.append(
                        f'{name}: flash_{name}.cfg names the right file but resolves outside '
                        f'its own directory ({os.path.dirname(tgt)})')
                elif md5(tgt) != (v.get('appimage_md5') or '').lower():
                    fl_s = 'HASH'
                    problems.append(
                        f'{name}: the image flash_{name}.cfg points at does not hash to '
                        f'the recorded appimage_md5')
                elif sbl:
                    fl_s = 'SBL'
                    problems.append(
                        f'{name}: flash_{name}.cfg also writes 0x{SBL_SLOT:X} ({os.path.basename(sbl[0])}). '
                        f'Keep the bootloader in a separate provision config -- re-writing it '
                        f'on every variant switch risks an unbootable board for no benefit')
                else:
                    fl_s = 'ok'
        else:
            fl_s = '-'

        print(f'  {name:<13}{der_s:>8}{img_s:>10}{cfg_s:>8}{rpc_s:>6}{fl_s:>8}  {shown}')

    # 2. two variants sharing an appimage
    for h, names in hashes.items():
        if len(names) > 1:
            problems.append(f'IDENTICAL appimage across {", ".join(names)} ({h}) '
                            f'-- two packages hold the same firmware')

    print()
    if problems:
        print('=' * 92)
        print('  PROBLEMS')
        print('=' * 92)
        for p in problems:
            print(f'  - {p}')
        print()

    filled = sum(1 for v in m['variants'].values() if v.get('appimage_md5'))
    n = len(m['variants'])
    print(f'  {filled}/{n} variants have an appimage hash')
    print(f'  {rprc_checked}/{n} variants had their .rprc fingerprint VERIFIED')
    if rprc_absent:
        print(f'  no .rprc found for: {", ".join(rprc_absent)}')
        print('    -> the deterministic build identity is UNVERIFIED for these.')
        print('       The .appimage hash proves the file did not change in transit;')
        print('       only the .rprc proves it came from the build this map names.')

    if a.write:
        if problems:
            print('\n  NOT WRITING - fix the problems above first, then re-run with --write.')
            return 1
        with open(mpath, 'w', encoding='utf-8') as f:
            json.dump(m, f, indent=2)
            f.write('\n')
        print(f'  written: {mpath}')
        return 0

    print('\n  dry run - re-run with --write to save the hashes.')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
