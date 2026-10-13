#!/usr/bin/env bash
# flash_variant.sh <variant> [serial-port]
#
# Flash one variant's .appimage to QSPI and RECORD WHAT WAS FLASHED.
#
# WHY A WRAPPER
#   Seven variants will be swapped repeatedly across J1-J3. Two steps in that
#   loop are easy to skip and silent when skipped:
#     1. confirming the .appimage on disk is the one firmware_map.json names
#     2. writing ~/.radar_flashed so processing can tell what is in flash
#   QSPI boot removed the loader log line that used to identify the image, so
#   if (2) is skipped there is no record at all -- the situation that cost the
#   whole May-data attribution exercise. This makes both unskippable.
#
#   It deliberately does NOT touch the SBL at 0x0. That lives in
#   flash_provision.cfg and is only needed if the board stops booting.
#
# USAGE
#   ./flash_variant.sh r075m
#   ./flash_variant.sh r050m /dev/ttyACM1
set -euo pipefail

VARIANT="${1:?usage: $0 <variant> [serial-port]}"
PORT="${2:-/dev/ttyACM0}"

REPO="$HOME/dev/CascadedRadarRoadSafetyOnTwoWheelers"
FW="$REPO/cascaded_radar_capture_pipeline/firmware"
BOOT="$REPO/flashing/boot"
MAP="$FW/firmware_map.json"
VDIR="$FW/$VARIANT"
RECORD="$HOME/.radar_flashed"

die() { printf '\n[ABORT] %s\n' "$*" >&2; exit 1; }

[ -f "$MAP" ]   || die "no firmware_map.json at $MAP"
[ -d "$VDIR" ]  || die "unknown variant '$VARIANT' (no directory $VDIR)"

# ---- pull this variant's expected identity straight from the map ----------
read -r APPIMAGE WANT_MD5 SLOPE MAXR FLOOR CFG <<EOF
$(python3 -I -c "
import json,sys
v=json.load(open('$MAP'))['variants'].get('$VARIANT')
if not v: sys.exit('variant not in map')
print(v['appimage'], v['appimage_md5'].lower(), v['slope_mhz_us'],
      v['max_range_m'], v['hpf_floor_m'], v['processing_cfg'])")
EOF

IMG="$VDIR/$APPIMAGE"
FCFG="$VDIR/flash_$VARIANT.cfg"
[ -f "$IMG" ]  || die "missing image: $IMG"
[ -f "$FCFG" ] || die "missing flash config: $FCFG"

# ---- 1. do the bytes on disk match what the map describes? ---------------
GOT_MD5="$(md5sum "$IMG" | cut -d' ' -f1)"
[ "$GOT_MD5" = "$WANT_MD5" ] || die \
  "$APPIMAGE hashes $GOT_MD5 but the map records $WANT_MD5.
        This image is NOT the build the map describes. Do not flash it."

# ---- 2. does the flash config point at THIS image? ----------------------
CFG_TARGET="$(grep -oP '(?<=--file=)\S+(?=.*0[xX]0*A0000)' "$FCFG" || true)"
[ -n "$CFG_TARGET" ] || die "no --file at offset 0xA0000 in $FCFG"
[ "$(readlink -f "$CFG_TARGET")" = "$(readlink -f "$IMG")" ] || die \
  "flash_$VARIANT.cfg points at
          $CFG_TARGET
        but this variant's image is
          $IMG"
if grep -qP -- '--flash-offset=0[xX]0+(\s|$)' "$FCFG"; then
  die "$FCFG also writes the SBL at 0x0. Use flash_provision.cfg for that."
fi

cat <<EOF

================================================================
  FLASH $VARIANT
================================================================
  image     $APPIMAGE
  md5       $GOT_MD5   (matches firmware_map.json)
  slope     $SLOPE MHz/us
  window    $FLOOR - $MAXR m   <-- nothing closer than $FLOOR m will be seen
  cfg       $CFG
  port      $PORT
================================================================

  Put the board in UART BOOT mode (boot jumper ON), then press Enter.
EOF
read -r _

cd "$BOOT"
python uart_uniflash.py -p "$PORT" --cfg="$FCFG"

# ---- 3. record it, in a form the processing interlock can parse ----------
{
  echo "variant=$VARIANT"
  echo "appimage=$IMG"
  echo "appimage_md5=$GOT_MD5"
  echo "processing_cfg=$VDIR/$CFG"
  echo "slope_mhz_us=$SLOPE"
  echo "max_range_m=$MAXR"
  echo "hpf_floor_m=$FLOOR"
  echo "flashed_at=$(date -Is)"
} > "$RECORD"

cat <<EOF

  recorded in $RECORD:
$(sed 's/^/    /' "$RECORD")

  NEXT
    1. Move the boot jumper back to QSPI boot and POWER CYCLE the board.
       (Boot mode is sampled at reset -- moving the jumper alone does nothing.)
    2. sudo sysctl -w net.core.rmem_max=33554432      # resets on reboot
    3. Capture. Remember: nothing inside $FLOOR m will appear, by design.
EOF
