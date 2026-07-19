#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKETCH_DIR="$SCRIPT_DIR/steering_firmware"

usage() {
  echo "Usage: $0 uno|nano|nano-old [port]" >&2
  exit 2
}

[ "$#" -ge 1 ] || usage

BOARD="$1"
PORT="${2:-}"

case "$BOARD" in
  uno)      FQBN="arduino:avr:uno" ;;
  nano)     FQBN="arduino:avr:nano" ;;
  nano-old) FQBN="arduino:avr:nano:cpu=atmega328old" ;;
  *) echo "ERROR: unknown board '$BOARD'" >&2; usage ;;
esac

pick_port() {
  local board="$1"
  local pref="" link
  local candidates=()

  case "$board" in
    uno)          pref='Arduino|ACM' ;;
    nano|nano-old) pref='USB_Serial|CH340|1a86|wch' ;;
  esac

  if [ -d /dev/serial/by-id ]; then
    for link in /dev/serial/by-id/*; do
      [ -e "$link" ] || continue
      if echo "$link" | grep -Eiq 'CP210|Silicon_Labs|cp210x'; then
        echo "  skip (lidar): $link" >&2
        continue
      fi
      candidates+=("$link")
    done
  fi

  for link in "${candidates[@]:-}"; do
    [ -n "$link" ] || continue
    if echo "$link" | grep -Eiq "$pref"; then
      echo "  matched (preferred): $link" >&2
      readlink -f "$link"
      return 0
    fi
  done

  for link in "${candidates[@]:-}"; do
    [ -n "$link" ] || continue
    echo "  matched (fallback): $link" >&2
    readlink -f "$link"
    return 0
  done

  if [ "$board" = "uno" ] && [ -e /dev/ttyACM0 ]; then
    echo "  matched (default): /dev/ttyACM0" >&2
    echo "/dev/ttyACM0"
    return 0
  fi

  return 1
}

if [ -z "$PORT" ]; then
  echo "No port given; auto-detecting for '$BOARD'..." >&2
  if ! PORT="$(pick_port "$BOARD")"; then
    echo "ERROR: could not auto-detect a $BOARD serial port." >&2
    echo "       Plug the board in, or pass the port explicitly:" >&2
    echo "       $0 $BOARD /dev/ttyACM0   (or /dev/ttyUSB1, etc.)" >&2
    exit 1
  fi
fi

echo "Board : $BOARD  ($FQBN)"
echo "Port  : $PORT"
echo "Sketch: $SKETCH_DIR"

if [ ! -f "$SKETCH_DIR/steering_firmware.ino" ]; then
  echo "ERROR: sketch not found at $SKETCH_DIR/steering_firmware.ino" >&2
  exit 1
fi

echo "==> Compiling..."
if ! arduino-cli compile --fqbn "$FQBN" "$SKETCH_DIR"; then
  echo "ERROR: compile failed." >&2
  exit 1
fi

echo "==> Uploading to $PORT..."
if ! arduino-cli upload -p "$PORT" --fqbn "$FQBN" "$SKETCH_DIR"; then
  echo "ERROR: upload failed (port $PORT, board $BOARD)." >&2
  echo "       If this is a Nano clone with the old bootloader, try:" >&2
  echo "       $0 nano-old $PORT" >&2
  exit 1
fi

echo "==> Done. Flashed $BOARD firmware on $PORT."
