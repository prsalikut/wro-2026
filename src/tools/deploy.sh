#!/usr/bin/env bash
# Push the repo's ROS package to the Pi, rebuild it in the container, and check
# the sensors. Every step is skippable so a failed run can be resumed rather
# than repeated from the top.
set -uo pipefail

PI="${PI_HOST:-pi@100.115.88.108}"
DEST="${PI_DEST:-/home/pi/sign_detector}"
CONTAINER="${PI_CONTAINER:-signstack}"
SECONDS_CHECK="${CHECK_SECONDS:-10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PKG_DIR="$REPO_DIR/src/ros2-package/sign_detector"

RUN_FLASH=0 RUN_DEPLOY=1 RUN_BUILD=1 RUN_CHECK=1 RUN_LOGS=0

usage() {
  cat >&2 <<EOF
Usage: $0 [options]

  --flash          also compile and upload the Nano firmware first
  --check-only     skip deploy/build, just run the sensor check
  --no-check       deploy and build, but do not run the sensor check
  --logs           print recent container logs at the end
  --host USER@HOST override the Pi (default $PI, or \$PI_HOST)
  -h, --help       this message

Steps run in order: [flash] -> deploy -> build -> check.
EOF
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --flash)      RUN_FLASH=1 ;;
    --check-only) RUN_DEPLOY=0; RUN_BUILD=0 ;;
    --no-check)   RUN_CHECK=0 ;;
    --logs)       RUN_LOGS=1 ;;
    --host)       shift; [ $# -gt 0 ] || usage; PI="$1" ;;
    -h|--help)    usage ;;
    *) echo "unknown option: $1" >&2; usage ;;
  esac
  shift
done

say()  { printf '\n=== %s ===\n' "$*"; }
fail() { printf '\nFAILED: %s\n' "$*" >&2; exit 1; }

SSH="ssh -o ConnectTimeout=10 -o BatchMode=yes"

say "reaching $PI"
if ! $SSH "$PI" true 2>/dev/null; then
  cat >&2 <<EOF
Cannot reach $PI over ssh.

  - is the Pi powered and on the network?
  - if it is on Tailscale, is this machine logged into the same tailnet?
  - try:  ssh $PI
  - override with:  $0 --host user@address
EOF
  exit 1
fi
echo "ok"

if [ "$RUN_FLASH" = 1 ]; then
  say "flashing the Nano"
  # The Nano hangs off this workstation, not the Pi, so flash.sh runs locally.
  "$REPO_DIR/src/arduino/flash.sh" nano \
    || fail "firmware upload; try 'nano-old' for a clone bootloader"
fi

if [ "$RUN_DEPLOY" = 1 ]; then
  say "copying the package to $PI:$DEST"
  [ -d "$PKG_DIR" ] || fail "package not found at $PKG_DIR"
  $SSH "$PI" "mkdir -p '$DEST'" || fail "could not create $DEST"
  # -r on the contents, so DEST mirrors the package rather than nesting it.
  scp -q -o ConnectTimeout=10 -r "$PKG_DIR"/* "$PI:$DEST/" \
    || fail "scp to $PI:$DEST"
  # tools/ lives outside the package but the check script is run from inside.
  scp -q -o ConnectTimeout=10 -r "$REPO_DIR/src/tools" "$PI:$DEST/" \
    || fail "scp tools to $PI:$DEST"
  echo "ok"
fi

if [ "$RUN_BUILD" = 1 ]; then
  say "building in the $CONTAINER container"
  $SSH "$PI" "docker exec $CONTAINER bash -lc \
    'source /opt/ros/humble/setup.bash && cd /ros2_ws && \
     colcon build --packages-select sign_detector --symlink-install'" \
    || fail "colcon build - run with --logs to see why"
  say "restarting $CONTAINER"
  $SSH "$PI" "docker restart $CONTAINER" || fail "docker restart"
  # The graph needs a moment before anything is publishing.
  echo "waiting for nodes to come up"
  sleep 12
fi

RC=0
if [ "$RUN_CHECK" = 1 ]; then
  say "checking sensors (${SECONDS_CHECK}s)"
  $SSH "$PI" "docker exec $CONTAINER bash -lc \
    'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && \
     python3 /ros2_ws/src/sign_detector/tools/check_all.py --seconds $SECONDS_CHECK'"
  RC=$?
fi

if [ "$RUN_LOGS" = 1 ]; then
  say "recent container logs"
  $SSH "$PI" "docker logs --tail 60 $CONTAINER 2>&1"
fi

say "done"
[ "$RC" -eq 0 ] || echo "sensor check reported a failure (exit $RC)"
exit "$RC"
