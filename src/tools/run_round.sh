#!/usr/bin/env bash
# Run one Open Challenge round, tied to the life of this terminal.
#
# Closing the terminal, dropping the SSH session, or Ctrl-C all stop the car.
# That is not automatic: `docker exec` does NOT forward a hangup to the process
# inside the container, so a round started the obvious way keeps driving after
# the operator's terminal has gone.  Here the round runs in the background
# inside the container and a `cat` holds the foreground reading stdin; when the
# terminal goes away stdin hits EOF, and the trap stops the round and zeroes
# the actuators.
#
#   ./run_round.sh            # dry run: steers, never turns the motor
#   ./run_round.sh --drive    # drives at the configured duty
#   ./run_round.sh --drive 35 # ... at 35%
#
# The Arduino's own watchdog is the backstop: the bridge stops sending, and the
# firmware cuts the motor about a second later even if everything above fails.
set -uo pipefail

CONTAINER="${PI_CONTAINER:-signstack}"
DRIVE=0.0
case "${1:-}" in
  --drive) DRIVE="${2:-45.0}" ;;
  --dry|"") DRIVE=0.0 ;;
  *) echo "usage: $0 [--dry | --drive [pct]]" >&2; exit 2 ;;
esac

echo "=== open round: drive_pct=${DRIVE} ==="
[ "$DRIVE" = "0.0" ] && echo "DRY RUN - the servo will steer, the motor will not turn."
echo "Close this terminal or press Ctrl-C to stop the car."
echo

docker exec -i "$CONTAINER" bash -lc '
  source /opt/ros/humble/setup.bash
  source /ros2_ws/install/setup.bash
  CFG=/ros2_ws/install/sign_detector/share/sign_detector/config/params.yaml

  stop() {
    kill -INT "$ROUND" 2>/dev/null
    for _ in $(seq 1 20); do kill -0 "$ROUND" 2>/dev/null || break; sleep 0.1; done
    kill -KILL "$ROUND" 2>/dev/null
    # Belt and braces: the driver zeroes these on the way out, but if it was
    # killed hard nothing else has said stop.
    ros2 topic pub -1 /drive_cmd std_msgs/msg/Float32 "{data: 0.0}" >/dev/null 2>&1
    ros2 topic pub -1 /steering_cmd std_msgs/msg/Float32 "{data: 0.0}" >/dev/null 2>&1
    echo; echo "round stopped, motor zeroed"
  }
  trap stop EXIT INT TERM HUP

  # --params-file MUST come before the -p overrides: ROS 2 applies --ros-args
  # in order, so a params file listed last silently wins. That turned a
  # "dry run" into a live 45% drive command once already.
  ros2 run sign_detector open_round --ros-args \
    --params-file "$CFG" \
    -p require_start:=false -p drive_pct:='"$DRIVE"' 2>&1 &
  ROUND=$!

  # Foreground read on stdin: EOF means the operator terminal has gone.
  cat > /dev/null
'
