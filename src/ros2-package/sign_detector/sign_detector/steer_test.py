"""
Standalone steering test CLI (NO ROS). Talks to the steering Arduino through
arduino_link.ArduinoLink. Run it inside the Pi's Docker container or on the host:

    python3 steer_test.py [--port PORT] center
    python3 steer_test.py [--port PORT] sweep
    python3 steer_test.py [--port PORT] angle <deg>
    python3 steer_test.py [--port PORT] raw <us>
    python3 steer_test.py [--port PORT] state

PORT defaults to "auto" (scan /dev/serial/by-id/*, skip the lidar). Positive
angles steer RIGHT. 'sweep' runs C, +15, -15, C with 0.8 s pauses, printing each
firmware reply.
"""
import argparse
import sys
import time

try:
    from .arduino_link import ArduinoLink
except ImportError:
    from arduino_link import ArduinoLink


def build_parser():
    p = argparse.ArgumentParser(description="Steering Arduino test CLI (no ROS).")
    p.add_argument("--port", default="auto", help="serial port or 'auto' (default)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("center", help="center the steering (C)")
    sub.add_parser("sweep", help="C, +15, -15, C with 0.8 s pauses")
    ap = sub.add_parser("angle", help="steer to signed degrees (+ = right)")
    ap.add_argument("deg", type=float)
    rp = sub.add_parser("raw", help="raw servo pulse in microseconds (U)")
    rp.add_argument("us", type=int)
    sub.add_parser("state", help="query STATE (GET)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        with ArduinoLink(port=args.port) as link:
            if link.pong:
                print(link.pong)
            if args.cmd == "center":
                print(link.center())
            elif args.cmd == "angle":
                print(link.steer(args.deg))
            elif args.cmd == "raw":
                print(link.raw(args.us))
            elif args.cmd == "state":
                print(link.state())
            elif args.cmd == "sweep":
                for step in ("C", "S 15", "S -15", "C"):
                    print(">> {}".format(step))
                    print(link.send(step))
                    time.sleep(0.8)
    except Exception as exc:
        print("steer_test error: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
