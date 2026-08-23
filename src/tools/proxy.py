"""TCP proxy: localhost:8090 -> Pi 8080 (over Tailscale).
Lets the Steam Deck or any browser reach the Pi dashboard via localhost."""
import socket, threading

LADDR = ("0.0.0.0", 8090)
RADDR = ("raspberrypi", 8080)


def pipe(a, b):
    try:
        while True:
            d = a.recv(65536)
            if not d:
                break
            b.sendall(d)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.close()
            except OSError:
                pass


def handle(c):
    try:
        r = socket.create_connection(RADDR, timeout=10)
    except OSError as e:
        c.close(); print("upstream error:", e, flush=True); return
    threading.Thread(target=pipe, args=(c, r), daemon=True).start()
    pipe(r, c)


def main():
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(LADDR)
    s.listen(64)
    print(f"proxy {LADDR[0]}:{LADDR[1]} -> {RADDR[0]}:{RADDR[1]}", flush=True)
    while True:
        c, _ = s.accept()
        threading.Thread(target=handle, args=(c,), daemon=True).start()


if __name__ == "__main__":
    main()
