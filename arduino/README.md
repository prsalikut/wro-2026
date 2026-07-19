# WRO 2026 -- Steering Firmware

ATmega328P steering controller for the Future Engineers robot car. One MG995
servo drives a rack-pivot steering linkage on the front wheels. The Raspberry
Pi sends ASCII commands over USB serial; this firmware turns them into servo
pulses and fails safe if the Pi goes quiet.

The **same sketch runs on the Arduino Uno and the Arduino Nano** (both
ATmega328P). Only the upload target (FQBN) differs -- `flash.sh` handles that.

## Wiring recap

| Signal            | Connection                                                        |
|-------------------|-------------------------------------------------------------------|
| Servo signal      | Arduino **D2** -> MG995 signal (orange)                            |
| Servo power (+)   | **External 5-6 V supply** -> MG995 V+ (red). **NOT** the Arduino 5V |
| Servo power (-)   | External supply ground -> MG995 ground (brown)                    |
| Common ground     | External supply ground **must** be tied to Arduino GND            |
| Arduino           | USB to Raspberry Pi (power + serial)                              |

> **Why external power:** an MG995 stalls at a couple of amps and *will* brown
> out the Arduino's onboard 5V regulator, resetting the board mid-run. Drive the
> servo from a dedicated 5-6 V supply and share grounds. The Arduino only emits
> the control pulse on D2.

### Drive motor (BTS7960 + Johnson 500, 3S) -- wiring v2, 2026-07-11

> **PIN WARNING:** `Servo.h` occupies **Timer1**, which kills `analogWrite` on
> **D9/D10** -- never put motor PWM there. **D5/D6 (Timer0)** PWM coexists with
> the servo and with `millis()`.

| Signal          | Connection                                                  |
|-----------------|-------------------------------------------------------------|
| RPWM (forward)  | Arduino **D5** -> BTS7960 RPWM (Timer0 OC0B)                 |
| LPWM (reverse)  | Arduino **D6** -> BTS7960 LPWM (Timer0 OC0A)                 |
| R_EN            | Arduino **D7** -> BTS7960 R_EN (digital kill switch)         |
| L_EN            | Arduino **D8** -> BTS7960 L_EN (digital kill switch)         |
| Logic power     | Arduino 5V -> BTS7960 VCC; grounds on the common bus         |
| Motor power     | 3S (+) -> fuse -> switch -> B+; 3S (-) -> B- + common ground |
| Servo           | signal -> **D2**; power from the EXTERNAL 5-6 V PSU whose    |
|                 | ground joins the common bus (never the Nano 5V pin)          |

R_EN/L_EN are raised once at boot and **stay HIGH** (per-command toggling kept
this board from driving). `M 0` (and the watchdog, and firmware boot) sets both
PWM inputs low with the bridge enabled = **active brake**. NOTE: the Johnson 500
needs roughly **>=40-50% duty to start** -- the validated bench value is ~59%
(150/255); low duties ACK but do not move the motor.

## Serial protocol

- **115200 baud, 8N1, newline-terminated ASCII.** CR, LF and CRLF all accepted.
- Verb is **case-insensitive**; args are space-separated; one command per line.
- **Sign convention: `+deg` = steer RIGHT, `-deg` = steer LEFT, `0` = straight.**

| Command             | Reply                                          | Notes |
|---------------------|------------------------------------------------|-------|
| `PING`              | `PONG steering-fw v1`                           | liveness check |
| `S <deg>`           | `OK S <applied_deg>`                             | signed deg from center (float ok), clamped to `LIM` |
| `U <us>`            | `OK U <us>`                                      | raw pulse, clamped `[850,2150]`; **calibration only** |
| `C`                 | `OK C`                                           | center (deg 0 + trim) |
| `LIM <left> <right>`| `OK LIM <l> <r>`                                 | positive magnitudes = max LEFT / max RIGHT; default `25 25`, args capped at 25 |
| `TRIM <deg>`        | `OK TRIM <deg>`                                  | center offset, clamp +/-4, re-applies current angle |
| `WD <ms>`           | `OK WD <ms>`                                     | watchdog timeout; `0` disables; default `1000`, capped at `60000` |
| `GET`               | `STATE angle=.. trim=.. lim=../.. wd=.. up=..`   | full state; `up` = uptime ms |
| `M <pct>`           | `OK M <applied_pct>`                             | drive duty, signed % clamped [-100,100]; + = forward, `0` = active-brake stop |
| *(anything else)*   | `ERR unknown cmd`                                | |

Unsolicited lines the Pi may see at any time:

| Line                    | Meaning |
|-------------------------|---------|
| `READY steering-fw v1`  | printed once, ~300 ms after boot |
| `WD center stop`        | watchdog tripped -> motor stopped + steering centered (once per trip) |
| `ERR line too long`     | an over-long line was discarded |

Example session:

```
> PING
PONG steering-fw v1
> S -12.5
OK S -12.5
> S 90
OK S 25.0        # clamped to the right limit
> GET
STATE angle=25.0 trim=0.0 lim=25.0/25.0 wd=1000 up=8423
> C
OK C
```

### Degrees -> microseconds

```
us = 1500 + trim_us + deg * DEG2US        DEG2US = 24.0 us/deg  (calibrated)
```

**Calibrated 2026-07-11** on the assembled rack-pivot linkage: a raw `U` sweep
was smooth (no binding) through **850-2150 us**, so full command authority
`S +/-25` (= default `LIM`) is mapped onto **1500 +/- 600 us** -> `600/25 = 24.0`.
The output is always hard-clamped to `[850, 2150]` us -- the verified envelope
itself, since with 24 us/deg the clamp is actually reachable. `DEG2US` lives at
the top of the sketch; re-run the `U` sweep and retune if the linkage changes.

## Flashing (on the Raspberry Pi)

Needs `arduino-cli` with the `arduino:avr` core installed:

```bash
arduino-cli core install arduino:avr
```

Then from the `arduino/` directory:

```bash
./flash.sh uno            # Arduino Uno,  auto-detect port
./flash.sh nano           # Arduino Nano (new bootloader), auto-detect port
./flash.sh nano-old       # Arduino Nano clone (OLD bootloader) -- try this if
                          # 'nano' upload fails with sync/timeout errors
./flash.sh uno /dev/ttyACM0   # force a specific port
```

FQBNs used: `arduino:avr:uno`, `arduino:avr:nano`,
`arduino:avr:nano:cpu=atmega328old`.

### Port notes (Uno vs Nano vs the lidar)

- **Uno** enumerates as **`/dev/ttyACM*`** (genuine USB CDC).
- **Nano CH340 clones** enumerate as **`/dev/ttyUSB*`** (USB-serial bridge).
- The **YDLIDAR X2** is a CP210x (Silicon Labs) device, also on `/dev/ttyUSB*`
  (typically `/dev/ttyUSB0`).

Because the Nano and the lidar can both land on `/dev/ttyUSB*`, `flash.sh`
detects through `/dev/serial/by-id/` and **always excludes anything matching
`CP210` / `Silicon_Labs`** -- it will never flash the lidar. It prefers
`Arduino`/`ACM` names for the Uno and `USB_Serial`/`CH340`/`1a86`/`wch` names for
the Nano, and prints the port it chose. If auto-detect guesses wrong, pass the
port explicitly.

### Old-bootloader note

Many Nano clones ship with the **old** bootloader and only program at 57600
baud. If `./flash.sh nano` fails during upload (avrdude sync / `not in sync`
timeouts), use `./flash.sh nano-old`.

## Safety notes

- **Find the real end-stops before driving.** With the linkage assembled, sweep
  the raw pulse with `U` (e.g. `U 1300`, `U 1700`) and watch the rack. Note the
  pulses just before it binds, then set software limits with `LIM` so `S` can
  never command past them. **Never force the rack against its mechanical stops**
  -- you will strip the servo gears or bend the linkage.
- **Limits are steering deflection, not raw pulse.** `LIM 25 25` means +/-25 deg.
  Start conservative and open up as you confirm clearance.
- **Watchdog is the Pi-crash failsafe.** If no command arrives for `WD` ms
  (default 1000), the drive motor stops, the servo re-centers once and it prints
  `WD center stop`. Any new
  command re-arms it. Set `WD 0` to disable (e.g. for bench calibration), but
  keep it on when driving.
- **Trim, don't fight geometry.** Use `TRIM` (max +/-4 deg = ~96 us) to null out a small
  mechanical center offset; large offsets mean the linkage needs adjusting.
- Servo power is external -- verify the common ground before powering on.
