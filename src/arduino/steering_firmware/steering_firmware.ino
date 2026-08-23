/* WRO 2026 -- steering + drive firmware (v2)
 *
 * ATmega328P (Uno / Nano).  MG995 rack-pivot steering servo + BTS7960 half-bridge
 * driving a Johnson 500 from a 3S LiPo.  The Pi sends newline-terminated ASCII at
 * 115200; this sketch actuates it and fails safe when the Pi (or the operator)
 * goes quiet.
 *
 * WIRING (v2, 2026-07-11 -- see arduino/README.md)
 *   Servo signal  D2   -> MG995 signal.  Servo V+ from an EXTERNAL 5-6V PSU,
 *                        never the Nano 5V pin; PSU ground joins the common bus.
 *   RPWM (fwd)    D5   -> BTS7960 RPWM   (Timer0 OC0B)
 *   LPWM (rev)    D6   -> BTS7960 LPWM   (Timer0 OC0A)
 *   R_EN          D7   -> BTS7960 R_EN
 *   L_EN          D8   -> BTS7960 L_EN
 *   Logic         Arduino 5V -> BTS7960 VCC; ALL grounds on one common bus.
 *   Motor power   3S(+) -> 20A fuse -> switch -> B+ ;  3S(-) -> B- + common bus.
 *
 * PIN WARNING: Servo.h owns Timer1, which kills analogWrite() on D9/D10.
 * Motor PWM must stay on D5/D6 (Timer0), which coexists with Servo and millis().
 *
 * R_EN/L_EN are raised once at boot and STAY HIGH -- toggling them per command
 * previously kept this board from driving at all.  Stop = both PWM low with the
 * bridge still enabled = active brake, which halts harder than coasting.
 *
 * The Johnson 500 needs roughly >=40-50% duty to break away; lower duties ACK
 * but only stall and heat.  Duty->motion mapping is the client's job, not ours.
 */

#include <Servo.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>

/* ---- pins ---------------------------------------------------------------- */
static const uint8_t SERVO_PIN = 2;
static const uint8_t RPWM_PIN  = 5;
static const uint8_t LPWM_PIN  = 6;
static const uint8_t REN_PIN   = 7;
static const uint8_t LEN_PIN   = 8;

/* ---- steering calibration (2026-07-11, rack-pivot linkage) --------------- */
static const float DEG2US    = 24.0f;   /* 600us / 25deg, from a clean U sweep */
static const int   US_CENTER = 1500;
static const int   US_MIN    = 850;     /* verified no-bind envelope */
static const int   US_MAX    = 2150;

static const float DEF_LIM    = 25.0f;
static const float TRIM_CLAMP = 4.0f;

/* ---- failsafe timing ----------------------------------------------------- */
/* WD: any traffic re-arms it. Catches "the Pi died / USB fell out". */
static const unsigned long DEF_WD_MS  = 1000UL;
/* DWD: ONLY an M command re-arms it. Catches "the operator let go / the UI
 * froze / the network stalled" -- cases where chatter (GET, PING) keeps WD
 * happy while nobody is actually asking for throttle any more. A keepalive
 * must never be able to hold the motor on.
 *
 * DEFAULT OFF, deliberately. The autonomy path only refreshes M every 2 s
 * (steering_node.py: d_refresh >= 2.0), so a short deadman enabled by default
 * would chop the motor mid-run. The manual/RC client opts in with `DWD 400`
 * when it engages -- it streams M at 20 Hz -- and `DWD 0` when it releases. */
static const unsigned long DEF_DWD_MS = 0UL;
static const unsigned long BOOT_SETTLE_MS = 300UL;

/* ---- drive shaping ------------------------------------------------------- */
/* Ramp rate for INCREASING |duty|. A gamepad stick snaps instantly; slamming a
 * 3S pack into a stalled Johnson 500 is how you pop a fuse or a BTS7960. */
static const float         SLEW_PCT_PER_MS = 0.5f;  /* 0->100% in ~200 ms */
/* Passing through zero on a direction reversal: brake, dwell, then drive the
 * other way. Straight fwd->rev flips the bridge against the motor's back-EMF. */
static const unsigned long REVERSE_DWELL_MS = 120UL;

Servo steer;

/* ---- state --------------------------------------------------------------- */
float curDeg  = 0.0f;
float trimDeg = 0.0f;
float limL    = DEF_LIM;
float limR    = DEF_LIM;

float driveTarget = 0.0f;   /* what the client asked for, %  */
float driveNow    = 0.0f;   /* what the bridge is getting, % */
unsigned long reverseHoldUntil = 0;

bool estop = false;         /* latched; only ARM clears it */

unsigned long wdMs      = DEF_WD_MS;
unsigned long dwdMs     = DEF_DWD_MS;
unsigned long lastCmdMs   = 0;
unsigned long lastDriveMs = 0;
bool wdTriggered  = false;
bool dwdTriggered = false;

bool          booted = false;
unsigned long bootMs = 0;
unsigned long lastSlewMs = 0;

static const uint8_t BUF_SZ = 48;
char    buf[BUF_SZ];
uint8_t bufLen = 0;
bool    bufOverflow = false;

/* ---- steering ------------------------------------------------------------ */
int angleToUs(float deg) {
  float us = (float)US_CENTER + (trimDeg + deg) * DEG2US;
  if (us < US_MIN) us = US_MIN;
  if (us > US_MAX) us = US_MAX;
  return (int)(us + 0.5f);
}

float clampAngle(float deg) {
  if (deg >  limR) deg =  limR;
  if (deg < -limL) deg = -limL;
  return deg;
}

void applyCurrent() { steer.writeMicroseconds(angleToUs(curDeg)); }

void centerServo() {
  curDeg = 0.0f;
  applyCurrent();
}

/* ---- drive --------------------------------------------------------------- */
/* Push a duty straight at the bridge. Sign: + = forward (RPWM), - = reverse. */
void driveRaw(float pct) {
  driveNow = pct;
  uint8_t duty = (uint8_t)(fabs(pct) * 2.55f + 0.5f);
  if (pct > 0)      { analogWrite(LPWM_PIN, 0);    analogWrite(RPWM_PIN, duty); }
  else if (pct < 0) { analogWrite(RPWM_PIN, 0);    analogWrite(LPWM_PIN, duty); }
  else              { analogWrite(RPWM_PIN, 0);    analogWrite(LPWM_PIN, 0); }
}

/* Immediate, unramped stop = active brake. Every failsafe path uses this: a
 * stop request must never be rate-limited. */
void driveStopNow() {
  driveTarget = 0.0f;
  reverseHoldUntil = 0;
  driveRaw(0.0f);
}

void driveRequest(float pct) {
  if (pct > 100.0f)  pct =  100.0f;
  if (pct < -100.0f) pct = -100.0f;

  if (pct == 0.0f) { driveStopNow(); return; }

  /* Reversal: brake to zero first, then dwell before the other direction. */
  if (driveNow != 0.0f && ((pct > 0) != (driveNow > 0))) {
    driveRaw(0.0f);
    reverseHoldUntil = millis() + REVERSE_DWELL_MS;
  }
  driveTarget = pct;
}

/* Called every loop: walk driveNow toward driveTarget at the slew limit. */
void driveSlew(unsigned long now) {
  if (now < reverseHoldUntil) { driveRaw(0.0f); return; }

  unsigned long dt = now - lastSlewMs;
  if (dt == 0) return;
  lastSlewMs = now;

  if (driveNow == driveTarget) return;

  float step = SLEW_PCT_PER_MS * (float)dt;
  float diff = driveTarget - driveNow;

  /* Only RAMP when the magnitude is growing. Backing off, and anything that
   * reduces current, applies at once. */
  if (fabs(driveTarget) < fabs(driveNow)) { driveRaw(driveTarget); return; }

  if (fabs(diff) <= step) driveRaw(driveTarget);
  else                    driveRaw(driveNow + (diff > 0 ? step : -step));
}

void feedWatchdog() {
  lastCmdMs   = millis();
  wdTriggered = false;
}

void feedDriveDeadman() {
  lastDriveMs  = millis();
  dwdTriggered = false;
}

bool parseFinite(const char *s, float *out) {
  char *end;
  double v = strtod(s, &end);
  if (end == s || *end != '\0') return false;
  if (isnan(v) || isinf(v)) return false;
  *out = (float)v;
  return true;
}


/* ---- ultrasonics ---------------------------------------------------------
 * Four HC-SR04 sharing one trigger on D12, echoes on A0..A3 (all PORTC, so a
 * single pin-change vector serves all four). Everything is interrupt driven:
 * pulseIn() would block up to 30 ms per sensor and starve the drive watchdog
 * and serial parser in loop(), which is the code that stops the car.
 */
static const uint8_t SONAR_TRIG_PIN = 12;
static const uint8_t SONAR_N = 4;
static const unsigned long SONAR_PERIOD_MS = 50;
static const unsigned long SONAR_TIMEOUT_US = 25000UL;  /* ~4.2 m */

volatile unsigned long sonarRise[SONAR_N];
volatile unsigned long sonarWidth[SONAR_N];
volatile uint8_t sonarPrev = 0;
static unsigned long sonarFiredUs = 0;
static unsigned long sonarLastMs = 0;
static bool sonarStream = false;
static unsigned long sonarReportMs = 0;

ISR(PCINT1_vect) {
  uint8_t now = PINC & 0x0F;
  uint8_t changed = now ^ sonarPrev;
  sonarPrev = now;
  if (!changed) return;
  unsigned long t = micros();
  for (uint8_t i = 0; i < SONAR_N; i++) {
    uint8_t m = (uint8_t)(1 << i);
    if (!(changed & m)) continue;
    if (now & m) sonarRise[i] = t;
    else if (sonarRise[i]) { sonarWidth[i] = t - sonarRise[i]; sonarRise[i] = 0; }
  }
}

void sonarBegin() {
  pinMode(SONAR_TRIG_PIN, OUTPUT);
  digitalWrite(SONAR_TRIG_PIN, LOW);
  for (uint8_t i = 0; i < SONAR_N; i++) {
    pinMode(A0 + i, INPUT);
    sonarRise[i] = 0;
    sonarWidth[i] = 0;
  }
  sonarPrev = PINC & 0x0F;
  PCICR |= (1 << PCIE1);
  PCMSK1 |= 0x0F;
}

/* Fire all four at once. They face outward in different directions, so
 * cross-talk is minimal; a stale reading is discarded by the age check. */
void sonarService(unsigned long now) {
  if ((now - sonarLastMs) < SONAR_PERIOD_MS) return;
  sonarLastMs = now;
  sonarFiredUs = micros();
  digitalWrite(SONAR_TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(SONAR_TRIG_PIN, LOW);

  /* Streaming is opt-in: unsolicited lines would otherwise interleave with
   * command replies on a link the bridge also uses for steering and drive. */
  if (sonarStream && (now - sonarReportMs) >= SONAR_PERIOD_MS) {
    sonarReportMs = now;
    sonarReport();
  }
}

/* cm, or -1 when the last ping produced no usable echo */
int sonarCm(uint8_t i) {
  unsigned long w;
  uint8_t sreg = SREG;
  cli();
  w = sonarWidth[i];
  SREG = sreg;
  if (w == 0 || w > SONAR_TIMEOUT_US) return -1;
  return (int)(w / 58UL);
}

void sonarReport() {
  Serial.print("US");
  for (uint8_t i = 0; i < SONAR_N; i++) {
    Serial.print(' ');
    Serial.print(sonarCm(i));
  }
  Serial.println();
}

/* ---- command parser ------------------------------------------------------ */
void handleLine(char *line) {
  char *verb = strtok(line, " \t");
  if (verb == NULL) return;
  for (char *p = verb; *p; ++p) *p = (char)toupper((unsigned char)*p);

  feedWatchdog();

  if (strcmp(verb, "PING") == 0) {
    Serial.println("PONG steering-fw v2");

  } else if (strcmp(verb, "S") == 0) {
    char *a = strtok(NULL, " \t");
    if (a == NULL) { Serial.println("ERR S needs <deg>"); return; }
    float v;
    if (!parseFinite(a, &v)) { Serial.println("ERR S bad number"); return; }
    curDeg = clampAngle(v);
    applyCurrent();
    Serial.print("OK S "); Serial.println(curDeg, 1);

  } else if (strcmp(verb, "U") == 0) {
    char *a = strtok(NULL, " \t");
    if (a == NULL) { Serial.println("ERR U needs <us>"); return; }
    float uf;
    if (!parseFinite(a, &uf)) { Serial.println("ERR U bad number"); return; }
    if (uf < US_MIN) uf = US_MIN;
    if (uf > US_MAX) uf = US_MAX;
    long us = (long)uf;
    steer.writeMicroseconds((int)us);
    Serial.print("OK U "); Serial.println(us);

  } else if (strcmp(verb, "C") == 0) {
    centerServo();
    Serial.println("OK C");

  } else if (strcmp(verb, "LIM") == 0) {
    char *a1 = strtok(NULL, " \t");
    char *a2 = strtok(NULL, " \t");
    if (a1 == NULL || a2 == NULL) { Serial.println("ERR LIM needs <left> <right>"); return; }
    float l, r;
    if (!parseFinite(a1, &l) || !parseFinite(a2, &r)) {
      Serial.println("ERR LIM bad number"); return;
    }
    if (l < 0) l = -l;
    if (r < 0) r = -r;
    if (l > DEF_LIM) l = DEF_LIM;
    if (r > DEF_LIM) r = DEF_LIM;
    limL = l; limR = r;
    curDeg = clampAngle(curDeg);
    applyCurrent();
    Serial.print("OK LIM "); Serial.print(limL, 1);
    Serial.print(' ');       Serial.println(limR, 1);

  } else if (strcmp(verb, "TRIM") == 0) {
    char *a = strtok(NULL, " \t");
    if (a == NULL) { Serial.println("ERR TRIM needs <deg>"); return; }
    float t;
    if (!parseFinite(a, &t)) { Serial.println("ERR TRIM bad number"); return; }
    if (t >  TRIM_CLAMP) t =  TRIM_CLAMP;
    if (t < -TRIM_CLAMP) t = -TRIM_CLAMP;
    trimDeg = t;
    applyCurrent();
    Serial.print("OK TRIM "); Serial.println(trimDeg, 1);

  } else if (strcmp(verb, "WD") == 0) {
    char *a = strtok(NULL, " \t");
    if (a == NULL) { Serial.println("ERR WD needs <ms>"); return; }
    float w;
    if (!parseFinite(a, &w)) { Serial.println("ERR WD bad number"); return; }
    if (w < 0) { Serial.println("ERR WD must be >= 0"); return; }
    if (w > 60000.0f) w = 60000.0f;
    wdMs = (unsigned long)w;
    feedWatchdog();
    Serial.print("OK WD "); Serial.println(wdMs);

  } else if (strcmp(verb, "DWD") == 0) {
    char *a = strtok(NULL, " \t");
    if (a == NULL) { Serial.println("ERR DWD needs <ms>"); return; }
    float w;
    if (!parseFinite(a, &w)) { Serial.println("ERR DWD bad number"); return; }
    if (w < 0) { Serial.println("ERR DWD must be >= 0"); return; }
    if (w > 60000.0f) w = 60000.0f;
    dwdMs = (unsigned long)w;
    feedDriveDeadman();
    Serial.print("OK DWD "); Serial.println(dwdMs);

  } else if (strcmp(verb, "M") == 0) {
    char *a = strtok(NULL, " \t");
    if (a == NULL) { Serial.println("ERR M needs <pct>"); return; }
    float m;
    if (!parseFinite(a, &m)) { Serial.println("ERR M bad number"); return; }
    if (estop) {
      driveStopNow();
      Serial.println("ERR ESTOP latched, send ARM");
      return;
    }
    driveRequest(m);
    feedDriveDeadman();
    Serial.print("OK M "); Serial.println(driveTarget, 0);

  } else if (strcmp(verb, "E") == 0) {
    /* Latched emergency stop. Deliberately NOT self-clearing: it stays until an
     * explicit ARM, so a stuck/idle client can't let the motor creep back. */
    estop = true;
    driveStopNow();
    centerServo();
    Serial.println("OK E estop latched");

  } else if (strcmp(verb, "ARM") == 0) {
    estop = false;
    driveStopNow();          /* re-arm at zero, never at the pre-estop duty */
    feedDriveDeadman();
    Serial.println("OK ARM");

  } else if (strcmp(verb, "US") == 0) {
    sonarReport();

  } else if (strcmp(verb, "USON") == 0) {
    sonarStream = true;
    Serial.println("OK USON");

  } else if (strcmp(verb, "USOFF") == 0) {
    sonarStream = false;
    Serial.println("OK USOFF");

  } else if (strcmp(verb, "GET") == 0) {
    Serial.print("STATE angle="); Serial.print(curDeg, 1);
    Serial.print(" trim=");       Serial.print(trimDeg, 1);
    Serial.print(" lim=");        Serial.print(limL, 1);
    Serial.print('/');            Serial.print(limR, 1);
    Serial.print(" wd=");         Serial.print(wdMs);
    Serial.print(" dwd=");        Serial.print(dwdMs);
    Serial.print(" drive=");      Serial.print(driveNow, 0);
    Serial.print(" target=");     Serial.print(driveTarget, 0);
    Serial.print(" estop=");      Serial.print(estop ? 1 : 0);
    Serial.print(" up=");         Serial.println(millis());

  } else {
    Serial.println("ERR unknown cmd");
  }
}

/* ---- lifecycle ----------------------------------------------------------- */
void setup() {
  Serial.begin(115200);

  /* Motor pins first, and explicitly stopped, before anything else can run.
   * Boot must never produce torque. */
  sonarBegin();
  pinMode(RPWM_PIN, OUTPUT);
  pinMode(LPWM_PIN, OUTPUT);
  pinMode(REN_PIN, OUTPUT);
  pinMode(LEN_PIN, OUTPUT);
  analogWrite(RPWM_PIN, 0);
  analogWrite(LPWM_PIN, 0);
  digitalWrite(REN_PIN, HIGH);   /* raised once, then left alone */
  digitalWrite(LEN_PIN, HIGH);
  driveStopNow();

  steer.attach(SERVO_PIN, 500, 2500);
  centerServo();

  bootMs      = millis();
  lastCmdMs   = bootMs;
  lastDriveMs = bootMs;
  lastSlewMs  = bootMs;
}

void loop() {
  unsigned long now = millis();

  if (!booted && (now - bootMs) >= BOOT_SETTLE_MS) {
    Serial.println("READY steering-fw v2");
    booted = true;
  }

  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (bufOverflow) {
        Serial.println("ERR line too long");
        bufLen = 0; bufOverflow = false;
      } else if (bufLen > 0) {
        buf[bufLen] = '\0';
        handleLine(buf);
        bufLen = 0;
      }
    } else {
      if (bufLen < BUF_SZ - 1) buf[bufLen++] = c;
      else bufOverflow = true;
    }
  }

  /* Drive deadman: no M in dwdMs -> cut throttle, leave steering alone. This is
   * the one that actually protects a human driving by gamepad. */
  if (dwdMs > 0 && !dwdTriggered && driveNow != 0.0f &&
      (now - lastDriveMs) >= dwdMs) {
    driveStopNow();
    dwdTriggered = true;
    Serial.println("DWD drive stop");
  }

  /* Link watchdog: no traffic at all -> stop and re-center. */
  if (wdMs > 0 && !wdTriggered && (now - lastCmdMs) >= wdMs) {
    driveStopNow();
    centerServo();
    wdTriggered = true;
    Serial.println("WD center stop");
  }

  sonarService(now);
  driveSlew(now);
}
