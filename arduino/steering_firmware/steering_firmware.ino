
#include <Servo.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>

static const uint8_t SERVO_PIN = 2;

static const uint8_t RPWM_PIN = 5;
static const uint8_t LPWM_PIN = 6;
static const uint8_t REN_PIN  = 7;
static const uint8_t LEN_PIN  = 8;

static const float DEG2US   = 24.0f;
static const int   US_CENTER = 1500;

static const int   US_MIN    = 850;
static const int   US_MAX    = 2150;

static const float         DEF_LIM        = 25.0f;
static const float         TRIM_CLAMP     = 4.0f;

static const unsigned long DEF_WD_MS      = 1000UL;
static const unsigned long BOOT_SETTLE_MS = 300UL;

Servo steer;

float curDeg  = 0.0f;
float trimDeg = 0.0f;
float limL    = DEF_LIM;
float limR    = DEF_LIM;
float drivePct = 0.0f;

unsigned long wdMs        = DEF_WD_MS;
unsigned long lastCmdMs   = 0;
bool          wdTriggered = false;

bool          booted = false;
unsigned long bootMs = 0;

static const uint8_t BUF_SZ = 48;
char    buf[BUF_SZ];
uint8_t bufLen = 0;
bool    bufOverflow = false;

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

void applyCurrent() {
  steer.writeMicroseconds(angleToUs(curDeg));
}

void centerServo() {
  curDeg = 0.0f;
  applyCurrent();
}

void driveApply(float pct) {
  drivePct = pct;
  uint8_t duty = (uint8_t)(fabs(pct) * 2.55f + 0.5f);
  if (pct > 0)      { analogWrite(LPWM_PIN, 0);    analogWrite(RPWM_PIN, duty); }
  else if (pct < 0) { analogWrite(RPWM_PIN, 0);    analogWrite(LPWM_PIN, duty); }
  else              { analogWrite(RPWM_PIN, 0);    analogWrite(LPWM_PIN, 0); }
}

void driveStop() {
  driveApply(0.0f);
}

void feedWatchdog() {
  lastCmdMs   = millis();
  wdTriggered = false;
}

bool parseFinite(const char *s, float *out) {
  char *end;
  double v = strtod(s, &end);
  if (end == s || *end != '\0') return false;
  if (isnan(v) || isinf(v)) return false;
  *out = (float)v;
  return true;
}

void handleLine(char *line) {
  char *verb = strtok(line, " \t");
  if (verb == NULL) return;
  for (char *p = verb; *p; ++p) *p = (char)toupper((unsigned char)*p);

  feedWatchdog();

  if (strcmp(verb, "PING") == 0) {
    Serial.println("PONG steering-fw v1");

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

  } else if (strcmp(verb, "GET") == 0) {
    Serial.print("STATE angle="); Serial.print(curDeg, 1);
    Serial.print(" trim=");       Serial.print(trimDeg, 1);
    Serial.print(" lim=");        Serial.print(limL, 1);
    Serial.print('/');            Serial.print(limR, 1);
    Serial.print(" wd=");         Serial.print(wdMs);
    Serial.print(" drive=");      Serial.print(drivePct, 0);
    Serial.print(" up=");         Serial.println(millis());

  } else if (strcmp(verb, "M") == 0) {
    char *a = strtok(NULL, " \t");
    if (a == NULL) { Serial.println("ERR M needs <pct>"); return; }
    float m;
    if (!parseFinite(a, &m)) { Serial.println("ERR M bad number"); return; }
    if (m >  100.0f) m =  100.0f;
    if (m < -100.0f) m = -100.0f;
    driveApply(m);
    Serial.print("OK M "); Serial.println(drivePct, 0);

  } else {
    Serial.println("ERR unknown cmd");
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(RPWM_PIN, OUTPUT);
  pinMode(LPWM_PIN, OUTPUT);
  pinMode(REN_PIN, OUTPUT);
  pinMode(LEN_PIN, OUTPUT);
  digitalWrite(REN_PIN, HIGH);
  digitalWrite(LEN_PIN, HIGH);
  driveStop();

  steer.attach(SERVO_PIN, 500, 2500);
  centerServo();
  bootMs    = millis();
  lastCmdMs = bootMs;
}

void loop() {
  unsigned long now = millis();

  if (!booted && (now - bootMs) >= BOOT_SETTLE_MS) {
    Serial.println("READY steering-fw v1");
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

  if (wdMs > 0 && !wdTriggered && (now - lastCmdMs) >= wdMs) {
    driveStop();
    centerServo();
    wdTriggered = true;
    Serial.println("WD center stop");
  }
}
