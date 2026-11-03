// hal/pico_main/pico_main.ino
// M1 - toolchain check: blink + a 1 kHz timed loop that reports its own period once per second.
//
// Arduino IDE needs the sketch folder and the .ino file to have the same name,
// which is why the path is hal/pico_main/pico_main.ino.
//
// Pass (task M1): loop period is 1000 us +/- 20 us over 10 s, and core/ builds on the Pico.
//
// Later tasks put the controller call where the "LATER" comment is.
#include <Arduino.h>

// core/ headers are found through the include path, not through a "../../" path (the Arduino
// toolchain copies the sketch to a temporary folder, so relative paths out of it break):
//   Arduino IDE / arduino-cli : core/ is added as a library (see the runbook, task M1)
//   PlatformIO                : platformio.ini adds `-I core`
#include "build_check.hpp"

constexpr uint32_t kPeriodUs = 1000;      // 1 kHz loop
constexpr uint32_t kTolUs = 20;           // pass band: 1000 +/- 20 us
constexpr uint32_t kReportLoops = 1000;   // print once per second (never faster)
constexpr uint32_t kBlinkLoops = 500;     // toggle the LED every 0.5 s
constexpr uint32_t kTestWindows = 10;     // verdict every 10 s

static uint32_t g_next_us = 0;            // when the next iteration is due
static uint32_t g_last_us = 0;            // start time of the previous iteration
static bool g_have_last = false;
static uint32_t g_total_loops = 0;
static bool g_led_on = false;

// statistics of the current 1 s window
static uint32_t g_n = 0, g_min = 0xFFFFFFFFu, g_max = 0, g_out = 0;
static uint64_t g_sum = 0;
// statistics of the current 10 s test
static uint32_t g_windows = 0, g_total_windows = 0;
static uint32_t g_tmin = 0xFFFFFFFFu, g_tmax = 0;

static void print_line(const char *buf, int len) {
  // Never block the control loop: skip the print if the USB buffer is too full.
  if (len > 0 && Serial.availableForWrite() >= len) {
    Serial.write(reinterpret_cast<const uint8_t *>(buf), static_cast<size_t>(len));
  }
}

static void report_window() {
  char buf[112];
  const uint32_t avg10 = static_cast<uint32_t>((g_sum * 10ULL) / g_n);  // average in tenths of us
  g_total_windows++;
  int len = snprintf(buf, sizeof buf, "t=%lus n=%lu avg=%lu.%lu min=%lu max=%lu out=%lu\n",
                     static_cast<unsigned long>(g_total_windows), static_cast<unsigned long>(g_n),
                     static_cast<unsigned long>(avg10 / 10), static_cast<unsigned long>(avg10 % 10),
                     static_cast<unsigned long>(g_min), static_cast<unsigned long>(g_max),
                     static_cast<unsigned long>(g_out));
  print_line(buf, len);

  if (g_min < g_tmin) g_tmin = g_min;
  if (g_max > g_tmax) g_tmax = g_max;
  if (++g_windows >= kTestWindows) {
    const bool pass = (g_tmin >= kPeriodUs - kTolUs) && (g_tmax <= kPeriodUs + kTolUs);
    len = snprintf(buf, sizeof buf, "RESULT 10s: %s (period %lu..%lu us, limit %lu..%lu)\n",
                   pass ? "PASS" : "FAIL", static_cast<unsigned long>(g_tmin),
                   static_cast<unsigned long>(g_tmax),
                   static_cast<unsigned long>(kPeriodUs - kTolUs),
                   static_cast<unsigned long>(kPeriodUs + kTolUs));
    print_line(buf, len);
    g_windows = 0;
    g_tmin = 0xFFFFFFFFu;
    g_tmax = 0;
  }
  g_n = 0; g_sum = 0; g_out = 0; g_min = 0xFFFFFFFFu; g_max = 0;
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
  const uint32_t t0 = millis();
  while (!Serial && (millis() - t0) < 3000) { delay(10); }   // give the serial monitor time to attach
  Serial.printf("pico-quad M1: 1 kHz timing test | core build_check = %ld\n",
                static_cast<long>(core::build_check()));
  Serial.println("columns: t=seconds n=periods avg/min/max in us, out=periods outside 1000+/-20 us");
  g_next_us = micros() + kPeriodUs;
}

void loop() {
  // 1) Wait until this iteration is due. Busy-waiting gives the lowest jitter.
  while (static_cast<int32_t>(micros() - g_next_us) < 0) { }
  const uint32_t now = micros();
  g_next_us += kPeriodUs;
  // If the loop body ever took more than a whole period, resynchronise instead of bursting.
  if (static_cast<int32_t>(now - g_next_us) >= 0) g_next_us = now + kPeriodUs;

  // 2) Measure the time since the previous iteration started.
  if (g_have_last) {
    const uint32_t period = now - g_last_us;
    g_sum += period;
    g_n++;
    if (period < g_min) g_min = period;
    if (period > g_max) g_max = period;
    if (period + kTolUs < kPeriodUs || period > kPeriodUs + kTolUs) g_out++;
  }
  g_last_us = now;
  g_have_last = true;

  // LATER: the controller update (core/) goes here, between the measurement and the report.

  // 3) Heartbeat LED. Note: on a Pico W the LED sits behind the Wi-Fi chip and the toggle is slower.
  if (++g_total_loops % kBlinkLoops == 0) {
    g_led_on = !g_led_on;
    digitalWrite(LED_BUILTIN, g_led_on ? HIGH : LOW);
  }

  // 4) Once per second, print the statistics of the last 1000 periods.
  if (g_n >= kReportLoops) report_window();
}
