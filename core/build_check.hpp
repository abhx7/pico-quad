#pragma once
// core/build_check.hpp - the smallest possible "core" header.
//
// It exists only to prove that core/ compiles in BOTH places:
//   1) on the PC with g++            (tests/test_build.cpp, run by tools/build_tests.py)
//   2) on the Pico (Arduino-Pico)    (hal/pico_main/pico_main.ino prints build_check() at boot)
//
// Rules for everything in core/: no Arduino.h, no Serial, no millis()/micros(),
// no heap, float not double (the RP2040 has no floating-point unit).
#include <stdint.h>

namespace core {

constexpr int32_t kBuildCheckVersion = 1;

inline int32_t build_check() { return kBuildCheckVersion; }

// A tiny float function so the compiler has something real to chew on.
inline float lerp(float a, float b, float t) { return a + (b - a) * t; }

}  // namespace core
