// tests/test_build.cpp - proves that a core/ header builds with g++ on the PC.
// Build and run:  python tools/build_tests.py
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../core/build_check.hpp"

// Our own CHECK so the test still works if someone compiles with -DNDEBUG (assert would vanish).
#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);               \
      return EXIT_FAILURE;                                                 \
    }                                                                      \
  } while (0)

int main() {
  CHECK(core::build_check() == 1);
  CHECK(fabsf(core::lerp(0.0f, 10.0f, 0.5f) - 5.0f) < 1e-6f);
  CHECK(fabsf(core::lerp(2.0f, 4.0f, 0.0f) - 2.0f) < 1e-6f);
  printf("test_build OK\n");
  return EXIT_SUCCESS;
}
