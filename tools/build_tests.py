#!/usr/bin/env python3
"""tools/build_tests.py - compile and run every tests/test_*.cpp with g++ on the PC.

Each test is built together with all core/*.cpp files (none exist yet in M1; the header-only
build_check.hpp is enough). Warnings are errors, and -Wdouble-promotion catches accidental
double maths, which is slow on the RP2040.

    python tools/build_tests.py            # build and run all tests
    python tools/build_tests.py test_build # only tests whose name contains "test_build"
"""
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "tests", "build")
FLAGS = ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Wdouble-promotion", "-Werror"]


def find_compiler():
    for name in ("g++", "clang++", "c++"):
        path = shutil.which(name)
        if path:
            return path
    return None


def main() -> int:
    cxx = find_compiler()
    if not cxx:
        print("No C++ compiler found. Install g++ (see the 'Downloads' list in the runbook).")
        return 2
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    tests = sorted(p for p in glob.glob(os.path.join(ROOT, "tests", "test_*.cpp")) if pattern in p)
    if not tests:
        print("No tests found in tests/")
        return 2
    core_sources = sorted(glob.glob(os.path.join(ROOT, "core", "*.cpp")))
    os.makedirs(BUILD, exist_ok=True)
    failed = 0
    for src in tests:
        name = os.path.splitext(os.path.basename(src))[0]
        exe = os.path.join(BUILD, name + (".exe" if os.name == "nt" else ""))
        cmd = [cxx, *FLAGS, src, *core_sources, "-o", exe]
        print(f"[build] {name}")
        if subprocess.run(cmd).returncode != 0:
            print(f"[FAIL ] {name}: compile error")
            failed += 1
            continue
        if subprocess.run([exe]).returncode != 0:
            print(f"[FAIL ] {name}: test failed")
            failed += 1
        else:
            print(f"[ ok  ] {name}")
    print("ALL TESTS PASSED" if failed == 0 else f"{failed} TEST(S) FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
