#!/usr/bin/env python3
"""tools/check_timing.py - read the Pico's timing lines and say PASS or FAIL for task M1.

The sketch prints one line per second, for example:
    t=3s n=1000 avg=1000.0 min=999 max=1001 out=0
Pass criterion: every period within 1000 +/- 20 us over at least 10 s.

Live from the Pico (needs pyserial:  pip install pyserial):
    python tools/check_timing.py                      # auto-detects a Pico
    python tools/check_timing.py --port COM5          # Windows
    python tools/check_timing.py --port /dev/ttyACM0  # Linux
From a saved log (copy the lines out of the serial monitor into a text file):
    python tools/check_timing.py --file timing.txt
"""
import argparse
import re
import sys

LINE = re.compile(r"t=(\d+)s n=(\d+) avg=([\d.]+) min=(\d+) max=(\d+) out=(\d+)")
NOMINAL_US, TOL_US, MIN_SECONDS = 1000, 20, 10


def analyse(lines):
    rows = [tuple(m.groups()) for m in map(LINE.search, lines) if m]
    if not rows:
        print("No timing lines found. Is the sketch running, and is the Serial Monitor closed?")
        return 2
    mins = [int(r[3]) for r in rows]
    maxs = [int(r[4]) for r in rows]
    outs = sum(int(r[5]) for r in rows)
    seconds = len(rows)
    lo, hi = min(mins), max(maxs)
    print(f"{seconds} one-second windows: period min {lo} us, max {hi} us, "
          f"{outs} periods outside {NOMINAL_US}+/-{TOL_US} us")
    if seconds < MIN_SECONDS:
        print(f"Need at least {MIN_SECONDS} s of data for a verdict.")
        return 2
    ok = lo >= NOMINAL_US - TOL_US and hi <= NOMINAL_US + TOL_US
    print("PASS" if ok else "FAIL: jitter above 50 us usually means Serial printing inside the loop")
    return 0 if ok else 1


def open_port(name):
    try:
        import serial
        from serial.tools import list_ports
    except ImportError:
        sys.exit("pyserial is missing: pip install pyserial")
    if not name:
        for p in list_ports.comports():
            if p.vid == 0x2E8A or "pico" in (p.description or "").lower():
                name = p.device
                break
    if not name:
        sys.exit("No Pico found. Pass --port, or check that the cable carries data.")
    print(f"reading {name} ...")
    return serial.Serial(name, 115200, timeout=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--file")
    ap.add_argument("--seconds", type=int, default=12, help="how long to listen (live mode)")
    args = ap.parse_args()
    if args.file:
        with open(args.file, encoding="utf-8", errors="replace") as f:
            return analyse(f.read().splitlines())
    import time
    port = open_port(args.port)
    lines, end = [], time.time() + args.seconds
    while time.time() < end:
        raw = port.readline().decode("utf-8", errors="replace").strip()
        if raw:
            print(raw)
            lines.append(raw)
    return analyse(lines)


if __name__ == "__main__":
    sys.exit(main())
