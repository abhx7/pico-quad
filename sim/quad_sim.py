#!/usr/bin/env python3
"""quad_sim.py - 6-DOF (six degrees of freedom) micro-quadcopter model for SITL.

SW-1 of the Pico quadcopter runbook. Needs only NumPy.

Frames
    World: NED  (x north, y east, z DOWN). Gravity is +z.
    Body : FRD  (x forward, y right, z DOWN). Thrust acts along body -z (up).
    Roll  +  = right side down.   Pitch + = nose up.   Yaw + = clockwise seen from above.

State (17 numbers): position(3), velocity(3), quaternion w,x,y,z body->world (4),
body rates p,q,r (3), thrust of each motor in newtons (4, first-order motor lag).

Motor numbering and spin directions (seen from ABOVE). Diagonals spin the same way:

    M4 (front-left, CCW)   M2 (front-right, CW)
                 \\          /
                  \\  FRONT /
                   \\      /
                   /      \\
                  /        \\
    M3 (rear-left, CW)     M1 (rear-right, CCW)

A CCW propeller pushes the body round CW (reaction torque), i.e. towards +yaw.

Tests performed by the self-test:
    - Hover: runs four motors at the calculated hover throttle for 5 seconds and checks position, velocity, and attitude drift.
    - Free-fall: turns all motors off for 0.5 seconds and compares downward displacement with 0.5*g*t^2.
    - Torque direction: applies differential motor thrust for yaw, roll, and pitch and verifies the expected body-rate sign.
    - Quaternion normalisation: checks that the integrated attitude quaternion still has unit length.
    - Sensor statistics: collects noisy IMU samples and checks gyro bias/noise plus accelerometer noise and gravity.
    The --sweep option repeats these checks at thrust-to-weight ratios of
    1.5, 2.0, and 3.0.

Run:
    python quad_sim.py              self-test (the SW-1 pass checks)
    python quad_sim.py --table      print the motor table for your notes
    python quad_sim.py --sweep      self-test at T/W 1.5, 2.0 and 3.0 (task M3)
    python quad_sim.py --tw 2.0     self-test with a chosen thrust-to-weight ratio
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass

import numpy as np

G = 9.80665  # m/s^2

# (name, position, forward sign, right sign, spin seen from above, spin sign)
# spin sign +1 = CCW prop = reaction torque along +z (yaw clockwise).
MOTORS = (
    ("M1", "rear-right", -1, +1, "CCW", +1),
    ("M2", "front-right", +1, +1, "CW", -1),
    ("M3", "rear-left", -1, -1, "CW", -1),
    ("M4", "front-left", +1, -1, "CCW", +1),
)


@dataclass
class Params:
    mass: float = 0.050            # kg (total mass)
    ixx: float = 2.0e-5            # kg m^2 (roll)  (placeholder)
    iyy: float = 2.0e-5            # kg m^2 (pitch) (placeholder)
    izz: float = 3.0e-5            # kg m^2 (yaw)   (placeholder)
    
    diagonal: float = 0.095        # m, motor-to-motor diagonal (QX95 frame)
    
    thrust_max_gf: float = 15.0    # gram-force per motor at full throttle (placeholder)
    tau_motor: float = 0.030       # s, motor + prop thrust lag
    km: float = 0.010              # m, yaw reaction torque per newton of thrust (placeholder)
    
    gyro_sigma_dps: float = 0.5    # deg/s, white noise per sample
    gyro_bias_max_dps: float = 1.0  # deg/s, constant bias drawn uniformly in +-this per axis
    accel_sigma: float = 0.1       # m/s^2, white noise per sample
    
    dt: float = 0.001              # s, 1 kHz

    @property
    def arm(self) -> float:
        """Per-axis lever arm: half the diagonal divided by sqrt(2) (47.5 mm / sqrt 2)."""
        return 0.5 * self.diagonal / math.sqrt(2.0)

    @property
    def thrust_max_n(self) -> float:
        """Return the maximum thrust of one motor in newtons."""
        return self.thrust_max_gf * 1e-3 * G

    @property
    def hover_thrust_n(self) -> float:
        """Return the per-motor thrust required to balance the vehicle's weight."""
        return self.mass * G / 4.0

    @property
    def hover_throttle(self) -> float:
        """Return the throttle fraction needed for steady hover."""
        return self.hover_thrust_n / self.thrust_max_n

    @property
    def thrust_to_weight(self) -> float:
        """Return the vehicle's total maximum thrust divided by its weight."""
        return 4.0 * self.thrust_max_n / (self.mass * G)

    @classmethod
    def from_thrust_to_weight(cls, tw: float, **kw) -> "Params":
        """Build parameters whose motor capacity matches the requested thrust ratio."""
        p = cls(**kw)
        p.thrust_max_gf = tw * p.mass * 1000.0 / 4.0
        return p


def quat_mult(a, b):
    """Multiply two scalar-first quaternions and return their product."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_to_dcm(q):
    """Convert a body-to-world scalar-first quaternion to a direction cosine matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def euler_to_quat(roll, pitch, yaw):
    """Convert roll, pitch, and yaw Euler angles in radians to a quaternion."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def quat_to_euler(q):
    """Convert a scalar-first quaternion to roll, pitch, and yaw in radians."""
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


class Quad:
    """Rigid-body quadcopter stepped with RK4 at params.dt, commands held for each step."""

    def __init__(self, params: Params | None = None, seed: int = 0):
        """Initialise the vehicle model, deterministic noise source, and zero-time state."""
        self.p = params or Params()
        self.rng = np.random.default_rng(seed)
        a = self.p.arm
        self.xs = np.array([m[2] * a for m in MOTORS])      # forward positions (m)
        self.ys = np.array([m[3] * a for m in MOTORS])      # right positions (m)
        self.spin = np.array([m[5] for m in MOTORS], float)
        self.I = np.diag([self.p.ixx, self.p.iyy, self.p.izz])
        self.Iinv = np.linalg.inv(self.I)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, pos=(0, 0, 0), vel=(0, 0, 0), euler_deg=(0, 0, 0),
              rates_dps=(0, 0, 0), motors="hover"):
        """Set the vehicle state and fixed gyro bias; motors start at hover or zero thrust."""
        q = euler_to_quat(*np.radians(euler_deg))
        T0 = self.p.hover_thrust_n if motors == "hover" else 0.0
        self.s = np.concatenate([
            np.asarray(pos, float), np.asarray(vel, float), q,
            np.radians(np.asarray(rates_dps, float)), np.full(4, T0)])
        b = self.p.gyro_bias_max_dps
        self.gyro_bias = np.radians(self.rng.uniform(-b, b, 3))  # rad/s, fixed for the run
        self.t = 0.0

    @property
    def pos(self):
        """Return the vehicle position in world NED coordinates."""
        return self.s[0:3]

    @property
    def vel(self):
        """Return the vehicle velocity in world NED coordinates."""
        return self.s[3:6]

    @property
    def quat(self):
        """Return the body-to-world orientation quaternion."""
        return self.s[6:10]

    @property
    def rates(self):
        """Return body angular rates p, q, and r in radians per second."""
        return self.s[10:13]

    @property
    def motor_thrust(self):
        """Return the four motor thrust values in newtons."""
        return self.s[13:17]

    def euler_deg(self):
        """Return the current roll, pitch, and yaw orientation in degrees."""
        return np.degrees(quat_to_euler(self.quat))

    # --------------------------------------------------------------- dynamics
    def _deriv(self, s, T_cmd):
        """Compute the state derivative for the current state and motor commands."""
        p = self.p
        vel, q, w, T = s[3:6], s[6:10], s[10:13], s[13:17]
        F_body = np.array([0.0, 0.0, -T.sum()])
        acc = np.array([0.0, 0.0, G]) + quat_to_dcm(q) @ F_body / p.mass
        qd = 0.5 * quat_mult(q, np.array([0.0, w[0], w[1], w[2]]))
        tau = np.array([
            -(self.ys * T).sum(),                 # right side more thrust -> roll negative
            (self.xs * T).sum(),                  # front more thrust -> pitch up
            (self.spin * p.km * T).sum(),         # CCW props -> yaw clockwise (+z)
        ])
        wd = self.Iinv @ (tau - np.cross(w, self.I @ w))
        Td = (T_cmd - T) / p.tau_motor
        return np.concatenate([vel, acc, qd, wd, Td])

    def step(self, u):
        """Advance the model by one timestep using four M1..M4 throttle commands."""
        u = np.clip(np.asarray(u, float), 0.0, 1.0)
        Tc = u * self.p.thrust_max_n
        h, s = self.p.dt, self.s
        k1 = self._deriv(s, Tc)
        k2 = self._deriv(s + 0.5 * h * k1, Tc)
        k3 = self._deriv(s + 0.5 * h * k2, Tc)
        k4 = self._deriv(s + h * k3, Tc)
        s = s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        s[6:10] /= np.linalg.norm(s[6:10])
        s[13:17] = np.maximum(s[13:17], 0.0)
        self.s, self.t = s, self.t + h

    # ---------------------------------------------------------------- sensors
    def imu(self):
        """Return one noisy body-frame IMU sample as gyro and accelerometer vectors.

        Accelerometer measures specific force: 0 in free-fall, -g on body z at hover.
        """
        p = self.p
        f_body = np.array([0.0, 0.0, -self.motor_thrust.sum()]) / p.mass
        gyro = self.rates + self.gyro_bias + self.rng.normal(
            0.0, math.radians(p.gyro_sigma_dps), 3)
        accel = f_body + self.rng.normal(0.0, p.accel_sigma, 3)
        return gyro, accel


# ====================================================================== tests
def check(name: str, ok: bool, detail: str) -> bool:
    """Print one self-test result and return whether the check passed."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def run_for(quad: Quad, u, seconds: float):
    """Advance a quadcopter by the requested duration using a fixed command."""
    for _ in range(int(round(seconds / quad.p.dt))):
        quad.step(u)


def selftest(params: Params) -> bool:
    """Run hover, free-fall, attitude, quaternion, and sensor-noise checks."""
    p = params
    print(f"mass {p.mass*1000:.0f} g | max thrust {p.thrust_max_gf:.1f} gf/motor | "
          f"T/W {p.thrust_to_weight:.2f} | hover throttle {p.hover_throttle*100:.0f}% | "
          f"lever arm {p.arm*1000:.1f} mm")
    ok = True
    q = Quad(p, seed=1)

    # 1. Hover holds position.
    if p.hover_throttle <= 1.0:
        q.reset(motors="hover")
        run_for(q, np.full(4, p.hover_throttle), 5.0)
        drift, speed = float(np.linalg.norm(q.pos)), float(np.linalg.norm(q.vel))
        tilt = float(np.max(np.abs(q.euler_deg())))
        ok &= check("hover holds position (5 s)", drift < 1e-3 and speed < 1e-3 and tilt < 1e-3,
                    f"drift {drift*1000:.4f} mm, speed {speed*1000:.4f} mm/s, tilt {tilt:.5f} deg")
    else:
        ok &= check("hover holds position (5 s)", False,
                    f"hover needs {p.hover_throttle*100:.0f}% throttle (T/W below 1): cannot lift off")

    # 2. Free-fall for 0.5 s: z = 0.5 g t^2 (NED, so z is positive downwards).
    q.reset(motors="off")
    run_for(q, np.zeros(4), 0.5)
    expected = 0.5 * G * 0.5 ** 2
    drop = float(q.pos[2])
    ok &= check("free-fall 0.5 s", abs(drop - expected) / expected < 0.02,
                f"dropped {drop:.4f} m, expected 0.5*g*t^2 = {expected:.4f} m")

    # 3-5. Torque sign tests from hover: differential throttle for 0.1 s.
    h = min(p.hover_throttle, 1.0)
    d = min(0.05, h, 1.0 - h)
    tests = [
        # name, command pattern (M1..M4), axis index, expected sign, description
        ("yaw command", (+d, -d, -d, +d), 2, +1, "more thrust on CCW props M1,M4 -> yaw clockwise (r > 0)"),
        ("roll command", (+d, +d, -d, -d), 0, -1, "more thrust on right motors M1,M2 -> right side up (p < 0)"),
        ("pitch command", (-d, +d, -d, +d), 1, +1, "more thrust on front motors M2,M4 -> nose up (q > 0)"),
    ]
    for name, pattern, axis, sign, why in tests:
        q.reset(motors="hover")
        run_for(q, h + np.array(pattern), 0.1)
        w = q.rates
        main = w[axis] * sign
        others = np.delete(np.abs(w), axis)
        good = d > 0 and main > 0 and np.all(others < 1e-6 * abs(w[axis]))
        ok &= check(name, bool(good),
                    f"{why}; rates p,q,r = {np.degrees(w).round(3)} deg/s")

    # 6. Quaternion stays normalised.
    err = abs(float(np.linalg.norm(q.quat)) - 1.0)
    ok &= check("quaternion norm", err < 1e-9, f"|norm-1| = {err:.2e}")

    # 7. Sensor noise statistics at hover (true rates are zero, so gyro mean = bias).
    q.reset(motors="hover")
    n = 5000
    samples = np.array([q.imu() for _ in range(n)])       # shape (n, 2, 3)
    gyro = np.degrees(samples[:, 0, :])
    accel = samples[:, 1, :]
    bias = np.degrees(q.gyro_bias)
    mean_err = float(np.max(np.abs(gyro.mean(axis=0) - bias)))
    g_std = gyro.std(axis=0)
    a_std = accel.std(axis=0)
    ok &= check("gyro noise", mean_err < 0.05 and np.all(np.abs(g_std / p.gyro_sigma_dps - 1) < 0.1),
                f"bias {bias.round(2)} deg/s, mean error {mean_err:.3f}, sigma {g_std.round(3)} deg/s")
    ok &= check("accel noise and gravity",
                np.all(np.abs(a_std / p.accel_sigma - 1) < 0.1) and abs(accel[:, 2].mean() + G) < 0.02,
                f"sigma {a_std.round(3)} m/s^2, mean z {accel[:, 2].mean():.3f} (expect {-G:.3f})")
    print("  ALL PASS" if ok else "  SOME CHECKS FAILED")
    return bool(ok)


def print_table(p: Params) -> None:
    """Print motor locations, spin directions, and body reaction directions."""
    print("| Motor | Position | Spin (from above) | Reaction on body | x fwd (mm) | y right (mm) |")
    print("|---|---|---|---|---|---|")
    for name, where, sx, sy, spin, s in MOTORS:
        react = "clockwise (+yaw)" if s > 0 else "counter-clockwise (-yaw)"
        print(f"| {name} | {where} | {spin} | {react} | {sx*p.arm*1000:+.1f} | {sy*p.arm*1000:+.1f} |")
    print("\nDiagonals match: M1+M4 are CCW, M2+M3 are CW.")
    print("+roll = right side down, +pitch = nose up, +yaw = clockwise from above.")


def main() -> int:
    """Parse command-line options, run the selected report or self-test, and exit."""
    ap = argparse.ArgumentParser(description="6-DOF micro-quadcopter model (SW-1)")
    ap.add_argument("--table", action="store_true", help="print motor numbering and spin table")
    ap.add_argument("--sweep", action="store_true", help="self-test at T/W 1.5, 2.0, 3.0 (task M3)")
    ap.add_argument("--tw", type=float, help="self-test at this thrust-to-weight ratio")
    args = ap.parse_args()

    if args.table:
        print_table(Params())
        return 0
    if args.sweep:
        results = []
        for tw in (1.5, 2.0, 3.0):
            print(f"\n== T/W {tw} ==")
            results.append(selftest(Params.from_thrust_to_weight(tw)))
        return 0 if all(results) else 1
    params = Params.from_thrust_to_weight(args.tw) if args.tw else Params()
    return 0 if selftest(params) else 1


if __name__ == "__main__":
    sys.exit(main())
