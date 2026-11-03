#!/usr/bin/env python3
"""viz3d.py - 3D replay of the quadcopter model in quad_sim.py (SW-1 visualisation).

Put this file next to quad_sim.py (sim/). Needs NumPy and Matplotlib.

Conventions (same as quad_sim.py)
    World NED (x north, y east, z DOWN), body FRD (x forward, y right, z down),
    quaternion w,x,y,z rotating body -> world.
    The 3D z-axis is flipped so that UP LOOKS UP: height above the start is towards the top.

What you see
    grey arms          from the centre to each motor (rotated by the quaternion)
    M1..M4 labels      red = CCW propeller, blue = CW propeller (seen from above)
    green nose line    points to the front of the quad (body +x)
    orange arrows      thrust of each motor, length proportional to its thrust
                       (one hover-thrust = 0.9 arm lengths)
    black trail        where the centre of the quad has been

Usage as a library
    from viz3d import run_and_log, animate
    log = run_and_log("roll")
    anim = animate(log)                    # keep the returned object alive until the window closes

Usage from the command line
    python viz3d.py roll                   # hover | freefall | yaw | roll | pitch | all
    python viz3d.py roll --speed 0.25      # slow motion
    python viz3d.py all --save out/        # writes viz_<scenario>.gif, no window
    python viz3d.py roll --tw 2.0 --follow # T/W 2.0, camera follows the quad

Scenarios (open loop, no controller, so the quad keeps rotating after the pulse):
    hover     all motors at hover thrust for 3 s: nothing should move
    freefall  motors off for 0.6 s: the quad drops straight down
    yaw       0.1 s pulse: more thrust on CCW props M1,M4 -> turns clockwise seen from above
    roll      0.1 s pulse: more thrust on right motors M1,M2 -> right side goes UP (negative roll)
    pitch     0.1 s pulse: more thrust on front motors M2,M4 -> nose goes up (positive pitch)
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

import quad_sim as qs

LOG_KEYS = ("t", "pos", "quat", "rates", "thrust")

# duration (s); pattern = throttle change for M1..M4 during the pulse;
# frac = pulse size as a fraction of hover thrust (so the torque does not depend on T/W).
SCENARIOS = {
    "hover":    dict(duration=3.0, motors="hover", pattern=None, frac=0.0),
    "freefall": dict(duration=0.6, motors="off",   pattern=None, frac=0.0),
    "yaw":      dict(duration=1.5, motors="hover", pattern=(+1, -1, -1, +1), frac=0.07),
    "roll":     dict(duration=0.8, motors="hover", pattern=(+1, +1, -1, -1), frac=0.010),
    "pitch":    dict(duration=0.8, motors="hover", pattern=(-1, +1, -1, +1), frac=0.010),
}
PULSE_START, PULSE_LEN = 0.1, 0.1  # seconds


# =============================================================== simulation log
def run_and_log(scenario: str = "hover", params: qs.Params | None = None,
                seed: int = 0, duration: float | None = None) -> dict:
    """Run one scenario at 1 kHz and return a log dict.

    Keys (N samples): t (N,), pos (N,3) NED metres, quat (N,4) w,x,y,z body->world,
    rates (N,3) body p,q,r in rad/s, thrust (N,4) newtons for M1..M4.
    Extra keys: u (N,4) throttle commands, scenario, params.
    Unpack the five main arrays with:  t, pos, quat, rates, thrust = (log[k] for k in LOG_KEYS)
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
    spec = SCENARIOS[scenario]
    p = params or qs.Params()
    quad = qs.Quad(p, seed=seed)
    quad.reset(motors=spec["motors"])

    n = int(round((duration or spec["duration"]) / p.dt)) + 1
    h = min(p.hover_throttle, 1.0)
    base = np.zeros(4) if spec["motors"] == "off" else np.full(4, h)
    amp = min(spec["frac"] * h, h, 1.0 - h)

    t = np.zeros(n)
    pos, quat = np.zeros((n, 3)), np.zeros((n, 4))
    rates, thrust, u_log = np.zeros((n, 3)), np.zeros((n, 4)), np.zeros((n, 4))
    for k in range(n):
        t[k], pos[k], quat[k] = quad.t, quad.pos, quad.quat
        rates[k], thrust[k] = quad.rates, quad.motor_thrust
        u = base.copy()
        if spec["pattern"] is not None and PULSE_START <= quad.t < PULSE_START + PULSE_LEN:
            u = u + amp * np.array(spec["pattern"], float)
        u_log[k] = u
        quad.step(u)
    return dict(t=t, pos=pos, quat=quat, rates=rates, thrust=thrust, u=u_log,
                scenario=scenario, params=p)


# ================================================================== animation
class _Scene:
    """Builds the 3D figure once; update(k) redraws sample k of the log."""

    def __init__(self, log: dict, follow: bool = False):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        self.log, self.follow = log, follow
        p = log.get("params") or qs.Params()
        self.p = p
        pos = np.asarray(log["pos"], float)

        # Plot coordinates are NED: x north, y east, z down (axis flipped below).
        ext = float(np.ptp(pos, axis=0).max())
        half = max(0.6 * ext, 0.12)
        arm_true = 0.5 * p.diagonal                              # centre to motor, metres
        self.scale = max(1.0, 0.45 * half / arm_true)            # enlarge the quad when the path is long
        self.arm = arm_true * self.scale
        self.half = half + self.arm
        self.center = 0.5 * (pos.max(axis=0) + pos.min(axis=0))
        self.hover_thrust = p.hover_thrust_n

        # Motor positions in the body frame (x forward, y right), already scaled for drawing.
        self.r_body = np.array([[m[2] * p.arm * self.scale, m[3] * p.arm * self.scale, 0.0]
                                for m in qs.MOTORS])
        self.colors = ["red" if m[4] == "CCW" else "blue" for m in qs.MOTORS]

        self.fig = plt.figure(figsize=(8, 7))
        self.ax = self.fig.add_subplot(111, projection="3d")
        ax = self.ax
        ax.set_xlabel("North x (m)")
        ax.set_ylabel("East y (m)")
        ax.set_zlabel("Down z (m), flipped: up is up")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22, azim=-58)
        self._set_limits(self.center)

        # artists
        self.trail, = ax.plot([], [], [], color="black", lw=1.2, alpha=0.7)
        self.start = ax.plot([pos[0, 0]], [pos[0, 1]], [pos[0, 2]], "k+", ms=10)[0]
        self.arms = [ax.plot([], [], [], color="0.45", lw=3)[0] for _ in range(4)]
        self.motors = [ax.plot([], [], [], "o", color=c, ms=9)[0] for c in self.colors]
        self.labels = [ax.text(0, 0, 0, m[0], color=c, fontsize=11, fontweight="bold")
                       for m, c in zip(qs.MOTORS, self.colors)]
        self.nose, = ax.plot([], [], [], color="green", lw=4)
        self.quiv = None
        self.info = ax.text2D(0.02, 0.97, "", transform=ax.transAxes, va="top",
                              family="monospace", fontsize=9)
        note = f" (quad drawn x{self.scale:.1f})" if self.scale > 1.05 else ""
        ax.set_title(f"Quad model: {log.get('scenario', 'run')}{note}")
        ax.legend(handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="red", ms=9, label="CCW prop"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", ms=9, label="CW prop"),
            Line2D([0], [0], color="green", lw=4, label="front"),
            Line2D([0], [0], color="orange", lw=2, label="thrust"),
            Line2D([0], [0], color="black", lw=1.2, label="trail"),
        ], loc="upper right", fontsize=8)

    def _set_limits(self, c):
        h, ax = self.half, self.ax
        ax.set_xlim(c[0] - h, c[0] + h)
        ax.set_ylim(c[1] - h, c[1] + h)
        ax.set_zlim(c[2] + h, c[2] - h)   # larger z (down) at the bottom: up looks up

    def update(self, k: int):
        log, ax = self.log, self.ax
        c = np.asarray(log["pos"][k], float)
        q = np.asarray(log["quat"][k], float)
        R = qs.quat_to_dcm(q)
        if self.follow:
            self._set_limits(c)

        tips = c + self.r_body @ R.T                    # motor positions in the world frame
        for i in range(4):
            self.arms[i].set_data([c[0], tips[i, 0]], [c[1], tips[i, 1]])
            self.arms[i].set_3d_properties([c[2], tips[i, 2]])
            self.motors[i].set_data([tips[i, 0]], [tips[i, 1]])
            self.motors[i].set_3d_properties([tips[i, 2]])
            self.labels[i].set_position((tips[i, 0], tips[i, 1]))
            self.labels[i].set_3d_properties(tips[i, 2] - 0.02 * self.half, "z")

        nose = c + R @ np.array([self.arm * 1.15, 0.0, 0.0])
        self.nose.set_data([c[0], nose[0]], [c[1], nose[1]])
        self.nose.set_3d_properties([c[2], nose[2]])

        trail = np.asarray(log["pos"][: k + 1], float)
        self.trail.set_data(trail[:, 0], trail[:, 1])
        self.trail.set_3d_properties(trail[:, 2])

        # Thrust arrows point along body -z (up); one hover thrust = 0.9 arm lengths.
        thrust = np.asarray(log["thrust"][k], float)
        up = R @ np.array([0.0, 0.0, -1.0])
        lengths = 0.9 * self.arm * thrust / self.hover_thrust
        if self.quiv is not None:
            self.quiv.remove()
        vec = lengths[:, None] * up[None, :]
        self.quiv = ax.quiver(tips[:, 0], tips[:, 1], tips[:, 2], vec[:, 0], vec[:, 1], vec[:, 2],
                              color="orange", arrow_length_ratio=0.35, linewidth=2)

        roll, pitch, yaw = np.degrees(qs.quat_to_euler(q))
        p_, q_, r_ = np.degrees(log["rates"][k])
        gf = thrust / (1e-3 * qs.G)
        self.info.set_text(
            f"t = {log['t'][k]:5.3f} s\n"
            f"roll {roll:7.1f}  pitch {pitch:7.1f}  yaw {yaw:7.1f} deg\n"
            f"p {p_:7.1f}  q {q_:7.1f}  r {r_:7.1f} deg/s\n"
            f"height {-c[2]:6.3f} m\n"
            f"thrust gf  M1 {gf[0]:5.1f}  M2 {gf[1]:5.1f}  M3 {gf[2]:5.1f}  M4 {gf[3]:5.1f}")


def animate(log: dict, fps: int = 30, speed: float = 1.0, follow: bool = False,
            save: str | None = None, show: bool = True):
    """Animate a log from run_and_log() with Matplotlib 3D FuncAnimation.

    fps     frames per second of the animation
    speed   1.0 = real time, 0.25 = four times slower
    follow  True: the camera box follows the quad instead of staying fixed
    save    path ending in .gif (Pillow) or .mp4 (needs ffmpeg); skips nothing else
    show    open a window (blocks until closed); set False when saving only
    Returns the FuncAnimation: keep a reference to it while the window is open.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    t = np.asarray(log["t"], float)
    dt = float(t[1] - t[0])
    step = max(1, int(round(speed / (fps * dt))))
    frames = list(range(0, len(t), step))
    if frames[-1] != len(t) - 1:
        frames.append(len(t) - 1)

    scene = _Scene(log, follow=follow)
    anim = FuncAnimation(scene.fig, scene.update, frames=frames,
                         interval=1000.0 / fps, repeat=True, blit=False)
    if save:
        folder = os.path.dirname(save)
        if folder:
            os.makedirs(folder, exist_ok=True)
        if save.lower().endswith(".gif"):
            anim.save(save, writer=PillowWriter(fps=fps))
        else:
            anim.save(save, fps=fps)   # needs ffmpeg on the PATH
        print(f"saved {save}")
    if show:
        plt.show()
    return anim


# ====================================================================== CLI
def main() -> int:
    ap = argparse.ArgumentParser(description="3D replay of the quad model")
    ap.add_argument("scenario", nargs="?", default="roll",
                    choices=sorted(SCENARIOS) + ["all"])
    ap.add_argument("--tw", type=float, help="thrust-to-weight ratio (default: quad_sim defaults)")
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 real time, 0.25 slow motion")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--follow", action="store_true", help="camera follows the quad")
    ap.add_argument("--save", metavar="PATH", help="file (.gif/.mp4) or a folder for 'all'")
    ap.add_argument("--no-show", action="store_true", help="do not open a window")
    args = ap.parse_args()

    import matplotlib
    if args.no_show or (args.save and args.scenario == "all"):
        matplotlib.use("Agg")

    params = qs.Params.from_thrust_to_weight(args.tw) if args.tw else None
    names = sorted(SCENARIOS) if args.scenario == "all" else [args.scenario]
    keep = []
    for name in names:
        log = run_and_log(name, params)
        save = args.save
        if save and args.scenario == "all":
            save = os.path.join(save, f"viz_{name}.gif")
        keep.append(animate(log, fps=args.fps, speed=args.speed, follow=args.follow,
                            save=save, show=not (args.no_show or (args.save and args.scenario == "all"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
