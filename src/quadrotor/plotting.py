"""Plots and 3D animations of the experiments (matplotlib, headless)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from quadrotor.control.base import Array  # noqa: E402
from quadrotor.dynamics import IDX_P, IDX_Q  # noqa: E402
from quadrotor.params import QuadParams  # noqa: E402
from quadrotor.planning.scenarios import Scenario  # noqa: E402
from quadrotor.rotations import quat_conj, quat_log, quat_mult, quat_to_rot  # noqa: E402
from quadrotor.sim import SimResult  # noqa: E402

COLORS = {"PID": "#d95f02", "Geometric": "#1b9e77", "LQR": "#7570b3", "MPC": "#e7298a"}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def _draw_gates(ax, sc: Scenario) -> None:
    th = np.linspace(0, 2 * np.pi, 40)
    for g in sc.gates:  # gates are vertical rings whose normal is the x axis
        ax.plot(
            np.full_like(th, g[0]),
            g[1] + sc.gate_radius * np.cos(th),
            g[2] + sc.gate_radius * np.sin(th),
            color="0.35",
            lw=2,
        )


def plot_trajectories(sc: Scenario, results: dict[str, SimResult], path: Path) -> None:
    """3D view and top view of the flown paths versus the reference."""
    _style()
    fig = plt.figure(figsize=(11, 4.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ref = next(iter(results.values())).ref_p
    ax.plot(*ref.T, "k--", lw=1.5, label="reference")
    for name, r in results.items():
        ax.plot(*r.x[:, IDX_P].T, color=COLORS[name], lw=1.0, label=name)
    _draw_gates(ax, sc)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(f"{sc.name}: 3D")
    ax.legend(loc="upper left", fontsize=8)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(ref[:, 0], ref[:, 1], "k--", lw=1.5, label="reference")
    for name, r in results.items():
        ax2.plot(r.x[:, 0], r.x[:, 1], color=COLORS[name], lw=1.0, label=name)
    for g in sc.gates:
        ax2.plot([g[0], g[0]], [g[1] - sc.gate_radius, g[1] + sc.gate_radius], color="0.35", lw=3)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_title("top view")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_errors(sc: Scenario, results: dict[str, SimResult], path: Path) -> None:
    """Position tracking error over time for all controllers."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for name, r in results.items():
        ax.plot(r.t, r.pos_error * 100, color=COLORS[name], lw=1.1, label=name)
    ax.axvline(sc.trajectory.duration, color="0.5", ls=":", lw=1)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("position error [cm]")
    ax.set_title(f"{sc.name}: tracking error (controllers fed by the ESKF estimate)")
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_metric_bars(
    rows: list[dict], path: Path, cases: list[str], controllers: list[str]
) -> None:
    """Grouped bars of the RMSE per disturbance case, one panel per scenario."""
    _style()
    scenarios = sorted({r["scenario"] for r in rows})
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.2 * len(scenarios), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    width = 0.8 / len(controllers)
    for ax, s in zip(axes, scenarios, strict=True):
        for i, c in enumerate(controllers):
            vals = []
            for case in cases:
                m = [
                    r["rmse_pos_m"]
                    for r in rows
                    if r["scenario"] == s and r["controller"] == c and r["case"] == case
                ]
                vals.append(m[0] * 100 if m else np.nan)
            ax.bar(np.arange(len(cases)) + i * width, vals, width, color=COLORS[c], label=c)
        ax.set_xticks(np.arange(len(cases)) + 0.4 - width / 2)
        ax.set_xticklabels(cases, rotation=30, ha="right", fontsize=8)
        ax.set_title(s)
    axes[0].set_ylabel("position RMSE [cm]")
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_step(step_data: dict[str, tuple[Array, Array]], path: Path) -> None:
    """Step response of |p - p_target| for all controllers."""
    _style()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for name, (t, e) in step_data.items():
        ax.plot(t, e, color=COLORS[name], label=name)
    ax.axhline(0.05, color="0.5", ls=":", lw=1)
    ax.set_xlim(0, 5)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("distance to target [m]")
    ax.set_title("1 m diagonal step (5 cm settling band dotted)")
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def attitude_error(res: SimResult) -> Array:
    """Body-frame attitude estimation error vector (N, 3) [rad]."""
    return np.array(
        [quat_log(quat_mult(quat_conj(res.x[i, IDX_Q]), res.est_q[i])) for i in range(len(res.t))]
    )


def plot_estimation(res: SimResult, path: Path, title: str) -> None:
    """Estimation error with 3-sigma bounds (position, velocity, attitude) and IMU biases."""
    _style()
    assert res.est_sigma is not None
    blocks = [
        ("position error [m]", res.est_p - res.x[:, IDX_P], res.est_sigma[:, 0:3], 1.0),
        ("velocity error [m/s]", res.est_v - res.x[:, 3:6], res.est_sigma[:, 3:6], 1.0),
        ("attitude error [deg]", attitude_error(res), res.est_sigma[:, 6:9], 180 / np.pi),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    for ax, (label, err, sig, k) in zip(axes.flat[:3], blocks, strict=False):
        for i, (axis, col) in enumerate(zip("xyz", ["C0", "C1", "C2"], strict=True)):
            ax.plot(res.t, err[:, i] * k, color=col, lw=0.8, label=f"e_{axis}")
            ax.fill_between(res.t, -3 * sig[:, i] * k, 3 * sig[:, i] * k, color=col, alpha=0.12)
        ax.set_ylabel(label)
        settled = res.t > 1.5  # the initial covariance transient would dwarf the steady state
        ax.set_ylim(-1.4 * np.max(3 * sig[settled] * k), 1.4 * np.max(3 * sig[settled] * k))
        ax.legend(ncol=3, fontsize=8, loc="upper right")
    ax = axes[1, 1]
    bias = res.extra["bias"]
    for i, col in enumerate(["C0", "C1", "C2"]):
        ax.plot(res.t, bias[:, i], color=col, lw=1.0)
        ax.plot(res.t, bias[:, 6 + i], color=col, lw=1.0, ls="--")
    ax.set_ylabel("accelerometer bias [m/s^2]\n(solid: estimate, dashed: true)")
    for ax in axes[1]:
        ax.set_xlabel("t [s]")
    fig.suptitle(f"{title}: shaded band = 3-sigma of the ESKF covariance")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def animate(
    sc: Scenario,
    results: dict[str, SimResult],
    params: QuadParams,
    path: Path,
    fps: int = 15,
    speed: float = 1.0,
) -> None:
    """3D GIF: quadrotors (X frames) of all controllers flying the scenario."""
    _style()
    names = list(results)
    t = next(iter(results.values())).t
    step = max(1, round(speed / (fps * (t[1] - t[0]))))
    frames = np.arange(0, len(t), step)
    ref = next(iter(results.values())).ref_p
    fig = plt.figure(figsize=(7.2, 4.8))
    ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=28, azim=-62)
    ax.plot(*ref.T, "k--", lw=1, alpha=0.6, label="reference")
    _draw_gates(ax, sc)
    allp = np.vstack([r.x[:, IDX_P] for r in results.values()] + [ref])
    lo, hi = allp.min(0) - 0.3, allp.max(0) + 0.3
    half = np.maximum((hi - lo) / 2, 0.8)
    half = np.maximum(half, 0.22 * half.max())  # keep short axes readable on long scenes
    mid = (lo + hi) / 2
    ax.set_xlim(mid[0] - half[0], mid[0] + half[0])
    ax.set_ylim(mid[1] - half[1], mid[1] + half[1])
    ax.set_zlim(mid[2] - half[2], mid[2] + half[2])
    ax.set_box_aspect(tuple(2 * half))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    # the vehicle is drawn enlarged so that it is visible on the scale of the scene
    scale = 0.07 * float((2 * half).max()) / params.arm_length
    arms = np.array([[1, -1, 0], [1, 1, 0], [-1, 1, 0], [-1, -1, 0]]) * scale * params.arm_length
    arms = arms / np.sqrt(2)
    trails, frames_art = {}, {}
    for n in names:
        (trails[n],) = ax.plot([], [], [], color=COLORS[n], lw=1.3, label=n)
        frames_art[n] = [ax.plot([], [], [], color=COLORS[n], lw=3)[0] for _ in range(2)]
    ax.legend(loc="upper left", fontsize=8)
    title = ax.set_title("")

    def update(idx: int):
        i = frames[idx]
        for n in names:
            r = results[n]
            p = r.x[i, IDX_P]
            R = quat_to_rot(r.x[i, IDX_Q])
            tr = r.x[max(0, i - 400) : i + 1, IDX_P]
            trails[n].set_data_3d(tr[:, 0], tr[:, 1], tr[:, 2])
            pts = (R @ arms.T).T + p
            for k, (a, b) in enumerate([(0, 2), (1, 3)]):
                seg = np.array([pts[a], pts[b]])
                frames_art[n][k].set_data_3d(seg[:, 0], seg[:, 1], seg[:, 2])
        title.set_text(f"{sc.name}   t = {t[i]:.1f} s")
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=70)
    plt.close(fig)
