"""Ready-made benchmark scenarios built on the min-snap generator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from quadrotor.control.base import Array
from quadrotor.planning.minsnap import MinSnapTrajectory, allocate_times

G = 9.81


@dataclass
class Scenario:
    """A benchmark: trajectory, simulated duration and optional gates."""

    name: str
    trajectory: MinSnapTrajectory
    duration: float
    gates: list[Array] = field(default_factory=list)  # gate centres
    gate_radius: float = 0.4

    @property
    def start(self) -> Array:
        return self.trajectory.waypoints[0]


def _finalize(traj: MinSnapTrajectory, a_max: float) -> MinSnapTrajectory:
    """Slow the trajectory down (same path) so that the horizontal acceleration is <= a_max."""
    a_peak = max(
        np.linalg.norm(traj.derivative(t, 2)[:2]) for t in np.arange(0, traj.duration, 0.02)
    )
    if a_peak > a_max:
        traj = traj.scaled_in_time(float(np.sqrt(a_peak / a_max)))
    return traj


def smooth_yaw(t0: float, t1: float, yaw1: float) -> Callable[[float], tuple[float, float, float]]:
    """Quintic (minimum-jerk) yaw change from 0 to ``yaw1`` between ``t0`` and ``t1``."""

    def f(t: float) -> tuple[float, float, float]:
        if t <= t0:
            return 0.0, 0.0, 0.0
        if t >= t1:
            return yaw1, 0.0, 0.0
        T = t1 - t0
        u = (t - t0) / T
        s = 10 * u**3 - 15 * u**4 + 6 * u**5
        ds = (30 * u**2 - 60 * u**3 + 30 * u**4) / T
        dds = (60 * u - 180 * u**2 + 120 * u**3) / T**2
        return yaw1 * s, yaw1 * ds, yaw1 * dds

    return f


def figure_eight(segment_time: float = 0.6, a_max: float = 6.0) -> Scenario:
    """3D lemniscate: 3 m x 1.6 m loops, vertical excursion of 1 m between the two crossings."""
    th = np.linspace(0.0, 2 * np.pi, 25)
    wp = np.column_stack([1.6 * np.sin(th), 0.8 * np.sin(2 * th), 1.5 + 0.5 * (1 - np.cos(th))])
    traj = _finalize(MinSnapTrajectory(wp, np.full(len(wp) - 1, segment_time)), a_max)
    return Scenario("figure8", traj, traj.duration + 2.0)


def helix(v_avg: float = 2.0, a_max: float = 6.0) -> Scenario:
    """Two turns of a helix (radius 1.5 m) climbing 2 m."""
    th = np.linspace(0.0, 4 * np.pi, 25)
    wp = np.column_stack([1.5 * np.cos(th) - 1.5, 1.5 * np.sin(th), 1.0 + 2.0 * th / (4 * np.pi)])
    traj = _finalize(MinSnapTrajectory(wp, allocate_times(wp, v_avg)), a_max)
    return Scenario("helix", traj, traj.duration + 2.0)


def slalom(v_avg: float = 2.5, a_max: float = 7.0) -> Scenario:
    """Slalom through six gates alternating left and right of the centre line."""
    gates = [np.array([2.5 * (i + 1), 1.0 * (-1) ** i, 1.5]) for i in range(6)]
    wp = np.vstack([[0.0, 0.0, 1.0], gates, [17.5, 0.0, 1.0]])
    traj = _finalize(MinSnapTrajectory(wp, allocate_times(wp, v_avg)), a_max)
    return Scenario("slalom", traj, traj.duration + 2.0, gates=gates)


def aggressive_turn(v_avg: float = 3.6, a_max: float = 8.0) -> Scenario:
    """Fast straight line followed by a hairpin turn with a simultaneous 90 deg yaw change."""
    wp = np.array(
        [
            [0.0, 0.0, 1.5],
            [3.0, 0.0, 1.5],
            [6.0, 0.0, 1.5],
            [7.5, 1.5, 1.7],
            [6.0, 3.0, 1.9],
            [3.0, 3.0, 1.9],
            [0.0, 3.0, 1.9],
        ]
    )
    times = allocate_times(wp, v_avg)
    yaw = smooth_yaw(float(times[:2].sum()), float(times[:5].sum()), np.pi / 2)
    traj = _finalize(MinSnapTrajectory(wp, times, yaw), a_max)
    # rebuild yaw timing for the (possibly) time-scaled trajectory
    k = traj.duration / times.sum()
    traj.yaw = smooth_yaw(float(times[:2].sum()) * k, float(times[:5].sum()) * k, np.pi / 2)
    return Scenario("aggressive_turn", traj, traj.duration + 2.0)


SCENARIOS: dict[str, Callable[[], Scenario]] = {
    "figure8": figure_eight,
    "helix": helix,
    "slalom": slalom,
    "aggressive_turn": aggressive_turn,
}
