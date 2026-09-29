"""Tests of the min-snap trajectory generator."""

from __future__ import annotations

import numpy as np
import pytest

from quadrotor.planning.minsnap import MinSnapTrajectory, allocate_times
from quadrotor.planning.scenarios import SCENARIOS


@pytest.fixture
def traj() -> MinSnapTrajectory:
    wp = np.array([[0, 0, 1], [2, 1, 1.5], [4, -1, 2], [6, 0, 1.0]], float)
    return MinSnapTrajectory(wp, allocate_times(wp, 1.5))


def test_passes_through_waypoints(traj: MinSnapTrajectory) -> None:
    for wp, t in zip(traj.waypoints, traj.knots, strict=True):
        assert np.allclose(traj.derivative(t, 0), wp, atol=1e-9)


def test_boundary_derivatives_zero(traj: MinSnapTrajectory) -> None:
    for n in (1, 2, 3):
        assert np.allclose(traj.derivative(0.0, n), 0.0, atol=1e-8)
        assert np.allclose(traj.derivative(traj.duration, n), 0.0, atol=1e-8)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_derivative_continuity_at_knots(traj: MinSnapTrajectory, n: int) -> None:
    """Position, velocity, acceleration, jerk (imposed) and snap (optimality) are continuous."""
    eps = 1e-7
    for t in traj.knots[1:-1]:
        left = traj.derivative(t - eps, n)
        right = traj.derivative(t + eps, n)
        assert np.allclose(left, right, atol=1e-3 * (1 + np.abs(left).max()))


def test_two_waypoints_match_analytic_solution() -> None:
    """Rest-to-rest min-snap between two points is the 7th-order smoothstep polynomial."""
    wp = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    line = MinSnapTrajectory(wp, np.array([2.0]))
    s = np.linspace(0, 1, 5)
    expected = 35 * s**4 - 84 * s**5 + 70 * s**6 - 20 * s**7
    got = np.array([line.derivative(2.0 * si, 0)[0] for si in s])
    assert np.allclose(got, expected, atol=1e-9)


def test_optimality_against_feasible_perturbations(traj: MinSnapTrajectory) -> None:
    """Moving the coefficients inside the constraint set (null space) never lowers the cost."""
    from scipy.linalg import block_diag, null_space

    from quadrotor.planning.minsnap import _snap_cost

    A, _ = traj._constraints(traj.waypoints[:, 0], np.zeros(3), np.zeros(3))
    H = block_diag(*[2 * _snap_cost(T) for T in traj.times])
    c = traj.coef[0].ravel()
    N = null_space(A)
    rng = np.random.default_rng(0)
    base = c @ H @ c
    for _ in range(20):
        c2 = c + N @ (0.05 * rng.standard_normal(N.shape[1]))
        assert c2 @ H @ c2 >= base - 1e-9
    assert np.allclose(N.T @ H @ c, 0.0, atol=1e-6)  # stationarity in the feasible subspace


def test_time_scaling(traj: MinSnapTrajectory) -> None:
    slow = traj.scaled_in_time(2.0)
    assert slow.duration == pytest.approx(2 * traj.duration)
    assert slow.max_derivative(2) == pytest.approx(traj.max_derivative(2) / 4, rel=1e-3)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenarios_are_feasible(name: str) -> None:
    sc = SCENARIOS[name]()
    tr = sc.trajectory
    assert tr.max_derivative(2) < 1.5 * 9.81  # within a plausible thrust envelope
    for wp, t in zip(tr.waypoints, tr.knots, strict=True):
        assert np.allclose(tr.derivative(t, 0), wp, atol=1e-8)
