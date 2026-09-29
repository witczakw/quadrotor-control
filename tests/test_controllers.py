"""Closed-loop tests of the controllers (ground-truth state feedback)."""

from __future__ import annotations

import numpy as np
import pytest

from quadrotor.control.base import Controller, hover_reference
from quadrotor.control.geometric import GeometricController
from quadrotor.control.pid import CascadedPID
from quadrotor.dynamics import IDX_P, IDX_Q, IDX_V, hover_state
from quadrotor.params import QuadParams
from quadrotor.rotations import euler_from_quat, quat_from_euler
from quadrotor.sim import Simulation


@pytest.fixture
def params() -> QuadParams:
    return QuadParams()


@pytest.mark.parametrize("cls", [CascadedPID, GeometricController])
def test_step_response_converges(cls: type[Controller], params: QuadParams) -> None:
    target = np.array([1.0, -1.0, 0.8])
    sim = Simulation(params, cls(params), lambda t: hover_reference(target))
    res = sim.run(7.0)
    assert np.linalg.norm(res.x[-1, IDX_P] - target) < 0.02
    assert np.linalg.norm(res.x[-1, IDX_V]) < 0.02


def test_geometric_recovers_from_large_attitude_error(params: QuadParams) -> None:
    """Convergence from a 120 deg tilt and a yaw offset (almost-global SO(3) property)."""
    x0 = hover_state(params, np.array([0.0, 0.0, 5.0]))
    x0[IDX_Q] = quat_from_euler(np.deg2rad(120.0), np.deg2rad(20.0), np.deg2rad(70.0))
    ctrl = GeometricController(params)
    sim = Simulation(params, ctrl, lambda t: hover_reference(np.array([0.0, 0.0, 5.0]), 0.5), x0=x0)
    res = sim.run(8.0)
    eul = euler_from_quat(res.x[-1, IDX_Q])
    assert np.allclose(eul[:2], 0.0, atol=0.02)
    assert eul[2] == pytest.approx(0.5, abs=0.02)
    assert np.linalg.norm(res.x[-1, IDX_P] - [0.0, 0.0, 5.0]) < 0.1


def test_geometric_yaw_tracking(params: QuadParams) -> None:
    sim = Simulation(
        params, GeometricController(params), lambda t: hover_reference(np.zeros(3), 2.0)
    )
    res = sim.run(4.0)
    assert euler_from_quat(res.x[-1, IDX_Q])[2] == pytest.approx(2.0, abs=0.02)


def test_hover_holds_position(params: QuadParams) -> None:
    for cls in (CascadedPID, GeometricController):
        sim = Simulation(params, cls(params), lambda t: hover_reference(np.array([0.0, 0.0, 2.0])))
        res = sim.run(3.0)
        assert np.max(res.pos_error) < 1e-3
