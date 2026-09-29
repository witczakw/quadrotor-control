"""Tests of the LQR and MPC controllers."""

from __future__ import annotations

import numpy as np
import pytest

from quadrotor.control.base import hover_reference
from quadrotor.control.linear_common import augmented_model
from quadrotor.control.lqr import LQRController
from quadrotor.control.mpc import LinearMPC, MPCConfig
from quadrotor.dynamics import IDX_P, IDX_Q, IDX_V, hover_state
from quadrotor.params import QuadParams
from quadrotor.rotations import euler_from_quat
from quadrotor.sim import Simulation


@pytest.fixture
def params() -> QuadParams:
    return QuadParams()


def test_lqr_weights_and_stability(params: QuadParams) -> None:
    ctrl = LQRController(params)
    assert np.all(np.linalg.eigvalsh(ctrl.Q) > 0) and np.all(np.linalg.eigvalsh(ctrl.R) > 0)
    A, B = augmented_model(params)
    poles = np.linalg.eigvals(A - B @ ctrl.K)
    assert np.all(poles.real < -0.5)


def test_lqr_hover_and_mass_mismatch(params: QuadParams) -> None:
    """Integral action removes the steady-state offset caused by a +20 % mass error."""
    plant = params.scaled(1.2)
    sim = Simulation(plant, LQRController(params), lambda t: hover_reference(np.array([0, 0, 2.0])))
    res = sim.run(10.0)
    assert np.linalg.norm(res.x[-1, IDX_P] - [0, 0, 2.0]) < 0.01


def test_lqr_small_perturbation_recovery(params: QuadParams) -> None:
    x0 = hover_state(params, np.array([0.3, -0.2, 1.8]))
    sim = Simulation(
        params, LQRController(params), lambda t: hover_reference(np.array([0, 0, 2.0])), x0=x0
    )
    res = sim.run(6.0)
    assert np.linalg.norm(res.x[-1, IDX_P] - [0, 0, 2.0]) < 0.01
    assert np.linalg.norm(res.x[-1, IDX_V]) < 0.01


def test_mpc_tracks_and_respects_tilt_limit(params: QuadParams) -> None:
    cfg = MPCConfig(tilt_max=np.deg2rad(20.0))
    target = np.array([3.0, 0.0, 1.0])
    ctrl = LinearMPC(params, cfg)
    res = Simulation(params, ctrl, lambda t: hover_reference(target)).run(6.0)
    tilts = np.array([np.abs(euler_from_quat(q)[:2]).max() for q in res.x[:, IDX_Q]])
    assert tilts.max() < np.deg2rad(20.0) + np.deg2rad(6.0)  # soft limit + linearisation error
    assert np.linalg.norm(res.x[-1, IDX_P] - target) < 0.05
    assert ctrl.n_fail == 0


def test_mpc_thrust_constraint(params: QuadParams) -> None:
    cfg = MPCConfig(thrust_max=14.0)
    ctrl = LinearMPC(params, cfg)
    res = Simulation(params, ctrl, lambda t: hover_reference(np.array([0, 0, 3.0]))).run(1.5)
    # commanded total thrust never exceeds the limit (thrust ~ sum kf w_cmd^2 before saturation)
    thrust_cmd = params.kf * np.sum(res.w_cmd**2, axis=1)
    assert thrust_cmd.max() <= 14.0 + 0.3
