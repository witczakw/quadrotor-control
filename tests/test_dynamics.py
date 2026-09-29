"""Tests of the rigid-body model, mixer and linearisation."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from quadrotor.dynamics import (
    IDX_P,
    IDX_Q,
    IDX_R,
    IDX_V,
    IDX_W,
    STATE_DIM,
    Quadrotor,
    euler_wrench_dynamics,
    hover_state,
    linearize_hover,
)
from quadrotor.params import QuadParams
from quadrotor.rotations import quat_from_euler, quat_to_rot


@pytest.fixture
def params() -> QuadParams:
    return QuadParams.from_yaml()


def test_yaml_matches_defaults(params: QuadParams) -> None:
    assert params == QuadParams()


def test_energy_conserved_without_drag_and_control(params: QuadParams) -> None:
    """Free flight (rotors off, no drag): kinetic + potential energy is constant."""
    p = dataclasses.replace(params, drag_coeff=0.0, w_min=0.0)
    quad = Quadrotor(p)
    x = np.zeros(STATE_DIM)
    x[IDX_Q] = quat_from_euler(0.3, -0.2, 1.0)
    x[IDX_V] = [2.0, -1.0, 3.0]
    x[IDX_P] = [0.0, 0.0, 10.0]
    x[IDX_W] = [4.0, -3.0, 5.0]  # tumbling asymmetric-inertia body (gyroscopic term active)
    e0 = quad.mechanical_energy(x)
    cmd = np.zeros(4)
    for _ in range(2000):  # 2 s
        x = quad.step(x, cmd, 1e-3)
    assert abs(quad.mechanical_energy(x) - e0) / e0 < 1e-8
    # ballistic flight: z(t) = z0 + vz t - g t^2 / 2
    assert x[2] == pytest.approx(10.0 + 3.0 * 2.0 - 0.5 * 9.81 * 4.0, abs=1e-6)


def test_quaternion_norm_preserved(params: QuadParams) -> None:
    quad = Quadrotor(params)
    x = hover_state(params)
    x[IDX_W] = [3.0, 2.0, -4.0]
    cmd = np.full(4, params.hover_speed)
    for _ in range(5000):
        x = quad.step(x, cmd, 1e-3)
    assert np.linalg.norm(x[IDX_Q]) == pytest.approx(1.0, abs=1e-12)


def test_hover_equilibrium(params: QuadParams) -> None:
    """Thrust equal to m g keeps the vehicle at rest."""
    quad = Quadrotor(params)
    assert quad.wrench(np.full(4, params.hover_speed))[0] == pytest.approx(params.mass * 9.81)
    x0 = hover_state(params, np.array([1.0, 2.0, 3.0]))
    cmd = np.full(4, params.hover_speed)
    assert np.allclose(quad.derivative(x0, cmd)[: IDX_W.stop], 0.0, atol=1e-12)
    x = x0.copy()
    for _ in range(3000):
        x = quad.step(x, cmd, 1e-3)
    assert np.allclose(x[IDX_P], x0[IDX_P], atol=1e-9)
    assert np.allclose(x[IDX_V], 0.0, atol=1e-9)


def test_rotor_speed_saturation(params: QuadParams) -> None:
    quad = Quadrotor(params)
    x = hover_state(params)
    for _ in range(3000):
        x = quad.step(x, np.full(4, 1e5), 1e-3)
    assert np.all(x[IDX_R] <= params.w_max + 1e-9)
    assert np.all(x[IDX_R] > 0.99 * params.w_max)


def test_motor_first_order_response(params: QuadParams) -> None:
    quad = Quadrotor(params)
    x = hover_state(params)
    target = params.hover_speed + 100.0
    for _ in range(int(params.tau_motor / 1e-3)):
        x = quad.step(x, np.full(4, target), 1e-3)
    frac = (x[13] - params.hover_speed) / 100.0
    assert frac == pytest.approx(1 - np.exp(-1), abs=1e-3)


def test_mixer_roundtrip(params: QuadParams) -> None:
    quad = Quadrotor(params)
    thrust, torque = 14.0, np.array([0.05, -0.08, 0.02])
    w = quad.mix(thrust, torque)
    assert np.allclose(quad.wrench(w), [thrust, *torque])


def test_mixer_saturation_prioritises_thrust(params: QuadParams) -> None:
    quad = Quadrotor(params)
    w = quad.mix(params.hover_thrust, np.array([50.0, 0.0, 0.0]))
    assert np.all(w >= params.w_min - 1e-9) and np.all(w <= params.w_max + 1e-9)
    assert quad.wrench(w)[0] == pytest.approx(params.hover_thrust, rel=1e-9)


def test_roll_torque_sign(params: QuadParams) -> None:
    """Positive tau_x rolls the vehicle right (+y goes down): body y-axis tilts to -z."""
    quad = Quadrotor(params)
    x = hover_state(params)
    w = quad.mix(params.hover_thrust, np.array([0.1, 0.0, 0.0]))
    for _ in range(100):
        x = quad.step(x, w, 1e-3)
    assert x[IDX_W][0] > 0
    assert quat_to_rot(x[IDX_Q])[2, 1] > 0  # body y axis rises -> R[2,1] = sin(roll) > 0


def test_linearization_matches_nonlinear_model(params: QuadParams) -> None:
    A, B = linearize_hover(params)
    u0 = np.array([params.hover_thrust, 0.0, 0.0, 0.0])
    x0 = np.zeros(12)
    assert np.allclose(euler_wrench_dynamics(x0, u0, params), 0.0, atol=1e-12)

    rng = np.random.default_rng(1)
    errs = []
    dx0 = rng.standard_normal(12)
    du0 = rng.standard_normal(4) * np.array([1.0, 0.1, 0.1, 0.1])
    for scale in (1e-2, 1e-3):
        dx, du = scale * dx0, scale * du0
        exact = euler_wrench_dynamics(x0 + dx, u0 + du, params)
        lin = A @ dx + B @ du
        errs.append(np.linalg.norm(exact - lin))
        assert errs[-1] < 5 * scale**2 * 20  # second-order error
    assert errs[1] < errs[0] / 50  # same direction, 10x smaller step  # error shrinks quadratically


def test_linearization_vs_quaternion_model(params: QuadParams) -> None:
    """Linearised translational/rotational accelerations match the full quaternion model."""
    A, B = linearize_hover(params)
    quad = Quadrotor(params)
    eul = np.array([0.02, -0.03, 0.0])
    dv = np.array([0.1, -0.2, 0.05])
    x = hover_state(params)
    x[IDX_Q] = quat_from_euler(*eul)
    x[IDX_V] = dv
    w_cmd = np.full(4, params.hover_speed)
    acc = quad.derivative(x, w_cmd)[IDX_V]
    lin = A[3:6, :] @ np.concatenate([np.zeros(3), dv, eul, np.zeros(3)])
    assert np.allclose(acc, lin, atol=1e-2)
