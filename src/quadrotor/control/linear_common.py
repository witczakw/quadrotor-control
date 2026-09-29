"""Shared pieces of the hover-linearised controllers (LQR and MPC).

The linear model (see :func:`quadrotor.dynamics.linearize_hover`) has the state
``x = [p, v, (roll, pitch, yaw), omega]`` (12) and the input ``u = [dT, tau_x, tau_y, tau_z]``.
It is augmented with the integral of the position error, ``z = int e_p dt``, giving 15 states and
offset-free tracking under model mismatch (mass error, rotor fault, wind).

Position and velocity errors are expressed in the yaw-aligned frame (``Rz(yaw)^T``), for which
the hover linearisation is valid for any heading.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_continuous_are

from quadrotor.control.base import Array, Estimate, Reference
from quadrotor.dynamics import linearize_hover
from quadrotor.params import QuadParams
from quadrotor.rotations import euler_from_quat, rot_z, wrap_angle

N_X = 12
N_Z = 15


def augmented_model(params: QuadParams) -> tuple[Array, Array]:
    """Hover model augmented with the position-error integral (15 states, 4 inputs)."""
    A, B = linearize_hover(params)
    Aa = np.zeros((N_Z, N_Z))
    Ba = np.zeros((N_Z, 4))
    Aa[:N_X, :N_X] = A
    Aa[N_X:, 0:3] = np.eye(3)
    Ba[:N_X] = B
    return Aa, Ba


@dataclass
class BrysonSpec:
    """Maximum acceptable deviations used by Bryson's rule ``Q_ii = 1/x_max_i^2``.

    Tuning rationale: tracking error of ~10 cm is acceptable, velocity errors of 0.7 m/s,
    tilt errors of ~8 deg and body-rate errors of 1 rad/s; the integral state is weighted so that
    its closed-loop pole is slower than the position loop. Inputs: thrust deviations up to 30 % of
    hover thrust, roll/pitch torque 0.4 N m, yaw torque 0.1 N m.
    """

    pos: float = 0.10
    vel: float = 0.7
    tilt: float = np.deg2rad(8.0)
    yaw: float = np.deg2rad(10.0)
    rate: float = 1.0
    yaw_rate: float = 1.0
    integ: float = 0.15
    thrust_frac: float = 0.30
    tau_xy: float = 0.40
    tau_z: float = 0.10


def bryson_weights(params: QuadParams, spec: BrysonSpec | None = None) -> tuple[Array, Array]:
    """Diagonal ``Q`` (15x15) and ``R`` (4x4) from Bryson's rule."""
    s = spec or BrysonSpec()
    x_max = np.concatenate(
        [
            np.full(3, s.pos),
            np.full(3, s.vel),
            [s.tilt, s.tilt, s.yaw],
            [s.rate, s.rate, s.yaw_rate],
            np.full(3, s.integ),
        ]
    )
    u_max = np.array([s.thrust_frac * params.hover_thrust, s.tau_xy, s.tau_xy, s.tau_z])
    return np.diag(1.0 / x_max**2), np.diag(1.0 / u_max**2)


def lqr_gain(A: Array, B: Array, Q: Array, R: Array) -> Array:
    """Continuous-time LQR gain ``K = R^-1 B^T P``."""
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P)


def reference_tilt(a_ref: Array, yaw: float, g: float) -> tuple[float, float]:
    """Small-angle roll/pitch that produce the reference acceleration (yaw-aligned frame)."""
    a_local = rot_z(yaw).T @ a_ref
    return float(-a_local[1] / g), float(a_local[0] / g)


def error_state(est: Estimate, ref: Reference, g: float) -> tuple[Array, Array]:
    """Error ``x - x_ref`` (12,) in the yaw-aligned frame and the thrust feed-forward vector.

    Returns:
        ``(e, f_ff)`` with ``f_ff = a_ref + g e3`` (acceleration the thrust must provide).
    """
    eul = euler_from_quat(est.q)
    Rz = rot_z(eul[2])
    roll_r, pitch_r = reference_tilt(ref.a, eul[2], g)
    e = np.concatenate(
        [
            Rz.T @ (est.p - ref.p),
            Rz.T @ (est.v - ref.v),
            [
                eul[0] - roll_r,
                eul[1] - pitch_r,
                wrap_angle(eul[2] - ref.yaw),
            ],
            est.omega - np.array([0.0, 0.0, ref.yaw_rate]),
        ]
    )
    return e, ref.a + np.array([0.0, 0.0, g])
