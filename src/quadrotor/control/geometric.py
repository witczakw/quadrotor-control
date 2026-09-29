"""Geometric tracking controller on SO(3) (Lee, Leok, McClamroch, CDC 2010)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quadrotor.control.base import Array, Controller, Estimate, Reference
from quadrotor.params import QuadParams
from quadrotor.rotations import hat, quat_to_rot, vee


@dataclass
class GeometricGains:
    """Gains of the geometric controller (force gains are per unit mass, torque per inertia)."""

    kx: Array = field(default_factory=lambda: np.array([9.0, 9.0, 14.0]))  # [1/s^2]
    kv: Array = field(default_factory=lambda: np.array([5.0, 5.0, 6.5]))  # [1/s]
    ki: Array = field(default_factory=lambda: np.array([2.0, 2.0, 4.0]))  # [1/s^3]
    kR: Array = field(default_factory=lambda: np.array([130.0, 130.0, 25.0]))  # [1/s^2]
    kOmega: Array = field(default_factory=lambda: np.array([16.0, 16.0, 8.0]))  # [1/s]
    kIR: float = 8.0  # attitude integral gain [1/(s^3)]
    int_pos_max: float = 4.0  # [m/s^2]
    int_zone: float = 0.5  # integrate the position error only within this distance [m]
    int_att_max: float = 30.0  # [rad/s^2]
    int_att_zone: float = 0.15  # integrate e_R only when the attitude error is small


class GeometricController(Controller):
    """Position and yaw tracking on SE(3).

    Thrust vector: ``F = -kx e_x - kv e_v - ki int(e_x) + m (a_d + g e3)``,  ``T = F . R e3``.
    Desired attitude ``R_d = [b1d b2d b3d]`` with ``b3d = F/|F|`` and ``b1d`` from the yaw
    reference. Torque::

        tau = -kR e_R - kOmega e_Omega + Omega x J Omega
              - J (hat(Omega) R^T R_d Omega_d - R^T R_d dOmega_d)

    with ``e_R = 1/2 vee(R_d^T R - R^T R_d)`` and ``e_Omega = Omega - R^T R_d Omega_d``.
    The desired angular velocity ``Omega_d`` is obtained analytically from the reference jerk and
    yaw rate; ``dOmega_d`` is neglected. A small integral term on ``e_R`` and on the position
    error removes steady-state error under model mismatch (Lee et al. 2011).
    """

    name = "Geometric"

    def __init__(self, params: QuadParams, gains: GeometricGains | None = None, dt: float = 0.004):
        self.params = params
        self.g = gains or GeometricGains()
        self.dt = dt
        self.reset()

    def reset(self) -> None:
        self.i_pos = np.zeros(3)
        self.i_att = np.zeros(3)

    def desired_rotation(self, f_vec: Array, yaw: float) -> Array:
        """Desired attitude matrix from the thrust vector and yaw angle."""
        b3 = f_vec / max(float(np.linalg.norm(f_vec)), 1e-6)
        c_yaw = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        b2 = np.cross(b3, c_yaw)
        b2 /= max(float(np.linalg.norm(b2)), 1e-9)
        b1 = np.cross(b2, b3)
        return np.column_stack([b1, b2, b3])

    def feedforward_rate(self, ref: Reference, R_d: Array) -> Array:
        """Desired body angular velocity from reference jerk and yaw rate."""
        prm = self.params
        f_ref = ref.a + np.array([0.0, 0.0, prm.gravity])
        n = max(float(np.linalg.norm(f_ref)), 1e-6)
        b3 = f_ref / n
        db3 = (ref.j - (b3 @ ref.j) * b3) / n
        w_tilt = np.cross(b3, db3)  # angular velocity (inertial) that tilts the thrust axis
        w_world = w_tilt + ref.yaw_rate * np.array([0.0, 0.0, 1.0])
        return R_d.T @ w_world

    def compute(self, t: float, est: Estimate, ref: Reference) -> tuple[float, Array]:
        prm, g, dt = self.params, self.g, self.dt
        R = quat_to_rot(est.q)
        J = prm.J

        e_x = est.p - ref.p
        e_v = est.v - ref.v
        if np.linalg.norm(e_x) < g.int_zone:  # conditional integration (anti-windup)
            self.i_pos = np.clip(self.i_pos + g.ki * e_x * dt, -g.int_pos_max, g.int_pos_max)
        a_cmd = -g.kx * e_x - g.kv * e_v - self.i_pos + ref.a + np.array([0.0, 0.0, prm.gravity])
        f_vec = prm.mass * a_cmd
        thrust = max(float(f_vec @ R[:, 2]), 0.0)

        R_d = self.desired_rotation(f_vec, ref.yaw)
        omega_d = self.feedforward_rate(ref, R_d)

        e_R = 0.5 * vee(R_d.T @ R - R.T @ R_d)
        e_W = est.omega - R.T @ R_d @ omega_d
        if np.linalg.norm(e_R) < g.int_att_zone:  # conditional integration (anti-windup)
            self.i_att = np.clip(self.i_att + g.kIR * e_R * dt, -g.int_att_max, g.int_att_max)

        ang_acc = -g.kR * e_R - g.kOmega * e_W - self.i_att
        # feed-forward: - J (hat(Omega) R^T R_d Omega_d) (dOmega_d neglected)
        ff = -hat(est.omega) @ (R.T @ R_d @ omega_d)
        torque = J @ (ang_acc + ff) + np.cross(est.omega, J @ est.omega)
        return thrust, torque
