"""Rigid-body 6-DOF quadrotor dynamics (Newton-Euler), rotor model, mixer and RK4 integrator.

State vector (17): ``[p(3), v(3), q(4), omega(3), rotor_speeds(4)]`` with position ``p`` and
velocity ``v`` in the inertial frame, quaternion ``q`` (body -> inertial, [w, x, y, z]),
angular velocity ``omega`` in the body frame and rotor speeds ``w_i`` [rad/s].
"""

from __future__ import annotations

import numpy as np

from quadrotor.params import QuadParams
from quadrotor.rotations import quat_from_euler, quat_mult, quat_to_rot

Array = np.ndarray

IDX_P = slice(0, 3)
IDX_V = slice(3, 6)
IDX_Q = slice(6, 10)
IDX_W = slice(10, 13)
IDX_R = slice(13, 17)
STATE_DIM = 17


def hover_state(params: QuadParams, position: Array | None = None) -> Array:
    """State of a quadrotor hovering at ``position`` with level attitude."""
    x = np.zeros(STATE_DIM)
    if position is not None:
        x[IDX_P] = position
    x[6] = 1.0
    x[IDX_R] = params.hover_speed
    return x


def allocation_matrix(params: QuadParams) -> Array:
    """Matrix ``A`` with ``[T, tau_x, tau_y, tau_z] = A @ [f_0 .. f_3]`` (f_i = rotor thrust)."""
    pos = params.rotor_positions
    spin = np.asarray(params.spin, float)
    return np.vstack(
        [
            np.ones(4),
            pos[:, 1],  # tau_x = sum y_i f_i
            -pos[:, 0],  # tau_y = -sum x_i f_i
            -spin * params.km / params.kf,  # reaction torque opposite to the rotor spin
        ]
    )


def mix(params: QuadParams, alloc_inv: Array, thrust: float, torque: Array) -> Array:
    """Wrench -> rotor speed commands with desaturation (see :meth:`Quadrotor.mix`).

    Rotor thrusts are ``f = alloc_inv @ [T, tau]``. If some rotor leaves ``[f_min, f_max]`` the
    common thrust ``T`` is first shifted as little as possible (torque has priority, so that
    attitude control survives at very low or very high thrust); if no feasible shift exists the
    torques are scaled down until the request fits.
    """
    f_min = params.kf * params.w_min**2
    f_max = params.kf * params.w_max**2
    thrust = float(np.clip(thrust, 4 * f_min, 4 * f_max))
    f_thrust = alloc_inv[:, 0] * thrust
    f_torque = alloc_inv[:, 1:] @ np.asarray(torque, float)
    # a shift of T by dT changes rotor i by alloc_inv[i, 0] * dT
    lo = max(
        (f_min - fq - ft) / a for ft, fq, a in zip(f_thrust, f_torque, alloc_inv[:, 0], strict=True)
    )
    hi = min(
        (f_max - fq - ft) / a for ft, fq, a in zip(f_thrust, f_torque, alloc_inv[:, 0], strict=True)
    )
    if lo <= hi:
        f = f_thrust + f_torque + alloc_inv[:, 0] * float(np.clip(0.0, lo, hi))
    else:
        scale = 1.0
        for ft, fq in zip(f_thrust, f_torque, strict=True):
            if fq > 1e-12:
                scale = min(scale, (f_max - ft) / fq)
            elif fq < -1e-12:
                scale = min(scale, (f_min - ft) / fq)
        f = f_thrust + max(scale, 0.0) * f_torque
    return np.sqrt(np.clip(f, f_min, f_max) / params.kf)


class Quadrotor:
    """Quadrotor plant: dynamics, rotor model and motor mixer.

    Args:
        params: physical parameters of the (true) plant.
        rotor_efficiency: multiplicative thrust/torque efficiency of each rotor (fault model).
    """

    def __init__(self, params: QuadParams, rotor_efficiency: Array | None = None) -> None:
        self.p = params
        self.rotor_efficiency = (
            np.ones(4) if rotor_efficiency is None else np.asarray(rotor_efficiency, float)
        )
        self._g = np.array([0.0, 0.0, -params.gravity])
        self._J = params.J
        self._Jinv = np.linalg.inv(self._J)
        self.allocation = allocation_matrix(params)
        self._alloc_inv = np.linalg.inv(self.allocation)

    def rotor_thrusts(self, w: Array) -> Array:
        """Thrust of each rotor [N] including efficiency loss."""
        return self.rotor_efficiency * self.p.kf * w * w

    def wrench(self, w: Array) -> Array:
        """Total thrust and body torques ``[T, tau_x, tau_y, tau_z]`` produced by rotor speeds."""
        return self.allocation @ self.rotor_thrusts(w)

    def mix(self, thrust: float, torque: Array) -> Array:
        """Map (total thrust, body torques) to rotor speed commands [rad/s].

        If the request is infeasible the torques are scaled down (thrust has priority), then
        rotor speeds are clipped to ``[w_min, w_max]``.
        """
        return mix(self.p, self._alloc_inv, thrust, torque)

    def derivative(self, x: Array, w_cmd: Array, v_wind: Array | None = None) -> Array:
        """Continuous-time state derivative for rotor speed commands ``w_cmd``."""
        p = self.p
        v, q, omega, w = x[IDX_V], x[IDX_Q], x[IDX_W], x[IDX_R]

        R = quat_to_rot(q)
        f = self.rotor_efficiency * p.kf * w * w
        thrust = f.sum()
        torque = self.allocation[1:] @ f

        v_rel = v if v_wind is None else v - v_wind
        acc = R[:, 2] * (thrust / p.mass) - (p.drag_coeff / p.mass) * v_rel + self._g

        dx = np.empty(STATE_DIM)
        dx[IDX_P] = v
        dx[IDX_V] = acc
        dx[IDX_Q] = 0.5 * quat_mult(q, np.array([0.0, *omega]))
        # Euler equations with the gyroscopic term omega x J omega
        dx[IDX_W] = self._Jinv @ (torque - np.cross(omega, self._J @ omega))
        dx[IDX_R] = (np.clip(w_cmd, p.w_min, p.w_max) - w) / p.tau_motor
        return dx

    def specific_force(self, x: Array, w_cmd: Array, v_wind: Array | None = None) -> Array:
        """Non-gravitational acceleration in the body frame (ideal accelerometer output)."""
        dx = self.derivative(x, w_cmd, v_wind)
        return quat_to_rot(x[IDX_Q]).T @ (dx[IDX_V] - self._g)

    def step(self, x: Array, w_cmd: Array, dt: float, v_wind: Array | None = None) -> Array:
        """One RK4 step with zero-order-hold inputs; the quaternion is renormalised."""
        k1 = self.derivative(x, w_cmd, v_wind)
        k2 = self.derivative(x + 0.5 * dt * k1, w_cmd, v_wind)
        k3 = self.derivative(x + 0.5 * dt * k2, w_cmd, v_wind)
        k4 = self.derivative(x + dt * k3, w_cmd, v_wind)
        xn = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        xn[IDX_Q] /= np.linalg.norm(xn[IDX_Q])
        return xn

    def mechanical_energy(self, x: Array) -> float:
        """Translational + rotational kinetic energy plus potential energy [J]."""
        p = self.p
        v, omega = x[IDX_V], x[IDX_W]
        return float(
            0.5 * p.mass * v @ v + 0.5 * omega @ self._J @ omega + p.mass * p.gravity * x[2]
        )


def euler_rate_matrix(roll: float, pitch: float) -> Array:
    """Matrix W with d(euler)/dt = W @ omega_body (ZYX convention)."""
    sr, cr = np.sin(roll), np.cos(roll)
    tp, cp = np.tan(pitch), np.cos(pitch)
    return np.array([[1.0, sr * tp, cr * tp], [0.0, cr, -sr], [0.0, sr / cp, cr / cp]])


def euler_wrench_dynamics(x: Array, u: Array, params: QuadParams) -> Array:
    """Nonlinear rigid-body dynamics with Euler angles and wrench input (ideal motors).

    Args:
        x: ``[p(3), v(3), (roll, pitch, yaw), omega_body(3)]`` (12).
        u: ``[T, tau_x, tau_y, tau_z]``.
    """
    v, eul, omega = x[3:6], x[6:9], x[9:12]
    R = quat_to_rot(quat_from_euler(*eul))
    acc = R[:, 2] * (u[0] / params.mass) - (params.drag_coeff / params.mass) * v
    acc = acc + np.array([0.0, 0.0, -params.gravity])
    J = params.J
    domega = np.linalg.solve(J, u[1:] - np.cross(omega, J @ omega))
    return np.concatenate([v, acc, euler_rate_matrix(eul[0], eul[1]) @ omega, domega])


def linearize_hover(params: QuadParams) -> tuple[Array, Array]:
    """Linearisation of :func:`euler_wrench_dynamics` about hover (zero state, ``T = m g``).

    Deviation state ``[dp, dv, d(euler), domega]`` and input ``[dT, tau_x, tau_y, tau_z]``.
    For small angles: ``dv_x = g*pitch``, ``dv_y = -g*roll``, ``dv_z = dT/m``.
    """
    m, g, cd = params.mass, params.gravity, params.drag_coeff
    A = np.zeros((12, 12))
    B = np.zeros((12, 4))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 3:6] = -(cd / m) * np.eye(3)
    A[3, 7] = g
    A[4, 6] = -g
    A[6:9, 9:12] = np.eye(3)
    B[5, 0] = 1.0 / m
    B[9:12, 1:4] = np.linalg.inv(params.J)
    return A, B
