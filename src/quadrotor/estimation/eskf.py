"""Error-state (multiplicative) extended Kalman filter for a quadrotor.

Nominal state: position ``p``, velocity ``v`` (inertial), attitude quaternion ``q`` (body ->
inertial) and IMU biases ``b_a``, ``b_g``. Error state (15): ``[dp, dv, dtheta, db_a, db_g]`` with
the attitude error expressed in the body frame, ``q = q_nom (x) exp(dtheta)`` (Sola, 2017).

Measurement models: accelerometer ``a_m = R^T (a - g) + b_a + n``, gyroscope
``w_m = w + b_g + n``.
"""

from __future__ import annotations

import numpy as np

from quadrotor.control.base import Array
from quadrotor.estimation.sensors import SensorParams
from quadrotor.rotations import (
    euler_from_quat,
    hat,
    quat_exp,
    quat_mult,
    quat_normalize,
    quat_to_rot,
    wrap_angle,
)

GRAVITY = np.array([0.0, 0.0, -9.81])
N = 15
SL_P, SL_V, SL_TH, SL_BA, SL_BG = (
    slice(0, 3),
    slice(3, 6),
    slice(6, 9),
    slice(9, 12),
    slice(12, 15),
)


class ESKF:
    """Error-state Kalman filter.

    The nominal state is propagated with every IMU sample (:meth:`propagate_nominal`); the error
    covariance is propagated and measurements are fused at the (slower) EKF rate.
    """

    def __init__(self, sp: SensorParams, gravity: float = 9.81) -> None:
        self.sp = sp
        self.g = np.array([0.0, 0.0, -gravity])
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)
        self.P = np.eye(N)

    def initialize(self, p: Array, v: Array, q: Array, P0_std: Array | None = None) -> None:
        """Set the nominal state and the initial covariance (1-sigma per error-state block)."""
        self.p, self.v, self.q = p.copy(), v.copy(), quat_normalize(q)
        self.ba, self.bg = np.zeros(3), np.zeros(3)
        std = np.array([0.5, 0.1, 0.05, 0.06, 0.015]) if P0_std is None else P0_std
        self.P = np.diag(np.repeat(std**2, 3))

    # ------------------------------------------------------------------ prediction
    def propagate_nominal(self, gyro: Array, accel: Array, dt: float) -> None:
        """Strapdown integration of the nominal state with bias-corrected IMU data."""
        R = quat_to_rot(self.q)
        a = R @ (accel - self.ba) + self.g
        self.p = self.p + self.v * dt + 0.5 * a * dt * dt
        self.v = self.v + a * dt
        self.q = quat_normalize(quat_mult(self.q, quat_exp((gyro - self.bg) * dt)))

    def propagate_covariance(self, gyro: Array, accel: Array, dt: float) -> None:
        """Propagate ``P`` over ``dt`` using the (averaged) IMU data of the interval."""
        R = quat_to_rot(self.q)
        F = np.zeros((N, N))
        F[SL_P, SL_V] = np.eye(3)
        F[SL_V, SL_TH] = -R @ hat(accel - self.ba)
        F[SL_V, SL_BA] = -R
        F[SL_TH, SL_TH] = -hat(gyro - self.bg)
        F[SL_TH, SL_BG] = -np.eye(3)
        Phi = np.eye(N) + F * dt + 0.5 * (F * dt) @ (F * dt)
        sp = self.sp
        Q = np.zeros((N, N))
        Q[SL_V, SL_V] = sp.accel_noise_density**2 * dt * np.eye(3)
        Q[SL_TH, SL_TH] = sp.gyro_noise_density**2 * dt * np.eye(3)
        Q[SL_BA, SL_BA] = sp.accel_bias_walk**2 * dt * np.eye(3)
        Q[SL_BG, SL_BG] = sp.gyro_bias_walk**2 * dt * np.eye(3)
        self.P = Phi @ self.P @ Phi.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    # ------------------------------------------------------------------ correction
    def _update(self, residual: Array, H: Array, R: Array) -> None:
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ residual
        I_KH = np.eye(N) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form
        # inject the error into the nominal state
        self.p = self.p + dx[SL_P]
        self.v = self.v + dx[SL_V]
        dth = dx[SL_TH]
        self.q = quat_normalize(quat_mult(self.q, quat_exp(dth)))
        self.ba = self.ba + dx[SL_BA]
        self.bg = self.bg + dx[SL_BG]
        # reset: the attitude error frame changed
        G = np.eye(N)
        G[SL_TH, SL_TH] = np.eye(3) - hat(0.5 * dth)
        self.P = G @ self.P @ G.T

    def update_gps(self, pos: Array, vel: Array) -> None:
        """Fuse a GPS position + velocity fix."""
        H = np.zeros((6, N))
        H[0:3, SL_P] = np.eye(3)
        H[3:6, SL_V] = np.eye(3)
        R = np.diag([*(np.array(self.sp.gps_pos_std) ** 2), *([self.sp.gps_vel_std**2] * 3)])
        self._update(np.concatenate([pos - self.p, vel - self.v]), H, R)

    def update_baro(self, z: float) -> None:
        """Fuse a barometric altitude."""
        H = np.zeros((1, N))
        H[0, 2] = 1.0
        self._update(np.array([z - self.p[2]]), H, np.array([[self.sp.baro_std**2]]))

    def update_yaw(self, yaw: float) -> None:
        """Fuse a heading measurement (magnetometer)."""
        R = quat_to_rot(self.q)
        D = R[0, 0] ** 2 + R[1, 0] ** 2
        H = np.zeros((1, N))
        H[0, 7] = (R[1, 0] * R[0, 2] - R[0, 0] * R[1, 2]) / D
        H[0, 8] = (R[0, 0] * R[1, 1] - R[1, 0] * R[0, 1]) / D
        res = wrap_angle(yaw - euler_from_quat(self.q)[2])
        self._update(np.array([res]), H, np.array([[self.sp.mag_yaw_std**2]]))

    @property
    def sigma(self) -> Array:
        """1-sigma of ``[dp, dv, dtheta]`` (9,)."""
        return np.sqrt(np.diag(self.P)[:9])
