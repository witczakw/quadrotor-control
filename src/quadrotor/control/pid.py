"""Baseline cascaded PID controller: position -> velocity -> attitude angles -> body rates."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quadrotor.control.base import Array, Controller, Estimate, Reference, clamp_norm
from quadrotor.params import QuadParams
from quadrotor.rotations import euler_from_quat, quat_to_rot, wrap_angle


@dataclass
class PIDGains:
    """Gains of the cascade (per axis where a 3-vector is given, order x, y, z)."""

    kp_pos: Array = field(default_factory=lambda: np.array([2.0, 2.0, 3.0]))
    kp_vel: Array = field(default_factory=lambda: np.array([4.5, 4.5, 6.0]))
    ki_vel: Array = field(default_factory=lambda: np.array([2.0, 2.0, 4.0]))
    kd_vel: Array = field(default_factory=lambda: np.zeros(3))
    kp_att: Array = field(default_factory=lambda: np.array([9.0, 9.0, 4.0]))
    kp_rate: Array = field(default_factory=lambda: np.array([28.0, 28.0, 12.0]))
    ki_rate: Array = field(default_factory=lambda: np.array([25.0, 25.0, 8.0]))
    kd_rate: Array = field(default_factory=lambda: np.array([0.35, 0.35, 0.0]))
    v_max: float = 8.0  # [m/s]
    tilt_max: float = np.deg2rad(45.0)
    int_vel_max: float = 4.0  # [m/s^2] integrator clamp (acceleration units)
    int_zone: float = 0.5  # integrate velocity error only within this position error [m]
    int_rate_max: float = 40.0  # [rad/s^2]
    rate_max: float = 12.0  # [rad/s]


class CascadedPID(Controller):
    """Cascaded PID with anti-windup (integrator clamping + conditional integration).

    Outer loops (position P, velocity PID) produce a desired acceleration and hence a desired
    thrust vector. Together with the desired yaw it defines a desired attitude, converted to
    roll/pitch/yaw errors (P) that yield body-rate commands; the rate PID outputs angular
    accelerations scaled by the nominal inertia to obtain torques.
    """

    name = "PID"

    def __init__(self, params: QuadParams, gains: PIDGains | None = None, dt: float = 0.004):
        self.params = params
        self.g = gains or PIDGains()
        self.dt = dt
        self.reset()

    def reset(self) -> None:
        self.i_vel = np.zeros(3)
        self.i_rate = np.zeros(3)
        self.prev_omega: Array | None = None
        self.d_filt = np.zeros(3)

    def compute(self, t: float, est: Estimate, ref: Reference) -> tuple[float, Array]:
        prm, g, dt = self.params, self.g, self.dt

        # --- position -> velocity
        v_cmd = clamp_norm(ref.v + g.kp_pos * (ref.p - est.p), g.v_max)
        # --- velocity PID -> acceleration (anti-windup: freeze integrator when clamped)
        e_v = v_cmd - est.v
        if np.linalg.norm(ref.p - est.p) < g.int_zone:
            self.i_vel = np.clip(self.i_vel + g.ki_vel * e_v * dt, -g.int_vel_max, g.int_vel_max)
        a_des = ref.a + g.kp_vel * e_v + self.i_vel
        f_vec = prm.mass * (a_des + np.array([0.0, 0.0, prm.gravity]))

        # --- desired thrust direction with tilt limit
        f_norm = np.linalg.norm(f_vec)
        b3_des = f_vec / max(f_norm, 1e-6)
        tilt = np.arccos(np.clip(b3_des[2], -1.0, 1.0))
        if tilt > g.tilt_max:
            horiz = b3_des[:2] / max(np.linalg.norm(b3_des[:2]), 1e-9)
            b3_des = np.array(
                [*(np.sin(g.tilt_max) * horiz), np.cos(g.tilt_max)],
            )
            # keep vertical thrust component demanded by the outer loop
            f_norm = f_vec[2] / max(b3_des[2], 1e-3)
        R = quat_to_rot(est.q)
        thrust = max(float(f_norm * (b3_des @ R[:, 2])), 0.0)

        # --- desired attitude -> Euler angle errors
        c_yaw = np.array([np.cos(ref.yaw), np.sin(ref.yaw), 0.0])
        b2 = np.cross(b3_des, c_yaw)
        b2 /= max(np.linalg.norm(b2), 1e-9)
        b1 = np.cross(b2, b3_des)
        roll_d = np.arctan2(b2[2], b3_des[2])
        pitch_d = np.arcsin(np.clip(-b1[2], -1.0, 1.0))
        eul = euler_from_quat(est.q)
        e_att = np.array(
            [
                wrap_angle(roll_d - eul[0]),
                wrap_angle(pitch_d - eul[1]),
                wrap_angle(ref.yaw - eul[2]),
            ]
        )
        # Euler-rate command -> body rate command
        eul_rate_cmd = g.kp_att * e_att + np.array([0.0, 0.0, ref.yaw_rate])
        sr, cr = np.sin(eul[0]), np.cos(eul[0])
        sp, cp = np.sin(eul[1]), np.cos(eul[1])
        omega_cmd = np.array(
            [
                eul_rate_cmd[0] - sp * eul_rate_cmd[2],
                cr * eul_rate_cmd[1] + sr * cp * eul_rate_cmd[2],
                -sr * eul_rate_cmd[1] + cr * cp * eul_rate_cmd[2],
            ]
        )
        omega_cmd = np.clip(omega_cmd, -g.rate_max, g.rate_max)

        # --- rate PID (D on measurement, low-pass filtered) with anti-windup
        e_w = omega_cmd - est.omega
        self.i_rate = np.clip(self.i_rate + g.ki_rate * e_w * dt, -g.int_rate_max, g.int_rate_max)
        if self.prev_omega is None:
            self.prev_omega = est.omega.copy()
        d_omega = (est.omega - self.prev_omega) / dt
        self.prev_omega = est.omega.copy()
        self.d_filt += 0.3 * (d_omega - self.d_filt)
        alpha = g.kp_rate * e_w + self.i_rate - g.kd_rate * self.d_filt
        torque = prm.J @ alpha + np.cross(est.omega, prm.J @ est.omega)
        return thrust, torque
