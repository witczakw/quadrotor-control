"""Linear MPC (cvxpy / OSQP) for trajectory tracking with thrust, torque and tilt limits."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy.linalg import expm, solve_discrete_are

from quadrotor.control.base import Array, Controller, Estimate, Reference
from quadrotor.control.linear_common import (
    N_Z,
    BrysonSpec,
    augmented_model,
    bryson_weights,
    error_state,
    lqr_gain,
    reference_tilt,
)
from quadrotor.params import QuadParams
from quadrotor.rotations import rot_z, wrap_angle


@dataclass
class MPCConfig:
    """MPC settings."""

    horizon: int = 20  # steps
    dt: float = 0.02  # prediction step / MPC update period [s] (50 Hz)
    tilt_max: float = np.deg2rad(40.0)  # roll / pitch limit
    thrust_min: float = 2.0  # [N]
    thrust_max: float = 24.0  # [N]
    tau_xy_max: float = 1.0  # [N m]
    tau_z_max: float = 0.2  # [N m]


def _discretize(A: Array, B: Array, dt: float) -> tuple[Array, Array]:
    """Exact zero-order-hold discretisation."""
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * dt)
    return Md[:n, :n], Md[:n, n:]


class LinearMPC(Controller):
    """Receding-horizon tracking on the hover-linearised model (with integral state).

    At every MPC update (50 Hz) the reference over the horizon is previewed and the QP::

        min  sum_k  e_k' Q e_k + v_k' R v_k  +  e_N' P e_N
        s.t. e_{k+1} = A e_k + B v_k + d_k,   e_0 = current error,
             T_min <= T_ref,k + v_T,k <= T_max,   |tau_ref,k + v_tau,k| <= tau_max,
             |angle_ref,k + e_angle,k| <= tilt_max

    is solved, where ``e = x - x_ref``, ``v = u - u_ref`` and ``d_k`` is the mismatch between the
    reference and the linear model. ``Q``, ``R`` are the Bryson weights shared with LQR, ``P`` is
    the DARE solution (terminal cost). The first input is applied (zero-order hold) until the next
    update. The horizon is limited to 20 steps to keep the problem fast.

    Args:
        params: nominal parameters.
        preview: function ``t -> Reference`` used to look ahead along the trajectory (if None the
            current reference is assumed constant over the horizon).
    """

    name = "MPC"

    def __init__(
        self,
        params: QuadParams,
        config: MPCConfig | None = None,
        spec: BrysonSpec | None = None,
        preview: Callable[[float], Reference] | None = None,
        dt: float = 0.004,
    ) -> None:
        self.params = params
        self.cfg = config or MPCConfig()
        self.preview = preview
        self.dt = dt
        A, B = augmented_model(params)
        self.Ad, self.Bd = _discretize(A, B, self.cfg.dt)
        Q, R = bryson_weights(params, spec)
        self.Q, self.R = Q, R
        self.P = solve_discrete_are(self.Ad, self.Bd, Q * self.cfg.dt, R * self.cfg.dt)
        self._lqr_fallback = lqr_gain(A, B, Q, R)
        self._build_problem()
        self.reset()

    def reset(self) -> None:
        self.integral = np.zeros(3)
        self.u_hold = np.zeros(4)
        self.thrust_hold: float | None = None
        self.next_update = 0.0
        self.last_solve_ok = True
        self.n_fail = 0

    # ------------------------------------------------------------------ QP
    def _build_problem(self) -> None:
        N = self.cfg.horizon
        c = self.cfg
        self.e0 = cp.Parameter(N_Z)
        self.d = cp.Parameter((N_Z, N))
        self.u_ref = cp.Parameter((4, N))
        self.ang_ref = cp.Parameter((2, N + 1))  # absolute roll/pitch reference
        e = cp.Variable((N_Z, N + 1))
        v = cp.Variable((4, N))
        slack = cp.Variable((2, N + 1), nonneg=True)  # soft tilt limit (keeps the QP feasible)
        cost = 0
        Qh = np.linalg.cholesky(self.Q * c.dt).T
        Rh = np.linalg.cholesky(self.R * c.dt).T
        Ph = np.linalg.cholesky(self.P + 1e-9 * np.eye(N_Z)).T
        cost += (
            cp.sum_squares(Qh @ e[:, :N]) + cp.sum_squares(Rh @ v) + cp.sum_squares(Ph @ e[:, N])
        )
        cons = [e[:, 0] == self.e0, e[:, 1:] == self.Ad @ e[:, :N] + self.Bd @ v + self.d]
        u_abs = self.u_ref + v
        m, g = self.params.mass, self.params.gravity
        cons += [
            u_abs[0, :] + m * g >= c.thrust_min,
            u_abs[0, :] + m * g <= c.thrust_max,
            cp.abs(u_abs[1:3, :]) <= c.tau_xy_max,
            cp.abs(u_abs[3, :]) <= c.tau_z_max,
            cp.abs(self.ang_ref + e[6:8, :]) <= c.tilt_max + slack,
        ]
        self.problem = cp.Problem(cp.Minimize(cost), cons)
        self.e_var, self.v_var = e, v

    def _reference_horizon(
        self, t: float, ref0: Reference, yaw0: float
    ) -> tuple[Array, Array, Array]:
        """Preview: reference state (N+1, 15-vector without integral), u_ref and tilt limits."""
        prm, c = self.params, self.cfg
        N = c.horizon
        Rz = rot_z(yaw0).T
        xs = np.zeros((N + 1, N_Z))
        f_ff = np.zeros((N + 1, 3))
        for k in range(N + 1):
            r = self.preview(t + k * c.dt) if self.preview is not None else ref0
            phi, theta = reference_tilt(r.a, yaw0, prm.gravity)
            xs[k, 0:3] = Rz @ (r.p - ref0.p)
            xs[k, 3:6] = Rz @ r.v
            xs[k, 6:9] = [phi, theta, wrap_angle(r.yaw - yaw0)]
            xs[k, 11] = r.yaw_rate
            f_ff[k] = r.a + np.array([0.0, 0.0, prm.gravity])
        # reference body rates from finite differences of the reference angles
        xs[:-1, 9:11] = np.diff(xs[:, 6:8], axis=0) / c.dt
        xs[-1, 9:11] = xs[-2, 9:11]
        u_ref = np.zeros((N, 4))
        u_ref[:, 0] = prm.mass * np.linalg.norm(f_ff[:N], axis=1) - prm.mass * prm.gravity
        alpha = np.diff(xs[:, 9:12], axis=0) / c.dt
        u_ref[:, 1:] = alpha @ prm.J.T
        return xs, u_ref, f_ff

    def _solve(self, t: float, est: Estimate, ref: Reference) -> np.ndarray | None:
        prm, c = self.params, self.cfg
        N = c.horizon
        e12, _ = error_state(est, ref, prm.gravity)
        yaw0 = float(
            np.arctan2(
                2 * (est.q[0] * est.q[3] + est.q[1] * est.q[2]),
                1 - 2 * (est.q[2] ** 2 + est.q[3] ** 2),
            )
        )
        xs, u_ref, _ = self._reference_horizon(t, ref, yaw0)
        # error dynamics residual d_k = A x_ref,k + B u_ref,k - x_ref,k+1 (integral of the
        # reference position error is zero by definition, ref state z-part is 0)
        xs[:, 12:15] = 0.0
        # integral-state ref is 0: e = x - x_ref; reference position relative to ref0 so shift
        d = np.zeros((N_Z, N))
        for k in range(N):
            d[:, k] = self.Ad @ xs[k] + self.Bd @ u_ref[k] - xs[k + 1]
        self.e0.value = np.concatenate([e12, self.integral])
        self.d.value = d
        self.u_ref.value = u_ref.T
        eul_ref = xs[:, 6:8].T
        self.ang_ref.value = eul_ref
        try:
            self.problem.solve(
                solver=cp.OSQP,
                warm_start=True,
                eps_abs=1e-4,
                eps_rel=1e-4,
                max_iter=4000,
                verbose=False,
            )
        except cp.error.SolverError:
            return None
        if self.problem.status not in ("optimal", "optimal_inaccurate") or self.v_var.value is None:
            return None
        return u_ref[0] + self.v_var.value[:, 0]

    def compute(self, t: float, est: Estimate, ref: Reference) -> tuple[float, Array]:
        prm = self.params
        e12, f_ff = error_state(est, ref, prm.gravity)
        if np.linalg.norm(e12[0:3]) < 0.5:
            self.integral = np.clip(self.integral + e12[0:3] * self.dt, -0.6, 0.6)
        if t >= self.next_update - 1e-9:
            u = self._solve(t, est, ref)
            self.next_update = t + self.cfg.dt
            if u is None:  # solver failure: fall back to the LQR feedback
                self.n_fail += 1
                z = np.concatenate([e12, self.integral])
                u = -self._lqr_fallback @ z
                u[0] += prm.mass * float(np.linalg.norm(f_ff)) - prm.mass * prm.gravity
            self.u_hold = u
        u = self.u_hold
        return prm.mass * prm.gravity + float(u[0]), u[1:].copy()
