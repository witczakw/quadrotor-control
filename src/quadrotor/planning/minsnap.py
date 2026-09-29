"""Minimum-snap trajectory through waypoints (Mellinger & Kumar, ICRA 2011).

Each axis is a piecewise polynomial of order 7 (8 coefficients per segment). Coefficients are
found by minimising the integral of the squared snap ``int (d^4 p / dt^4)^2 dt`` subject to
waypoint interpolation and continuity of velocity, acceleration and jerk between segments. This
is an equality-constrained QP solved exactly through its KKT system. Polynomials are expressed
in the normalised segment time ``s = (t - t_i) / T_i`` for good numerical conditioning.
"""

from __future__ import annotations

from math import factorial

import numpy as np

from quadrotor.control.base import Array, Reference

ORDER = 7
N_COEF = ORDER + 1
N_CONT = 3  # derivatives (1..3) forced to be continuous / prescribed at the ends


def _basis(s: float, n: int) -> Array:
    """Row vector of the n-th derivative of the monomials ``[1, s, ..., s^7]`` at ``s``."""
    b = np.zeros(N_COEF)
    for k in range(n, N_COEF):
        b[k] = factorial(k) / factorial(k - n) * s ** (k - n)
    return b


def _snap_cost(T: float) -> Array:
    """Hessian of ``int_0^T (p'''')^2 dt`` w.r.t. the normalised-time coefficients."""
    Q = np.zeros((N_COEF, N_COEF))
    for k in range(4, N_COEF):
        for m in range(4, N_COEF):
            ck = factorial(k) / factorial(k - 4)
            cm = factorial(m) / factorial(m - 4)
            Q[k, m] = ck * cm / (k + m - 7)
    return Q / T**7


def allocate_times(waypoints: Array, v_avg: float, t_min: float = 0.4) -> Array:
    """Segment durations proportional to the segment length (constant average speed)."""
    dist = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    return np.maximum(dist / v_avg, t_min)


class MinSnapTrajectory:
    """Rest-to-rest minimum-snap trajectory through 3D waypoints.

    Args:
        waypoints: array (M+1, 3).
        times: segment durations (M,).
        yaw: optional function ``t -> (yaw, yaw_rate, yaw_acc)``.
        start_derivs / end_derivs: velocity, acceleration, jerk at the first / last waypoint,
            each of shape (3, 3) as ``[[vx, vy, vz], [ax, ...], [jx, ...]]``; default zero.
    """

    def __init__(
        self,
        waypoints: Array,
        times: Array,
        yaw=None,
        start_derivs: Array | None = None,
        end_derivs: Array | None = None,
    ) -> None:
        self.waypoints = np.asarray(waypoints, float)
        self.times = np.asarray(times, float)
        if len(self.times) != len(self.waypoints) - 1:
            raise ValueError("need one duration per segment")
        self.yaw = yaw
        self.knots = np.concatenate([[0.0], np.cumsum(self.times)])
        self.duration = float(self.knots[-1])
        sd = np.zeros((N_CONT, 3)) if start_derivs is None else np.asarray(start_derivs, float)
        ed = np.zeros((N_CONT, 3)) if end_derivs is None else np.asarray(end_derivs, float)
        self.coef = np.stack(
            [self._solve_axis(self.waypoints[:, a], sd[:, a], ed[:, a]) for a in range(3)]
        )  # (3, M, 8)

    # ------------------------------------------------------------------ QP
    def _constraints(self, w: Array, sd: Array, ed: Array) -> tuple[Array, Array]:
        M = len(self.times)
        T = self.times
        rows, rhs = [], []

        def row(seg: int, s: float, n: int, scale: float = 1.0) -> Array:
            r = np.zeros(N_COEF * M)
            r[seg * N_COEF : (seg + 1) * N_COEF] = _basis(s, n) * scale
            return r

        for i in range(M):  # interpolation of both segment ends
            rows += [row(i, 0.0, 0), row(i, 1.0, 0)]
            rhs += [w[i], w[i + 1]]
        for n in range(1, N_CONT + 1):  # boundary derivatives
            rows.append(row(0, 0.0, n, T[0] ** -n))
            rhs.append(sd[n - 1])
            rows.append(row(M - 1, 1.0, n, T[-1] ** -n))
            rhs.append(ed[n - 1])
        for i in range(M - 1):  # continuity of derivatives 1..3 at interior knots
            for n in range(1, N_CONT + 1):
                rows.append(row(i, 1.0, n, T[i] ** -n) - row(i + 1, 0.0, n, T[i + 1] ** -n))
                rhs.append(0.0)
        return np.array(rows), np.array(rhs)

    def _solve_axis(self, w: Array, sd: Array, ed: Array) -> Array:
        M = len(self.times)
        n = N_COEF * M
        H = np.zeros((n, n))
        for i, T in enumerate(self.times):
            H[i * N_COEF : (i + 1) * N_COEF, i * N_COEF : (i + 1) * N_COEF] = 2 * _snap_cost(T)
        A, b = self._constraints(w, sd, ed)
        kkt = np.block([[H, A.T], [A, np.zeros((len(b), len(b)))]])
        sol = np.linalg.solve(kkt, np.concatenate([np.zeros(n), b]))
        return sol[:n].reshape(M, N_COEF)

    # ------------------------------------------------------------------ evaluation
    def derivative(self, t: float, n: int) -> Array:
        """n-th time derivative of the position (3,) at time ``t`` (clamped to the horizon)."""
        t = float(np.clip(t, 0.0, self.duration))
        i = int(np.clip(np.searchsorted(self.knots, t, side="right") - 1, 0, len(self.times) - 1))
        T = self.times[i]
        basis = _basis((t - self.knots[i]) / T, n) * T**-n
        return self.coef[:, i, :] @ basis

    def __call__(self, t: float) -> Reference:
        """Reference (position ... snap and yaw) at time ``t``; holds the end point afterwards."""
        if t >= self.duration:
            ref = Reference(p=self.derivative(self.duration, 0))
        else:
            ref = Reference(
                p=self.derivative(t, 0),
                v=self.derivative(t, 1),
                a=self.derivative(t, 2),
                j=self.derivative(t, 3),
                s=self.derivative(t, 4),
            )
        if self.yaw is not None:
            ref.yaw, ref.yaw_rate, ref.yaw_acc = self.yaw(t)
        return ref

    def sample(self, dt: float = 0.01) -> tuple[Array, Array]:
        """Time vector and positions (N, 3) sampled uniformly."""
        t = np.arange(0.0, self.duration + 1e-9, dt)
        return t, np.array([self.derivative(ti, 0) for ti in t])

    def max_derivative(self, n: int, dt: float = 0.01) -> float:
        """Maximum norm of the n-th derivative over the trajectory."""
        ts = np.arange(0.0, self.duration, dt)
        return float(max(np.linalg.norm(self.derivative(t, n)) for t in ts))

    def scaled_in_time(self, factor: float) -> MinSnapTrajectory:
        """Same path flown ``factor`` times slower (velocities /factor, accelerations /factor^2)."""
        return MinSnapTrajectory(
            self.waypoints,
            self.times * factor,
            self.yaw,
        )
