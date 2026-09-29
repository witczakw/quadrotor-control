"""LQR with integral action on the hover-linearised model."""

from __future__ import annotations

import numpy as np

from quadrotor.control.base import Array, Controller, Estimate, Reference
from quadrotor.control.linear_common import (
    N_X,
    N_Z,
    BrysonSpec,
    augmented_model,
    bryson_weights,
    error_state,
    lqr_gain,
)
from quadrotor.params import QuadParams


class LQRController(Controller):
    """Full-state LQR ``u = -K [e; int e_p]`` around hover, plus reference feed-forward.

    The model is the hover linearisation (12 states: position, velocity, roll/pitch/yaw, body
    rates) augmented with the integral of the position error. The weights follow Bryson's rule
    (see :class:`~quadrotor.control.linear_common.BrysonSpec`): each state/input is penalised by the
    inverse square of its acceptable deviation, which gives interpretable and well-scaled Q and R.
    Tracking uses the error with respect to the reference (position, velocity, tilt from the
    reference acceleration) and the feed-forward thrust ``m |a_ref + g e3|``.
    """

    name = "LQR"

    def __init__(
        self, params: QuadParams, spec: BrysonSpec | None = None, dt: float = 0.004
    ) -> None:
        self.params = params
        self.dt = dt
        A, B = augmented_model(params)
        self.Q, self.R = bryson_weights(params, spec)
        self.K = lqr_gain(A, B, self.Q, self.R)
        self.reset()

    def reset(self) -> None:
        self.integral = np.zeros(3)

    def compute(self, t: float, est: Estimate, ref: Reference) -> tuple[float, Array]:
        prm = self.params
        e, f_ff = error_state(est, ref, prm.gravity)
        if np.linalg.norm(e[0:3]) < 0.5:  # conditional integration (anti-windup)
            self.integral = np.clip(self.integral + e[0:3] * self.dt, -0.6, 0.6)
        z = np.concatenate([e, self.integral])
        assert z.shape == (N_Z,) and N_X == 12
        u = -self.K @ z
        thrust = prm.mass * float(np.linalg.norm(f_ff)) + u[0]
        return thrust, u[1:]
