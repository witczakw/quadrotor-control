"""Closed-loop multi-rate simulation (physics 1 kHz, control 250 Hz, EKF 100 Hz)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from quadrotor.control.base import Array, Controller, Estimate, Reference
from quadrotor.dynamics import IDX_P, IDX_Q, IDX_R, IDX_V, IDX_W, Quadrotor, hover_state
from quadrotor.params import QuadParams
from quadrotor.wind import WindModel

PHYSICS_DT = 0.001
CONTROL_DT = 0.004
EKF_DT = 0.01
IMU_DT = 0.002


@dataclass
class Fault:
    """Loss of effectiveness of one rotor starting at ``t_start``."""

    rotor: int = 0
    efficiency: float = 0.7
    t_start: float = 0.0


@dataclass
class SimResult:
    """Logged signals (sampled at the control rate)."""

    t: Array
    x: Array  # true state (N, 17)
    ref_p: Array
    ref_v: Array
    est_p: Array
    est_v: Array
    est_q: Array
    est_sigma: Array | None  # 1-sigma of estimated [p, v, att] (N, 9)
    w_cmd: Array
    power: Array  # shaft power [W]
    extra: dict[str, Array] = field(default_factory=dict)

    @property
    def pos_error(self) -> Array:
        """Norm of the true tracking error."""
        return np.linalg.norm(self.x[:, IDX_P] - self.ref_p, axis=1)


class TruthEstimator:
    """Ideal estimator: hands the true state to the controller."""

    sigma: Array | None = None

    def reset(self, x0: Array) -> None:
        """Nothing to initialise."""

    def observe(self, x: Array, t: float, k: int, quad: Quadrotor, w_cmd: Array, v_wind: Array):
        """Called every physics step before integration (sensors are not needed here)."""

    def estimate(self, x: Array, **_: object) -> Estimate:
        """Return the true state as the estimate."""
        return Estimate(x[IDX_P].copy(), x[IDX_V].copy(), x[IDX_Q].copy(), x[IDX_W].copy())


class Simulation:
    """Simulate a controller tracking a trajectory.

    Args:
        plant_params: true physical parameters (may differ from the controller's model).
        controller: controller instance (built with the *nominal* parameters).
        trajectory: function ``t -> Reference``.
        wind: wind model (None -> calm).
        fault: rotor fault (None -> healthy).
        estimator: object with ``reset(x0)`` and ``estimate(...)``; default ground truth. An
            estimator with the attribute ``is_ekf`` is driven with sensor data (see
            :mod:`quadrotor.estimation.pipeline`).
        x0: initial state (default hover at the first reference point).
    """

    def __init__(
        self,
        plant_params: QuadParams,
        controller: Controller,
        trajectory: Callable[[float], Reference],
        wind: WindModel | None = None,
        fault: Fault | None = None,
        estimator=None,
        x0: Array | None = None,
    ) -> None:
        self.params = plant_params
        self.controller = controller
        self.traj = trajectory
        self.wind = wind or WindModel()
        self.fault = fault
        self.estimator = estimator or TruthEstimator()
        self.x0 = hover_state(plant_params, trajectory(0.0).p) if x0 is None else x0.copy()

    def run(self, duration: float) -> SimResult:
        """Run the simulation for ``duration`` seconds."""
        n_ctrl = round(CONTROL_DT / PHYSICS_DT)
        n_steps = round(duration / PHYSICS_DT)
        quad = Quadrotor(self.params)
        ctrl = self.controller
        ctrl.dt = CONTROL_DT
        ctrl.reset()
        # controller-side mixer uses the nominal model
        nominal = getattr(ctrl, "params", self.params)
        mixer = Quadrotor(nominal)

        x = self.x0.copy()
        est = self.estimator
        est.reset(x)
        w_cmd = np.full(4, nominal.hover_speed)
        n_log = n_steps // n_ctrl
        log = {
            k: np.zeros((n_log, d))
            for k, d in [("x", 17), ("rp", 3), ("rv", 3), ("ep", 3), ("ev", 3), ("eq", 4), ("w", 4)]
        }
        sig = np.zeros((n_log, 9))
        power = np.zeros(n_log)
        t_log = np.zeros(n_log)
        v_wind = self.wind.mean.copy()
        i_log = 0
        for k in range(n_steps):
            t = k * PHYSICS_DT
            if self.fault is not None and t >= self.fault.t_start and k % n_ctrl == 0:
                eff = np.ones(4)
                eff[self.fault.rotor] = self.fault.efficiency
                quad.rotor_efficiency = eff
            if k % n_ctrl == 0:
                estimate = est.estimate(x, t=t, quad=quad, w_cmd=w_cmd, v_wind=v_wind)
                ref = self.traj(t)
                thrust, torque = ctrl.compute(t, estimate, ref)
                w_cmd = mixer.mix(thrust, torque)
                if i_log < n_log:
                    t_log[i_log] = t
                    log["x"][i_log] = x
                    log["rp"][i_log] = ref.p
                    log["rv"][i_log] = ref.v
                    log["ep"][i_log] = estimate.p
                    log["ev"][i_log] = estimate.v
                    log["eq"][i_log] = estimate.q
                    log["w"][i_log] = w_cmd
                    if est.sigma is not None:
                        sig[i_log] = est.sigma
                    power[i_log] = self.params.km * float(np.sum(x[IDX_R] ** 3))
                    i_log += 1
            v_wind = self.wind.step(PHYSICS_DT)
            est.observe(x, t, k, quad, w_cmd, v_wind)
            x = quad.step(x, w_cmd, PHYSICS_DT, v_wind)
        return SimResult(
            t=t_log,
            x=log["x"],
            ref_p=log["rp"],
            ref_v=log["rv"],
            est_p=log["ep"],
            est_v=log["ev"],
            est_q=log["eq"],
            est_sigma=sig if est.sigma is not None else None,
            w_cmd=log["w"],
            power=power,
        )
