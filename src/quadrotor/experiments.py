"""Experiment definitions: controller factory, disturbance cases, metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quadrotor.control.base import Controller, hover_reference
from quadrotor.control.geometric import GeometricController
from quadrotor.control.lqr import LQRController
from quadrotor.control.mpc import LinearMPC
from quadrotor.control.pid import CascadedPID
from quadrotor.estimation.pipeline import ESKFEstimator
from quadrotor.params import QuadParams
from quadrotor.planning.scenarios import SCENARIOS, Scenario
from quadrotor.sim import Fault, SimResult, Simulation
from quadrotor.wind import WindModel

CONTROLLERS = ["PID", "Geometric", "LQR", "MPC"]
CASES = ["nominal", "wind", "mass+20%", "mass-20%", "rotor_fault_30%"]
FAIL_ERROR = 1.0  # [m] tracking RMSE above which a run counts as failed


def make_controller(name: str, params: QuadParams, scenario: Scenario | None = None) -> Controller:
    """Build a controller with the *nominal* parameters."""
    if name == "PID":
        return CascadedPID(params)
    if name == "Geometric":
        return GeometricController(params)
    if name == "LQR":
        return LQRController(params)
    if name == "MPC":
        return LinearMPC(params, preview=scenario.trajectory if scenario else None)
    raise ValueError(name)


def make_case(case: str, params: QuadParams) -> tuple[QuadParams, WindModel | None, Fault | None]:
    """Plant parameters, wind model and fault of a robustness case."""
    if case == "nominal":
        return params, None, None
    if case == "wind":
        # 5 m/s mean crosswind with 1.5 m/s OU gusts
        return params, WindModel(np.array([4.0, -3.0, 0.0]), gust_sigma=1.5, seed=7), None
    if case == "mass+20%":
        return params.scaled(1.2), None, None
    if case == "mass-20%":
        return params.scaled(0.8), None, None
    if case == "rotor_fault_30%":
        return params, None, Fault(rotor=0, efficiency=0.7, t_start=1.0)
    raise ValueError(case)


@dataclass
class RunSummary:
    """Metrics of one simulation run."""

    scenario: str
    controller: str
    case: str
    estimator: str
    rmse: float
    max_err: float
    energy: float
    final_err: float
    gate_miss: float
    failed: bool

    def as_row(self) -> dict[str, object]:
        return {
            "scenario": self.scenario,
            "controller": self.controller,
            "case": self.case,
            "estimator": self.estimator,
            "rmse_pos_m": round(self.rmse, 4),
            "max_err_m": round(self.max_err, 4),
            "energy_J": round(self.energy, 1),
            "final_err_m": round(self.final_err, 4),
            "gate_miss_m": round(self.gate_miss, 3) if np.isfinite(self.gate_miss) else "",
            "failed": self.failed,
        }


def run_case(
    scenario_name: str,
    controller: str,
    case: str,
    estimator: str = "ekf",
    seed: int = 1,
    keep_result: bool = False,
) -> tuple[RunSummary, SimResult | None]:
    """Simulate one (scenario, controller, disturbance case) with the ESKF in the loop."""
    params = QuadParams.from_yaml()
    sc = SCENARIOS[scenario_name]()
    plant, wind, fault = make_case(case, params)
    sim = Simulation(
        plant,
        make_controller(controller, params, sc),
        sc.trajectory,
        wind=wind,
        fault=fault,
        estimator=ESKFEstimator(seed=seed) if estimator == "ekf" else None,
    )
    res = sim.run(sc.duration)
    summary = summarize(res, sc, controller, case, estimator)
    return summary, res if keep_result else None


def summarize(
    res: SimResult, sc: Scenario, controller: str, case: str, estimator: str
) -> RunSummary:
    """Compute tracking RMSE, maximum error, energy and gate miss distance."""
    mask = res.t <= sc.trajectory.duration
    err = res.pos_error
    dt = float(res.t[1] - res.t[0])
    energy = float(np.sum(res.power) * dt)
    miss = np.nan
    if sc.gates:
        pos = res.x[:, :3]
        miss = max(float(np.min(np.linalg.norm(pos - g, axis=1))) for g in sc.gates)
    rmse = float(np.sqrt(np.mean(err[mask] ** 2)))
    bad = (not np.isfinite(rmse)) or rmse > FAIL_ERROR
    return RunSummary(
        sc.name,
        controller,
        case,
        estimator,
        rmse,
        float(np.max(err[mask])),
        energy,
        float(err[-1]),
        miss,
        bad,
    )


def step_response(controller: str, step: np.ndarray | None = None) -> dict[str, float]:
    """1 m diagonal position step (ground-truth feedback): settling time (5 cm) and overshoot."""
    params = QuadParams.from_yaml()
    target = np.array([1.0, 1.0, 1.0]) if step is None else step
    t_step = 0.5
    sim = Simulation(
        params,
        make_controller(controller, params),
        lambda t: hover_reference(target if t >= t_step else np.zeros(3)),
    )
    res = sim.run(8.0)
    err = np.linalg.norm(res.x[:, :3] - target, axis=1)
    outside = np.where((err > 0.05) & (res.t >= t_step))[0]
    settle = float(res.t[outside[-1]] - t_step) if len(outside) else 0.0
    direction = target / np.linalg.norm(target)
    overshoot = float(max(0.0, np.max((res.x[:, :3] - target) @ direction)))
    return {
        "controller": controller,
        "settling_time_s": round(settle, 2),
        "overshoot_m": round(overshoot, 3),
        "energy_J": round(float(np.sum(res.power) * (res.t[1] - res.t[0])), 1),
    }
