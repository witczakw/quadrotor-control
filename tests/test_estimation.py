"""Tests of the sensors and the error-state Kalman filter."""

from __future__ import annotations

import numpy as np

from quadrotor.control.base import hover_reference
from quadrotor.control.geometric import GeometricController
from quadrotor.estimation.eskf import ESKF
from quadrotor.estimation.pipeline import ESKFEstimator
from quadrotor.estimation.sensors import SensorParams
from quadrotor.params import QuadParams
from quadrotor.planning.scenarios import figure_eight
from quadrotor.rotations import quat_conj, quat_log, quat_mult
from quadrotor.sim import Simulation


def test_strapdown_static_no_drift() -> None:
    """Perfect IMU at rest (accelerometer reads +g up): nominal state does not move."""
    f = ESKF(SensorParams())
    f.initialize(np.array([1.0, 2.0, 3.0]), np.zeros(3), np.array([1.0, 0, 0, 0]))
    for _ in range(500):
        f.propagate_nominal(np.zeros(3), np.array([0.0, 0.0, 9.81]), 0.002)
    assert np.allclose(f.p, [1.0, 2.0, 3.0], atol=1e-9)
    assert np.allclose(f.v, 0.0, atol=1e-9)


def test_covariance_grows_without_updates_and_shrinks_with_gps() -> None:
    f = ESKF(SensorParams())
    f.initialize(np.zeros(3), np.zeros(3), np.array([1.0, 0, 0, 0]))
    for _ in range(50):
        f.propagate_covariance(np.zeros(3), np.array([0.0, 0.0, 9.81]), 0.01)
    grown = f.P[0, 0]
    f.update_gps(np.zeros(3), np.zeros(3))
    assert f.P[0, 0] < grown


def test_filter_tracks_truth_in_flight() -> None:
    """Closed loop on the estimate along a figure-eight: accurate, consistent, biases found."""
    params = QuadParams()
    sc = figure_eight()
    est = ESKFEstimator(seed=0)
    res = Simulation(params, GeometricController(params), sc.trajectory, estimator=est).run(9.0)
    e_p = res.est_p - res.x[:, :3]
    e_v = res.est_v - res.x[:, 3:6]
    e_att = np.array(
        [quat_log(quat_mult(quat_conj(res.x[i, 6:10]), res.est_q[i])) for i in range(len(res.t))]
    )
    assert np.sqrt(np.mean(e_p**2)) < 0.12
    assert np.sqrt(np.mean(e_v**2)) < 0.06
    assert np.degrees(np.sqrt(np.mean(e_att**2))) < 1.0
    # consistency: errors stay inside the 3-sigma bounds
    assert np.mean(np.abs(e_p) < 3 * res.est_sigma[:, 0:3]) > 0.97
    assert np.mean(np.abs(e_v) < 3 * res.est_sigma[:, 3:6]) > 0.97
    assert np.mean(np.abs(e_att) < 3 * res.est_sigma[:, 6:9]) > 0.97
    # accelerometer / gyro bias estimates converge
    bias = res.extra["bias"][-1]
    assert np.all(np.abs(bias[0:3] - bias[6:9]) < 0.1)
    assert np.all(np.abs(bias[3:6] - bias[9:12]) < 0.005)


def test_hover_on_estimate() -> None:
    params = QuadParams()
    res = Simulation(
        params,
        GeometricController(params),
        lambda t: hover_reference(np.array([0.0, 0.0, 2.0])),
        estimator=ESKFEstimator(seed=5),
    ).run(4.0)
    assert np.max(res.pos_error) < 0.35
