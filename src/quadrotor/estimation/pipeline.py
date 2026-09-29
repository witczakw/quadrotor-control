"""Glue between the plant simulation, the simulated sensors and the ESKF."""

from __future__ import annotations

import numpy as np

from quadrotor.control.base import Array, Estimate
from quadrotor.dynamics import IDX_P, IDX_Q, IDX_V, IDX_W, Quadrotor
from quadrotor.estimation.eskf import ESKF
from quadrotor.estimation.sensors import Barometer, Gps, Imu, Magnetometer, SensorParams
from quadrotor.rotations import quat_exp, quat_mult

PHYSICS_DT = 0.001
IMU_STEPS = 2  # IMU at 500 Hz
EKF_STEPS = 10  # covariance propagation and updates at 100 Hz


class ESKFEstimator:
    """Sensor suite + ESKF driven by the simulation loop.

    * IMU at 500 Hz drives the nominal-state strapdown integration (trapezoidal).
    * At 100 Hz the covariance is propagated with the averaged IMU data and the due measurements
      (GPS 10 Hz, barometer 50 Hz, magnetometer 25 Hz) are fused.
    * Controllers receive the position/velocity/attitude of the filter and the bias-corrected
      gyro of the latest IMU sample.
    """

    def __init__(self, sp: SensorParams | None = None, seed: int = 0) -> None:
        self.sp = sp or SensorParams()
        self.seed = seed
        self.reset(None)

    def reset(self, x0: Array | None) -> None:
        """Create sensors and initialise the filter around the true initial state."""
        self.rng = np.random.default_rng(self.seed)
        sp = self.sp
        self.imu = Imu(sp, PHYSICS_DT * IMU_STEPS, self.rng)
        self.gps = Gps(sp, self.rng)
        self.baro = Barometer(sp, self.rng)
        self.mag = Magnetometer(sp, self.rng)
        self.filter = ESKF(sp)
        self.prev_sample: tuple[Array, Array] | None = None
        self.buf_g = np.zeros(3)
        self.buf_a = np.zeros(3)
        self.n_buf = 0
        self.last_gyro = np.zeros(3)
        self.sigma = self.filter.sigma
        self.aux = np.zeros(12)
        if x0 is not None:
            p0 = x0[IDX_P] + 0.2 * self.rng.standard_normal(3)
            q0 = quat_mult(x0[IDX_Q], quat_exp(0.02 * self.rng.standard_normal(3)))
            self.filter.initialize(p0, x0[IDX_V].copy(), q0)
            self.sigma = self.filter.sigma

    def observe(
        self, x: Array, t: float, k: int, quad: Quadrotor, w_cmd: Array, v_wind: Array
    ) -> None:
        """Sample sensors at physics step ``k`` and run the filter when due."""
        if k % IMU_STEPS:
            return
        dt_imu = PHYSICS_DT * IMU_STEPS
        f = self.filter
        gyro, acc = self.imu.measure(x[IDX_W], quad.specific_force(x, w_cmd, v_wind))
        self.last_gyro = gyro
        if self.prev_sample is not None:
            g_avg = 0.5 * (gyro + self.prev_sample[0])
            a_avg = 0.5 * (acc + self.prev_sample[1])
            f.propagate_nominal(g_avg, a_avg, dt_imu)
            self.buf_g += g_avg
            self.buf_a += a_avg
            self.n_buf += 1
        self.prev_sample = (gyro, acc)

        if k % EKF_STEPS == 0 and self.n_buf > 0:
            f.propagate_covariance(
                self.buf_g / self.n_buf, self.buf_a / self.n_buf, self.n_buf * dt_imu
            )
            self.buf_g[:] = 0.0
            self.buf_a[:] = 0.0
            self.n_buf = 0
            sp = self.sp
            if k % round(1.0 / (sp.gps_rate * PHYSICS_DT)) == 0:
                f.update_gps(*self.gps.measure(x[IDX_P], x[IDX_V]))
            if k % round(1.0 / (sp.baro_rate * PHYSICS_DT)) == 0:
                f.update_baro(self.baro.measure(x[2]))
            if k % round(1.0 / (sp.mag_rate * PHYSICS_DT)) == 0:
                f.update_yaw(self.mag.measure(x[IDX_Q]))
            self.sigma = f.sigma
        self.aux = np.concatenate([f.ba, f.bg, self.imu.bias_a, self.imu.bias_g])

    def estimate(self, x: Array, **_: object) -> Estimate:
        """State estimate handed to the controller (never uses the true state)."""
        f = self.filter
        return Estimate(f.p.copy(), f.v.copy(), f.q.copy(), self.last_gyro - f.bg)
