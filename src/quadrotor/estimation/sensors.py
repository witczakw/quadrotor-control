"""Simulated sensors: IMU (noise + bias random walk), GPS, barometer and magnetometer (yaw)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quadrotor.control.base import Array
from quadrotor.rotations import euler_from_quat, wrap_angle


@dataclass(frozen=True)
class SensorParams:
    """Noise levels. Densities are continuous-time (per sqrt(Hz)), the rest are discrete 1-sigma."""

    gyro_noise_density: float = 1.0e-3  # [rad/s/sqrt(Hz)]
    accel_noise_density: float = 1.0e-2  # [m/s^2/sqrt(Hz)]
    gyro_bias_walk: float = 2.0e-5  # [rad/s^2/sqrt(Hz)]
    accel_bias_walk: float = 2.0e-4  # [m/s^3/sqrt(Hz)]
    gyro_bias_init: float = 0.01  # 1-sigma of the initial bias [rad/s]
    accel_bias_init: float = 0.05  # [m/s^2]
    gps_pos_std: tuple[float, float, float] = (0.30, 0.30, 0.50)  # [m]
    gps_vel_std: float = 0.05  # [m/s]
    baro_std: float = 0.15  # [m]
    mag_yaw_std: float = 0.03  # [rad]
    gps_rate: float = 10.0  # [Hz]
    baro_rate: float = 50.0
    mag_rate: float = 25.0


class Imu:
    """Gyroscope and accelerometer: ``m = true + bias + white noise``, bias is a random walk."""

    def __init__(self, sp: SensorParams, dt: float, rng: np.random.Generator) -> None:
        self.sp, self.dt, self.rng = sp, dt, rng
        self.bias_g = sp.gyro_bias_init * rng.standard_normal(3)
        self.bias_a = sp.accel_bias_init * rng.standard_normal(3)

    def measure(self, omega: Array, specific_force: Array) -> tuple[Array, Array]:
        """Return ``(gyro, accel)`` readings; advances the bias random walk by one sample."""
        sp, dt, rng = self.sp, self.dt, self.rng
        self.bias_g = self.bias_g + sp.gyro_bias_walk * np.sqrt(dt) * rng.standard_normal(3)
        self.bias_a = self.bias_a + sp.accel_bias_walk * np.sqrt(dt) * rng.standard_normal(3)
        gyro = omega + self.bias_g + sp.gyro_noise_density / np.sqrt(dt) * rng.standard_normal(3)
        acc = (
            specific_force
            + self.bias_a
            + sp.accel_noise_density / np.sqrt(dt) * rng.standard_normal(3)
        )
        return gyro, acc


class Gps:
    """GPS receiver: position and velocity with white noise."""

    def __init__(self, sp: SensorParams, rng: np.random.Generator) -> None:
        self.sp, self.rng = sp, rng

    def measure(self, p: Array, v: Array) -> tuple[Array, Array]:
        pos = p + np.array(self.sp.gps_pos_std) * self.rng.standard_normal(3)
        vel = v + self.sp.gps_vel_std * self.rng.standard_normal(3)
        return pos, vel


class Barometer:
    """Barometric altitude with white noise."""

    def __init__(self, sp: SensorParams, rng: np.random.Generator) -> None:
        self.sp, self.rng = sp, rng

    def measure(self, z: float) -> float:
        return float(z + self.sp.baro_std * self.rng.standard_normal())


class Magnetometer:
    """Heading sensor (yaw angle), needed to make yaw observable."""

    def __init__(self, sp: SensorParams, rng: np.random.Generator) -> None:
        self.sp, self.rng = sp, rng

    def measure(self, q: Array) -> float:
        return wrap_angle(euler_from_quat(q)[2] + self.sp.mag_yaw_std * self.rng.standard_normal())
