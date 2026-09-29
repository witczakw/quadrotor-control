"""Wind model: constant mean wind plus Ornstein-Uhlenbeck gusts."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


class WindModel:
    """Wind velocity in the inertial frame: ``v_w = mean + gust`` with OU gust.

    The gust follows ``dg = -g/tau dt + sigma sqrt(2/tau) dW`` (stationary std ``sigma``), a
    first-order approximation of Dryden turbulence.

    Args:
        mean: constant wind velocity [m/s].
        gust_sigma: standard deviation of the gust per axis [m/s]; 0 disables gusts.
        gust_tau: correlation time [s].
        seed: random seed (reproducible experiments).
    """

    def __init__(
        self,
        mean: Array | None = None,
        gust_sigma: float = 0.0,
        gust_tau: float = 2.0,
        seed: int = 0,
    ) -> None:
        self.mean = np.zeros(3) if mean is None else np.asarray(mean, float)
        self.sigma = float(gust_sigma)
        self.tau = float(gust_tau)
        self.rng = np.random.default_rng(seed)
        self.gust = self.sigma * self.rng.standard_normal(3) if self.sigma > 0 else np.zeros(3)

    def step(self, dt: float) -> Array:
        """Advance the gust by ``dt`` (exact OU discretisation) and return the wind velocity."""
        if self.sigma > 0:
            a = np.exp(-dt / self.tau)
            noise = self.sigma * np.sqrt(1 - a * a) * self.rng.standard_normal(3)
            self.gust = a * self.gust + noise
        return self.mean + self.gust
