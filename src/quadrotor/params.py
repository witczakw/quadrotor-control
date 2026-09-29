"""Physical parameters of the quadrotor (single source of truth)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "quad.yaml"


@dataclass(frozen=True)
class QuadParams:
    """Parameters of an X-configuration quadrotor.

    Frame conventions: inertial frame is ENU-like with +z up (gravity along -z), body frame is
    FLU (x forward, y left, z up). Rotor thrust acts along body +z.

    Rotor layout (top view, x forward, y left)::

        1 (CW)   0 (CCW)         rotor i at (x_i, y_i) = L/sqrt(2) * (+-1, +-1)
            \\   /
             body
            /   \\
        2 (CCW)  3 (CW)
    """

    mass: float = 1.2
    inertia: tuple[float, float, float] = (0.0123, 0.0123, 0.0224)
    arm_length: float = 0.225
    kf: float = 3.0e-6
    km: float = 1.1e-7
    tau_motor: float = 0.025
    w_min: float = 100.0
    w_max: float = 1500.0
    drag_coeff: float = 0.15
    gravity: float = 9.81
    # Spin direction of each rotor: +1 -> CCW seen from above (body reacts with -z torque).
    spin: tuple[int, int, int, int] = field(default=(1, -1, 1, -1))

    @property
    def J(self) -> np.ndarray:
        """Inertia tensor (3x3, diagonal)."""
        return np.diag(self.inertia)

    @property
    def rotor_positions(self) -> np.ndarray:
        """Rotor positions in the body frame, shape (4, 2) (x, y)."""
        d = self.arm_length / np.sqrt(2.0)
        return np.array([[d, -d], [d, d], [-d, d], [-d, -d]])

    @property
    def hover_thrust(self) -> float:
        """Total thrust needed to hover [N]."""
        return self.mass * self.gravity

    @property
    def hover_speed(self) -> float:
        """Rotor speed in hover [rad/s]."""
        return float(np.sqrt(self.hover_thrust / (4.0 * self.kf)))

    @property
    def max_total_thrust(self) -> float:
        """Maximum total thrust with all rotors saturated [N]."""
        return 4.0 * self.kf * self.w_max**2

    def scaled(self, mass_scale: float = 1.0) -> QuadParams:
        """Return a copy with mass (and inertia, proportionally) scaled - model mismatch."""
        return dataclasses.replace(
            self,
            mass=self.mass * mass_scale,
            inertia=tuple(i * mass_scale for i in self.inertia),  # type: ignore[arg-type]
        )

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG) -> QuadParams:
        """Load parameters from a YAML file (missing keys keep their defaults)."""
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        if "inertia" in data:
            data["inertia"] = tuple(float(i) for i in data["inertia"])
        if "spin" in data:
            data["spin"] = tuple(int(s) for s in data["spin"])
        return cls(**data)
