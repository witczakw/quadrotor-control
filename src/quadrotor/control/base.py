"""Common controller interfaces and data containers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

Array = np.ndarray


@dataclass
class Reference:
    """Desired trajectory sample (inertial frame) with feed-forward derivatives."""

    p: Array
    v: Array = field(default_factory=lambda: np.zeros(3))
    a: Array = field(default_factory=lambda: np.zeros(3))
    j: Array = field(default_factory=lambda: np.zeros(3))
    s: Array = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0
    yaw_rate: float = 0.0
    yaw_acc: float = 0.0


@dataclass
class Estimate:
    """State estimate consumed by the controllers.

    Attributes:
        p, v: position and velocity, inertial frame.
        q: attitude quaternion [w, x, y, z] (body -> inertial).
        omega: angular velocity in the body frame (bias-corrected gyro).
    """

    p: Array
    v: Array
    q: Array
    omega: Array


def hover_reference(p: Array, yaw: float = 0.0) -> Reference:
    """Reference for holding a position."""
    return Reference(p=np.asarray(p, float), yaw=yaw)


class Controller(ABC):
    """Maps an estimated state and a reference to total thrust [N] and body torques [N m]."""

    name: str = "controller"
    #: control period [s]
    dt: float = 0.004

    def reset(self) -> None:  # noqa: B027 - optional hook
        """Reset internal states (integrators, filters)."""

    @abstractmethod
    def compute(self, t: float, est: Estimate, ref: Reference) -> tuple[float, Array]:
        """Return ``(thrust, torque_body)``."""


def clamp_norm(v: Array, limit: float) -> Array:
    """Scale ``v`` down so that its norm does not exceed ``limit``."""
    n = float(np.linalg.norm(v))
    return v if n <= limit else v * (limit / n)
