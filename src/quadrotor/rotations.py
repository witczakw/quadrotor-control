"""Rotation utilities. Quaternions are stored as [w, x, y, z] (body -> inertial)."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def hat(v: Array) -> Array:
    """Skew-symmetric matrix such that hat(a) @ b = a x b."""
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def vee(M: Array) -> Array:
    """Inverse of :func:`hat`."""
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def quat_mult(a: Array, b: Array) -> Array:
    """Hamilton product a (x) b."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_conj(q: Array) -> Array:
    """Quaternion conjugate (inverse for unit quaternions)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q: Array) -> Array:
    """Normalise to unit length."""
    return q / np.linalg.norm(q)


def quat_to_rot(q: Array) -> Array:
    """Rotation matrix R (body -> inertial) of a unit quaternion."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_to_quat(R: Array) -> Array:
    """Unit quaternion (w >= 0) of a rotation matrix (Shepperd's method)."""
    tr = np.trace(R)
    if tr > 0.0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = np.array(
            [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        )
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = np.array(
            [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        )
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = np.array(
            [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        )
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = np.array(
            [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
        )
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def quat_exp(rotvec: Array) -> Array:
    """Quaternion of the rotation vector (axis * angle)."""
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return quat_normalize(np.array([1.0, *(0.5 * rotvec)]))
    axis = rotvec / angle
    return np.array([np.cos(angle / 2), *(np.sin(angle / 2) * axis)])


def quat_log(q: Array) -> Array:
    """Rotation vector of a unit quaternion (shortest rotation)."""
    if q[0] < 0:
        q = -q
    v = q[1:]
    n = np.linalg.norm(v)
    if n < 1e-12:
        return 2.0 * v
    return 2.0 * np.arctan2(n, q[0]) * v / n


def quat_from_euler(roll: float, pitch: float, yaw: float) -> Array:
    """ZYX (yaw-pitch-roll) Euler angles -> quaternion."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def euler_from_rot(R: Array) -> Array:
    """Rotation matrix -> (roll, pitch, yaw), ZYX convention."""
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def euler_from_quat(q: Array) -> Array:
    """Quaternion -> (roll, pitch, yaw), ZYX convention."""
    return euler_from_rot(quat_to_rot(q))


def wrap_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def rot_z(yaw: float) -> Array:
    """Rotation about the z axis."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
