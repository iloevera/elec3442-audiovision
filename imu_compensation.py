"""Lightweight IMU helpers for camera motion compensation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from realsense_driver import IMUSample


@dataclass(frozen=True)
class EgoMotionEstimate:
    """Approximate camera-frame ego motion over a short interval."""

    delta_rotation_rad: np.ndarray
    confidence: float


def estimate_gyro_rotation(imu_samples: Iterable[IMUSample], dt_s: float) -> EgoMotionEstimate:
    """Estimate short-horizon camera rotation from gyro samples.

    The estimate is deliberately simple: average gyroscope angular velocity,
    then multiply by frame interval for a small-angle rotation increment.
    """

    gyro_vectors: list[np.ndarray] = []
    for sample in imu_samples:
        if sample.stream_name == "gyro":
            gyro_vectors.append(sample.xyz.astype(np.float64, copy=False))

    if not gyro_vectors or dt_s <= 0.0:
        return EgoMotionEstimate(delta_rotation_rad=np.zeros(3, dtype=np.float64), confidence=0.0)

    avg_omega = np.mean(np.stack(gyro_vectors, axis=0), axis=0)
    delta_rotation = avg_omega * float(dt_s)

    # Confidence drops with too few samples and unrealistically high angular rate.
    sample_factor = min(1.0, len(gyro_vectors) / 6.0)
    omega_norm = float(np.linalg.norm(avg_omega))
    rate_penalty = math.exp(-max(0.0, omega_norm - 4.0))
    confidence = float(max(0.0, min(1.0, sample_factor * rate_penalty)))

    return EgoMotionEstimate(delta_rotation_rad=delta_rotation, confidence=confidence)


def apply_small_angle_rotation(points_xyz: np.ndarray, delta_rotation_rad: np.ndarray) -> np.ndarray:
    """Rotate Nx3 points using first-order small-angle approximation."""

    if points_xyz.size == 0:
        return points_xyz

    rx, ry, rz = delta_rotation_rad.astype(np.float64, copy=False)
    skew = np.array(
        [
            [0.0, -rz, ry],
            [rz, 0.0, -rx],
            [-ry, rx, 0.0],
        ],
        dtype=np.float64,
    )
    rotation_approx = np.eye(3, dtype=np.float64) + skew
    return points_xyz @ rotation_approx.T
