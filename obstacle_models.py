"""Shared data contracts for depth-based obstacle tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

TrackerState = Literal["stopped", "running", "degraded"]


@dataclass(frozen=True)
class ObstacleTrackerConfig:
    """Runtime configuration for depth segmentation and tracking."""

    min_depth_m: float = 0.15
    max_depth_m: float = 6.0
    depth_downsample: int = 2
    median_blur_ksize: int = 5
    min_cluster_pixels: int = 120
    association_gate_m: float = 0.75
    stale_track_seconds: float = 0.9
    warning_ttc_s: float = 2.5
    urgent_ttc_s: float = 1.0
    max_tracks: int = 64
    enable_imu_compensation: bool = True


@dataclass(frozen=True)
class ObstacleDetection:
    """Single-frame obstacle estimate in camera coordinates (meters)."""

    centroid_xyz_m: np.ndarray
    radius_m: float
    pixel_count: int
    confidence: float


@dataclass(frozen=True)
class TrackedObstacle:
    """Temporally tracked obstacle with collision metadata."""

    obstacle_id: int
    xyz_m: np.ndarray
    velocity_mps: np.ndarray
    distance_m: float
    approach_rate_mps: float
    ttc_s: float
    collision_score: float
    confidence: float
    is_collision_course: bool
    age_frames: int
    last_seen_s: float


@dataclass(frozen=True)
class ObstacleUpdate:
    """Snapshot payload emitted to downstream consumers and audio systems."""

    timestamp_s: float
    frame_number: int
    state: TrackerState
    imu_compensated: bool
    obstacles: tuple[TrackedObstacle, ...] = field(default_factory=tuple)


ObstacleUpdateCallback = Callable[[ObstacleUpdate], None]
