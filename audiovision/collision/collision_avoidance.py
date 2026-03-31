"""Collision avoidance layer.

Maps tracked objects to collision risk levels and generates the associated
audio cue parameters (pitch, volume, urgency).

Collision-risk model
--------------------
For each tracked object we estimate:

* **Time-to-collision (TTC)** using the depth and depth velocity::

      TTC = depth_m / (-velocity_depth)   # positive TTC means approaching

* An object is flagged as a **collision threat** if its predicted 2-D
  trajectory (constant-velocity extrapolation) places it within the
  user's *threat corridor* (a rectangle centred on the image).
* Objects predicted *not* to intersect the threat corridor are still
  reported but with their volume capped (the "non-collision logic" from the
  problem statement).
"""

from __future__ import annotations

import dataclasses
import math
from enum import Enum
from typing import List, Optional

import numpy as np

from audiovision.tracking.tracker import TrackedObject


class RiskLevel(Enum):
    """Qualitative collision risk level."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclasses.dataclass
class CollisionRisk:
    """Collision risk assessment for one tracked object.

    Attributes:
        tracked_object: The underlying :class:`TrackedObject`.
        risk_level: Qualitative :class:`RiskLevel`.
        ttc_s: Estimated time-to-collision in seconds.  ``inf`` when the
            object is moving away or is stationary.
        will_collide: Whether the object is predicted to pass through the
            threat corridor.
        pitch_hz: Suggested audio pitch in Hz (higher = closer).
        pan: Stereo/HRTF pan value in ``[-1, 1]`` (left = -1, right = +1).
        volume: Suggested normalised volume in ``[0, 1]``.
    """

    tracked_object: TrackedObject
    risk_level: RiskLevel
    ttc_s: float
    will_collide: bool
    pitch_hz: float
    pan: float
    volume: float


class CollisionAvoidance:
    """Assess collision risk for a list of tracked objects.

    Args:
        image_width: Width of the camera image in pixels.
        image_height: Height of the camera image in pixels.
        threat_corridor_fraction: Fraction of the image width/height that
            defines the *threat corridor* (centred on the image).
        max_depth_m: Objects beyond this depth are considered low risk.
        pitch_min_hz: Minimum audio pitch (deepest sound, far objects).
        pitch_max_hz: Maximum audio pitch (highest sound, near objects).
        volume_cap_non_collision: Volume multiplier for objects that are
            predicted *not* to collide (the "non-collision logic").
        fps: Camera frame rate, used to convert velocity (px/frame) to
            velocity (px/s) for TTC estimation.

    Example::

        avoidance = CollisionAvoidance(image_width=640, image_height=480)
        risks = avoidance.assess(tracked_objects)
        for risk in risks:
            if risk.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                audio.play_urgent_alert(risk)
    """

    def __init__(
        self,
        image_width: int = 640,
        image_height: int = 480,
        threat_corridor_fraction: float = 0.3,
        max_depth_m: float = 10.0,
        pitch_min_hz: float = 200.0,
        pitch_max_hz: float = 2000.0,
        volume_cap_non_collision: float = 0.4,
        fps: float = 30.0,
    ) -> None:
        self._w = image_width
        self._h = image_height
        self._corridor_fraction = threat_corridor_fraction
        self._max_depth = max_depth_m
        self._pitch_min = pitch_min_hz
        self._pitch_max = pitch_max_hz
        self._vol_cap = volume_cap_non_collision
        self._fps = fps

    def assess(self, objects: List[TrackedObject]) -> List[CollisionRisk]:
        """Return a :class:`CollisionRisk` for each tracked object.

        Results are sorted by risk (highest first).
        """
        risks = [self._assess_one(obj) for obj in objects]
        risks.sort(key=lambda r: self._risk_sort_key(r), reverse=True)
        return risks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assess_one(self, obj: TrackedObject) -> CollisionRisk:
        depth = obj.depth_m if not math.isnan(obj.depth_m) else self._max_depth
        depth = min(depth, self._max_depth)

        ttc = self._compute_ttc(obj)
        will_collide = self._predict_collision(obj, ttc)
        risk_level = self._classify_risk(depth, ttc, will_collide)

        pitch = self._depth_to_pitch(depth)
        pan = self._centre_x_to_pan(obj.centre_x)
        volume = self._compute_volume(depth, ttc, will_collide, obj)

        return CollisionRisk(
            tracked_object=obj,
            risk_level=risk_level,
            ttc_s=ttc,
            will_collide=will_collide,
            pitch_hz=pitch,
            pan=pan,
            volume=volume,
        )

    def _compute_ttc(self, obj: TrackedObject) -> float:
        vd = obj.velocity_depth * self._fps  # m/s
        depth = obj.depth_m if not math.isnan(obj.depth_m) else self._max_depth
        if vd >= 0:
            return math.inf
        return abs(depth / vd)

    def _predict_collision(self, obj: TrackedObject, ttc: float) -> bool:
        if math.isinf(ttc):
            return False

        corridor_half_w = self._w * self._corridor_fraction / 2
        corridor_half_h = self._h * self._corridor_fraction / 2
        cx = self._w / 2
        cy = self._h / 2

        future_x = obj.centre_x + obj.velocity_x * self._fps * ttc
        future_y = obj.centre_y + obj.velocity_y * self._fps * ttc

        return (
            abs(future_x - cx) <= corridor_half_w
            and abs(future_y - cy) <= corridor_half_h
        )

    def _classify_risk(
        self,
        depth: float,
        ttc: float,
        will_collide: bool,
    ) -> RiskLevel:
        if not will_collide:
            if depth < 1.0:
                return RiskLevel.LOW
            return RiskLevel.NONE

        if ttc <= 1.0 or depth <= 0.5:
            return RiskLevel.CRITICAL
        if ttc <= 2.5 or depth <= 1.5:
            return RiskLevel.HIGH
        if ttc <= 5.0 or depth <= 3.0:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _depth_to_pitch(self, depth: float) -> float:
        ratio = 1.0 - min(depth / self._max_depth, 1.0)
        return self._pitch_min + ratio * (self._pitch_max - self._pitch_min)

    def _centre_x_to_pan(self, centre_x: float) -> float:
        return float(np.clip((centre_x / self._w) * 2.0 - 1.0, -1.0, 1.0))

    def _compute_volume(
        self,
        depth: float,
        ttc: float,
        will_collide: bool,
        obj: TrackedObject,
    ) -> float:
        base = 1.0 - min(depth / self._max_depth, 1.0)
        vd = abs(obj.velocity_depth) * self._fps
        urgency_boost = min(vd / 2.0, 0.3)
        vol = min(base + urgency_boost, 1.0)

        if not will_collide:
            vol = min(vol, self._vol_cap)

        return float(vol)

    @staticmethod
    def _risk_sort_key(risk: CollisionRisk) -> int:
        order = {
            RiskLevel.NONE: 0,
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4,
        }
        return order.get(risk.risk_level, 0)
