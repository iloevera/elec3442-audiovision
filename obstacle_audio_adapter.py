"""Adapter that maps tracked obstacles to SpatialTone voices."""

from __future__ import annotations

import math
import threading
from typing import Optional

from audio_spatial_tone import SpatialTone
from obstacle_models import ObstacleUpdate, TrackedObstacle


class ObstacleAudioAdapter:
    """Convert obstacle tracking updates into spatial audio control signals."""

    def __init__(
        self,
        min_pitch_hz: float = 220.0,
        max_pitch_hz: float = 1200.0,
        warning_ttc_s: float = 2.5,
        max_active_tones: int = 16,
    ) -> None:
        self._min_pitch_hz = float(min_pitch_hz)
        self._max_pitch_hz = float(max_pitch_hz)
        self._warning_ttc_s = float(max(0.2, warning_ttc_s))
        self._max_active_tones = int(max(1, max_active_tones))

        self._tones: dict[int, SpatialTone] = {}
        self._lock = threading.RLock()

    def close(self) -> None:
        with self._lock:
            for tone in self._tones.values():
                tone.stop()
            self._tones.clear()

    def on_update(self, update: ObstacleUpdate) -> None:
        if update.state == "stopped":
            self.close()
            return

        # Keep only collision-relevant obstacles, highest risk first.
        ranked = sorted(
            (obs for obs in update.obstacles if obs.is_collision_course),
            key=lambda obs: (obs.collision_score, -obs.ttc_s if math.isfinite(obs.ttc_s) else -1e9),
            reverse=True,
        )

        selected = ranked[: self._max_active_tones]
        keep_ids = {obs.obstacle_id for obs in selected}

        with self._lock:
            stale_ids = [obstacle_id for obstacle_id in self._tones if obstacle_id not in keep_ids]
            for obstacle_id in stale_ids:
                tone = self._tones.pop(obstacle_id)
                tone.stop()

            for obstacle in selected:
                tone = self._tones.get(obstacle.obstacle_id)
                if tone is None:
                    tone = SpatialTone(initial_pitch_hz=self._min_pitch_hz, initial_volume=0.0, initial_azimuth_deg=0.0)
                    tone.start()
                    self._tones[obstacle.obstacle_id] = tone

                azimuth_deg = self._compute_azimuth_deg(obstacle)
                pitch_hz = self._compute_pitch_hz(obstacle)
                volume = self._compute_volume(obstacle)
                tone.set_params(pitch_hz=pitch_hz, volume=volume, azimuth_deg=azimuth_deg)

    def _compute_azimuth_deg(self, obstacle: TrackedObstacle) -> float:
        x_m, _, z_m = obstacle.xyz_m
        azimuth_deg = math.degrees(math.atan2(x_m, max(1e-6, z_m)))
        return float(max(-90.0, min(90.0, azimuth_deg)))

    def _compute_pitch_hz(self, obstacle: TrackedObstacle) -> float:
        # Closer physical distance maps to higher pitch.
        near_m = 0.25
        far_m = 5.0
        clamped_dist = min(far_m, max(near_m, obstacle.distance_m))
        ratio = 1.0 - ((clamped_dist - near_m) / (far_m - near_m))
        eased = ratio ** 0.85
        return self._min_pitch_hz + (self._max_pitch_hz - self._min_pitch_hz) * eased

    def _compute_volume(self, obstacle: TrackedObstacle) -> float:
        # Sooner projected collision (lower TTC) maps to higher loudness.
        if math.isfinite(obstacle.ttc_s):
            ttc_ratio = max(0.0, min(1.0, (self._warning_ttc_s - obstacle.ttc_s) / self._warning_ttc_s))
        else:
            ttc_ratio = 0.0

        confidence = max(0.0, min(1.0, obstacle.confidence))
        score = max(0.0, min(1.0, obstacle.collision_score))
        volume = 0.05 + 0.75 * ttc_ratio + 0.20 * score
        volume *= 0.7 + 0.3 * confidence
        return float(max(0.0, min(1.0, volume)))
