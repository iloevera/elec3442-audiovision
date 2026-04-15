"""Depth-based obstacle detection and collision-course tracking service."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Optional

import cv2
import numpy as np

from imu_compensation import apply_small_angle_rotation, estimate_gyro_rotation
from obstacle_models import (
    ObstacleDetection,
    ObstacleTrackerConfig,
    ObstacleUpdate,
    ObstacleUpdateCallback,
    TrackedObstacle,
)
from realsense_driver import CameraIntrinsics, D435iDriver, FrameBundle


@dataclass
class _MutableTrack:
    obstacle_id: int
    xyz_m: np.ndarray
    velocity_mps: np.ndarray
    confidence: float
    age_frames: int
    last_seen_s: float


class DepthObstacleTracker:
    """Per-frame depth processing and temporal obstacle tracking."""

    def __init__(self, config: Optional[ObstacleTrackerConfig] = None) -> None:
        self._config = config or ObstacleTrackerConfig()
        self._tracks: dict[int, _MutableTrack] = {}
        self._next_id = 1
        self._last_timestamp_s: Optional[float] = None

    def process_bundle(self, bundle: FrameBundle) -> ObstacleUpdate:
        timestamp_s = float(bundle.depth.host_timestamp_s)
        dt_s = self._compute_dt(timestamp_s)

        detections = self._extract_detections(
            depth_image=bundle.depth.image,
            depth_scale=bundle.depth.depth_scale,
            intrinsics=bundle.depth.intrinsics,
        )

        imu_compensated = False
        if self._config.enable_imu_compensation and bundle.imu_samples and dt_s > 0.0:
            imu_est = estimate_gyro_rotation(bundle.imu_samples, dt_s)
            if imu_est.confidence > 0.0 and self._tracks:
                predicted_points = np.stack([track.xyz_m for track in self._tracks.values()], axis=0)
                compensated = apply_small_angle_rotation(predicted_points, -imu_est.delta_rotation_rad)
                for track, adjusted_xyz in zip(self._tracks.values(), compensated):
                    track.xyz_m = adjusted_xyz.astype(np.float64, copy=False)
                imu_compensated = imu_est.confidence >= 0.2

        self._associate_and_update(detections=detections, now_s=timestamp_s, dt_s=dt_s)
        self._drop_stale_tracks(now_s=timestamp_s)

        tracked = tuple(
            self._to_tracked_obstacle(track=track, dt_s=dt_s, now_s=timestamp_s)
            for track in sorted(self._tracks.values(), key=lambda t: t.obstacle_id)
        )

        state = "running" if tracked else "degraded"
        return ObstacleUpdate(
            timestamp_s=timestamp_s,
            frame_number=int(bundle.depth.frame_number),
            state=state,
            imu_compensated=imu_compensated,
            obstacles=tracked,
        )

    def _compute_dt(self, timestamp_s: float) -> float:
        previous = self._last_timestamp_s
        self._last_timestamp_s = timestamp_s
        if previous is None:
            return 0.0
        return max(0.0, timestamp_s - previous)

    def _extract_detections(
        self,
        depth_image: np.ndarray,
        depth_scale: float,
        intrinsics: CameraIntrinsics,
    ) -> list[ObstacleDetection]:
        cfg = self._config
        downsample = max(1, int(cfg.depth_downsample))

        depth_m = depth_image.astype(np.float32) * float(depth_scale)
        if cfg.median_blur_ksize >= 3 and cfg.median_blur_ksize % 2 == 1:
            depth_m = cv2.medianBlur(depth_m, cfg.median_blur_ksize)

        depth_ds = depth_m[::downsample, ::downsample]
        valid = (depth_ds >= cfg.min_depth_m) & (depth_ds <= cfg.max_depth_m)
        mask = (valid.astype(np.uint8) * 255)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        detections: list[ObstacleDetection] = []

        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < cfg.min_cluster_pixels:
                continue

            cx_ds, cy_ds = centroids[label]
            cx = float(cx_ds * downsample)
            cy = float(cy_ds * downsample)

            x_px = int(round(cx))
            y_px = int(round(cy))
            if y_px < 0 or x_px < 0 or y_px >= depth_m.shape[0] or x_px >= depth_m.shape[1]:
                continue

            cluster_depths = depth_ds[labels == label]
            cluster_depths = cluster_depths[np.isfinite(cluster_depths)]
            cluster_depths = cluster_depths[(cluster_depths >= cfg.min_depth_m) & (cluster_depths <= cfg.max_depth_m)]
            if cluster_depths.size == 0:
                continue

            z_m = float(np.median(cluster_depths))
            if z_m <= 0.0:
                continue

            xyz_m = self._project_pixel_to_xyz(x_px, y_px, z_m, intrinsics)
            radius_px = math.sqrt(area / math.pi) * downsample
            radius_m = float((radius_px / intrinsics.fx) * z_m)

            depth_std = float(np.std(cluster_depths))
            conf_area = min(1.0, area / float(cfg.min_cluster_pixels * 4))
            conf_depth = math.exp(-depth_std / 0.15)
            confidence = float(max(0.05, min(1.0, 0.5 * conf_area + 0.5 * conf_depth)))

            detections.append(
                ObstacleDetection(
                    centroid_xyz_m=xyz_m,
                    radius_m=radius_m,
                    pixel_count=area,
                    confidence=confidence,
                )
            )

        return detections

    @staticmethod
    def _project_pixel_to_xyz(
        x_px: int,
        y_px: int,
        z_m: float,
        intrinsics: CameraIntrinsics,
    ) -> np.ndarray:
        x_m = ((float(x_px) - intrinsics.ppx) / intrinsics.fx) * z_m
        y_m = ((float(y_px) - intrinsics.ppy) / intrinsics.fy) * z_m
        return np.array([x_m, y_m, z_m], dtype=np.float64)

    def _associate_and_update(
        self,
        detections: list[ObstacleDetection],
        now_s: float,
        dt_s: float,
    ) -> None:
        if not detections:
            return

        unmatched_track_ids = set(self._tracks.keys())

        for det in detections:
            best_id: Optional[int] = None
            best_dist = float("inf")

            for track_id in unmatched_track_ids:
                track = self._tracks[track_id]
                dist = float(np.linalg.norm(det.centroid_xyz_m - track.xyz_m))
                if dist < best_dist:
                    best_dist = dist
                    best_id = track_id

            if best_id is None or best_dist > self._config.association_gate_m:
                self._create_track(detection=det, now_s=now_s)
                continue

            track = self._tracks[best_id]
            unmatched_track_ids.remove(best_id)

            if dt_s > 0.0:
                measured_velocity = (det.centroid_xyz_m - track.xyz_m) / dt_s
                track.velocity_mps = 0.7 * track.velocity_mps + 0.3 * measured_velocity

            track.xyz_m = det.centroid_xyz_m.astype(np.float64, copy=False)
            track.confidence = float(max(0.05, min(1.0, 0.5 * track.confidence + 0.5 * det.confidence)))
            track.age_frames += 1
            track.last_seen_s = now_s

    def _create_track(self, detection: ObstacleDetection, now_s: float) -> None:
        if len(self._tracks) >= self._config.max_tracks:
            oldest_id = min(self._tracks.values(), key=lambda t: t.last_seen_s).obstacle_id
            del self._tracks[oldest_id]

        obstacle_id = self._next_id
        self._next_id += 1

        self._tracks[obstacle_id] = _MutableTrack(
            obstacle_id=obstacle_id,
            xyz_m=detection.centroid_xyz_m.astype(np.float64, copy=False),
            velocity_mps=np.zeros(3, dtype=np.float64),
            confidence=detection.confidence,
            age_frames=1,
            last_seen_s=now_s,
        )

    def _drop_stale_tracks(self, now_s: float) -> None:
        stale_seconds = self._config.stale_track_seconds
        stale_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if (now_s - track.last_seen_s) > stale_seconds
        ]
        for track_id in stale_ids:
            del self._tracks[track_id]

    def _to_tracked_obstacle(self, track: _MutableTrack, dt_s: float, now_s: float) -> TrackedObstacle:
        xyz = track.xyz_m.astype(np.float64, copy=False)
        distance_m = float(np.linalg.norm(xyz))

        if distance_m > 1e-6:
            radial_unit = xyz / distance_m
            approach_rate_mps = float(max(0.0, -np.dot(track.velocity_mps, radial_unit)))
        else:
            approach_rate_mps = 0.0

        ttc_s = float("inf") if approach_rate_mps <= 1e-4 else distance_m / approach_rate_mps
        in_front = xyz[2] > 0.0
        is_collision_course = bool(
            in_front and approach_rate_mps > 0.03 and ttc_s <= self._config.warning_ttc_s
        )

        ttc_term = 0.0
        if math.isfinite(ttc_s):
            ttc_term = float(max(0.0, min(1.0, (self._config.warning_ttc_s - ttc_s) / self._config.warning_ttc_s)))

        distance_term = float(max(0.0, min(1.0, 1.0 - (distance_m / self._config.max_depth_m))))
        urgent_boost = 1.0 if ttc_s <= self._config.urgent_ttc_s else 0.0
        collision_score = float(
            max(
                0.0,
                min(
                    1.0,
                    (0.55 * ttc_term + 0.25 * distance_term + 0.20 * track.confidence)
                    * (1.0 + 0.25 * urgent_boost),
                ),
            )
        )

        if not in_front:
            collision_score *= 0.25

        return TrackedObstacle(
            obstacle_id=track.obstacle_id,
            xyz_m=xyz.copy(),
            velocity_mps=track.velocity_mps.astype(np.float64, copy=True),
            distance_m=distance_m,
            approach_rate_mps=approach_rate_mps,
            ttc_s=ttc_s,
            collision_score=collision_score,
            confidence=track.confidence,
            is_collision_course=is_collision_course,
            age_frames=track.age_frames,
            last_seen_s=now_s,
        )


class ObstacleTrackerService:
    """Background service that produces obstacle updates via callback."""

    def __init__(
        self,
        driver: D435iDriver,
        callback: ObstacleUpdateCallback,
        config: Optional[ObstacleTrackerConfig] = None,
        wait_timeout_s: float = 1.0,
    ) -> None:
        self._driver = driver
        self._callback = callback
        self._config = config or ObstacleTrackerConfig()
        self._wait_timeout_s = float(wait_timeout_s)

        self._tracker = DepthObstacleTracker(config=self._config)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        with self._lock:
            if self.is_running:
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._loop, name="ObstacleTracker", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            self._stop_event.set()
            self._thread = None

        if thread is not None:
            thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            bundle = self._driver.wait_for_bundle(timeout_s=self._wait_timeout_s)
            if bundle is None:
                update = ObstacleUpdate(
                    timestamp_s=time.monotonic(),
                    frame_number=-1,
                    state="degraded",
                    imu_compensated=False,
                    obstacles=(),
                )
                self._safe_emit(update)
                continue

            update = self._tracker.process_bundle(bundle)
            self._safe_emit(update)

    def _safe_emit(self, update: ObstacleUpdate) -> None:
        try:
            self._callback(update)
        except Exception:
            # Ignore callback exceptions to keep tracking alive.
            return
