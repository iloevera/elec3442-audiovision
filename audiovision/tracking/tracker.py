"""Kalman-filter-based multi-object tracker.

Each tracked object maintains its own Kalman filter using a constant-velocity
model in ``[x, y, depth]`` state space.  The tracker associates new detections
to existing tracks via IoU matching.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

import numpy as np

from audiovision.detection.object_detector import Detection


@dataclasses.dataclass
class TrackedObject:
    """A detection enriched with tracking identity and motion estimate.

    Attributes:
        track_id: Unique integer identifier for this track.
        class_id: COCO class index.
        class_name: Human-readable class label.
        centre_x: Estimated horizontal centre (pixels).
        centre_y: Estimated vertical centre (pixels).
        depth_m: Estimated depth in metres (``NaN`` if unavailable).
        velocity_x: Estimated horizontal velocity (pixels/frame).
        velocity_y: Estimated vertical velocity (pixels/frame).
        velocity_depth: Estimated depth velocity (m/frame).
        bbox_xyxy: Bounding box ``[x1, y1, x2, y2]`` (pixels).
        age: Number of frames since the track was first created.
        hits: Number of consecutive frames with a matched detection.
        confidence: Latest detection confidence.
    """

    track_id: int
    class_id: int
    class_name: str
    centre_x: float
    centre_y: float
    depth_m: float
    velocity_x: float
    velocity_y: float
    velocity_depth: float
    bbox_xyxy: np.ndarray
    age: int
    hits: int
    confidence: float


class _KalmanTrack:
    """Internal per-object Kalman filter using a constant-velocity model.

    State vector: ``[x, y, d, vx, vy, vd]``
    where ``(x, y)`` is the bbox centre (pixels), ``d`` is depth (metres),
    and ``(vx, vy, vd)`` are the corresponding velocities.
    """

    _next_id: int = 0

    def __init__(
        self,
        detection: Detection,
        depth: float,
        dt: float = 1.0,
    ) -> None:
        self.track_id = _KalmanTrack._next_id
        _KalmanTrack._next_id += 1

        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.bbox_xyxy = detection.bbox_xyxy.copy()
        self.confidence = detection.confidence
        self.age = 1
        self.hits = 1
        self.time_since_update = 0

        n = 6
        m = 3
        self._kf_x = np.array(
            [detection.centre_x, detection.centre_y, depth, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        self._kf_P = np.eye(n, dtype=np.float64) * 10.0
        self._kf_F = self._build_transition(dt)
        self._kf_H = np.zeros((m, n), dtype=np.float64)
        self._kf_H[:m, :m] = np.eye(m)
        self._kf_Q = np.eye(n, dtype=np.float64) * 0.1
        self._kf_R = np.eye(m, dtype=np.float64) * 1.0

    @staticmethod
    def _build_transition(dt: float) -> np.ndarray:
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F

    def predict(self) -> None:
        self._kf_x = self._kf_F @ self._kf_x
        self._kf_P = self._kf_F @ self._kf_P @ self._kf_F.T + self._kf_Q
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: Detection, depth: float) -> None:
        z = np.array(
            [detection.centre_x, detection.centre_y, depth],
            dtype=np.float64,
        )
        S = self._kf_H @ self._kf_P @ self._kf_H.T + self._kf_R
        K = self._kf_P @ self._kf_H.T @ np.linalg.inv(S)
        innovation = z - self._kf_H @ self._kf_x
        self._kf_x = self._kf_x + K @ innovation
        I = np.eye(len(self._kf_x), dtype=np.float64)
        self._kf_P = (I - K @ self._kf_H) @ self._kf_P

        self.bbox_xyxy = detection.bbox_xyxy.copy()
        self.confidence = detection.confidence
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.hits += 1
        self.time_since_update = 0

    @property
    def state(self) -> np.ndarray:
        return self._kf_x.copy()

    def to_tracked_object(self) -> TrackedObject:
        x = self._kf_x
        return TrackedObject(
            track_id=self.track_id,
            class_id=self.class_id,
            class_name=self.class_name,
            centre_x=float(x[0]),
            centre_y=float(x[1]),
            depth_m=float(x[2]),
            velocity_x=float(x[3]),
            velocity_y=float(x[4]),
            velocity_depth=float(x[5]),
            bbox_xyxy=self.bbox_xyxy.copy(),
            age=self.age,
            hits=self.hits,
            confidence=self.confidence,
        )


def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two ``[x1, y1, x2, y2]`` boxes."""
    xi1 = max(box_a[0], box_b[0])
    yi1 = max(box_a[1], box_b[1])
    xi2 = min(box_a[2], box_b[2])
    yi2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


class ObjectTracker:
    """Multi-object tracker using Kalman filters and IoU-based association.

    Args:
        max_age: Maximum number of frames a track survives without a
            matched detection before it is removed.
        min_hits: Minimum number of consecutive matched frames before a
            track is considered confirmed and returned to callers.
        iou_threshold: Minimum IoU required to associate a detection with
            an existing track.

    Example::

        tracker = ObjectTracker()
        for left, right in camera_stream:
            detections = detector.detect(left)
            depth_map  = estimator.compute(left, right)
            tracks     = tracker.update(detections, depth_map)
            for obj in tracks:
                print(obj.track_id, obj.depth_m)
    """

    def __init__(
        self,
        max_age: int = 5,
        min_hits: int = 2,
        iou_threshold: float = 0.3,
    ) -> None:
        self._max_age = max_age
        self._min_hits = min_hits
        self._iou_threshold = iou_threshold
        self._tracks: List[_KalmanTrack] = []

    def update(
        self,
        detections: List[Detection],
        depth_map: Optional[np.ndarray] = None,
    ) -> List[TrackedObject]:
        """Advance the tracker by one frame.

        Args:
            detections: Detections from the current frame.
            depth_map: Float32 depth map (metres, NaN for invalid pixels)
                of the same spatial resolution as the image used to produce
                *detections*.  If ``None``, depth is treated as ``NaN``.

        Returns:
            List of confirmed :class:`TrackedObject` instances.
        """
        for track in self._tracks:
            track.predict()

        depths = self._extract_depths(detections, depth_map)

        matched, unmatched_dets, unmatched_tracks = self._associate(
            detections, self._tracks
        )

        for det_idx, trk_idx in matched:
            self._tracks[trk_idx].update(detections[det_idx], depths[det_idx])

        for det_idx in unmatched_dets:
            self._tracks.append(
                _KalmanTrack(detections[det_idx], depths[det_idx])
            )

        self._tracks = [
            t
            for t in self._tracks
            if t.time_since_update <= self._max_age
        ]

        confirmed: List[TrackedObject] = [
            t.to_tracked_object()
            for t in self._tracks
            if t.hits >= self._min_hits
        ]
        return confirmed

    def reset(self) -> None:
        """Clear all active tracks and reset the track ID counter."""
        self._tracks.clear()
        _KalmanTrack._next_id = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_depths(
        detections: List[Detection],
        depth_map: Optional[np.ndarray],
    ) -> List[float]:
        if depth_map is None:
            return [float("nan")] * len(detections)

        h, w = depth_map.shape[:2]
        depths: List[float] = []

        for det in detections:
            cx = int(np.clip(det.centre_x, 0, w - 1))
            cy = int(np.clip(det.centre_y, 0, h - 1))
            d = float(depth_map[cy, cx])
            depths.append(d)

        return depths

    def _associate(
        self,
        detections: List[Detection],
        tracks: List[_KalmanTrack],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not tracks:
            return [], list(range(len(detections))), []
        if not detections:
            return [], [], list(range(len(tracks)))

        iou_matrix = np.zeros((len(detections), len(tracks)), dtype=np.float64)
        for d_idx, det in enumerate(detections):
            for t_idx, trk in enumerate(tracks):
                iou_matrix[d_idx, t_idx] = _compute_iou(
                    det.bbox_xyxy, trk.bbox_xyxy
                )

        matched_indices: List[Tuple[int, int]] = []
        used_dets: set = set()
        used_trks: set = set()

        flat_order = np.argsort(-iou_matrix, axis=None)
        for flat_idx in flat_order:
            d_idx, t_idx = divmod(int(flat_idx), len(tracks))
            if iou_matrix[d_idx, t_idx] < self._iou_threshold:
                break
            if d_idx in used_dets or t_idx in used_trks:
                continue
            matched_indices.append((d_idx, t_idx))
            used_dets.add(d_idx)
            used_trks.add(t_idx)

        unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]
        unmatched_trks = [i for i in range(len(tracks)) if i not in used_trks]
        return matched_indices, unmatched_dets, unmatched_trks
