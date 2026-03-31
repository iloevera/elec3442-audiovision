"""Tests for the Kalman-filter-based object tracker."""

import numpy as np
import pytest

from audiovision.detection.object_detector import Detection
from audiovision.tracking.tracker import ObjectTracker, TrackedObject, _KalmanTrack


def make_detection(
    cx: float = 100.0,
    cy: float = 100.0,
    w: float = 60.0,
    h: float = 120.0,
    class_id: int = 0,
    class_name: str = "person",
    confidence: float = 0.9,
) -> Detection:
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
    )


@pytest.fixture(autouse=True)
def reset_track_id():
    """Reset the global track ID counter before each test."""
    _KalmanTrack._next_id = 0
    yield
    _KalmanTrack._next_id = 0


@pytest.fixture
def tracker() -> ObjectTracker:
    return ObjectTracker(max_age=3, min_hits=2, iou_threshold=0.3)


class TestObjectTracker:
    def test_empty_detections_returns_empty(self, tracker):
        result = tracker.update([])
        assert result == []

    def test_single_detection_appears_after_min_hits(self, tracker):
        det = make_detection(cx=100, cy=100)
        result1 = tracker.update([det])
        result2 = tracker.update([det])
        assert len(result2) == 1

    def test_track_id_assigned(self, tracker):
        det = make_detection()
        tracker.update([det])
        result = tracker.update([det])
        assert len(result) == 1
        assert isinstance(result[0].track_id, int)

    def test_track_persists_when_detection_continues(self, tracker):
        det = make_detection(cx=200, cy=200)
        for _ in range(5):
            results = tracker.update([det])
        ids = [r.track_id for r in results]
        assert len(set(ids)) == 1

    def test_track_removed_after_max_age(self, tracker):
        det = make_detection()
        tracker.update([det])
        tracker.update([det])
        for _ in range(4):
            tracker.update([])
        result = tracker.update([])
        assert len(result) == 0

    def test_depth_from_depth_map(self, tracker):
        depth_map = np.full((200, 200), 3.5, dtype=np.float32)
        det = make_detection(cx=100, cy=100)
        tracker.update([det], depth_map)
        result = tracker.update([det], depth_map)
        assert len(result) >= 1

    def test_multiple_detections_tracked_independently(self, tracker):
        det1 = make_detection(cx=50, cy=50)
        det2 = make_detection(cx=300, cy=300)
        for _ in range(3):
            results = tracker.update([det1, det2])
        track_ids = [r.track_id for r in results]
        assert len(set(track_ids)) == 2

    def test_reset_clears_tracks(self, tracker):
        det = make_detection()
        tracker.update([det])
        tracker.update([det])
        tracker.reset()
        result = tracker.update([det])
        assert all(r.age <= 1 for r in result)

    def test_tracked_object_has_velocity_fields(self, tracker):
        det = make_detection()
        tracker.update([det])
        results = tracker.update([det])
        if results:
            obj = results[0]
            assert hasattr(obj, "velocity_x")
            assert hasattr(obj, "velocity_y")
            assert hasattr(obj, "velocity_depth")

    def test_iou_helper(self):
        from audiovision.tracking.tracker import _compute_iou

        box_a = np.array([0, 0, 10, 10], dtype=np.float32)
        box_b = np.array([5, 5, 15, 15], dtype=np.float32)
        iou = _compute_iou(box_a, box_b)
        assert iou == pytest.approx(25.0 / 175.0, rel=1e-4)

    def test_iou_no_overlap(self):
        from audiovision.tracking.tracker import _compute_iou

        box_a = np.array([0, 0, 5, 5], dtype=np.float32)
        box_b = np.array([10, 10, 15, 15], dtype=np.float32)
        assert _compute_iou(box_a, box_b) == pytest.approx(0.0)

    def test_iou_full_overlap(self):
        from audiovision.tracking.tracker import _compute_iou

        box = np.array([0, 0, 10, 10], dtype=np.float32)
        assert _compute_iou(box, box.copy()) == pytest.approx(1.0)
