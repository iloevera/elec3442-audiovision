"""Tests for the collision avoidance module."""

import math

import numpy as np
import pytest

from audiovision.collision.collision_avoidance import (
    CollisionAvoidance,
    CollisionRisk,
    RiskLevel,
)
from audiovision.tracking.tracker import TrackedObject


def make_tracked_object(
    track_id: int = 0,
    centre_x: float = 320.0,
    centre_y: float = 240.0,
    depth_m: float = 5.0,
    velocity_x: float = 0.0,
    velocity_y: float = 0.0,
    velocity_depth: float = 0.0,
    class_name: str = "person",
) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        class_id=0,
        class_name=class_name,
        centre_x=centre_x,
        centre_y=centre_y,
        depth_m=depth_m,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        velocity_depth=velocity_depth,
        bbox_xyxy=np.array(
            [centre_x - 30, centre_y - 60, centre_x + 30, centre_y + 60],
            dtype=np.float32,
        ),
        age=5,
        hits=5,
        confidence=0.9,
    )


@pytest.fixture
def avoidance() -> CollisionAvoidance:
    return CollisionAvoidance(image_width=640, image_height=480, fps=30.0)


class TestCollisionAvoidance:
    def test_empty_input(self, avoidance):
        risks = avoidance.assess([])
        assert risks == []

    def test_stationary_far_object_low_risk(self, avoidance):
        obj = make_tracked_object(depth_m=8.0, velocity_depth=0.0)
        risks = avoidance.assess([obj])
        assert len(risks) == 1
        assert risks[0].risk_level in (RiskLevel.NONE, RiskLevel.LOW)

    def test_fast_approaching_central_object_high_risk(self, avoidance):
        obj = make_tracked_object(
            centre_x=320.0,
            centre_y=240.0,
            depth_m=2.0,
            velocity_depth=-0.1,
        )
        risks = avoidance.assess([obj])
        assert risks[0].risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    def test_very_close_critical(self, avoidance):
        obj = make_tracked_object(
            centre_x=320.0,
            centre_y=240.0,
            depth_m=0.3,
            velocity_depth=-0.05,
        )
        risks = avoidance.assess([obj])
        assert risks[0].risk_level == RiskLevel.CRITICAL

    def test_will_collide_flag(self, avoidance):
        central_obj = make_tracked_object(
            centre_x=320.0,
            centre_y=240.0,
            depth_m=3.0,
            velocity_depth=-0.1,
            velocity_x=0.0,
        )
        risks = avoidance.assess([central_obj])
        assert risks[0].will_collide is True

    def test_non_collision_volume_capped(self, avoidance):
        side_obj = make_tracked_object(
            centre_x=620.0,
            centre_y=240.0,
            depth_m=3.0,
            velocity_depth=-0.05,
            velocity_x=0.0,
        )
        risks = avoidance.assess([side_obj])
        if not risks[0].will_collide:
            assert risks[0].volume <= avoidance._vol_cap + 1e-6

    def test_pitch_increases_for_closer_objects(self, avoidance):
        far = make_tracked_object(track_id=0, depth_m=8.0)
        near = make_tracked_object(track_id=1, depth_m=1.0)
        r_far = avoidance.assess([far])[0]
        r_near = avoidance.assess([near])[0]
        assert r_near.pitch_hz > r_far.pitch_hz

    def test_pan_left_for_left_object(self, avoidance):
        left_obj = make_tracked_object(centre_x=50.0)
        risk = avoidance.assess([left_obj])[0]
        assert risk.pan < 0

    def test_pan_right_for_right_object(self, avoidance):
        right_obj = make_tracked_object(centre_x=590.0)
        risk = avoidance.assess([right_obj])[0]
        assert risk.pan > 0

    def test_sorted_by_risk_descending(self, avoidance):
        low = make_tracked_object(track_id=0, depth_m=9.0, centre_x=10.0)
        critical = make_tracked_object(
            track_id=1,
            depth_m=0.4,
            centre_x=320.0,
            velocity_depth=-0.1,
        )
        risks = avoidance.assess([low, critical])
        levels = [r.risk_level for r in risks]
        assert levels[0].value >= levels[-1].value or levels[0] == RiskLevel.CRITICAL

    def test_ttc_infinite_for_receding_object(self, avoidance):
        obj = make_tracked_object(depth_m=5.0, velocity_depth=0.02)
        risk = avoidance.assess([obj])[0]
        assert math.isinf(risk.ttc_s)
