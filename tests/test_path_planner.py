"""Tests for the spatial navigation / path planner module."""

import numpy as np
import pytest

from audiovision.navigation.path_planner import CueType, NavigationCue, PathPlanner


@pytest.fixture
def planner() -> PathPlanner:
    return PathPlanner(image_width=320, image_height=240, num_strips=9)


def make_depth_map(
    h: int = 240,
    w: int = 320,
    fill: float = 5.0,
) -> np.ndarray:
    return np.full((h, w), fill, dtype=np.float32)


class TestPathPlanner:
    def test_all_nan_depth_returns_no_cues(self, planner):
        depth = np.full((240, 320), np.nan, dtype=np.float32)
        cues = planner.plan(depth)
        path_cues = [c for c in cues if c.cue_type == CueType.PATH_TONE]
        assert len(path_cues) == 0

    def test_clear_corridor_generates_path_tone(self, planner):
        depth = make_depth_map(fill=5.0)
        cues = planner.plan(depth)
        path_cues = [c for c in cues if c.cue_type == CueType.PATH_TONE]
        assert len(path_cues) == 1

    def test_path_tone_azimuth_near_centre_for_central_corridor(self, planner):
        depth = make_depth_map(fill=5.0)
        depth[:, :80] = 0.5
        depth[:, 240:] = 0.5
        cues = planner.plan(depth)
        path_cues = [c for c in cues if c.cue_type == CueType.PATH_TONE]
        assert len(path_cues) == 1
        assert abs(path_cues[0].azimuth_deg) < 40.0

    def test_shallow_depth_below_min_produces_no_path_tone(self, planner):
        depth = make_depth_map(fill=0.5)
        cues = planner.plan(depth)
        path_cues = [c for c in cues if c.cue_type == CueType.PATH_TONE]
        assert len(path_cues) == 0

    def test_edge_detected_at_depth_boundary(self, planner):
        depth = make_depth_map(fill=5.0)
        depth[:, 160:] = 0.3
        cues = planner.plan(depth)
        edge_cues = [c for c in cues if c.cue_type == CueType.EDGE]
        assert len(edge_cues) >= 1

    def test_intersection_cues_for_two_corridors(self, planner):
        depth = make_depth_map(fill=0.5)
        depth[:, :60] = 5.0
        depth[:, 260:] = 5.0
        cues = planner.plan(depth)
        inter_cues = [c for c in cues if c.cue_type == CueType.INTERSECTION]
        assert len(inter_cues) >= 2

    def test_no_intersection_for_single_corridor(self, planner):
        depth = make_depth_map(fill=5.0)
        cues = planner.plan(depth)
        inter_cues = [c for c in cues if c.cue_type == CueType.INTERSECTION]
        assert len(inter_cues) == 0

    def test_cue_volumes_in_range(self, planner):
        depth = make_depth_map(fill=5.0)
        cues = planner.plan(depth)
        for cue in cues:
            assert 0.0 <= cue.volume <= 1.0

    def test_navigation_cue_dataclass(self):
        cue = NavigationCue(
            cue_type=CueType.PATH_TONE,
            azimuth_deg=10.0,
            pitch_hz=440.0,
            volume=0.7,
        )
        assert cue.label == ""
        assert cue.cue_type == CueType.PATH_TONE
