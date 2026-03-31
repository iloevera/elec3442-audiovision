"""Tests for the stereo depth estimation module."""

import math

import cv2
import numpy as np
import pytest

from audiovision.depth.stereo_depth import DepthConfig, StereoDepthEstimator


@pytest.fixture
def default_estimator() -> StereoDepthEstimator:
    return StereoDepthEstimator()


@pytest.fixture
def synthetic_stereo_pair():
    """Generate a simple synthetic stereo pair using a checkerboard pattern."""
    h, w = 240, 320
    left = np.zeros((h, w), dtype=np.uint8)
    right = np.zeros((h, w), dtype=np.uint8)

    square_size = 20
    for row in range(h // square_size):
        for col in range(w // square_size):
            if (row + col) % 2 == 0:
                y0 = row * square_size
                x0 = col * square_size
                left[y0 : y0 + square_size, x0 : x0 + square_size] = 200

    shift = 8
    right[:, : w - shift] = left[:, shift:]

    return left, right


class TestDepthConfig:
    def test_defaults(self):
        cfg = DepthConfig()
        assert cfg.baseline_m == pytest.approx(0.06)
        assert cfg.focal_length_px == pytest.approx(700.0)
        assert cfg.num_disparities % 16 == 0

    def test_custom_values(self):
        cfg = DepthConfig(baseline_m=0.12, focal_length_px=500.0)
        assert cfg.baseline_m == pytest.approx(0.12)
        assert cfg.focal_length_px == pytest.approx(500.0)


class TestStereoDepthEstimator:
    def test_output_shape(self, default_estimator, synthetic_stereo_pair):
        left, right = synthetic_stereo_pair
        depth = default_estimator.compute(left, right)
        assert depth.shape == left.shape

    def test_output_dtype(self, default_estimator, synthetic_stereo_pair):
        left, right = synthetic_stereo_pair
        depth = default_estimator.compute(left, right)
        assert depth.dtype == np.float32

    def test_invalid_pixels_are_nan(self, default_estimator, synthetic_stereo_pair):
        left, right = synthetic_stereo_pair
        depth = default_estimator.compute(left, right)
        assert np.any(np.isnan(depth)), "Expected some NaN pixels for invalid disparity"

    def test_valid_pixels_are_positive(self, default_estimator, synthetic_stereo_pair):
        left, right = synthetic_stereo_pair
        depth = default_estimator.compute(left, right)
        valid = depth[~np.isnan(depth)]
        assert np.all(valid > 0), "Valid depth values must be positive"

    def test_mismatched_shapes_raises(self, default_estimator):
        left = np.zeros((240, 320), dtype=np.uint8)
        right = np.zeros((480, 640), dtype=np.uint8)
        with pytest.raises(ValueError, match="identical height/width"):
            default_estimator.compute(left, right)

    def test_colour_input_accepted(self, default_estimator, synthetic_stereo_pair):
        gray_l, gray_r = synthetic_stereo_pair
        bgr_l = cv2.cvtColor(gray_l, cv2.COLOR_GRAY2BGR)
        bgr_r = cv2.cvtColor(gray_r, cv2.COLOR_GRAY2BGR)
        depth = default_estimator.compute(bgr_l, bgr_r)
        assert depth.shape == gray_l.shape

    def test_compute_with_disparity_returns_tuple(
        self, default_estimator, synthetic_stereo_pair
    ):
        left, right = synthetic_stereo_pair
        depth, disp = default_estimator.compute_with_disparity(left, right)
        assert depth.shape == left.shape
        assert disp.shape == left.shape
        assert depth.dtype == np.float32
        assert disp.dtype == np.float32

    def test_depth_formula(self):
        """Verify depth = focal * baseline / disparity for a known disparity."""
        cfg = DepthConfig(focal_length_px=700.0, baseline_m=0.07)
        estimator = StereoDepthEstimator(cfg)
        expected_depth = 700.0 * 0.07 / 10.0
        actual_depth = estimator._disparity_to_depth(
            np.array([[10.0 * 16]], dtype=np.int16)
        )[0, 0]
        assert actual_depth == pytest.approx(expected_depth, rel=1e-4)

    def test_config_accessor(self):
        cfg = DepthConfig(baseline_m=0.10)
        est = StereoDepthEstimator(cfg)
        assert est.config is cfg
