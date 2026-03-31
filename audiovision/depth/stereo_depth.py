"""Stereo depth map estimation using OpenCV StereoSGBM.

Follows the approach described in:
https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclasses.dataclass
class DepthConfig:
    """Configuration parameters for the stereo depth estimator.

    Attributes:
        baseline_m: Physical distance between the two camera centres, in metres.
        focal_length_px: Focal length of the (rectified) cameras, in pixels.
        min_disparity: Minimum disparity value (can be 0 or negative when
            cameras are not perfectly rectified).
        num_disparities: Range of disparity values to search; must be a
            positive multiple of 16.
        block_size: Matched block size; must be an odd number >= 1.
        uniqueness_ratio: Margin (%) by which the best match must beat the
            second-best to be accepted.
        speckle_window_size: Maximum area (pixels) of speckle regions to
            remove; 0 disables speckle filtering.
        speckle_range: Maximum disparity variation within each speckle region.
        disp12_max_diff: Maximum allowed difference (in integer pixel units)
            in the left-right disparity check; -1 disables the check.
        pre_filter_cap: Truncation value for pre-filtered image pixels.
        mode: StereoSGBM mode.  One of ``cv2.StereoSGBM_MODE_SGBM``,
            ``cv2.StereoSGBM_MODE_HH``, ``cv2.StereoSGBM_MODE_SGBM_3WAY``,
            or ``cv2.StereoSGBM_MODE_HH4``.
    """

    baseline_m: float = 0.06
    focal_length_px: float = 700.0
    min_disparity: int = 0
    num_disparities: int = 128
    block_size: int = 11
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 32
    disp12_max_diff: int = 1
    pre_filter_cap: int = 63
    mode: int = cv2.StereoSGBM_MODE_SGBM_3WAY


class StereoDepthEstimator:
    """Compute dense depth maps from a rectified stereo image pair.

    The depth map is expressed in metres.  Pixels where disparity could not
    be reliably estimated are set to ``NaN``.

    Example usage::

        cfg = DepthConfig(baseline_m=0.06, focal_length_px=700.0)
        estimator = StereoDepthEstimator(cfg)

        left_frame  = cv2.imread("left.png",  cv2.IMREAD_GRAYSCALE)
        right_frame = cv2.imread("right.png", cv2.IMREAD_GRAYSCALE)

        depth_map = estimator.compute(left_frame, right_frame)

    """

    def __init__(self, config: Optional[DepthConfig] = None) -> None:
        self._cfg = config or DepthConfig()
        self._stereo = self._build_matcher()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        """Return a float32 depth map (metres) for a rectified stereo pair.

        Args:
            left: Left camera image.  Colour (BGR) or grayscale accepted.
            right: Right camera image.  Must have the same shape as *left*.

        Returns:
            ``np.ndarray`` of shape ``(H, W)`` and dtype ``float32`` where
            each element is the estimated depth in metres.  Invalid pixels
            (no reliable disparity) are ``NaN``.

        Raises:
            ValueError: If *left* and *right* do not share the same spatial
                dimensions.
        """
        if left.shape[:2] != right.shape[:2]:
            raise ValueError(
                f"left and right images must have identical height/width; "
                f"got {left.shape[:2]} vs {right.shape[:2]}"
            )

        gray_left = self._to_gray(left)
        gray_right = self._to_gray(right)

        disparity_raw = self._stereo.compute(gray_left, gray_right)

        depth = self._disparity_to_depth(disparity_raw)
        return depth

    def compute_with_disparity(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Like :meth:`compute`, but also returns the raw disparity map.

        Returns:
            Tuple of ``(depth_map, disparity_map)`` where:

            * *depth_map* – float32 array of depths in metres (NaN for
              invalid pixels).
            * *disparity_map* – float32 array of raw disparity values in
              pixels (negative values for invalid pixels, as returned by
              StereoSGBM after fixed-point conversion).
        """
        if left.shape[:2] != right.shape[:2]:
            raise ValueError(
                f"left and right images must have identical height/width; "
                f"got {left.shape[:2]} vs {right.shape[:2]}"
            )

        gray_left = self._to_gray(left)
        gray_right = self._to_gray(right)

        disparity_raw = self._stereo.compute(gray_left, gray_right)

        disparity_float = disparity_raw.astype(np.float32) / 16.0
        depth = self._disparity_to_depth(disparity_raw)
        return depth, disparity_float

    @property
    def config(self) -> DepthConfig:
        """Return the active :class:`DepthConfig`."""
        return self._cfg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """Convert *image* to uint8 grayscale if it is not already."""
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _build_matcher(self) -> cv2.StereoSGBM:
        cfg = self._cfg
        p1 = 8 * 3 * cfg.block_size ** 2
        p2 = 32 * 3 * cfg.block_size ** 2
        return cv2.StereoSGBM.create(
            minDisparity=cfg.min_disparity,
            numDisparities=cfg.num_disparities,
            blockSize=cfg.block_size,
            P1=p1,
            P2=p2,
            disp12MaxDiff=cfg.disp12_max_diff,
            uniquenessRatio=cfg.uniqueness_ratio,
            speckleWindowSize=cfg.speckle_window_size,
            speckleRange=cfg.speckle_range,
            preFilterCap=cfg.pre_filter_cap,
            mode=cfg.mode,
        )

    def _disparity_to_depth(self, disparity_raw: np.ndarray) -> np.ndarray:
        """Convert StereoSGBM fixed-point disparity to a depth map in metres.

        StereoSGBM returns disparity values multiplied by 16 (fixed-point).
        Invalid pixels are marked with a value less than
        ``(minDisparity * 16)``.
        """
        cfg = self._cfg
        disp = disparity_raw.astype(np.float32) / 16.0
        invalid_mask = disp <= cfg.min_disparity

        with np.errstate(divide="ignore", invalid="ignore"):
            depth = (cfg.focal_length_px * cfg.baseline_m) / disp

        depth[invalid_mask] = np.nan
        return depth
