"""Audio-Vision main pipeline.

Ties together:
1. Stereo depth estimation (OpenCV StereoSGBM)
2. Object detection (YOLOv8n)
3. Kalman-filter-based object tracking
4. Collision avoidance assessment
5. Spatial navigation / path planning
6. Binaural audio rendering

Usage (CLI)::

    python -m audiovision.main --left /dev/video0 --right /dev/video1

Usage (API)::

    pipeline = AudioVisionPipeline()
    pipeline.run()           # blocking loop, press Ctrl-C to exit
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional, Union

import cv2
import numpy as np

from audiovision.audio.audio_renderer import AudioCue, AudioRenderer
from audiovision.collision.collision_avoidance import (
    CollisionAvoidance,
    CollisionRisk,
    RiskLevel,
)
from audiovision.depth.stereo_depth import DepthConfig, StereoDepthEstimator
from audiovision.detection.object_detector import ObjectDetector
from audiovision.navigation.path_planner import CueType, NavigationCue, PathPlanner
from audiovision.tracking.tracker import ObjectTracker, TrackedObject

logger = logging.getLogger(__name__)


class AudioVisionPipeline:
    """End-to-end Audio-Vision processing pipeline.

    Args:
        left_source: Camera index or video file path for the *left* camera.
        right_source: Camera index or video file path for the *right* camera.
        depth_config: :class:`~audiovision.depth.stereo_depth.DepthConfig`
            used to configure the stereo depth estimator.
        show_debug: If ``True``, display an OpenCV window with a false-colour
            depth map and detection overlays.
    """

    def __init__(
        self,
        left_source: Union[int, str] = 0,
        right_source: Union[int, str] = 1,
        depth_config: Optional[DepthConfig] = None,
        show_debug: bool = False,
    ) -> None:
        self._left_src = left_source
        self._right_src = right_source
        self._show_debug = show_debug

        self._depth_estimator = StereoDepthEstimator(depth_config)
        self._detector = ObjectDetector()
        self._tracker = ObjectTracker()
        self._renderer = AudioRenderer()
        self._collision: Optional[CollisionAvoidance] = None
        self._planner: Optional[PathPlanner] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame_pair(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        """Process one stereo frame pair and return a stereo PCM audio buffer.

        This method is the core of the pipeline and is also exposed for
        unit-testing without requiring a live camera.

        Args:
            left: Left camera BGR frame.
            right: Right camera BGR frame.

        Returns:
            Stereo int16 PCM array ``(N, 2)`` ready for audio output.
        """
        if self._collision is None or self._planner is None:
            h, w = left.shape[:2]
            self._collision = CollisionAvoidance(
                image_width=w, image_height=h
            )
            self._planner = PathPlanner(image_width=w, image_height=h)

        depth_map = self._depth_estimator.compute(left, right)

        detections = self._detector.detect(left)
        tracked_objects = self._tracker.update(detections, depth_map)

        risks = self._collision.assess(tracked_objects)
        nav_cues = self._planner.plan(depth_map)

        audio_cues = self._build_audio_cues(risks, nav_cues)
        pcm = self._renderer.render(audio_cues)

        if self._show_debug:
            self._render_debug(left, depth_map, tracked_objects, risks)

        return pcm

    def run(self) -> None:
        """Open cameras and run the pipeline until interrupted."""
        cap_left = cv2.VideoCapture(self._left_src)
        cap_right = cv2.VideoCapture(self._right_src)

        if not cap_left.isOpened():
            raise RuntimeError(f"Cannot open left camera: {self._left_src!r}")
        if not cap_right.isOpened():
            raise RuntimeError(f"Cannot open right camera: {self._right_src!r}")

        logger.info("Audio-Vision pipeline started. Press Ctrl-C to exit.")
        try:
            while True:
                ret_l, frame_l = cap_left.read()
                ret_r, frame_r = cap_right.read()

                if not ret_l or not ret_r:
                    logger.warning("Frame capture failed; retrying …")
                    continue

                _pcm = self.process_frame_pair(frame_l, frame_r)

                if self._show_debug and cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            cap_left.release()
            cap_right.release()
            if self._show_debug:
                cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_audio_cues(
        risks: List[CollisionRisk],
        nav_cues: List[NavigationCue],
    ) -> List[AudioCue]:
        cues: List[AudioCue] = []

        for risk in risks:
            if risk.risk_level == RiskLevel.NONE:
                continue
            cues.append(
                AudioCue(
                    frequency_hz=risk.pitch_hz,
                    pan=risk.pan,
                    volume=risk.volume,
                )
            )

        for nc in nav_cues:
            cues.append(
                AudioCue(
                    frequency_hz=nc.pitch_hz,
                    pan=float(np.clip(nc.azimuth_deg / 45.0, -1.0, 1.0)),
                    volume=nc.volume * 0.5,
                )
            )

        return cues

    @staticmethod
    def _render_debug(
        frame: np.ndarray,
        depth_map: np.ndarray,
        objects: List[TrackedObject],
        risks: List[CollisionRisk],
    ) -> None:
        depth_vis = depth_map.copy()
        depth_vis = np.nan_to_num(depth_vis, nan=0.0)
        depth_vis = np.clip(depth_vis / 10.0, 0, 1)
        depth_colour = cv2.applyColorMap(
            (depth_vis * 255).astype(np.uint8), cv2.COLORMAP_JET
        )

        risk_by_id = {r.tracked_object.track_id: r for r in risks}

        overlay = frame.copy()
        for obj in objects:
            x1, y1, x2, y2 = obj.bbox_xyxy.astype(int)
            risk = risk_by_id.get(obj.track_id)
            colour = (0, 255, 0)
            if risk is not None:
                if risk.risk_level == RiskLevel.CRITICAL:
                    colour = (0, 0, 255)
                elif risk.risk_level == RiskLevel.HIGH:
                    colour = (0, 128, 255)
                elif risk.risk_level == RiskLevel.MEDIUM:
                    colour = (0, 255, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, 2)
            label = f"#{obj.track_id} {obj.class_name} {obj.depth_m:.1f}m"
            cv2.putText(
                overlay,
                label,
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colour,
                1,
            )

        combined = np.hstack([overlay, depth_colour])
        cv2.imshow("Audio-Vision Debug", combined)


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audio-Vision: synthetic audio-vision for the visually impaired"
    )
    parser.add_argument(
        "--left",
        default=0,
        help="Left camera index or video file path (default: 0)",
    )
    parser.add_argument(
        "--right",
        default=1,
        help="Right camera index or video file path (default: 1)",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=0.06,
        help="Stereo baseline in metres (default: 0.06)",
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=700.0,
        help="Focal length in pixels (default: 700.0)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show OpenCV debug window",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    """Entry point for the ``python -m audiovision.main`` invocation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)

    left_src = int(args.left) if str(args.left).isdigit() else args.left
    right_src = int(args.right) if str(args.right).isdigit() else args.right

    depth_config = DepthConfig(
        baseline_m=args.baseline,
        focal_length_px=args.focal_length,
    )
    pipeline = AudioVisionPipeline(
        left_source=left_src,
        right_source=right_src,
        depth_config=depth_config,
        show_debug=args.debug,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
