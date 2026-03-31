"""Object detection using YOLOv8n.

Wraps the Ultralytics inference API to provide a lightweight, consistent
interface for the rest of the Audio-Vision pipeline.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional

import numpy as np


@dataclasses.dataclass
class Detection:
    """A single detected object in one video frame.

    Attributes:
        class_id: Integer COCO class index.
        class_name: Human-readable class label (e.g. ``"person"``).
        confidence: Detection confidence in [0, 1].
        bbox_xyxy: Bounding box as ``[x1, y1, x2, y2]`` in pixel coordinates.
        centre_x: Horizontal centre of the bounding box (pixels).
        centre_y: Vertical centre of the bounding box (pixels).
    """

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: np.ndarray

    @property
    def centre_x(self) -> float:
        return float((self.bbox_xyxy[0] + self.bbox_xyxy[2]) / 2)

    @property
    def centre_y(self) -> float:
        return float((self.bbox_xyxy[1] + self.bbox_xyxy[3]) / 2)

    @property
    def width(self) -> float:
        return float(self.bbox_xyxy[2] - self.bbox_xyxy[0])

    @property
    def height(self) -> float:
        return float(self.bbox_xyxy[3] - self.bbox_xyxy[1])


class ObjectDetector:
    """Run YOLOv8n inference on a single frame.

    The detector is intentionally lazy-loaded on first use so that importing
    this module is cheap even when the ``ultralytics`` package is not
    installed (e.g. during unit tests that mock the detector).

    Args:
        model_path: Path to a YOLOv8 ``.pt`` weights file, or a model name
            such as ``"yolov8n.pt"`` to trigger automatic download.
        confidence_threshold: Minimum confidence score to keep a detection.
        device: PyTorch device string (``"cpu"``, ``"cuda:0"``, etc.).

    Example::

        detector = ObjectDetector()
        frame = cv2.imread("scene.jpg")
        detections = detector.detect(frame)
        for det in detections:
            print(det.class_name, det.confidence)
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.4,
        device: str = "cpu",
    ) -> None:
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._device = device
        self._model: Optional[object] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference and return a list of :class:`Detection` objects.

        Args:
            frame: BGR image as a ``numpy`` array (the standard OpenCV format).

        Returns:
            List of :class:`Detection` instances, sorted by confidence
            (highest first).
        """
        model = self._get_model()
        results = model.predict(
            frame,
            conf=self._confidence_threshold,
            device=self._device,
            verbose=False,
        )
        return self._parse_results(results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model(self) -> object:
        if self._model is None:
            from ultralytics import YOLO  # lazy import
            self._model = YOLO(self._model_path)
        return self._model

    def _parse_results(self, results: list) -> List[Detection]:
        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return detections

        class_names = result.names

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].cpu().numpy()

            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=class_names.get(cls_id, str(cls_id)),
                    confidence=conf,
                    bbox_xyxy=xyxy,
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
