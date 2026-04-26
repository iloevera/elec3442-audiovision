from __future__ import annotations

from dataclasses import dataclass
from ultralytics import YOLO
import numpy as np


@dataclass(frozen=True)
class TripHazardDetection:
    label: str
    confidence: float
    column: int
    azimuth_deg: float
    bbox_xyxy: tuple[int, int, int, int]
    urgency: float


class PiTripHazardDetector:
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        column_count: int = 5,
        conf_threshold: float = 0.35,
    ) -> None:
        self.model = YOLO(model_path)
        self.column_count = column_count
        self.conf_threshold = conf_threshold

        self.target_labels = {
            "person",
            "chair",
            "potted plant",
            "backpack",
            "suitcase",
            "bottle",
            "cup",
            "box",
            "wire"
        }

    def detect(self, image: np.ndarray) -> tuple[TripHazardDetection, ...]:
        h, w = image.shape[:2]
        results = self.model.predict(image, verbose=False)
        detections: list[TripHazardDetection] = []

        if not results:
            return ()

        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return ()

        for box in boxes:
            cls_id = int(box.cls.item())
            label = self.model.names[cls_id]
            confidence = float(box.conf.item())
            if confidence < self.conf_threshold:
                continue

            if label not in self.target_labels:
                continue

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cx = 0.5 * (x1 + x2)
            box_area = max(1, (x2 - x1) * (y2 - y1))
            lower_bias = min(1.0, y2 / max(1, h))
            size_bias = min(1.0, box_area / float(w * h * 0.2))

            column = self._x_to_column(cx, w)
            azimuth_deg = self._x_to_azimuth(cx, w)
            urgency = float(np.clip(0.5 * confidence + 0.3 * lower_bias + 0.2 * size_bias, 0.0, 1.0))

            detections.append(
                TripHazardDetection(
                    label=label,
                    confidence=confidence,
                    column=column,
                    azimuth_deg=azimuth_deg,
                    bbox_xyxy=(x1, y1, x2, y2),
                    urgency=urgency,
                )
            )

        detections.sort(key=lambda d: d.urgency, reverse=True)
        return tuple(detections[:3])

    def _x_to_column(self, x_px: float, width_px: int) -> int:
        normalized = max(0.0, min(1.0, x_px / max(1, width_px - 1)))
        col = int(normalized * self.column_count)
        return min(self.column_count - 1, max(0, col))

    def _x_to_azimuth(self, x_px: float, width_px: int) -> float:
        normalized = max(0.0, min(1.0, x_px / max(1, width_px - 1)))
        return -90.0 + normalized * 180.0