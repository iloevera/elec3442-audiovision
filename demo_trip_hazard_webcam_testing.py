from __future__ import annotations

import argparse
from contextlib import suppress
import time

import cv2
import numpy as np

from src.navigation_processing import NavigationColumnState
from src.navigation_audio import NavigationAudioController, NavigationAudioConfig
from src.pi_trip_camera import PiTripCamera
from src.pi_trip_hazard import PiTripHazardDetector
from src.pi_trip_hazard_states import build_trip_column_states

WINDOW_NAME = "Separate Audio Test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Separate RealSense/Pi audio test")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--cols", type=int, default=5)
    return parser.parse_args()


def make_fake_realsense_states(cols: int, now_s: float) -> tuple[NavigationColumnState, ...]:
    if cols <= 1:
        azimuths = [0.0]
    else:
        azimuths = np.linspace(-90.0, 90.0, cols)

    active_col = int((now_s * 0.8) % cols)

    states = []
    for col in range(cols):
        if col == active_col:
            states.append(
                NavigationColumnState(
                    col=col,
                    azimuth_deg=float(azimuths[col]),
                    sample_count=20,
                    risk_score=0.8,
                    percentile_depth_m=0.45,
                    ttc_s=1.2,
                    pitch_hz=520.0,
                    pulse_hz=2.2,
                    volume=0.75,
                )
            )
        else:
            states.append(
                NavigationColumnState(
                    col=col,
                    azimuth_deg=float(azimuths[col]),
                    sample_count=0,
                    risk_score=0.0,
                    percentile_depth_m=None,
                    ttc_s=None,
                    pitch_hz=260.0,
                    pulse_hz=0.0,
                    volume=0.0,
                )
            )
    return tuple(states)


def draw_trip_detections(image: np.ndarray, detections) -> None:
    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(
            image,
            f"{det.label} col={det.column} urg={det.urgency:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )


def run_demo() -> None:
    args = parse_args()

    detector = PiTripHazardDetector(column_count=args.cols)
    realsense_audio = NavigationAudioController(
    column_count=args.cols,
    config=NavigationAudioConfig(use_pulse_gating=False),
    )

    pi_audio = NavigationAudioController(
        column_count=args.cols,
        config=NavigationAudioConfig(use_pulse_gating=True),
    )

    realsense_audio.start()
    pi_audio.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        with PiTripCamera(
            camera_index=args.camera_index,
            width=args.width,
            height=args.height,
        ) as cam:
            while True:
                frame = cam.read()
                if frame is None:
                    continue

                now_s = time.monotonic()
                detections = detector.detect(frame.image)

                trip_states = build_trip_column_states(detections, cols=args.cols)
                fake_realsense_states = make_fake_realsense_states(args.cols, now_s)

                realsense_audio.apply(fake_realsense_states, now_s=now_s)
                pi_audio.apply(trip_states, now_s=now_s)

                vis = frame.image.copy()
                draw_trip_detections(vis, detections)
                cv2.imshow(WINDOW_NAME, vis)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        realsense_audio.stop()
        pi_audio.stop()
        with suppress(Exception):
            cv2.destroyWindow(WINDOW_NAME)
        cv2.destroyAllWindows()
        cv2.waitKey(1)


if __name__ == "__main__":
    run_demo()