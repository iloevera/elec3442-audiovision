from __future__ import annotations

import argparse
from contextlib import suppress
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

from navigation_processing import NavigationFrameAnalysis, NavigationProcessor
from realsense_driver import D435iDriver

if TYPE_CHECKING:
    from navigation_audio import NavigationAudioController


WINDOW_NAME = "Assistive Navigation Debug"
WINDOW_SIZE = (1520, 980)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Depth+IMU assistive navigation demo")
    parser.add_argument("--no-preview", action="store_true", help="Disable the OpenCV debug window")
    parser.add_argument("--no-audio", action="store_true", help="Disable spatial audio output")
    return parser.parse_args()


def run_demo() -> None:
    args = parse_args()
    preview_enabled = not args.no_preview
    audio_enabled = not args.no_audio

    if preview_enabled:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, *WINDOW_SIZE)

    processor = NavigationProcessor()
    audio: NavigationAudioController | None = None

    try:
        with D435iDriver() as driver:
            while True:
                bundle = driver.wait_for_bundle(timeout_s=1.0)
                if bundle is None:
                    if driver.last_error is not None:
                        raise RuntimeError("Capture thread stopped after an error") from driver.last_error
                    continue

                if audio_enabled and audio is None:
                    # Delay audio import/startup until after first camera bundle arrives.
                    from navigation_audio import NavigationAudioController

                    audio = NavigationAudioController(column_count=processor.config.cols)
                    audio.start()

                analysis = processor.process_bundle(bundle)
                if audio is not None:
                    audio.apply(analysis.column_states, now_s=time.monotonic())

                if preview_enabled:
                    frame = compose_debug_frame(bundle.color.image, bundle.depth.image, analysis)
                    cv2.imshow(WINDOW_NAME, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
    finally:
        if audio is not None:
            audio.stop()
        if preview_enabled:
            with suppress(Exception):
                cv2.destroyWindow(WINDOW_NAME)
            cv2.destroyAllWindows()


def compose_debug_frame(
    color_image: np.ndarray,
    depth_raw: np.ndarray,
    analysis: NavigationFrameAnalysis,
) -> np.ndarray:
    depth_vis = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_raw, alpha=0.03),
        cv2.COLORMAP_TURBO,
    )
    overlay = depth_vis.copy()
    overlay[analysis.ground_mask] = (48, 96, 48)
    overlay[analysis.obstacle_mask] = (40, 40, 220)

    color_panel = color_image.copy()
    depth_panel = cv2.addWeighted(depth_vis, 0.45, overlay, 0.55, 0.0)

    if color_panel.shape[:2] != depth_panel.shape[:2]:
        depth_panel = cv2.resize(depth_panel, (color_panel.shape[1], color_panel.shape[0]))

    draw_grid(color_panel, analysis)
    draw_grid(depth_panel, analysis)

    top = np.hstack((color_panel, depth_panel))
    info = build_info_panel(analysis, width=top.shape[1])
    return np.vstack((top, info))


def draw_grid(image: np.ndarray, analysis: NavigationFrameAnalysis) -> None:
    rows, cols = analysis.risk_grid.shape
    height, width = image.shape[:2]
    row_edges = np.linspace(0, height, rows + 1, dtype=int)
    col_edges = np.linspace(0, width, cols + 1, dtype=int)

    for row in range(rows):
        for col in range(cols):
            x0 = col_edges[col]
            x1 = col_edges[col + 1]
            y0 = row_edges[row]
            y1 = row_edges[row + 1]
            risk = float(analysis.risk_grid[row, col])
            color = risk_to_bgr(risk)
            cv2.rectangle(image, (x0, y0), (x1, y1), color, 1)

            depth_value = analysis.q1_depth_grid_m[row, col]
            ttc_value = analysis.ttc_grid_s[row, col]
            label = f"R{risk:0.2f}"
            if np.isfinite(depth_value):
                label += f" D{depth_value:0.2f}"
            if np.isfinite(ttc_value):
                label += f" T{ttc_value:0.1f}"

            cv2.putText(
                image,
                label,
                (x0 + 8, y0 + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )


def build_info_panel(analysis: NavigationFrameAnalysis, width: int) -> np.ndarray:
    panel_height = 210
    panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
    panel[:] = (18, 18, 18)

    cv2.putText(
        panel,
        "Assistive Navigation: depth + IMU + Q1/TTC grid",
        (18, 28),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    plane_text = "Ground plane: unavailable"
    if analysis.ground_plane is not None:
        plane_text = (
            f"Ground plane inliers={analysis.ground_plane.inlier_count}  "
            f"residual={analysis.ground_plane.mean_residual_m:0.3f} m"
        )
    cv2.putText(
        panel,
        plane_text,
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    gravity_text = "Gravity: unavailable"
    if analysis.gravity_unit is not None:
        gravity = analysis.gravity_unit
        gravity_text = f"Gravity unit: [{gravity[0]: .2f}, {gravity[1]: .2f}, {gravity[2]: .2f}]"
    cv2.putText(
        panel,
        gravity_text,
        (18, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    start_x = 18
    start_y = 118
    col_spacing = max(200, width // max(1, len(analysis.column_states)))
    for column_state in analysis.column_states:
        x = start_x + column_state.col * col_spacing
        lines = [
            f"Col {column_state.col}  az={column_state.azimuth_deg: .0f}",
            f"risk={column_state.risk_score:0.2f}  vol={column_state.volume:0.2f}",
            f"q1={format_optional(column_state.q1_depth_m, 'm')}  ttc={format_optional(column_state.ttc_s, 's')}",
            f"pitch={column_state.pitch_hz:0.0f}Hz  urgency={column_state.pulse_hz:0.1f}",
        ]
        for index, line in enumerate(lines):
            cv2.putText(
                panel,
                line,
                (x, start_y + index * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (236, 236, 236),
                1,
                cv2.LINE_AA,
            )

    return panel


def risk_to_bgr(risk: float) -> tuple[int, int, int]:
    clamped = float(np.clip(risk, 0.0, 1.0))
    blue = int((1.0 - clamped) * 80)
    green = int((1.0 - clamped) * 165)
    red = int(90 + clamped * 165)
    return blue, green, red


def format_optional(value: float | None, units: str) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:0.2f}{units}"


if __name__ == "__main__":
    run_demo()