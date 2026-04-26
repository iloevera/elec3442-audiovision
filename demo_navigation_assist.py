from __future__ import annotations

import argparse
from contextlib import suppress
from dataclasses import dataclass
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.navigation_processing import NavigationFrameAnalysis, NavigationProcessor, NavigationProcessorConfig
from src.realsense_driver import D435iDriver

from src.pi_trip_camera import PiTripCamera
from src.pi_trip_hazard import PiTripHazardDetector
from src.pi_trip_hazard_states import build_trip_column_states

if TYPE_CHECKING:
    from src.navigation_audio import NavigationAudioController


WINDOW_NAME = "Assistive Navigation Debug"
WINDOW_SIZE = (1520, 980)


@dataclass(frozen=True)
class RuntimeModeSettings:
    depth_size: tuple[int, int]
    color_size: tuple[int, int]
    depth_fps: int
    color_fps: int
    align_depth_to_color: bool
    processor_config: NavigationProcessorConfig
    preview_default: bool
    window_size: tuple[int, int]


def resolve_mode_settings(mode: str) -> RuntimeModeSettings:
    if mode == "pi_normal":
        return RuntimeModeSettings(
            depth_size=(424, 240),
            color_size=(424, 240),
            depth_fps=30,
            color_fps=30,
            align_depth_to_color=False,
            processor_config=NavigationProcessorConfig(
                downsample_step=3,
                ransac_iterations=30,
                min_plane_inliers=180,
                ground_plane_refit_interval_frames=3,
            ),
            preview_default=False,
            window_size=(960, 600),
        )

    if mode == "pi_debug":
        return RuntimeModeSettings(
            depth_size=(424, 240),
            color_size=(424, 240),
            depth_fps=30,
            color_fps=30,
            align_depth_to_color=False,
            processor_config=NavigationProcessorConfig(
                downsample_step=3,
                ransac_iterations=35,
                min_plane_inliers=200,
                ground_plane_refit_interval_frames=2,
            ),
            preview_default=True,
            window_size=(1100, 700),
        )

    return RuntimeModeSettings(
        depth_size=(640, 480),
        color_size=(640, 480),
        depth_fps=60,
        color_fps=60,
        align_depth_to_color=True,
        processor_config=NavigationProcessorConfig(),
        preview_default=True,
        window_size=WINDOW_SIZE,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Depth+IMU assistive navigation demo")
    parser.add_argument(
        "--mode",
        choices=("desktop_debug", "pi_normal", "pi_debug"),
        default="desktop_debug",
        help="Runtime profile for latency/quality trade-offs",
    )
    parser.add_argument(
        "--profile",
        choices=("desktop_debug", "pi_normal", "pi_debug"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--preview", action="store_true", help="Force-enable preview window")
    parser.add_argument("--no-preview", action="store_true", help="Disable the OpenCV debug window")
    parser.add_argument(
        "--preview-fps",
        type=float,
        default=10.0,
        help="Preview refresh rate target (Pi debug mode uses this to throttle rendering)",
    )
    parser.add_argument("--no-audio", action="store_true", help="Disable spatial audio output")
    return parser.parse_args()


def run_demo() -> None:
    args = parse_args()
    mode = args.profile or args.mode
    settings = resolve_mode_settings(mode)

    preview_enabled = settings.preview_default
    if args.preview:
        preview_enabled = True
    if args.no_preview:
        preview_enabled = False

    audio_enabled = not args.no_audio
    preview_stride = 1
    if preview_enabled and mode == "pi_debug":
        camera_fps = float(settings.color_fps)
        target_preview_fps = max(1.0, float(args.preview_fps))
        preview_stride = max(1, int(round(camera_fps / target_preview_fps)))

    if preview_enabled:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, *settings.window_size)

    processor = NavigationProcessor(config=settings.processor_config)
    audio: NavigationAudioController | None = None
    frame_index = 0
    
    pi_audio: NavigationAudioController | None = None
    trip_camera = PiTripCamera(camera_index=0)
    trip_detector = PiTripHazardDetector(column_count=processor.config.cols)

    try:
        with D435iDriver(
            depth_size=settings.depth_size,
            color_size=settings.color_size,
            depth_fps=settings.depth_fps,
            color_fps=settings.color_fps,
            align_depth_to_color=settings.align_depth_to_color,
        ) as driver:
            while True:
                bundle = driver.wait_for_bundle(timeout_s=1.0)
                if bundle is None:
                    if driver.last_error is not None:
                        raise RuntimeError("Capture thread stopped after an error") from driver.last_error
                    continue

                if audio_enabled and audio is None:
                    # Delay audio import/startup until after first camera bundle arrives.
                    from src.navigation_audio import NavigationAudioController, NavigationAudioConfig

                    audio = NavigationAudioController(column_count=processor.config.cols, config= NavigationAudioConfig(use_pulse_gating=False),)
                    audio.start()

                    pi_audio = NavigationAudioController(column_count=processor.config.cols, config= NavigationAudioConfig(use_pulse_gating=True),)
                    pi_audio.start()

                analysis = processor.process_bundle(bundle)

                trip_frame = trip_camera.read()
                trip_detections = ()
                if trip_frame is not None:
                    trip_detections = trip_detector.detect(trip_frame.image)

                trip_states = build_trip_column_states(
                    trip_detections=trip_detections,
                    cols=processor.config.cols,
                )
    
                if pi_audio is not None:
                    pi_audio.apply(trip_states, now_s=time.monotonic())

                if audio is not None:
                    audio.apply(analysis.column_states, now_s=time.monotonic())
                if preview_enabled:
                    if frame_index % preview_stride == 0:
                        frame = compose_debug_frame(bundle.color.image, bundle.depth.image, analysis)
                        cv2.imshow(WINDOW_NAME, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                frame_index += 1
    finally:
        if audio is not None:
            audio.stop()
        if pi_audio is not None:
            pi_audio.stop()
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

            depth_value = analysis.percentile_depth_grid_m[row, col]
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
        f"Assistive Navigation: depth + IMU + P{analysis.depth_percentile:g}/TTC grid",
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
            f"p{analysis.depth_percentile:g}={format_optional(column_state.percentile_depth_m, 'm')}  ttc={format_optional(column_state.ttc_s, 's')}",
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