"""End-to-end demo: RealSense depth obstacle tracking driving spatial audio."""

from __future__ import annotations

import argparse
import math
import signal
import threading
import time

import cv2
import numpy as np

from obstacle_audio_adapter import ObstacleAudioAdapter
from obstacle_models import ObstacleTrackerConfig, ObstacleUpdate
from obstacle_tracker import ObstacleTrackerService
from realsense_driver import CameraIntrinsics, D435iDriver


class DemoVisualizer:
    def __init__(self, window_name: str = "Obstacle Audio Tracking") -> None:
        self._window_name = window_name

    def render(
        self,
        update: ObstacleUpdate,
        driver: D435iDriver,
        *,
        imu_enabled: bool,
    ) -> bool:
        bundle = driver.get_latest_bundle()
        frame = self._build_frame(update=update, bundle=bundle, imu_enabled=imu_enabled)

        try:
            cv2.imshow(self._window_name, frame)
            key = cv2.waitKey(1) & 0xFF
        except cv2.error:
            return False

        return key in (27, ord("q"))

    def close(self) -> None:
        try:
            cv2.destroyWindow(self._window_name)
        except cv2.error:
            return

    def _build_frame(
        self,
        update: ObstacleUpdate,
        bundle,
        *,
        imu_enabled: bool,
    ) -> np.ndarray:
        if bundle is None:
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
            self._draw_status(canvas, update=update, imu_enabled=imu_enabled)
            cv2.putText(
                canvas,
                "Waiting for depth frames...",
                (16, 96),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 200, 200),
                2,
                cv2.LINE_AA,
            )
            return canvas

        frame = cv2.applyColorMap(
            cv2.convertScaleAbs(bundle.depth.image, alpha=0.03),
            cv2.COLORMAP_JET,
        )

        self._draw_status(frame, update=update, imu_enabled=imu_enabled)

        for obstacle in sorted(update.obstacles, key=lambda item: item.collision_score, reverse=True):
            pixel = self._project_xyz_to_pixel(obstacle.xyz_m, bundle.depth.intrinsics)
            if pixel is None:
                continue

            x_px, y_px = pixel
            color = self._risk_color(obstacle.collision_score, obstacle.is_collision_course)
            radius_px = self._marker_radius(obstacle.distance_m)

            cv2.circle(frame, (x_px, y_px), radius_px, color, 2, cv2.LINE_AA)
            cv2.circle(frame, (x_px, y_px), 3, color, -1, cv2.LINE_AA)

            ttc_text = self._format_ttc(obstacle.ttc_s)
            label = (
                f"id={obstacle.obstacle_id} d={obstacle.distance_m:.2f}m "
                f"ttc={ttc_text} s={obstacle.collision_score:.2f}"
            )
            cv2.putText(
                frame,
                label,
                (x_px + 10, max(22, y_px - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

        return frame

    def _draw_status(self, frame: np.ndarray, *, update: ObstacleUpdate, imu_enabled: bool) -> None:
        header = (
            f"state={update.state} frame={update.frame_number} obstacles={len(update.obstacles)} "
            f"imu={'on' if imu_enabled else 'off'} compensated={'yes' if update.imu_compensated else 'no'}"
        )
        cv2.putText(
            frame,
            header,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Press q or Esc in this window to stop",
            (12, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _project_xyz_to_pixel(xyz_m: np.ndarray, intrinsics: CameraIntrinsics) -> tuple[int, int] | None:
        z_m = float(xyz_m[2])
        if z_m <= 0.0:
            return None

        x_px = int(round((float(xyz_m[0]) / z_m) * intrinsics.fx + intrinsics.ppx))
        y_px = int(round((float(xyz_m[1]) / z_m) * intrinsics.fy + intrinsics.ppy))
        if x_px < 0 or y_px < 0 or x_px >= intrinsics.width or y_px >= intrinsics.height:
            return None
        return x_px, y_px

    @staticmethod
    def _risk_color(collision_score: float, is_collision_course: bool) -> tuple[int, int, int]:
        if is_collision_course:
            return (0, 0, 255)
        if collision_score >= 0.5:
            return (0, 165, 255)
        return (0, 255, 0)

    @staticmethod
    def _marker_radius(distance_m: float) -> int:
        return max(8, min(28, int(round(40.0 / max(distance_m, 0.5)))))

    @staticmethod
    def _format_ttc(ttc_s: float) -> str:
        if not math.isfinite(ttc_s):
            return "inf"
        return f"{ttc_s:.2f}s"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Depth obstacle tracking audio demo.")
    parser.add_argument(
        "--disable-imu",
        action="store_true",
        help="Force depth-only mode (useful for D435 without IMU).",
    )
    parser.add_argument(
        "--require-imu",
        action="store_true",
        help="Exit if IMU is unavailable instead of falling back to depth-only mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stop_event = threading.Event()
    visualizer = DemoVisualizer()

    def _handle_sigint(_signum, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_sigint)

    adapter = ObstacleAudioAdapter(warning_ttc_s=2.5)

    last_print_s = 0.0
    driver: D435iDriver | None = None

    def on_update(update: ObstacleUpdate) -> None:
        nonlocal last_print_s
        adapter.on_update(update)

        if driver is not None and visualizer.render(update, driver, imu_enabled=driver.imu_enabled_runtime):
            stop_event.set()

        now_s = time.monotonic()
        if now_s - last_print_s < 0.5:
            return
        last_print_s = now_s

        risk = sorted(update.obstacles, key=lambda obs: obs.collision_score, reverse=True)
        if not risk:
            print("state=degraded obstacles=0")
            return

        top = risk[0]
        ttc_text = "inf" if top.ttc_s == float("inf") else f"{top.ttc_s:.2f}s"
        print(
            "state={state} count={count} top_id={obstacle_id} "
            "xyz=({x:.2f},{y:.2f},{z:.2f})m ttc={ttc} score={score:.2f} conf={conf:.2f}".format(
                state=update.state,
                count=len(update.obstacles),
                obstacle_id=top.obstacle_id,
                x=top.xyz_m[0],
                y=top.xyz_m[1],
                z=top.xyz_m[2],
                ttc=ttc_text,
                score=top.collision_score,
                conf=top.confidence,
            )
        )

    request_imu = not args.disable_imu

    with D435iDriver(enable_imu=request_imu) as active_driver:
        driver = active_driver
        if args.require_imu and not driver.imu_enabled_runtime:
            raise RuntimeError("IMU is required but unavailable. Remove --require-imu or connect a D435i.")

        imu_comp_enabled = request_imu and driver.imu_enabled_runtime
        config = ObstacleTrackerConfig(
            enable_imu_compensation=imu_comp_enabled,
            warning_ttc_s=2.5,
            urgent_ttc_s=1.0,
        )

        if imu_comp_enabled:
            print("IMU: enabled (gyro compensation active)")
        else:
            print("IMU: disabled/unavailable (running in depth-only mode)")

        service = ObstacleTrackerService(driver=driver, callback=on_update, config=config)
        service.start()

        print("Running obstacle audio tracking demo. Press Ctrl+C to stop, or q/Esc in the OpenCV window.")
        try:
            while not stop_event.is_set():
                time.sleep(0.1)
        finally:
            service.stop()
            adapter.close()
            visualizer.close()


if __name__ == "__main__":
    main()
