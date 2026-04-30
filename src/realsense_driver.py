# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Optional

import numpy as np
import pyrealsense2 as rs


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    coeffs: tuple[float, ...]
    model: str


@dataclass(frozen=True)
class CameraExtrinsics:
    rotation: tuple[float, ...]
    translation: tuple[float, ...]


@dataclass(frozen=True)
class DepthFrameData:
    image: np.ndarray
    intrinsics: CameraIntrinsics
    timestamp_ms: float
    host_timestamp_s: float
    frame_number: int
    depth_scale: float


@dataclass(frozen=True)
class ColorFrameData:
    image: np.ndarray
    intrinsics: CameraIntrinsics
    timestamp_ms: float
    host_timestamp_s: float
    frame_number: int


@dataclass(frozen=True)
class FrameBundle:
    depth: DepthFrameData
    color: ColorFrameData


class D435iDriver:
    """Background-threaded Intel RealSense D435 driver without IMU support."""

    def __init__(
        self,
        *,
        depth_size: tuple[int, int] = (640, 480),
        color_size: tuple[int, int] = (640, 480),
        depth_fps: int = 30,
        color_fps: int = 30,
        align_depth_to_color: bool = True,
        warmup_frames: int = 15,
        frame_timeout_ms: int = 5000,
    ) -> None:
        self.depth_size = tuple(depth_size)
        self.color_size = tuple(color_size)
        self.depth_fps = int(depth_fps)
        self.color_fps = int(color_fps)
        self.align_depth_to_color = bool(align_depth_to_color)
        self.warmup_frames = int(warmup_frames)
        self.frame_timeout_ms = int(frame_timeout_ms)

        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._align = rs.align(rs.stream.color) if self.align_depth_to_color else None
        self._pipeline_profile: Optional[rs.pipeline_profile] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._bundle_condition = threading.Condition(self._lock)

        self._running = False
        self._latest_bundle: Optional[FrameBundle] = None
        self._bundle_sequence = 0
        self._last_error: Optional[BaseException] = None

        self._depth_scale: Optional[float] = None
        self._depth_intrinsics: Optional[CameraIntrinsics] = None
        self._color_intrinsics: Optional[CameraIntrinsics] = None
        self._depth_to_color_extrinsics: Optional[CameraExtrinsics] = None
        self._color_to_depth_extrinsics: Optional[CameraExtrinsics] = None
        self._serial_number: Optional[str] = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def depth_scale(self) -> Optional[float]:
        with self._lock:
            return self._depth_scale

    @property
    def serial_number(self) -> Optional[str]:
        with self._lock:
            return self._serial_number

    @property
    def last_error(self) -> Optional[BaseException]:
        with self._lock:
            return self._last_error

    def __enter__(self) -> "D435iDriver":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        with self._lock:
            if self._running:
                return

        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_stream(
            rs.stream.depth,
            self.depth_size[0],
            self.depth_size[1],
            rs.format.z16,
            self.depth_fps,
        )
        self._config.enable_stream(
            rs.stream.color,
            self.color_size[0],
            self.color_size[1],
            rs.format.bgr8,
            self.color_fps,
        )

        try:
            profile = self._pipeline.start(self._config)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to start the RealSense depth/color pipeline. "
                "Check that the RealSense camera is connected over USB 3 and no other application is using the device."
            ) from exc

        try:
            self._initialize_device_metadata(profile)
            self._warm_up_pipeline()
        except Exception:
            self._pipeline.stop()
            raise

        with self._lock:
            self._pipeline_profile = profile
            self._latest_bundle = None
            self._bundle_sequence = 0
            self._last_error = None
            self._running = True
            self._stop_event.clear()

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="D435iCapture",
            daemon=True,
        )
        self._capture_thread.start()

    def stop(self) -> None:
        with self._lock:
            if not self._running and self._capture_thread is None:
                return
            self._running = False
            self._stop_event.set()
            capture_thread = self._capture_thread
            self._capture_thread = None
            self._bundle_condition.notify_all()

        if capture_thread is not None:
            capture_thread.join(timeout=2.0)

        try:
            self._pipeline.stop()
        except RuntimeError:
            pass

        with self._lock:
            self._pipeline_profile = None

    def get_latest_bundle(self) -> Optional[FrameBundle]:
        with self._lock:
            return self._latest_bundle

    def wait_for_bundle(self, timeout_s: float = 1.0) -> Optional[FrameBundle]:
        timeout_s = float(timeout_s)
        deadline = time.monotonic() + timeout_s

        with self._bundle_condition:
            start_sequence = self._bundle_sequence
            while self._running and self._bundle_sequence == start_sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._bundle_condition.wait(timeout=remaining)

            return self._latest_bundle

    def get_intrinsics(self) -> tuple[Optional[CameraIntrinsics], Optional[CameraIntrinsics]]:
        with self._lock:
            return self._depth_intrinsics, self._color_intrinsics

    def get_extrinsics(self) -> tuple[Optional[CameraExtrinsics], Optional[CameraExtrinsics]]:
        with self._lock:
            return self._depth_to_color_extrinsics, self._color_to_depth_extrinsics

    def get_depth_at_pixel(
        self,
        x: int,
        y: int,
        bundle: Optional[FrameBundle] = None,
    ) -> Optional[float]:
        current_bundle = bundle or self.get_latest_bundle()
        if current_bundle is None:
            return None

        height, width = current_bundle.depth.image.shape
        if x < 0 or y < 0 or x >= width or y >= height:
            raise IndexError(f"Pixel ({x}, {y}) is outside depth frame bounds {width}x{height}")

        return float(current_bundle.depth.image[y, x]) * current_bundle.depth.depth_scale

    def _initialize_device_metadata(self, profile: rs.pipeline_profile) -> None:
        device = profile.get_device()
        self._serial_number = device.get_info(rs.camera_info.serial_number)

        depth_sensor = device.first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())

        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()

        self._depth_intrinsics = self._convert_intrinsics(depth_profile.get_intrinsics())
        self._color_intrinsics = self._convert_intrinsics(color_profile.get_intrinsics())
        self._depth_to_color_extrinsics = self._convert_extrinsics(
            depth_profile.get_extrinsics_to(color_profile)
        )
        self._color_to_depth_extrinsics = self._convert_extrinsics(
            color_profile.get_extrinsics_to(depth_profile)
        )

    def _warm_up_pipeline(self) -> None:
        for _ in range(self.warmup_frames):
            self._pipeline.wait_for_frames(timeout_ms=self.frame_timeout_ms)

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=self.frame_timeout_ms)
                host_timestamp_s = time.monotonic()

                processed_frames = self._align.process(frames) if self._align is not None else frames
                depth_frame = processed_frames.get_depth_frame()
                color_frame = processed_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                bundle = self._build_bundle(depth_frame, color_frame, host_timestamp_s)
                with self._bundle_condition:
                    self._latest_bundle = bundle
                    self._bundle_sequence += 1
                    self._bundle_condition.notify_all()
            except RuntimeError as exc:
                with self._bundle_condition:
                    self._last_error = exc
                    self._running = False
                    self._bundle_condition.notify_all()
                break
            except Exception as exc:
                with self._bundle_condition:
                    self._last_error = exc
                    self._running = False
                    self._bundle_condition.notify_all()
                break

    def _build_bundle(
        self,
        depth_frame: rs.depth_frame,
        color_frame: rs.video_frame,
        host_timestamp_s: float,
    ) -> FrameBundle:
        depth_intrinsics = self._depth_intrinsics
        color_intrinsics = self._color_intrinsics
        depth_scale = self._depth_scale
        if depth_intrinsics is None or color_intrinsics is None or depth_scale is None:
            raise RuntimeError("Driver metadata was not initialized before capture")

        # Handle upside-down camera mounting by rotating 180 degrees (axis 0 and 1)
        depth_image = np.asanyarray(depth_frame.get_data()).copy()
        color_image = np.asanyarray(color_frame.get_data()).copy()
        depth_image = np.flip(depth_image, axis=(0, 1))
        color_image = np.flip(color_image, axis=(0, 1))

        depth_data = DepthFrameData(
            image=depth_image,
            intrinsics=depth_intrinsics,
            timestamp_ms=float(depth_frame.get_timestamp()),
            host_timestamp_s=host_timestamp_s,
            frame_number=int(depth_frame.get_frame_number()),
            depth_scale=depth_scale,
        )
        color_data = ColorFrameData(
            image=color_image,
            intrinsics=color_intrinsics,
            timestamp_ms=float(color_frame.get_timestamp()),
            host_timestamp_s=host_timestamp_s,
            frame_number=int(color_frame.get_frame_number()),
        )

        return FrameBundle(
            depth=depth_data,
            color=color_data,
        )

    @staticmethod
    def _convert_intrinsics(intrinsics: rs.intrinsics) -> CameraIntrinsics:
        return CameraIntrinsics(
            width=int(intrinsics.width),
            height=int(intrinsics.height),
            fx=float(intrinsics.fx),
            fy=float(intrinsics.fy),
            ppx=float(intrinsics.ppx),
            ppy=float(intrinsics.ppy),
            coeffs=tuple(float(value) for value in intrinsics.coeffs),
            model=str(intrinsics.model),
        )

    @staticmethod
    def _convert_extrinsics(extrinsics: rs.extrinsics) -> CameraExtrinsics:
        return CameraExtrinsics(
            rotation=tuple(float(value) for value in extrinsics.rotation),
            translation=tuple(float(value) for value in extrinsics.translation),
        )


if __name__ == "__main__":
    raise SystemExit("Run demo_realsense_preview.py for the OpenCV preview demo.")
