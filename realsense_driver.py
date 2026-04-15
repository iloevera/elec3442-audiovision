# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Optional

import numpy as np
import pyrealsense2 as rs


PREFERRED_ACCEL_FPS = (250, 200, 100, 400)
PREFERRED_GYRO_FPS = (200, 400)


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
class IMUSample:
    stream_name: str
    xyz: np.ndarray
    timestamp_ms: float
    host_timestamp_s: float
    frame_number: int


@dataclass(frozen=True)
class FrameBundle:
    depth: DepthFrameData
    color: ColorFrameData
    imu_samples: tuple[IMUSample, ...]
    latest_accel: Optional[IMUSample]
    latest_gyro: Optional[IMUSample]


@dataclass(frozen=True)
class MotionRuntimeConfig:
    serial_number: Optional[str]
    device_name: Optional[str]
    accel_fps: int
    gyro_fps: int


class D435iDriver:
    """Background-threaded Intel RealSense D435i driver for tracking workflows."""

    def __init__(
        self,
        *,
        depth_size: tuple[int, int] = (640, 480),
        color_size: tuple[int, int] = (640, 480),
        depth_fps: int = 30,
        color_fps: int = 30,
        enable_imu: bool = True,
        accel_fps: int = 250,
        gyro_fps: int = 200,
        align_depth_to_color: bool = True,
        warmup_frames: int = 15,
        frame_timeout_ms: int = 5000,
        imu_buffer_size: int = 512,
    ) -> None:
        self.depth_size = tuple(depth_size)
        self.color_size = tuple(color_size)
        self.depth_fps = int(depth_fps)
        self.color_fps = int(color_fps)
        self.enable_imu = bool(enable_imu)
        self.accel_fps = int(accel_fps)
        self.gyro_fps = int(gyro_fps)
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
        self._imu_enabled_runtime = False
        self._latest_bundle: Optional[FrameBundle] = None
        self._bundle_sequence = 0
        self._pending_imu_samples: list[IMUSample] = []
        self._imu_history: deque[IMUSample] = deque(maxlen=int(imu_buffer_size))
        self._latest_accel: Optional[IMUSample] = None
        self._latest_gyro: Optional[IMUSample] = None
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

    @property
    def imu_enabled_runtime(self) -> bool:
        with self._lock:
            return self._imu_enabled_runtime

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
        runtime_config = self._resolve_motion_runtime_config()

        self._config = self._build_config(
            enable_imu=self.enable_imu,
            serial_number=runtime_config.serial_number,
            accel_fps=runtime_config.accel_fps,
            gyro_fps=runtime_config.gyro_fps,
        )

        try:
            profile = self._pipeline.start(self._config)
        except RuntimeError as exc:
            if not self.enable_imu:
                raise

            raise RuntimeError(
                "Failed to start the RealSense depth/color/IMU pipeline. "
                f"Selected accel={runtime_config.accel_fps} Hz, gyro={runtime_config.gyro_fps} Hz "
                f"on device {runtime_config.device_name or 'unknown'} ({runtime_config.serial_number}). "
                "Check that the D435i is connected over USB 3 and no other application is using the device."
            ) from exc

        try:
            self._imu_enabled_runtime = self.enable_imu
            self._initialize_device_metadata(profile)
            self._warm_up_pipeline(require_imu=self.enable_imu)
        except Exception:
            self._pipeline.stop()
            self._imu_enabled_runtime = False
            raise

        with self._lock:
            self._pipeline_profile = profile
            self._imu_enabled_runtime = self.enable_imu
            self._latest_bundle = None
            self._bundle_sequence = 0
            self._pending_imu_samples.clear()
            self._imu_history.clear()
            self._latest_accel = None
            self._latest_gyro = None
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
            self._imu_enabled_runtime = False

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

    def get_recent_imu_samples(self) -> tuple[IMUSample, ...]:
        with self._lock:
            return tuple(self._imu_history)

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

    def _build_config(
        self,
        *,
        enable_imu: bool,
        serial_number: Optional[str],
        accel_fps: int,
        gyro_fps: int,
    ) -> rs.config:
        config = rs.config()
        if serial_number is not None:
            config.enable_device(serial_number)
        config.enable_stream(
            rs.stream.depth,
            self.depth_size[0],
            self.depth_size[1],
            rs.format.z16,
            self.depth_fps,
        )
        config.enable_stream(
            rs.stream.color,
            self.color_size[0],
            self.color_size[1],
            rs.format.bgr8,
            self.color_fps,
        )
        if enable_imu:
            config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, accel_fps)
            config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, gyro_fps)
        return config

    def _resolve_motion_runtime_config(self) -> MotionRuntimeConfig:
        if not self.enable_imu:
            return MotionRuntimeConfig(
                serial_number=None,
                device_name=None,
                accel_fps=self.accel_fps,
                gyro_fps=self.gyro_fps,
            )

        motion_device = self._find_motion_device()
        return MotionRuntimeConfig(
            serial_number=motion_device.get_info(rs.camera_info.serial_number),
            device_name=motion_device.get_info(rs.camera_info.name),
            accel_fps=self._choose_stream_fps(
                motion_device,
                stream_type=rs.stream.accel,
                requested_fps=self.accel_fps,
                preferred_fps=PREFERRED_ACCEL_FPS,
            ),
            gyro_fps=self._choose_stream_fps(
                motion_device,
                stream_type=rs.stream.gyro,
                requested_fps=self.gyro_fps,
                preferred_fps=PREFERRED_GYRO_FPS,
            ),
        )

    @staticmethod
    def _find_motion_device() -> rs.device:
        context = rs.context()
        devices = context.query_devices()

        if len(devices) == 0:
            raise RuntimeError("No Intel RealSense device detected. Connect a D435i and try again.")

        for device in devices:
            if D435iDriver._device_has_stream(device, rs.stream.accel) and D435iDriver._device_has_stream(
                device, rs.stream.gyro
            ):
                return device

        raise RuntimeError(
            "A RealSense device was found, but it does not expose both accelerometer and gyroscope streams. "
            "Use an Intel RealSense D435i."
        )

    @staticmethod
    def _device_has_stream(device: rs.device, stream_type: rs.stream) -> bool:
        for sensor in device.sensors:
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() == stream_type:
                    return True
        return False

    @staticmethod
    def _stream_fps_options(device: rs.device, stream_type: rs.stream) -> set[int]:
        options: set[int] = set()
        for sensor in device.sensors:
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() == stream_type:
                    options.add(profile.fps())
        return options

    @staticmethod
    def _stream_label(stream_type: rs.stream) -> str:
        if stream_type == rs.stream.accel:
            return "accel"
        if stream_type == rs.stream.gyro:
            return "gyro"
        return str(stream_type)

    @classmethod
    def _choose_stream_fps(
        cls,
        device: rs.device,
        *,
        stream_type: rs.stream,
        requested_fps: int,
        preferred_fps: tuple[int, ...],
    ) -> int:
        options = cls._stream_fps_options(device, stream_type)
        if requested_fps in options:
            return requested_fps

        selected_fps = next((fps for fps in preferred_fps if fps in options), None)
        if selected_fps is not None:
            return selected_fps

        stream_label = cls._stream_label(stream_type)
        raise RuntimeError(
            f"Could not find a supported {stream_label} stream configuration on this device. "
            f"Requested {stream_label}={requested_fps}; available options: {sorted(options)}"
        )

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

    def _warm_up_pipeline(self, *, require_imu: bool) -> None:
        saw_accel = False
        saw_gyro = False

        for _ in range(self.warmup_frames):
            frames = self._pipeline.wait_for_frames(timeout_ms=self.frame_timeout_ms)
            motion_samples = self._extract_imu_samples(frames, time.monotonic())
            for sample in motion_samples:
                if sample.stream_name == "accel":
                    saw_accel = True
                elif sample.stream_name == "gyro":
                    saw_gyro = True

        if require_imu and (not saw_accel or not saw_gyro):
            missing_streams: list[str] = []
            if not saw_accel:
                missing_streams.append("accel")
            if not saw_gyro:
                missing_streams.append("gyro")
            raise RuntimeError(
                "IMU pipeline started but did not produce all required motion streams during warmup. "
                f"Missing: {', '.join(missing_streams)}"
            )

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=self.frame_timeout_ms)
                host_timestamp_s = time.monotonic()
                motion_samples = self._extract_imu_samples(frames, host_timestamp_s)

                processed_frames = self._align.process(frames) if self._align is not None else frames
                depth_frame = processed_frames.get_depth_frame()
                color_frame = processed_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                bundle = self._build_bundle(depth_frame, color_frame, motion_samples, host_timestamp_s)
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
        motion_samples: list[IMUSample],
        host_timestamp_s: float,
    ) -> FrameBundle:
        depth_intrinsics = self._depth_intrinsics
        color_intrinsics = self._color_intrinsics
        depth_scale = self._depth_scale
        if depth_intrinsics is None or color_intrinsics is None or depth_scale is None:
            raise RuntimeError("Driver metadata was not initialized before capture")

        depth_data = DepthFrameData(
            image=np.asanyarray(depth_frame.get_data()).copy(),
            intrinsics=depth_intrinsics,
            timestamp_ms=float(depth_frame.get_timestamp()),
            host_timestamp_s=host_timestamp_s,
            frame_number=int(depth_frame.get_frame_number()),
            depth_scale=depth_scale,
        )
        color_data = ColorFrameData(
            image=np.asanyarray(color_frame.get_data()).copy(),
            intrinsics=color_intrinsics,
            timestamp_ms=float(color_frame.get_timestamp()),
            host_timestamp_s=host_timestamp_s,
            frame_number=int(color_frame.get_frame_number()),
        )

        with self._lock:
            self._pending_imu_samples.extend(motion_samples)
            imu_for_bundle = tuple(self._pending_imu_samples)
            self._pending_imu_samples = []
            latest_accel = self._latest_accel
            latest_gyro = self._latest_gyro

        return FrameBundle(
            depth=depth_data,
            color=color_data,
            imu_samples=imu_for_bundle,
            latest_accel=latest_accel,
            latest_gyro=latest_gyro,
        )

    def _extract_imu_samples(
        self,
        frames: rs.composite_frame,
        host_timestamp_s: float,
    ) -> list[IMUSample]:
        motion_samples: list[IMUSample] = []
        if not self._imu_enabled_runtime:
            return motion_samples

        for frame in frames:
            if not frame.is_motion_frame():
                continue

            motion_frame = frame.as_motion_frame()
            sample_stream_name = self._stream_label(frame.get_profile().stream_type())
            if sample_stream_name not in {"accel", "gyro"}:
                continue

            motion_data = motion_frame.get_motion_data()
            sample = IMUSample(
                stream_name=sample_stream_name,
                xyz=np.array([motion_data.x, motion_data.y, motion_data.z], dtype=np.float32),
                timestamp_ms=float(motion_frame.get_timestamp()),
                host_timestamp_s=host_timestamp_s,
                frame_number=int(motion_frame.get_frame_number()),
            )
            motion_samples.append(sample)

            with self._lock:
                self._imu_history.append(sample)
                if sample_stream_name == "accel":
                    self._latest_accel = sample
                else:
                    self._latest_gyro = sample

        return motion_samples

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
