"""SpatialTone voice class for continuous, controllable tone playback."""

from __future__ import annotations

import math
import threading
from typing import Optional

import numpy as np

from .audio_mixer import get_shared_mixer


class SpatialTone:
    """Continuously plays a synthetic tone with live spatial controls."""

    MIN_PITCH_HZ = 20.0
    MAX_PITCH_HZ = 20_000.0
    _MIXER = get_shared_mixer()

    def __init__(
        self,
        initial_pitch_hz: float = 440.0,
        initial_volume: float = 0.2,
        initial_azimuth_deg: float = 0.0,
        sample_rate: int = 48_000,
        block_size: int = 512,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)

        self._lock = threading.Lock()
        self._phase = 0.0
        self._active = False

        self._pitch_hz = self._clamp_pitch(initial_pitch_hz)
        self._volume = self._clamp_volume(initial_volume)
        self._azimuth_deg = self._normalize_azimuth(initial_azimuth_deg)
        self._left_gain, self._right_gain = self._azimuth_to_equal_power_gains(self._azimuth_deg)
        self._sample_idx = np.arange(self.block_size, dtype=np.float64)
        self._silence_block = np.zeros((self.block_size, 2), dtype=np.float32)

    @property
    def pitch_hz(self) -> float:
        with self._lock:
            return self._pitch_hz

    @property
    def volume(self) -> float:
        with self._lock:
            return self._volume

    @property
    def azimuth_deg(self) -> float:
        with self._lock:
            return self._azimuth_deg

    def start(self) -> None:
        """Start this tone voice inside the shared mixer."""
        with self._lock:
            if self._active:
                return
            self._active = True

        try:
            self._MIXER.register(self)
        except Exception:
            with self._lock:
                self._active = False
            raise

    def stop(self) -> None:
        """Stop this tone voice and unregister it from the shared mixer."""
        with self._lock:
            if not self._active:
                return
            self._active = False

        self._MIXER.unregister(self)

    def set_pitch(self, pitch_hz: float) -> None:
        with self._lock:
            self._pitch_hz = self._clamp_pitch(pitch_hz)

    def set_volume(self, volume: float) -> None:
        with self._lock:
            self._volume = self._clamp_volume(volume)

    def set_azimuth(self, azimuth_deg: float) -> None:
        with self._lock:
            self._azimuth_deg = self._normalize_azimuth(azimuth_deg)
            self._left_gain, self._right_gain = self._azimuth_to_equal_power_gains(self._azimuth_deg)

    def set_params(
        self,
        *,
        pitch_hz: Optional[float] = None,
        volume: Optional[float] = None,
        azimuth_deg: Optional[float] = None,
    ) -> None:
        """Atomically update any subset of parameters."""
        with self._lock:
            if pitch_hz is not None:
                self._pitch_hz = self._clamp_pitch(pitch_hz)
            if volume is not None:
                self._volume = self._clamp_volume(volume)
            if azimuth_deg is not None:
                self._azimuth_deg = self._normalize_azimuth(azimuth_deg)
                self._left_gain, self._right_gain = self._azimuth_to_equal_power_gains(self._azimuth_deg)

    def _render_stereo_block(self, frames: int) -> np.ndarray:
        with self._lock:
            active = self._active
            pitch_hz = self._pitch_hz
            volume = self._volume
            left_gain = self._left_gain
            right_gain = self._right_gain
            phase_start = self._phase

            phase_inc = (2.0 * math.pi * pitch_hz) / float(self.sample_rate)
            self._phase = (self._phase + frames * phase_inc) % (2.0 * math.pi)

        if not active or volume <= 0.0:
            if frames <= self.block_size:
                return self._silence_block[:frames]
            return np.zeros((frames, 2), dtype=np.float32)

        sample_idx = self._sample_idx[:frames] if frames <= self.block_size else np.arange(frames, dtype=np.float64)
        phases = phase_start + sample_idx * phase_inc
        mono = np.sin(phases).astype(np.float32)
        stereo = np.empty((frames, 2), dtype=np.float32)
        stereo[:, 0] = mono * np.float32(left_gain * volume)
        stereo[:, 1] = mono * np.float32(right_gain * volume)
        return stereo

    @classmethod
    def _clamp_pitch(cls, pitch_hz: float) -> float:
        pitch = float(pitch_hz)
        if pitch < cls.MIN_PITCH_HZ:
            return cls.MIN_PITCH_HZ
        if pitch > cls.MAX_PITCH_HZ:
            return cls.MAX_PITCH_HZ
        return pitch

    @staticmethod
    def _clamp_volume(volume: float) -> float:
        vol = float(volume)
        if vol < 0.0:
            return 0.0
        if vol > 1.0:
            return 1.0
        return vol

    @staticmethod
    def _normalize_azimuth(azimuth_deg: float) -> float:
        # Keep azimuth in [-180, 180).
        return ((float(azimuth_deg) + 180.0) % 360.0) - 180.0

    @staticmethod
    def _azimuth_to_equal_power_gains(azimuth_deg: float) -> tuple[float, float]:
        # Map [-90, 90] to pan [-1, 1]; clamp rear hemisphere to nearest side.
        pan = max(-1.0, min(1.0, float(azimuth_deg) / 90.0))
        theta = (pan + 1.0) * (math.pi / 4.0)
        return math.cos(theta), math.sin(theta)
