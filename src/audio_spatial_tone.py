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
        waveform: str = "sine",
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
        self._waveform = waveform
        self._left_gain, self._right_gain = self._calculate_spatial_params(self._azimuth_deg)
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

    @property
    def waveform(self) -> str:
        with self._lock:
            return self._waveform

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
            # Constrain azimuth to +/- 90 degrees as requested
            self._azimuth_deg = max(-90.0, min(90.0, float(azimuth_deg)))
            self._left_gain, self._right_gain = self._calculate_spatial_params(self._azimuth_deg)

    def set_waveform(self, waveform: str) -> None:
        with self._lock:
            self._waveform = waveform

    def set_params(
        self,
        *,
        pitch_hz: Optional[float] = None,
        volume: Optional[float] = None,
        azimuth_deg: Optional[float] = None,
        waveform: Optional[str] = None,
    ) -> None:
        """Atomically update any subset of parameters."""
        with self._lock:
            if pitch_hz is not None:
                self._pitch_hz = self._clamp_pitch(pitch_hz)
            if volume is not None:
                self._volume = self._clamp_volume(volume)
            if azimuth_deg is not None:
                # Constrain azimuth to +/- 90 degrees
                self._azimuth_deg = max(-90.0, min(90.0, float(azimuth_deg)))
                self._left_gain, self._right_gain = self._calculate_spatial_params(self._azimuth_deg)
            if waveform is not None:
                self._waveform = waveform

    def _render_stereo_block(self, frames: int) -> np.ndarray:
        with self._lock:
            active = self._active
            pitch_hz = self._pitch_hz
            volume = self._volume
            left_gain = self._left_gain
            right_gain = self._right_gain
            phase_start = self._phase
            waveform = self._waveform
            azimuth_deg = self._azimuth_deg

            phase_inc = (2.0 * math.pi * pitch_hz) / float(self.sample_rate)
            self._phase = (self._phase + frames * phase_inc) % (2.0 * math.pi)

        if not active or volume <= 0.0:
            if frames <= self.block_size:
                return self._silence_block[:frames]
            return np.zeros((frames, 2), dtype=np.float32)

        sample_idx = self._sample_idx[:frames] if frames <= self.block_size else np.arange(frames, dtype=np.float64)
        phases = phase_start + sample_idx * phase_inc

        if waveform == "triangle":
            # Map phase [0, 2pi] to [0, 1]
            x = (phases / (2.0 * math.pi)) % 1.0
            mono = (4.0 * np.abs(x - 0.5) - 1.0).astype(np.float32)
        else:  # default to sine
            mono = np.sin(phases).astype(np.float32)

        # Removed ITD (Interaural Time Difference) to fix buzzing/clicking artifacts.
        # Focused only on ILD (Interaural Level Difference) with azimuth constrained to +/- 90 deg.
        left = mono
        right = mono

        # Apply ILD (Interaural Level Difference) and Volume
        stereo = np.empty((frames, 2), dtype=np.float32)
        stereo[:, 0] = left * np.float32(left_gain * volume)
        stereo[:, 1] = right * np.float32(right_gain * volume)
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

    def _calculate_spatial_params(self, azimuth_deg: float) -> tuple[float, float]:
        """Calculates gains for ILD."""
        theta = np.deg2rad(azimuth_deg)
        # Simple ILD approximation up to +/- 6 dB as in the attached file
        ild_db = 6.0 * np.sin(theta)
        right_gain = 10.0 ** (ild_db / 20.0)
        left_gain = 10.0 ** (-ild_db / 20.0)
        return float(left_gain), float(right_gain)

