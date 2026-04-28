"""Shared audio mixer for layering multiple tone voices."""

from __future__ import annotations

import threading
from typing import Optional, Protocol

import numpy as np
import sounddevice as sd


class AudioVoice(Protocol):
    """Structural interface required by the mixer."""

    sample_rate: int
    block_size: int

    def _render_stereo_block(self, frames: int) -> np.ndarray:
        ...


class AudioMixer:
    """Process-wide mixer that owns a single output stream."""

    _instance: Optional["AudioMixer"] = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> "AudioMixer":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._initialized = True
        self._registry_lock = threading.RLock()
        self._stream: Optional["sd.OutputStream"] = None
        self._voices: set[AudioVoice] = set()
        self._voices_snapshot: tuple[AudioVoice, ...] = ()
        self._sample_rate = 48_000
        self._block_size = 512
        self._mix_buffer = np.zeros((self._block_size, 2), dtype=np.float32)

    def register(self, voice: AudioVoice) -> None:
        if sd is None:
            raise RuntimeError(
                "sounddevice is not installed. Install it with: pip install sounddevice"
            )

        with self._registry_lock:
            if self._voices:
                if (
                    voice.sample_rate != self._sample_rate
                    or voice.block_size != self._block_size
                ):
                    raise ValueError(
                        "All active SpatialTone instances must share sample_rate and block_size"
                    )
            else:
                self._sample_rate = voice.sample_rate
                self._block_size = voice.block_size

            already_present = voice in self._voices
            self._voices.add(voice)
            self._voices_snapshot = tuple(self._voices)
            if not already_present and self._stream is None:
                self._start_stream_locked()

    def unregister(self, voice: AudioVoice) -> None:
        with self._registry_lock:
            self._voices.discard(voice)
            self._voices_snapshot = tuple(self._voices)
            if not self._voices:
                self._stop_stream_locked()

    def _start_stream_locked(self) -> None:
        try:
            stream = sd.OutputStream(
                samplerate=self._sample_rate,
                channels=2,
                dtype="float32",
                blocksize=self._block_size,
                callback=self._audio_callback,
            )
            stream.start()
        except Exception as exc:  # pragma: no cover - device dependent
            raise RuntimeError(f"Failed to start output stream: {exc}") from exc

        self._stream = stream

    def _stop_stream_locked(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            stream.stop()
            stream.close()

    def _audio_callback(self, outdata, frames, _time_info, _status) -> None:
        voices = self._voices_snapshot

        if not voices:
            outdata.fill(0)
            return

        if self._mix_buffer.shape[0] < frames:
            self._mix_buffer = np.zeros((frames, 2), dtype=np.float32)

        mixed = self._mix_buffer[:frames]
        mixed.fill(0.0)
        for voice in voices:
            mixed += voice._render_stereo_block(frames)

        # Keep overall level bounded as voice count grows.
        mixed *= 1.0 / max(1.0, float(len(voices)))
        np.clip(mixed, -1.0, 1.0, out=mixed)
        outdata[:] = mixed


def get_shared_mixer() -> AudioMixer:
    """Return the singleton audio mixer instance."""

    return AudioMixer()
