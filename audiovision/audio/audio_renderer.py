"""Binaural audio rendering using synthesised tones.

Generates stereo (binaural) PCM audio frames from a list of :class:`AudioCue`
objects.  Each cue maps to a sine-wave tone with:

* **Pitch** – controlled by ``frequency_hz``.
* **Panning** – controlled by ``pan`` via a simple constant-power panner that
  approximates HRTF azimuth cues.  A full HRTF implementation (e.g. via
  OpenAL-Soft / SOFA) can replace the panning stage without changing the
  public interface.
* **Volume** – controlled by ``volume`` (normalised to ``[0, 1]``).

All mixed cues are summed, normalised to prevent clipping, and returned as a
``numpy`` int16 stereo array suitable for writing to an audio device or file
with ``soundfile`` / ``pyaudio``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import List

import numpy as np


@dataclasses.dataclass
class AudioCue:
    """A single audio event to be rendered.

    Attributes:
        frequency_hz: Tone pitch in Hz.
        pan: Stereo pan in ``[-1, 1]`` (left = -1, centre = 0, right = +1).
        volume: Normalised amplitude in ``[0, 1]``.
        duration_s: Duration of the tone in seconds (default: one frame).
    """

    frequency_hz: float
    pan: float
    volume: float
    duration_s: float = 0.033


class AudioRenderer:
    """Mix multiple :class:`AudioCue` objects into a stereo PCM buffer.

    Args:
        sample_rate: Audio sample rate in Hz.
        frame_duration_s: Default frame duration in seconds.  Individual cues
            can override this via ``AudioCue.duration_s``.
        amplitude_scale: Global amplitude scale factor applied before
            int16 conversion.

    Example::

        renderer = AudioRenderer(sample_rate=44100)
        cues = [
            AudioCue(frequency_hz=440.0, pan=0.0,  volume=0.8),
            AudioCue(frequency_hz=880.0, pan=-0.5, volume=0.4),
        ]
        pcm = renderer.render(cues)
        # write pcm to an audio device or file …
    """

    INT16_MAX: int = 32767

    def __init__(
        self,
        sample_rate: int = 44100,
        frame_duration_s: float = 0.033,
        amplitude_scale: float = 0.9,
    ) -> None:
        self._sr = sample_rate
        self._frame_dur = frame_duration_s
        self._scale = amplitude_scale

    def render(self, cues: List[AudioCue]) -> np.ndarray:
        """Render *cues* to a stereo int16 PCM array.

        Args:
            cues: List of :class:`AudioCue` objects to mix.

        Returns:
            ``np.ndarray`` of shape ``(N, 2)`` and dtype ``int16`` where
            column 0 is the left channel and column 1 is the right channel.
            Returns silence (all zeros) when *cues* is empty.
        """
        n_samples = int(self._sr * self._frame_dur)

        if not cues:
            return np.zeros((n_samples, 2), dtype=np.int16)

        mix_left = np.zeros(n_samples, dtype=np.float64)
        mix_right = np.zeros(n_samples, dtype=np.float64)

        for cue in cues:
            tone_len = min(int(self._sr * cue.duration_s), n_samples)
            tone = self._sine_wave(cue.frequency_hz, tone_len)

            gain_l, gain_r = self._pan_gains(cue.pan)
            mix_left[:tone_len] += tone * gain_l * cue.volume
            mix_right[:tone_len] += tone * gain_r * cue.volume

        mix_left, mix_right = self._normalise(mix_left, mix_right)

        stereo = np.stack([mix_left, mix_right], axis=1)
        return (stereo * self.INT16_MAX).astype(np.int16)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sine_wave(self, frequency_hz: float, n_samples: int) -> np.ndarray:
        t = np.arange(n_samples, dtype=np.float64) / self._sr
        return np.sin(2.0 * math.pi * frequency_hz * t)

    @staticmethod
    def _pan_gains(pan: float) -> tuple:
        """Constant-power panning law.

        *pan* is in ``[-1, 1]``.  Returns ``(gain_left, gain_right)``.
        """
        angle = (np.clip(pan, -1.0, 1.0) + 1.0) * (math.pi / 4.0)
        gain_left = math.cos(angle)
        gain_right = math.sin(angle)
        return gain_left, gain_right

    def _normalise(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> tuple:
        peak = max(np.max(np.abs(left)), np.max(np.abs(right)), 1e-9)
        factor = self._scale / peak if peak > self._scale else 1.0
        return left * factor, right * factor
