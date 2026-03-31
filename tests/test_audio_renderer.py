"""Tests for the binaural audio renderer."""

import math

import numpy as np
import pytest

from audiovision.audio.audio_renderer import AudioCue, AudioRenderer


@pytest.fixture
def renderer() -> AudioRenderer:
    return AudioRenderer(sample_rate=44100, frame_duration_s=0.033)


class TestAudioCue:
    def test_defaults(self):
        cue = AudioCue(frequency_hz=440.0, pan=0.0, volume=1.0)
        assert cue.duration_s == pytest.approx(0.033)

    def test_custom_duration(self):
        cue = AudioCue(frequency_hz=440.0, pan=0.0, volume=1.0, duration_s=0.1)
        assert cue.duration_s == pytest.approx(0.1)


class TestAudioRenderer:
    def test_silence_on_empty_cues(self, renderer):
        pcm = renderer.render([])
        assert pcm.dtype == np.int16
        assert pcm.ndim == 2
        assert pcm.shape[1] == 2
        assert np.all(pcm == 0)

    def test_output_shape(self, renderer):
        cue = AudioCue(frequency_hz=440.0, pan=0.0, volume=0.8)
        pcm = renderer.render([cue])
        expected_samples = int(44100 * 0.033)
        assert pcm.shape == (expected_samples, 2)
        assert pcm.dtype == np.int16

    def test_single_cue_produces_nonzero_output(self, renderer):
        cue = AudioCue(frequency_hz=440.0, pan=0.0, volume=1.0)
        pcm = renderer.render([cue])
        assert np.any(pcm != 0)

    def test_centre_pan_equal_channels(self, renderer):
        cue = AudioCue(frequency_hz=440.0, pan=0.0, volume=1.0)
        pcm = renderer.render([cue])
        left_rms = float(np.sqrt(np.mean(pcm[:, 0].astype(np.float64) ** 2)))
        right_rms = float(np.sqrt(np.mean(pcm[:, 1].astype(np.float64) ** 2)))
        assert left_rms == pytest.approx(right_rms, rel=0.01)

    def test_right_pan_louder_right(self, renderer):
        cue = AudioCue(frequency_hz=440.0, pan=1.0, volume=1.0)
        pcm = renderer.render([cue])
        left_rms = float(np.sqrt(np.mean(pcm[:, 0].astype(np.float64) ** 2)))
        right_rms = float(np.sqrt(np.mean(pcm[:, 1].astype(np.float64) ** 2)))
        assert right_rms > left_rms

    def test_left_pan_louder_left(self, renderer):
        cue = AudioCue(frequency_hz=440.0, pan=-1.0, volume=1.0)
        pcm = renderer.render([cue])
        left_rms = float(np.sqrt(np.mean(pcm[:, 0].astype(np.float64) ** 2)))
        right_rms = float(np.sqrt(np.mean(pcm[:, 1].astype(np.float64) ** 2)))
        assert left_rms > right_rms

    def test_no_clipping(self, renderer):
        cues = [
            AudioCue(frequency_hz=440.0, pan=0.0, volume=1.0),
            AudioCue(frequency_hz=880.0, pan=0.5, volume=1.0),
            AudioCue(frequency_hz=1320.0, pan=-0.5, volume=1.0),
        ]
        pcm = renderer.render(cues)
        assert np.max(np.abs(pcm)) <= 32767

    def test_multiple_cues_mixed(self, renderer):
        cue1 = AudioCue(frequency_hz=440.0, pan=0.0, volume=0.5)
        cue2 = AudioCue(frequency_hz=880.0, pan=0.0, volume=0.5)
        pcm_mixed = renderer.render([cue1, cue2])
        pcm_single = renderer.render([cue1])
        assert not np.array_equal(pcm_mixed, pcm_single)

    def test_zero_volume_cue_produces_silence(self, renderer):
        cue = AudioCue(frequency_hz=440.0, pan=0.0, volume=0.0)
        pcm = renderer.render([cue])
        assert np.all(pcm == 0)

    def test_pan_gains_constant_power(self):
        for pan in np.linspace(-1.0, 1.0, 9):
            gl, gr = AudioRenderer._pan_gains(pan)
            power = gl ** 2 + gr ** 2
            assert power == pytest.approx(1.0, abs=1e-10)
