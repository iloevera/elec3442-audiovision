"""Rotating two-tone demo for SpatialTone."""

from __future__ import annotations

import math
import time

from src.audio_spatial_tone import SpatialTone


def run_demo() -> None:
    """Run two layered tones orbiting the listener at different rates."""

    left_base_hz = 196.0  # G3
    right_base_hz = 523.25  # C5
    left_speed_deg_per_sec = 50.0
    right_speed_deg_per_sec = -85.0

    left_tone = SpatialTone(
        initial_pitch_hz=left_base_hz,
        initial_volume=0.25,
        initial_azimuth_deg=-90,
    )
    right_tone = SpatialTone(
        initial_pitch_hz=right_base_hz,
        initial_volume=0.20,
        initial_azimuth_deg=90,
    )
    left_tone.start()
    right_tone.start()

    print("Two tones rotating at different paces. Press Ctrl+C to stop.")
    try:
        start_t = time.perf_counter()
        while True:
            t = time.perf_counter() - start_t

            left_azimuth = -90.0 + left_speed_deg_per_sec * t
            right_azimuth = 90.0 + right_speed_deg_per_sec * t

            # Different tone character per voice via unique vibrato rates/depths.
            left_pitch = left_base_hz + 18.0 * math.sin(2.0 * math.pi * 0.40 * t)
            right_pitch = right_base_hz + 35.0 * math.sin(2.0 * math.pi * 0.17 * t)

            left_tone.set_params(pitch_hz=left_pitch, azimuth_deg=left_azimuth)
            right_tone.set_params(pitch_hz=right_pitch, azimuth_deg=right_azimuth)
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        left_tone.stop()
        right_tone.stop()


if __name__ == "__main__":
    run_demo()
