from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Final

from .audio_spatial_tone import SpatialTone
from .navigation_processing import NavigationColumnState


@dataclass(frozen=True)
class NavigationAudioConfig:
    sample_rate: int = 48_000
    block_size: int = 512
    base_volume: float = 0.22
    output_gain: float = 2.0
    min_audible_volume: float = 0.08
    near_priority_distance_m: float = 0.50
    ttc_priority_horizon_s: float = 5.0
    max_simultaneous_columns: int | None = 3
    use_pulse_gating: bool = False


@dataclass(frozen=True)
class TexturePresetSettings:
    pitch_scale: float
    gain_scale: float
    brightness: float
    roughness_depth: float
    roughness_hz: float
    pulse_rate_scale: float
    pulse_duty_cycle: float


TEXTURE_PRESETS: Final[dict[str, TexturePresetSettings]] = {
    "original": TexturePresetSettings(
        pitch_scale=1.0,
        gain_scale=1.0,
        brightness=0.0,
        roughness_depth=0.0,
        roughness_hz=0.0,
        pulse_rate_scale=1.0,
        pulse_duty_cycle=0.25,
    ),
    "smooth": TexturePresetSettings(
        pitch_scale=0.94,
        gain_scale=0.93,
        brightness=0.12,
        roughness_depth=0.01,
        roughness_hz=2.6,
        pulse_rate_scale=0.80,
        pulse_duty_cycle=0.40,
    ),
    "alert": TexturePresetSettings(
        pitch_scale=1.16,
        gain_scale=1.18,
        brightness=0.60,
        roughness_depth=0.24,
        roughness_hz=11.0,
        pulse_rate_scale=1.55,
        pulse_duty_cycle=0.14,
    ),
}

class NavigationAudioController:
    def __init__(self, column_count: int, config: NavigationAudioConfig | None = None) -> None:
        self.config = config or NavigationAudioConfig()
        self._voices = [
            SpatialTone(
                initial_pitch_hz=320.0,
                initial_volume=0.0,
                initial_azimuth_deg=0.0,
                initial_brightness=TEXTURE_PRESETS["original"].brightness,
                initial_roughness_depth=TEXTURE_PRESETS["original"].roughness_depth,
                initial_roughness_hz=TEXTURE_PRESETS["original"].roughness_hz,
                sample_rate=self.config.sample_rate,
                block_size=self.config.block_size,
            )
            for _ in range(column_count)
        ]
        self._running = False
        self._preset_name = "original"

    @property
    def texture_preset(self) -> str:
        return self._preset_name

    def set_texture_preset(self, preset_name: str) -> str:
        normalized = str(preset_name).strip().lower()
        if normalized not in TEXTURE_PRESETS:
            valid = ", ".join(sorted(TEXTURE_PRESETS))
            raise ValueError(f"Unknown preset '{preset_name}'. Valid presets: {valid}")
        self._preset_name = normalized
        return self._preset_name

    def start(self) -> None:
        if self._running:
            return
        for voice in self._voices:
            voice.start()
        self._running = True

    def stop(self) -> None:
        if not self._running:
            return
        for voice in self._voices:
            voice.stop()
        self._running = False

    def apply(self, column_states: tuple[NavigationColumnState, ...], now_s: float | None = None) -> None:
        if not self._running:
            raise RuntimeError("NavigationAudioController.start() must be called before apply()")

        _ = time.monotonic() if now_s is None else float(now_s)
        ranked_columns = sorted(column_states, key=lambda state: state.risk_score, reverse=True)

        high_priority_columns: list[int] = []
        for state in ranked_columns:
            if state.volume < self.config.min_audible_volume:
                continue

            near_hazard = (
                state.percentile_depth_m is not None
                and state.percentile_depth_m <= self.config.near_priority_distance_m
            )
            approaching_hazard = (
                state.ttc_s is not None
                and state.ttc_s <= self.config.ttc_priority_horizon_s
            )
            if near_hazard or approaching_hazard:
                high_priority_columns.append(state.col)

        if self.config.max_simultaneous_columns is not None:
            high_priority_columns = high_priority_columns[: self.config.max_simultaneous_columns]
        active_columns = set(high_priority_columns)
        preset = TEXTURE_PRESETS[self._preset_name]
        pulse_gating = self.config.use_pulse_gating

        for state in column_states:
            voice = self._voices[state.col]
            target_volume = 0.0
            pitch_hz = state.pitch_hz * preset.pitch_scale
            if state.col in active_columns:
                base_target = min(
                    1.0,
                    self.config.base_volume * self.config.output_gain * preset.gain_scale * state.volume,
                )
                if pulse_gating and state.pulse_hz > 0.0:
                    effective_pulse_hz = state.pulse_hz * preset.pulse_rate_scale
                    phase = (_ * effective_pulse_hz) % 1.0
                    duty_cycle = preset.pulse_duty_cycle
                    gate = 1.0 if phase < duty_cycle else 0.0
                    target_volume = base_target * gate
                else:
                    target_volume = base_target

            voice.set_params(
                pitch_hz=pitch_hz,
                volume=target_volume,
                azimuth_deg=state.azimuth_deg,
                brightness=preset.brightness,
                roughness_depth=preset.roughness_depth,
                roughness_hz=preset.roughness_hz,
            )