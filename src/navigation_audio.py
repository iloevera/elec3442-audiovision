from __future__ import annotations

from dataclasses import dataclass
import time

from .audio_spatial_tone import SpatialTone
from .navigation_processing import NavigationColumnState


@dataclass(frozen=True)
class NavigationAudioConfig:
    sample_rate: int = 48_000
    block_size: int = 512
    base_volume: float = 0.50
    output_gain: float = 8.0
    min_audible_volume: float = 0.08
    near_priority_distance_m: float = 0.50
    ttc_priority_horizon_s: float = 5.0
    max_simultaneous_columns: int | None = 3


class NavigationAudioController:
    def __init__(self, column_count: int, config: NavigationAudioConfig | None = None) -> None:
        self.config = config or NavigationAudioConfig()
        self._voices = [
            SpatialTone(
                initial_pitch_hz=320.0,
                initial_volume=0.0,
                initial_azimuth_deg=0.0,
                sample_rate=self.config.sample_rate,
                block_size=self.config.block_size,
            )
            for _ in range(column_count)
        ]
        self._running = False

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

        for state in column_states:
            voice = self._voices[state.col]
            target_volume = 0.0
            if state.col in active_columns:
                target_volume = min(
                    1.0,
                    self.config.base_volume * self.config.output_gain * state.volume,
                )
            voice.set_params(
                pitch_hz=state.pitch_hz,
                volume=target_volume,
                azimuth_deg=state.azimuth_deg,
            )