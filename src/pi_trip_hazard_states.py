from __future__ import annotations

import numpy as np

from src.navigation_processing import NavigationColumnState


def build_trip_column_states(
    trip_detections,
    cols: int,
) -> tuple[NavigationColumnState, ...]:
    if cols <= 1:
        azimuths = [0.0]
    else:
        azimuths = np.linspace(-90.0, 90.0, cols)

    strongest_by_col: dict[int, object] = {}
    for det in trip_detections:
        if det.column < 0 or det.column >= cols:
            continue
        current = strongest_by_col.get(det.column)
        if current is None or det.urgency > current.urgency:
            strongest_by_col[det.column] = det

    states: list[NavigationColumnState] = []
    for col in range(cols):
        det = strongest_by_col.get(col)

        if det is None:
            states.append(
                NavigationColumnState(
                    col=col,
                    azimuth_deg=float(azimuths[col]),
                    sample_count=0,
                    risk_score=0.0,
                    percentile_depth_m=None,
                    ttc_s=None,
                    pitch_hz=260.0,
                    pulse_hz=0,
                    volume=0.0,
                )
            )
            continue

        urgency = float(np.clip(det.urgency, 0.0, 1.0))
        confidence = float(np.clip(det.confidence, 0.0, 1.0))

        risk_score = max(0.20, urgency)
        score = urgency * confidence
        if score >= 0.75:
            pulse_hz = 8.0
            volume = 0.85
            pitch_hz = 1200.0
        elif score >= 0.45:
            pulse_hz = 6.0
            volume = 0.65
            pitch_hz = 1050.0
        elif score >= 0.20:
            pulse_hz = 4.5
            volume = 0.50
            pitch_hz = 950.0
        else:
            pulse_hz = 0.0
            volume = 0.0
            pitch_hz = 260.0
        synthetic_depth_m = 0.40

        states.append(
            NavigationColumnState(
                col=col,
                azimuth_deg=float(azimuths[col]),
                sample_count=1,
                risk_score=risk_score,
                percentile_depth_m=synthetic_depth_m,
                ttc_s=None,
                pitch_hz=pitch_hz,
                pulse_hz=pulse_hz,
                volume=volume * confidence,
            )
        )

    return tuple(states)