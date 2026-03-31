"""Spatial navigation and path-planning layer.

Analyses the depth map to identify:

* The best forward walking direction (the "path tone" azimuth).
* Edge / drop-off cues at the left and right periphery.
* Intersection / branching beacons when multiple corridor openings exist.

All outputs are encoded as :class:`NavigationCue` objects that are passed to
the audio renderer.
"""

from __future__ import annotations

import dataclasses
import math
from enum import Enum
from typing import List, Optional

import numpy as np


class CueType(Enum):
    """Type of navigation audio cue."""
    PATH_TONE = "path_tone"
    EDGE = "edge"
    INTERSECTION = "intersection"


@dataclasses.dataclass
class NavigationCue:
    """A single navigation audio hint.

    Attributes:
        cue_type: :class:`CueType` describing what the cue represents.
        azimuth_deg: Direction of the cue relative to straight ahead, in
            degrees.  Negative = left, positive = right.
        pitch_hz: Suggested tone pitch in Hz.
        volume: Normalised volume in ``[0, 1]``.
        label: Optional human-readable label for debugging.
    """

    cue_type: CueType
    azimuth_deg: float
    pitch_hz: float
    volume: float
    label: str = ""


class PathPlanner:
    """Generate navigation cues from a depth map.

    The depth map is divided into vertical *column strips*.  For each strip
    the median depth is computed; clear corridors have large median depths.

    Args:
        image_width: Camera image width in pixels.
        image_height: Camera image height in pixels.
        hfov_deg: Horizontal field of view of the camera in degrees.
        num_strips: Number of vertical column strips to analyse.
        min_clear_depth_m: Minimum median depth (m) for a strip to be
            considered a clear corridor.
        edge_depth_drop_m: Minimum drop in median depth between adjacent
            strips to flag as an edge.
        path_tone_pitch_hz: Pitch of the forward path tone.
        edge_pitch_hz: Pitch of edge / drop-off cues.
        intersection_pitch_hz: Pitch of intersection beacon tones.

    Example::

        planner = PathPlanner(image_width=640, image_height=480)
        cues = planner.plan(depth_map)
        for cue in cues:
            audio.play_navigation_cue(cue)
    """

    def __init__(
        self,
        image_width: int = 640,
        image_height: int = 480,
        hfov_deg: float = 70.0,
        num_strips: int = 9,
        min_clear_depth_m: float = 1.5,
        edge_depth_drop_m: float = 0.5,
        path_tone_pitch_hz: float = 440.0,
        edge_pitch_hz: float = 120.0,
        intersection_pitch_hz: float = 660.0,
    ) -> None:
        self._w = image_width
        self._h = image_height
        self._hfov = hfov_deg
        self._num_strips = num_strips
        self._min_clear = min_clear_depth_m
        self._edge_drop = edge_depth_drop_m
        self._path_pitch = path_tone_pitch_hz
        self._edge_pitch = edge_pitch_hz
        self._inter_pitch = intersection_pitch_hz

    def plan(self, depth_map: np.ndarray) -> List[NavigationCue]:
        """Analyse *depth_map* and return navigation cues.

        Args:
            depth_map: Float32 array ``(H, W)`` of depths in metres.

        Returns:
            List of :class:`NavigationCue` instances.
        """
        cues: List[NavigationCue] = []

        strip_medians = self._compute_strip_medians(depth_map)
        azimuths = self._strip_azimuths()

        path_cue = self._path_tone(strip_medians, azimuths)
        if path_cue is not None:
            cues.append(path_cue)

        edge_cues = self._edge_cues(strip_medians, azimuths)
        cues.extend(edge_cues)

        inter_cues = self._intersection_cues(strip_medians, azimuths)
        cues.extend(inter_cues)

        return cues

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_strip_medians(self, depth_map: np.ndarray) -> np.ndarray:
        h, w = depth_map.shape[:2]
        strip_w = max(1, w // self._num_strips)
        medians = np.full(self._num_strips, np.nan)

        for i in range(self._num_strips):
            x0 = i * strip_w
            x1 = x0 + strip_w if i < self._num_strips - 1 else w
            strip = depth_map[:, x0:x1]
            valid = strip[np.isfinite(strip)]
            if valid.size > 0:
                medians[i] = float(np.median(valid))

        return medians

    def _strip_azimuths(self) -> np.ndarray:
        centres = (np.arange(self._num_strips) + 0.5) / self._num_strips
        return (centres - 0.5) * self._hfov

    def _path_tone(
        self,
        medians: np.ndarray,
        azimuths: np.ndarray,
    ) -> Optional[NavigationCue]:
        clear = np.where(
            np.isfinite(medians) & (medians >= self._min_clear)
        )[0]
        if clear.size == 0:
            return None

        best = clear[np.argmax(medians[clear])]
        depth = medians[best]
        vol = min(depth / 10.0, 1.0)

        return NavigationCue(
            cue_type=CueType.PATH_TONE,
            azimuth_deg=float(azimuths[best]),
            pitch_hz=self._path_pitch,
            volume=float(vol),
            label="best_path",
        )

    def _edge_cues(
        self,
        medians: np.ndarray,
        azimuths: np.ndarray,
    ) -> List[NavigationCue]:
        cues: List[NavigationCue] = []
        for i in range(len(medians) - 1):
            m0, m1 = medians[i], medians[i + 1]
            if not (np.isfinite(m0) and np.isfinite(m1)):
                continue
            drop = m0 - m1
            if drop >= self._edge_drop:
                az = float((azimuths[i] + azimuths[i + 1]) / 2)
                vol = min(drop / 3.0, 1.0)
                cues.append(
                    NavigationCue(
                        cue_type=CueType.EDGE,
                        azimuth_deg=az,
                        pitch_hz=self._edge_pitch,
                        volume=float(vol),
                        label=f"edge_{i}_{i+1}",
                    )
                )
        return cues

    def _intersection_cues(
        self,
        medians: np.ndarray,
        azimuths: np.ndarray,
    ) -> List[NavigationCue]:
        clear_strips = np.where(
            np.isfinite(medians) & (medians >= self._min_clear)
        )[0]

        if clear_strips.size < 2:
            return []

        groups: List[List[int]] = []
        current: List[int] = [int(clear_strips[0])]
        for idx in clear_strips[1:]:
            if idx == current[-1] + 1:
                current.append(int(idx))
            else:
                groups.append(current)
                current = [int(idx)]
        groups.append(current)

        if len(groups) < 2:
            return []

        cues: List[NavigationCue] = []
        for group in groups:
            centre_idx = group[len(group) // 2]
            az = float(azimuths[centre_idx])
            depth = float(medians[centre_idx])
            vol = min(depth / 10.0, 0.5)
            cues.append(
                NavigationCue(
                    cue_type=CueType.INTERSECTION,
                    azimuth_deg=az,
                    pitch_hz=self._inter_pitch,
                    volume=float(vol),
                    label=f"corridor_{centre_idx}",
                )
            )
        return cues
