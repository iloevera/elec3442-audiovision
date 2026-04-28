# SpatialTone is used for path detection and guidance
# Hazard detection is given priority over path detection and guidance

from __future__ import annotations

import time 

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.audio_spatial_tone import SpatialTone
from src.navigation_processing import NavigationFrameAnalysis

@dataclass
class PathGuidanceConfig:
    # Azimuth mapping
    max_guidance_deg: float = 90.0
    num_columns: int = 5

    # Threshold for guidance trigger (No guidance when walking straight enough)
    deadzone_deg: float = 8.0

    # Pitch mapping
    pitch_center_hz: float = 440.0
    pitch_extreme_hz: float = 660.0

    # Column mapping
    volume_normal: float = 0.25
    volume_on_path: float = 0.08

    # Hazard override: reduce guidance volume when hazards are present
    hazard_volume_reduction: float = 0.7 # Volume is multiplied by this factor when hazard detected

    # Bottom portion of ground mask
    bottom_fraction: float = 0.33

    # Minimum walkable pixels for valid path (Adjust based on real-world tests)
    min_walkable_pixels: int = 80

    # Number of consecutive frames of no ground before fading out
    ground_loss_frames_before_fade: int = 5

class PathGuidanceAudio:
    def __init__(self, config: Optional[PathGuidanceConfig] = None):
        self.config = config or PathGuidanceConfig()

        # Create the spatial tone 
        self._tone = SpatialTone(
            initial_pitch_hz = self.config.pitch_center_hz,
            initial_volume = 0.0,
            initial_azimuth_deg = 0.0,
        )
        self._tone.start()

        self._no_ground_counter = 0
        self._last_azimuth = 0.0
        self._last_volume = 0.0

    def update(self, analysis: NavigationFrameAnalysis) -> None:
        '''
        Call this function at every frame to update guidance tone based on current ground plane and hazards.
        '''
        # Get hazard levels per column from the analysis
        column_hazards = self._get_column_hazard_levels(analysis)

        # Find walkable centre from ground mask
        walkable_centre_col = self._find_walkable_center_column(analysis.ground_mask)

        if walkable_centre_col is None:
            # If no ground is detected, audio fades eventually
            self._no_ground_counter += 1
            if self._no_ground_counter >= self.config.ground_loss_frames_before_fade:
                self._update_tone(azimuth_deg=self._last_azimuth, volume=0.0)
            return

        self._no_ground_counter = 0

        # Calculate ideal guidance direction
        ideal_azimuth = self._col_to_azimuth(walkable_centre_col)

        # Apply hazard override: shift guidance away from dangerous columns
        safe_azimuth, hazard_factor = self._apply_hazard_override(
            ideal_azimuth,
            column_hazards
        )

        # Compute pitch (higher when further off path, but always smooth)
        pitch = self._compute_pitch(safe_azimuth)

        # Compute volume (reduced if hazards present)
        base_volume = self._compute_pitch(safe_azimuth)
        final_volume = base_volume * (1.0 - hazard_factor * self.config.hazard_volume_reduction) # Reduce volume when hazards are present/closer

        self._update_tone(azimuth_deg=safe_azimuth, pitch_hz=pitch, volume=final_volume)

        self._last_azimuth = safe_azimuth
        self._last_volume = final_volume


    def _get_column_hazard_levels(self, analysis: NavigationFrameAnalysis) -> np.ndarray:
        '''
        Extract hazard (risk) per column from analysis
        '''
        hazards = np.zeros(self.config.num_columns, dtype=np.float32)

        for col_state in analysis.column_states:
            if col_state.col < self.config.num_columns:
                hazards[col_state.col] = col_state.risk_score

        return hazards

    def _find_walkable_center_column(self, ground_mask: np.ndarray) -> Optional[int]:
        '''
        Find which column (0-4) contains the center of walkable area (Bottom portion of depth map only)
        '''
        if ground_mask is None or not np.any(ground_mask):
            return None

        rows, cols = ground_mask.shape

        bottom_rows = int(rows * self.config.bottom_fraction)
        bottom_ground = ground_mask[-bottom_rows:,:]

        # Divide into columns
        col_width = cols // self.config.num_columns

        walkable_counts = []
        for col_ind in range(self.config.num_columns):
            start_x = col_ind * col_width
            if col_ind < (self.config.num_columns - 1):
                end_x = start_x + col_width
            else: 
                end_x = cols
            col_region = bottom_ground[:, start_x:end_x]   
            walkable_counts.append(np.sum(col_region))
        total_walkable = sum(walkable_counts)

        if total_walkable < self.config.min_walkable_pixels:
            return None

        # Weighted center of mass (the best possible postition to go next, i.e. the area with the most walkable pixels))
        col_indices = np.arange(self.config.num_columns)
        center_of_mass = np.sum(col_indices * walkable_counts) / total_walkable
        
        return int(round(center_of_mass))

    def _col_to_azimuth(self, col: int) -> float:
        ''' 
        Convert column index to azimuth angle
        '''
        centre = (self.config.num_columns - 1) / 2.0
        normalized = (col - centre) / centre
        return normalized * self.config.max_guidance_deg

    def _apply_hazard_override(self, ideal_azimuth: float, column_hazards: np.ndarray) -> tuple[float, float]:
        '''
        Adjust guidance to avoid hazardous columns
        Returns (adjusted_azimuth, hazard_factor) where hazard_factor is 0-1
        '''
        # Find which column the ideal azimuth points to    
        ideal_col = self._azimuth_to_col(ideal_azimuth)

        # Check if that column is hazardous
        if column_hazards[ideal_col] < 0.3:
            # Ideal direction is safe
            return ideal_azimuth, 0.0

        # If hazardous, find safest neighbouring column
        best_col = ideal_col
        best_hazard = column_hazards[ideal_col]

        for offset in [-1,1]: # Check left and right
            neighbour = ideal_col + offset
            if 0 <= neighbour < self.config.num_columns:
                if column_hazards[neighbour] < best_hazard:
                    best_hazard = column_hazards[neighbour]
                    best_col = neighbour

            # If still hazardous, return back to centre
            if best_hazard >= 0.5:
                best_col = self.config.num_columns // 2

            adjusted_azimuth = self._col_to_azimuth(best_col)
            hazard_factor = min(1.0/best_hazard)

            return adjusted_azimuth, hazard_factor

    def _azimuth_to_col(self, azimuth_deg: float) -> int:
        '''
        Convert azimuth angle to nearest column index
        '''
        normalized = azimuth_deg / self.config.max_guidance_deg
        centre = (self.config.num_columns - 1) / 2.0
        col = int(round(centre + normalized * centre))
        return np.clip(col, 0, self.config.num_columns - 1)

    def _compute_pitch(self, azimuth_deg: float) -> float:
        '''
        Compute pitch based on how far off-centre we are
        '''
        off_centre_factor = abs(azimuth_deg) / self.config.max_guidance_deg
        off_centre_factor = min(1.0, off_centre_factor)

        pitch_range = self.config.pitch_extreme_hz - self.config.pitch_center_hz
        pitch = self.config.pitch_center_hz + (off_centre_factor * pitch_range)

        return pitch

    def _compute_volume(self, azimuth_deg: float) -> float:
        '''
        Compute volume based on how far off-centre we are
        '''
        if abs(azimuth_deg) <= self.config.deadzone_deg:
            # On path: quiet confirmation
            return self.config.volume_on_path
        else:
            # Off path: louder as error increases
            off_center_factor = abs(azimuth_deg) / self.config.max_guidance_deg
            off_center_factor = min(1.0, off_center_factor)
            return self.config.volume_normal * off_center_factor

    def _update_tone(self, azimuth_deg: float, pitch_hz: float, volume:float) -> None:
        '''
        Update Spatial tone paramaeters
        '''
        self._tone.set_params(pitch_hz = pitch_hz, azimuth_deg = azimuth_deg)
        self._tone.set_volume(volume)

    def stop(self) -> None:
        '''
        Stop tone
        '''
        self._tone.stop()

    def get_debug_info(self) -> dict:
        '''
        Return current state for debugging
        '''
        return{
            'azimuth_deg': self._last_azimuth,
            'volume': self._last_volume,
        }
