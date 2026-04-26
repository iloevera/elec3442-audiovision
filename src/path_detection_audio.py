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

    # Minimum walkable pixels for valid path
    min_walkable_pixels: int = 80

    # Number of consecutive frames of no ground before fading out
    ground_loss_frames_before_fade: int = 5
    
  
