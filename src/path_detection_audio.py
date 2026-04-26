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
  
