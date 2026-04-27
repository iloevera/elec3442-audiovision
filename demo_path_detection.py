import numpy as np
import cv2
from src.path_guidance_audio import PathGuidanceAudio, PathGuidanceConfig
from src.navigation_processing import NavigationFrameAnalysis, NavigationColumnState

def create_mock_analysis_data(image_shape = (480,640), path_center_column=2):
    '''
    Fake NavigationFrameAnalysis for testing path detection logic
    '''

    # Create a mock ground mask
    rows, cols = image_shape
    ground_mask = np.zeros((rows, cols), dtype=bool)

    # Simulate walkable path in certain columns
    col_width = cols // 5
    center_start = path_center_column * col_width
    ground_mask[:, centre_start:centre_start+col_width] = True

    # Create mock column states (needed by PathGuidanceAudio)
    column_states = []
    for col in range(5):
        column_states.append(NavigationColumnState(
            col=col,
            azimuth_deg=-45 + col*22.5,
            sample_count=100,
            risk_score=0.1, 
            percentile_depth_m = 2.0,
            ttc_s=5.0,
            pitch_hz=440,
            pulse_hz=0,
            volume=0.3
        ))
        
