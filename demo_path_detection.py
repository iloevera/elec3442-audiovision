import numpy as np
import cv2
from src.path_guidance_audio import PathGuidanceAudio, PathGuidanceConfig
from src.navigation_processing import NavigationFrameAnalysis, NavigationColumnState

def create_mock_analysis_data(image_shape = (480,640), path_centre_column=2):
    '''
    Fake NavigationFrameAnalysis for testing path detection logic
    '''

    # Create a mock ground mask
    rows, cols = image_shape
    ground_mask = np.zeros((rows, cols), dtype=bool)

    # Simulate walkable path in certain columns
    col_width = cols // 5
    centre_start = path_centre_column * col_width
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

    # Create mock analysis object
    analysis = NavigationFrameAnalysis(
        timestamp_s=0, 
        depth_m=np.zeros(image_shape),
        valid_mask=np.ones(image_shape, dtype=bool),
        ground_mask=ground_mask,
        obstacle_mask=np.zeros(image_shape, dtype=bool),
        height_above_ground_m=None,
        gravity_unit=None,
        ground_plane=None,
        cell_states=(),
        column_states=tuple(column_states),
        depth_percentile=5.0,
        percentile_depth_grid_m=np.zeros((3, 5)),
        ttc_grid_s=np.zeros((3, 5)),
        risk_grid=np.zeros((3, 5))
    )
    return analysis

# Test your path guidance
guide = PathGuidanceAudio(config=PathGuidanceConfig())

# Test with path centred
analysis = create_mock_analysis(path_centre_column=2)
guide.update(analysis)
print("Centre path test - should be quiet/centred")

# Test with path far right
analysis = create_mock_analysis(path_centre_column=4)
guide.update(analysis)
print("Right path test - should guide right")
