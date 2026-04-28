"""Test path guidance with webcam + simulated ground mask."""

import time
import cv2
import numpy as np
from src.path_detection_audio import PathGuidanceAudio, PathGuidanceConfig
from src.navigation_processing import NavigationFrameAnalysis, NavigationColumnState

from src.realsense_driver import D435iDriver, FrameBundle, IMUSample

class MockWebcamAnalysis:
    """Creates fake NavigationFrameAnalysis from webcam feed."""
    
    def __init__(self, num_columns=5):
        self.num_columns = num_columns
        
    def create_analysis(self, frame, path_center_col=2):
        """Create mock analysis with simulated ground mask."""
        h, w = frame.shape[:2]
        
        # Create a mock ground mask
        ground_mask = np.zeros((h, w), dtype=bool)
        
        # Simulate walkable area in a specific column
        col_width = w // self.num_columns
        start_x = path_center_col * col_width
        end_x = start_x + col_width
        ground_mask[:, start_x:end_x] = True
        
        # Add some noise to make it realistic
        noise = np.random.random((h, w)) > 0.95
        ground_mask[noise] = False
        
        # Create mock column states
        column_states = []
        for col in range(self.num_columns):
            column_states.append(NavigationColumnState(
                col=col,
                azimuth_deg=-45 + col * 22.5,
                sample_count=100,
                risk_score=0.1,
                percentile_depth_m=2.0,
                ttc_s=5.0,
                pitch_hz=440,
                pulse_hz=0,
                volume=0.3
            ))
        
        return NavigationFrameAnalysis(
            timestamp_s=time.time(),
            depth_m=np.zeros((h, w)),
            valid_mask=np.ones((h, w), dtype=bool),
            ground_mask=ground_mask,
            obstacle_mask=np.zeros((h, w), dtype=bool),
            height_above_ground_m=None,
            gravity_unit=None,
            ground_plane=None,
            cell_states=(),
            column_states=tuple(column_states),
            depth_percentile=5.0,
            percentile_depth_grid_m=np.zeros((3, self.num_columns)),
            ttc_grid_s=np.zeros((3, self.num_columns)),
            risk_grid=np.zeros((3, self.num_columns))
        )

def detect_path_from_webcam(frame):
    """
    Simple path detection using edge detection and color.
    Returns predicted center column (0-4).
    """
    h, w = frame.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Look at bottom half of image (where the path is)
    bottom_half = edges[h//2:, :]
    
    # Divide into columns
    num_columns = 5
    col_width = w // num_columns
    
    # Count edges per column
    edge_counts = []
    for col in range(num_columns):
        start_x = col * col_width
        end_x = start_x + col_width
        col_region = bottom_half[:, start_x:end_x]
        edge_counts.append(np.sum(col_region))
    
    # Column with fewest edges = most clear path
    # (Fewer edges = less texture/obstacles)
    if sum(edge_counts) > 0:
        # Find column with minimum edges (clearest path)
        best_col = np.argmin(edge_counts)
    else:
        best_col = 2  # Default to center
    
    return best_col, edge_counts

def main():
    print("=" * 60)
    print("Webcam Path Guidance Test")
    print("=" * 60)
    print("\n⚠️  Make sure your speakers/headphones are connected!")
    print("⚠️  Press 'q' to quit")
    print("\nInstructions:")
    print("  - Point webcam at a clear path (hallway, sidewalk, floor)")
    print("  - The tone will guide you toward the clearest area")
    print("  - Left/right panning indicates direction to turn\n")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam!")
        return
    
    # Initialize path guidance
    guide = PathGuidanceAudio(config=PathGuidanceConfig())
    
    # Create mock analyzer
    mock_analysis = MockWebcamAnalysis()
    
    # For smoothing (avoid jitter)
    history = []
    history_size = 5
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Detect path from webcam
            center_col, edge_counts = detect_path_from_webcam(frame)
            
            # Smooth the column detection
            history.append(center_col)
            if len(history) > history_size:
                history.pop(0)
            smoothed_col = int(round(np.mean(history)))
            
            # Create mock analysis with detected path
            analysis = mock_analysis.create_analysis(frame, path_center_col=smoothed_col)
            
            # Update guidance tone
            guide.update(analysis)
            
            # Visualize on frame
            h, w = frame.shape[:2]
            col_width = w // 5
            
            # Draw column overlay
            for col in range(5):
                start_x = col * col_width
                end_x = start_x + col_width
                
                # Color based on danger (red) or clear (green)
                if col == smoothed_col:
                    color = (0, 255, 0)  # Green = recommended path
                    thickness = 3
                else:
                    # Intensity based on edge count
                    intensity = min(255, edge_counts[col] // 50)
                    color = (0, 0, 255)  # Red
                    thickness = 1
                
                cv2.rectangle(frame, (start_x, 0), (end_x, h), color, thickness)
                cv2.putText(frame, f"Col {col}", (start_x + 10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show guidance info
            debug = guide.get_debug_info()
            cv2.putText(frame, f"Path: Col {smoothed_col}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Azimuth: {debug['azimuth_deg']:.1f}°", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(frame, f"Volume: {debug['volume']:.2f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Show instructions
            cv2.putText(frame, "Press 'q' to quit", (w - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Path Guidance Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            time.sleep(0.05)  # Small delay
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        guide.stop()
        print("\n✓ Test complete")

if __name__ == "__main__":
    main()
