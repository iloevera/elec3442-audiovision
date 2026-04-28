"""Test path guidance with RealSense camera (actual ground mask)."""

import time
import cv2
import numpy as np
from src.path_detection_audio import PathGuidanceAudio, PathGuidanceConfig
from src.realsense_driver import D435iDriver
from src.navigation_processing import NavigationProcessor

def find_best_column_from_ground_mask(ground_mask: np.ndarray, num_columns: int = 5) -> int | None:
    """
    Find which column has the most walkable area using actual ground_mask.
    Returns column index (0-4) or None if no walkable area.
    """
    if ground_mask is None or not np.any(ground_mask):
        return None
    
    h, w = ground_mask.shape
    
    # Focus on bottom portion (closest to user)
    bottom_start = int(h * 0.6)  # Bottom 40%
    bottom_ground = ground_mask[bottom_start:, :]
    
    # Divide into columns
    col_width = w // num_columns
    walkable_counts = []
    
    for col in range(num_columns):
        start_x = col * col_width
        end_x = start_x + col_width if col < num_columns - 1 else w
        col_region = bottom_ground[:, start_x:end_x]
        walkable_counts.append(np.sum(col_region))
    
    total_walkable = sum(walkable_counts)
    
    if total_walkable < 100:  # Minimum threshold
        return None
    
    # Find column with most walkable pixels
    best_col = np.argmax(walkable_counts)
    return best_col

def main():
    print("=" * 60)
    print("RealSense Path Guidance Test")
    print("=" * 60)
    print("\n⚠️  Make sure your speakers/headphones are connected!")
    print("⚠️  Press 'q' to quit")
    print("\nInstructions:")
    print("  - Point RealSense at a clear path (hallway, sidewalk, floor)")
    print("  - The tone will guide you toward the walkable area")
    print("  - Left/right panning indicates direction to turn")
    print("  - Green overlay = walkable, Red = obstacles\n")
    
    # Initialize path guidance
    guide = PathGuidanceAudio(config=PathGuidanceConfig())
    
    # Initialize navigation processor (this gives real ground_mask)
    processor = NavigationProcessor()
    
    # For smoothing (avoid jitter)
    history = []
    history_size = 5
    
    try:
        # Initialize RealSense driver
        with D435iDriver() as driver:
            print("✅ RealSense connected! Processing frames...\n")
            
            while True:
                # Get frame bundle from RealSense
                bundle = driver.wait_for_bundle(timeout_s=1.0)
                if bundle is None:
                    if driver.last_error is not None:
                        raise RuntimeError("Capture thread stopped after an error") from driver.last_error
                    continue
                
                # Process to get REAL ground_mask and analysis
                analysis = processor.process_bundle(bundle)
                
                # Extract the real ground_mask
                ground_mask = analysis.ground_mask
                
                # Find best column using actual ground_mask
                center_col = find_best_column_from_ground_mask(ground_mask)
                
                # Smooth the column detection
                if center_col is not None:
                    history.append(center_col)
                    if len(history) > history_size:
                        history.pop(0)
                    smoothed_col = int(round(np.mean(history)))
                else:
                    smoothed_col = None
                    history.clear()
                
                # Update guidance tone with REAL data
                guide.update(analysis)
                
                # Visualize on frame (using RealSense color image)
                frame = bundle.color.image.copy()
                h, w = frame.shape[:2]
                col_width = w // 5
                
                # Calculate edge counts per column from depth data (for visualization)
                edge_counts = []
                depth_normalized = cv2.normalize(bundle.depth.image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                edges = cv2.Canny(depth_normalized, 50, 150)
                bottom_half = edges[h//2:, :]
                
                for col in range(5):
                    start_x = col * col_width
                    end_x = start_x + col_width
                    col_region = bottom_half[:, start_x:end_x]
                    edge_counts.append(np.sum(col_region))
                
                # Draw column overlay
                for col in range(5):
                    start_x = col * col_width
                    end_x = start_x + col_width
                    
                    # Color based on walkable area or recommended path
                    if smoothed_col is not None and col == smoothed_col:
                        color = (0, 255, 0)  # Green = recommended path
                        thickness = 3
                    else:
                        # Check if this column has walkable pixels
                        col_ground = ground_mask[:, start_x:end_x]
                        if np.any(col_ground):
                            # Has walkable area but not the best
                            color = (0, 200, 100)  # Light green
                            thickness = 1
                        else:
                            # Mostly obstacles
                            intensity = min(255, edge_counts[col] // 50) if edge_counts else 0
                            color = (0, 0, 255)  # Red
                            thickness = 1
                    
                    cv2.rectangle(frame, (start_x, 0), (end_x, h), color, thickness)
                    
                    # Column label
                    label = f"Col {col}"
                    if smoothed_col is not None and col == smoothed_col:
                        label = f">>{col}<<"
                    cv2.putText(frame, label, (start_x + 10, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show guidance info
                debug = guide.get_debug_info()
                walkable_pixels = np.sum(ground_mask) if ground_mask is not None else 0
                
                cv2.putText(frame, f"Path: Col {smoothed_col if smoothed_col is not None else 'N/A'}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Azimuth: {debug['azimuth_deg']:.1f}°", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Volume: {debug['volume']:.2f}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Walkable pixels: {walkable_pixels}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Display ground mask info
                if ground_mask is not None:
                    ground_percent = (walkable_pixels / ground_mask.size) * 100
                    cv2.putText(frame, f"Ground: {ground_percent:.1f}%", (10, 145),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Show instructions
                cv2.putText(frame, "Press 'q' to quit", (w - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Optional: Show small ground mask preview in corner
                if ground_mask is not None:
                    # Resize ground mask for preview
                    preview_h, preview_w = 60, 80
                    mask_preview = cv2.resize(ground_mask.astype(np.uint8) * 255, (preview_w, preview_h))
                    mask_preview = cv2.cvtColor(mask_preview, cv2.COLOR_GRAY2BGR)
                    frame[10:10+preview_h, w-preview_w-10:w-10] = mask_preview
                    cv2.putText(frame, "Ground Mask", (w-preview_w-10, 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                cv2.imshow("RealSense Path Guidance", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                time.sleep(0.03)  # Small delay
                
    except KeyboardInterrupt:
        print("\n\n⚠️  Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        guide.stop()
        print("\n✓ Test complete")

if __name__ == "__main__":
    main()
