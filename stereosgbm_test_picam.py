"""
Demo for: Approaching Object Detection using Simulated Stereo Depth
Improved stability for distance and closeness values
Use only the PiCam 3 for demonstration purposes
"""

import cv2
import numpy as np
import time
from picamera2 import Picamera2

BASELINE = 0.06
FOCAL_LENGTH_PX = 650
MAX_DEPTH = 5.0

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16 * 8,
    blockSize=11,
    P1=8 * 3 * 11**2,
    P2=32 * 3 * 11**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

prev_gray = None
last_time = time.time()
frame_count = 0

smoothed_closeness = 0.0
smoothed_min_depth = 5.0
ALPHA = 0.25

print("Demo running. Walk toward the camera or have someone approach quickly.")
print("Press 'q' to quit.\n")

try:
    while True:
        array = picam2.capture_array("main")
        frame = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            prev_gray = gray.copy()
            continue

        gray_l = gray
        gray_r = prev_gray

        disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

        depth_map = (BASELINE * FOCAL_LENGTH_PX) / (disparity + 1e-5)
        depth_map = np.clip(depth_map, 0.2, MAX_DEPTH)

        h, w = depth_map.shape
        center_roi = depth_map[h//3 : 2*h//3, w//3 : 2*w//3]

        valid = (center_roi > 0.3) & (center_roi < 3.5)

        if np.any(valid):
            min_depth = float(center_roi[valid].min())
            closeness = max(0, (3.0 - min_depth) * 45)
        else:
            min_depth = MAX_DEPTH
            closeness = 0.0

        smoothed_min_depth = ALPHA * min_depth + (1 - ALPHA) * smoothed_min_depth
        smoothed_closeness = ALPHA * closeness + (1 - ALPHA) * smoothed_closeness

        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.65, disp_color, 0.35, 0)

        if smoothed_min_depth < 3.0:
            cv2.putText(overlay, "OBJECT APPROACHING!", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)
            cv2.putText(overlay, f"Distance: {smoothed_min_depth:.2f}m", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.05, (0, 0, 255), 3)
            cv2.putText(overlay, f"Closeness: {smoothed_closeness:.1f}", (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        else:
            cv2.putText(overlay, "No close approaching object", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2)
            cv2.putText(overlay, f"Distance: {smoothed_min_depth:.2f}m", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 2)
            cv2.putText(overlay, f"Closeness: {smoothed_closeness:.1f}", (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 2)

        frame_count += 1
        current_time = time.time()
        if current_time - last_time > 1.0:
            fps = frame_count / (current_time - last_time)
            cv2.putText(overlay, f"FPS: {fps:.1f}", (w-160, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            last_time = current_time
            frame_count = 0

        cv2.imshow("Approaching Object Demo (Smoothed)", overlay)
        cv2.imshow("Disparity", disp_vis)

        prev_gray = gray.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Demo stopped.")