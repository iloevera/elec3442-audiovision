"""
Demo: Approaching Object Detection using Simulated Stereo Depth
Single laptop webcam only (uses previous frame as "right" image)
Focus: Detect fast approaching objects (like someone walking toward the camera)
"""

import cv2
import numpy as np
import time

print("Starting Approaching Object Demo (Single Webcam Simulation on Laptop)")

BASELINE = 0.06          # simulated baseline (smaller because frames are close in time)
FOCAL_LENGTH_PX = 650
MAX_DEPTH = 5.0

# StereoSGBM
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

#camera setup (laptop webcam)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_gray = None
last_time = time.time()
frame_count = 0

print("Demo is running. Please walk toward the camera or have someone approach it quickly.")
print("Press 'q' to quit.\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #skip the first frame (no previous frame yet)
        if prev_gray is None:
            prev_gray = gray.copy()
            continue

        #simulate stereo: current = left, previous = right
        gray_l = gray
        gray_r = prev_gray

        #compute disparity
        disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

        #convert to depth (in meters)
        depth_map = (BASELINE * FOCAL_LENGTH_PX) / (disparity + 1e-5)
        depth_map = np.clip(depth_map, 0.2, MAX_DEPTH)

        #approaching object detection (only care about things getting closer for now)
        h, w = depth_map.shape
        
        #focus on central area (where approaching objects are likely to appear)
        center_roi = depth_map[h//4 : 3*h//4, w//4 : 3*w//4]
        valid = (center_roi > 0.2) & (center_roi < 3.0)
        
        if np.any(valid):
            min_depth = float(center_roi[valid].min())
            # Simple "approaching" score based on how close + how suddenly it appeared
            closeness = max(0, (3.0 - min_depth) * 40)
        else:
            min_depth = 999
            closeness = 0

        #visualisation
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(frame, 0.65, disp_color, 0.35, 0)

        #big alert when something is approaching fast
        if min_depth < 3.0 and closeness > 50:          # Tune these thresholds while testing
            cv2.putText(overlay, "FAST APPROACHING OBJECT!", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)
            cv2.putText(overlay, f"Distance: {min_depth:.2f}m", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(overlay, f"Closeness: {closeness:.2f}", (30, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        #elif min_depth < 2.5:
        #    cv2.putText(overlay, f"Object at {min_depth:.2f}m", (30, 70),
        #                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        else:
            cv2.putText(overlay, "No close approaching object", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        #FPS
        frame_count += 1
        current_time = time.time()
        if current_time - last_time > 1.0:
            fps = frame_count / (current_time - last_time)
            cv2.putText(overlay, f"FPS: {fps:.1f}", (w-160, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            last_time = current_time
            frame_count = 0

        cv2.imshow("Approaching Object Demo (Simulated Stereo)", overlay)
        cv2.imshow("Disparity", disp_vis)   #raw disparity (the black and white stuff)

        prev_gray = gray.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Demo stopped.")
