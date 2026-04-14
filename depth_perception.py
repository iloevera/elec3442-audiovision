import cv2
from picamera2 import Picamera2
import time
import numpy as np
from matplotlib import pyplot as plt

left_cam = Picamera2(cam_num=0)
right_cam = Picamera2(cam_num=1)

left_cam_config = left_cam.create_preview_configuration(main={"size": (640, 480)})
right_cam_config = right_cam.create_preview_configuration(main={"size": (640, 480)})

left_cam.configure(left_cam_config)
right_cam.configure(right_cam_config)

left_cam.start()
right_cam.start()
time.sleep(2)

left_frame = left_cam.capture_array()
right_frame = right_cam.capture_array()

# Convert from RGB to BGR for OpenCV
left_frame = cv2.cvtColor(left_frame, cv2.COLOR_RGB2BGR)
right_frame = cv2.cvtColor(right_frame, cv2.COLOR_RGB2BGR)

# Display the frames side-by-side for a preview
combined_frame = np.hstack((left_frame, right_frame))
cv2.imshow("Stereo Camera (Left | Right)", combined_frame)

key = cv2.waitKey(1) & 0xFF
if key == ord('c'):
    # Save captured image pair for calibration
    cv2.imwrite("capture_left.jpg", left_frame)
    cv2.imwrite("capture_right.jpg", right_frame)

# We read in grayscale since we it's more tedious to calculate disparities on RGB values (would need to calculate disparities for all 3 values)
left_image = cv2.imread('capture_left.jpg', cv.IMREAD_GRAYSCALE)
right_image = cv2.imread('capture_right.jpg', cv2.IMREAD_GRAYSCALE)

stereo = cv2.STEREOBM_create(numDisparities=18, blockSize=21)
# For each pixel, algorithm will find the best disparity from 0
# Larger block size implies smoother, though less accurate disparity
depth = stereo.compute(left_image, right_image)

cv2.imshow("Left", left_image)
cv2.imshow("Right", right_image)

plt.imshow(depth)
plt.axis('off')
plt.show()

# MIGHT NEED TO ADD CAMERA CALIBRATION STUFF FOR MORE ACCURATE DISPARITY CALCULATION


