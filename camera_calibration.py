import numpy as np
import cv2 as cv
import glob

chessboard_size = (9,6)
frame_size = (h,w)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)

# initialize arrays to story object points (3d points in real world space) and image points (2d points in image plane) from all the images
obj_points = []
img_points = []

images = glob.glob('*.jpg') # Load up image files in the same directory beforehand

for image in images:
  print(image)
  img = cv.imread(image)
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  # Find chessboard corners
  ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

  # If found, add object points, image points 
  if ret == True:
    obj_points.append(objp)
    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    img_points.append(corners)

    # Draw and display the corners
    cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
    cv.imsshow('img', img)
    cv.waitKey(1000)

cv.destroyAllWindows()

# Calibration
ret, camera_matrix, distortion, rotation_vectors, translation_vectors = cv.calibrateCamera(obj_points, img_points, frame_size, None, None)

# Undistortion
img = cv.imread({new_image})
h, w = img.shape[:2]
new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, distortion, (w, h), 1, (w, h))
dst = cv.undistort(img, camera_matrix, distortion, None, new_camera_matrix)

# Crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibration_result.png', dst)

# Reprojection error
mean_error = 0

for i in range(len(obj_points)):
  img_points_2, _ = cv.projectPoints(objPoints[i], rotation_vectors[i], translation_vectors[i], camera_matrix, distortion)
  error = cv.norm(img_points[i], img_points_2, cv.NORM_L2)/len(img_points_2)
  mean_error += error
