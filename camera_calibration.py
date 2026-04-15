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


