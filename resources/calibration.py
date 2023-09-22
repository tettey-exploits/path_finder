import cv2
import numpy as np
import cv2 as cv
import glob
import pickle

# ---------------- FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS -------------

chessboardSize = (8, 6)
frameSize = (640, 480)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

size_of_chessboard_squares_mm = 30
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
obj_points = []  # 3d point in real world space
img_points = []  # 2d points in image plane.

folder_name = "images"
images = glob.glob(f'{folder_name}/*.png')


for image in images:
    print(image)
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # if found, add object points, image points (after refining them)
    if ret:
        obj_points.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)

cv.destroyAllWindows()

#  ---------------- CALIBRATION -----------------

ret, cameraMatrix, dist, r_vecs, t_vecs = cv.calibrateCamera(obj_points, img_points, frameSize, None, None)

print(f"Camera matrix:")
print(cameraMatrix)
# print(f"dist: {dist}")
# print(f"r_vecs: {r_vecs}")
# print(f"t_vecs: {t_vecs}")
# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
pickle.dump((cameraMatrix, dist), open("calibration.pkl", "wb"))
pickle.dump(cameraMatrix, open("cameraMatrix.pkl", "wb"))
pickle.dump(dist, open("dist.pkl", "wb"))

#  ----------------- UN-DISTORTION -----------------

img = cv.imread(f'{folder_name}/img0.png')
h, w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))
pickle.dump(newCameraMatrix, open("newCameraMatrix.pkl", "wb"))
pickle.dump(roi, open("roi.pkl", "wb"))

# Un-distort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv.imwrite('caliResult1.png', dst)

#  Un-distort with Remapping
map_x, map_y = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w, h), 5)
dst = cv.remap(img, map_x, map_y, cv.INTER_LINEAR)

#  crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv.imwrite('caliResult2.png', dst)

# Re-projection Error
# mean_error = 0
#
# for i in range(len(obj_points)):
#     img_points_2, _ = cv.projectPoints(obj_points[i], r_vecs[i], t_vecs[i], cameraMatrix, dist)
#     error = cv.norm(img_points[i], img_points_2, cv.NORM_L2) / len(img_points_2)
#     mean_error += error
#
# print("total error: {}".format(mean_error / len(obj_points)))


cv2.waitKey(0)
# dist is distortion coeficients
