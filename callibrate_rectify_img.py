import numpy as np
import cv2 as cv
import glob
import pickle  # object serialiser

# ---------------- FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS -------------

resources_path = "resources"
cameraMatrix = pickle.load(open(f"{resources_path}/cameraMatrix.pkl", "rb"))
dist = pickle.load(open(f"{resources_path}/dist.pkl", "rb"))
newCameraMatrix = pickle.load(open(f"{resources_path}/newCameraMatrix.pkl", "rb"))
roi = pickle.load(open(f"{resources_path}/roi.pkl", "rb"))


def calibrate_cam():  # function to calibrate cameras

    global cameraMatrix, newCameraMatrix, roi, dist

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

    ret, cameraMatrix, dist, r_vecs, t_vecs = cv.calibrateCamera(obj_points, img_points, frameSize, None, None)
    calib_img = cv.imread(f'{folder_name}/img0.png')
    h, w = calib_img.shape[:2]  # resolve image dimensions and use in new calibration
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

    pickle.dump(open(f"{resources_path}/cameraMatrix.pkl", "rb"))
    pickle.dump(open(f"{resources_path}/dist.pkl", "rb"))
    pickle.dump(open(f"{resources_path}/newCameraMatrix.pkl", "rb"))
    pickle.dump(open(f"{resources_path}/roi.pkl", "rb"))

    cv.destroyAllWindows()


def rectify_img(img_left, img_right, img_dimensions: tuple):

    # img_dimensions is in (h, w)
    # Un-distort
    # dst_left = cv.undistort(img_left, cameraMatrix, dist, None, newCameraMatrix)
    # dst_right = cv.undistort(img_right, cameraMatrix, dist, None, newCameraMatrix)
    #
    # # crop the image
    x, y, w, h = roi
    # dst_left = dst_left[y:y + h, x:x + w]
    # dst_right = dst_right[y:y + h, x:x + w]

    #  Un-distort with Remapping
    map_x, map_y = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w, h), 5)
    dst_left = cv.remap(img_left, map_x, map_y, cv.INTER_LINEAR)
    dst_right = cv.remap(img_right, map_x, map_y, cv.INTER_LINEAR)

    #  crop the image
    x, y, w, h = roi
    dst_left = dst_left[y:y + h, x:x + w]
    dst_right = dst_right[y:y + h, x:x + w]

    return dst_left, dst_right
    # cv.imwrite('caliResult2.png', dst)
    # cv.imwrite('caliResult1.png', dst)
