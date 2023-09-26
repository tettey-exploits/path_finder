import numpy as np
import cv2
import glob
import pickle


resources_path = "resources"
cameraMatrix = pickle.load(open(f"{resources_path}/cameraMatrix.pkl", "rb"))
dist = pickle.load(open(f"{resources_path}/dist.pkl", "rb"))
newCameraMatrix = pickle.load(open(f"{resources_path}/newCameraMatrix.pkl", "rb"))
roi = pickle.load(open(f"{resources_path}/roi.pkl", "rb"))


def match_template(template_rgb, img, box):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template_rgb, cv2.COLOR_BGR2GRAY)

    # Perform match operations.
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    return max_loc, max_val

    # loc = np.where(res >= 0.6)
    # # print(res)
    # # loc = np.max(res)
    # return loc


def calibrate_cam():  # function to calibrate cameras
    global cameraMatrix, newCameraMatrix, roi, dist

    chessboardSize = (8, 6)
    frameSize = (640, 480)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

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
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

        # if found, add object points, image points (after refining them)
        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(1000)

    ret, cameraMatrix, dist, r_vecs, t_vecs = cv2.calibrateCamera(obj_points, img_points, frameSize, None, None)
    calib_img = cv2.imread(f'{folder_name}/img0.png')
    h, w = calib_img.shape[:2]  # resolve image dimensions and use in new calibration
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

    pickle.dump(open(f"{resources_path}/cameraMatrix.pkl", "rb"))
    pickle.dump(open(f"{resources_path}/dist.pkl", "rb"))
    pickle.dump(open(f"{resources_path}/newCameraMatrix.pkl", "rb"))
    pickle.dump(open(f"{resources_path}/roi.pkl", "rb"))

    cv2.destroyAllWindows()


def rectify_img(img_left, img_right, img_dimensions: tuple):
    # # crop the image
    x, y, w, h = roi  # region of interest

    #  Un-distort with Remapping
    map_x, map_y = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w, h), 5)
    dst_left = cv2.remap(img_left, map_x, map_y, cv2.INTER_LINEAR)
    dst_right = cv2.remap(img_right, map_x, map_y, cv2.INTER_LINEAR)

    #  crop the image
    x, y, w, h = roi
    dst_left = dst_left[y:y + h, x:x + w]
    dst_right = dst_right[y:y + h, x:x + w]

    return dst_left, dst_right
