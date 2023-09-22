# Sunday, 09 July 2023
# Visual Assistant for the Visually Impaired

# Import libraries
import sys
import cv2
import time
import pyttsx3
from estimate_distance import estimate_distance as dist
import callibrate_rectify_img as rec_img
from detect_objects import detect_objects as perform_object_detection
import helper

try:
    cap_left = cv2.VideoCapture(2)
    cap_right = cv2.VideoCapture(4)
except Exception:
    raise Exception("Failed to initialize camera")

img_dimensions = (640, 480)

# Set and define properties
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, img_dimensions[0])
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, img_dimensions[1])
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, img_dimensions[0])
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, img_dimensions[1])

frame_rate = cap_right.get(cv2.CAP_PROP_FPS)

classFile = 'resources/coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


def eva(ssd_image, detection_info):
    for classId, confidence, box in detection_info:
        distance = dist(img_dimensions[0], box[0])  # box[0] represents the width of detected object
        cv2.rectangle(ssd_image, box, color=(0, 255, 0), thickness=2)

        cv2.putText(ssd_image, distance, (box[0] + (box[0] + 10), box[1] + 120),  # Indicate distance to object
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(ssd_image, classNames[classId - 1], (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        cv2.putText(ssd_image, str(round(confidence * 100, 2)), (box[0] + 10, box[1] + 70),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)


if __name__ == "__main__":
    while True:

        if cap_left.isOpened() and cap_right.isOpened():
            try:
                ret_l, img_left = cap_left.read()
                ret_r, img_right = cap_right.read()
            except Exception:
                print("[ERROR] Could read from camera")
                break

        else:
            print("Camera not initialized.")
            break

        if ret_r == ret_l and ret_l:  # ensure that image capture was successful

            # Transform left and right images and project
            trans_img_left, trans_img_right = rec_img.rectify_img(img_left, img_right, img_dimensions)

            # Perform image detection on images
            isObjectsDetected, detection_results = perform_object_detection(img_left, confidence_threshold=0.3)

            if isObjectsDetected:
                eva(trans_img_left, detection_results)  # Compute object distance

            cv2.imshow("Left Cam", trans_img_left)
            cv2.imshow("Right Cam", trans_img_right)

            # stacked_images = helper.stack_images([img_lef, trans_img_left], 60)
            # cv2.imshow("Right Cam", stacked_images)

            cv2.waitKey(1)
    cv2.destroyAllWindows()

