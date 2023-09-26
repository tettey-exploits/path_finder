# Sunday, 09 July 2023
# Visual Assistant for the Visually Impaired

# Import libraries
import cv2
import pyttsx3
import numpy as np
from estimate_distance import estimate_distance as dist
from estimate_distance import calc_centroids
import callibrate_rectify_img as rec_img
from detect_objects import detect_objects as perform_object_detection
import helper

try:
    cap_principal = cv2.VideoCapture(2)
    cap_secondary = cv2.VideoCapture(4)
except Exception:
    raise Exception("Failed to initialize camera")

img_dimensions = (640, 480)

# Set and define properties
cap_secondary.set(cv2.CAP_PROP_FRAME_WIDTH, img_dimensions[0])
cap_secondary.set(cv2.CAP_PROP_FRAME_HEIGHT, img_dimensions[1])
cap_principal.set(cv2.CAP_PROP_FRAME_WIDTH, img_dimensions[0])
cap_principal.set(cv2.CAP_PROP_FRAME_HEIGHT, img_dimensions[1])

frame_rate = cap_secondary.get(cv2.CAP_PROP_FPS)

classFile = 'resources/coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


def annotate_imgs(ssd_image, detection_info, distance):
    confidence, box, classId = detection_info
    # distance = dist(img_dimensions[0], box[0])  # box[0] represents the width of detected object
    cv2.rectangle(ssd_image, box, color=(0, 255, 0), thickness=2)

    cv2.putText(ssd_image, distance, (600, 80),  # Indicate distance to object
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)
    cv2.putText(ssd_image, classNames[classId - 1], (box[0] + 10, box[1] + 30),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
    cv2.putText(ssd_image, str(round(confidence * 100, 2)), (box[0] + 10, box[1] + 70),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)


if __name__ == "__main__":
    while True:
        if cap_principal.isOpened() and cap_secondary.isOpened():
            try:
                ret_l, img_principal = cap_principal.read()
                ret_r, img_secondary = cap_secondary.read()
            except Exception:
                print("[ERROR] Could read from camera")
                break

        else:
            print("Camera not initialized.")
            break

        if ret_r and ret_l:  # ensure that image capture was successful

            # Transform left and right images and project
            trans_img_principal, trans_img_secondary = rec_img.rectify_img(img_principal, img_secondary, img_dimensions)
            trans_img_list = [trans_img_principal]  # , trans_img_right]

            # Perform image detection on images
            detection_results = perform_object_detection(trans_img_list, confidence_threshold=0.5)

            for i in range(len(detection_results)):
                if len(detection_results[i]) != 0:
                    x, y, w, h = detection_results[i][1]
                    interested_object = trans_img_principal[y-10:y+h+10, x-10:x+w+10]

                    loc = rec_img.match_template(interested_object, trans_img_secondary, (x, y, w, h))

                    distance = "dist: "
                    for pt in zip(*loc[::-1]):
                        # print(f"pt {pt}")
                        cv2.rectangle(trans_img_secondary, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

                        disparity = calc_centroids((x, y, w, h), (pt[0], pt[1], w, h))
                        distance = dist(disparity, img_dimensions[0])  # box[0] represents the width of detected object

                    annotate_imgs(trans_img_principal, detection_results[i], distance)  # Compute object distance

            # cv2.imshow("Principal Cam", trans_img_principal)
            # cv2.imshow("Secondary Cam", trans_img_secondary)

            stacked_images = helper.stack_images(([trans_img_principal, trans_img_secondary],),  0.2)
            cv2.imshow("Stacked Images", stacked_images)

            cv2.waitKey(1)
        else:
            print("[Warning]Camera read returned False")
    cv2.destroyAllWindows()

