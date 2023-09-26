# Sunday, 09 July 2023
# Visual Assistant for the Visually Impaired

# Import libraries
import cv2
from detected_objects_class import DetectedObjectClass
import callibrate_rectify_img as rec_img
from detect_objects import detect_objects as perform_object_detection
import globals
import threading

try:
    cap_principal = cv2.VideoCapture(2)
    cap_secondary = cv2.VideoCapture(4)
except Exception:
    raise Exception("Failed to initialize camera")

img_dimensions = globals.IMG_DIMENSIONS

# Set and define properties
cap_secondary.set(cv2.CAP_PROP_FRAME_WIDTH, img_dimensions[0])
cap_secondary.set(cv2.CAP_PROP_FRAME_HEIGHT, img_dimensions[1])
cap_principal.set(cv2.CAP_PROP_FRAME_WIDTH, img_dimensions[0])
cap_principal.set(cv2.CAP_PROP_FRAME_HEIGHT, img_dimensions[1])

frame_rate = cap_secondary.get(cv2.CAP_PROP_FPS)

classFile = 'resources/coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


def annotate_imgs(ssd_image, detected_objects_list):

    for detected_object in detected_objects_list:
        distance = detected_object.detected_object_distance
        confidence = detected_object.detected_object_conf
        box = detected_object.detected_object_bbox
        classId = detected_object.detected_object_name

        cv2.rectangle(ssd_image, box, color=(0, 255, 0), thickness=2)
        cv2.putText(ssd_image, distance, (600, 80),  # Indicate distance to object
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(ssd_image, classNames[classId - 1], (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        cv2.putText(ssd_image, str(round(confidence * 100, 2)), (box[0] + 10, box[1] + 70),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)

        print(f"Detected object is: {classNames[classId - 1]}")
        print(f"Estimated distance to detected object is {distance}")


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

            # Transform left and right images to lie on the same plane
            trans_img_principal, trans_img_secondary = rec_img.rectify_img(img_principal, img_secondary, img_dimensions)

            trans_img_list = [trans_img_principal]

            # Perform image detection on images
            detection_results = perform_object_detection(trans_img_list, confidence_threshold=0.5)
            detected_objects_object_list = []

            for i in range(len(detection_results)):
                if len(detection_results[i]) != 0:
                    x, y, w, h = detection_results[i][1]

                    # Crop location of detected object and use it to find a match in the second camera
                    interested_object = trans_img_principal[y - 10:y + h + 10, x - 10:x + w + 10]
                    pt, _ = rec_img.match_template(interested_object, trans_img_secondary, detection_results[i][1])

                    cv2.rectangle(trans_img_secondary, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
                    detected_objects_object_list.append(DetectedObjectClass(detection_results[i], (pt[0], pt[1], w, h)))

            annotate_imgs(trans_img_principal, detected_objects_object_list)  # Compute object distance

            stacked_images = globals.stack_images(([trans_img_principal, trans_img_secondary],), 0.2)
            cv2.imshow("Stacked Images", stacked_images)

            cv2.waitKey(1)
        else:
            print("[Warning]Camera read Failed")

    cv2.destroyAllWindows()
