# Sunday, 09 July 2023
# Visual Assistant for the Visually Impaired

# Import libraries
import sys
import cv2
import time
import pyttsx3

message = ""
cap = cv2.VideoCapture(0)

frame_width = 480
frame_height = 240

classFile = 'resources/coco.names'
configPath = 'resources/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'resources/frozen_inference_graph.pb'

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load deep neural network
net = cv2.dnn_DetectionModel(weightPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def write_text(img, lane, num_cars=0):
    cv2.putText(img=img, text="Number of cars: " + str(num_cars), org=(20, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=0.5,
                color=(0, 255, 0), thickness=1)
    cv2.putText(img=img, text="Lane: " + str(lane), org=(20, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 255, 0), thickness=1)


def detect_objects(img, confidence_threshold=0.5):
    ssd_image = img
    class_ids, confs, bbox = net.detect(img, confThreshold=confidence_threshold)
    isObjectsDetected = False

    if len(class_ids) != 0:
        isObjectsDetected = True
        # for classId, confidence, box in zip(class_ids.flatten(), confs.flatten(), bbox):
        #     cv2.rectangle(ssd_image, box, color=(0, 255, 0), thickness=2)
        #
        #     cv2.putText(ssd_image, classNames[classId - 1], (box[0] + 10, box[1] + 30),
        #                 cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        #     cv2.putText(ssd_image, str(round(confidence * 100, 2)), (box[0] + 10, box[1] + 70),
        #                 cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)

    # cv2.imshow("Detections", ssd_image)

    return isObjectsDetected, zip(class_ids, confs, bbox)


cap_failure = 0

if __name__ == "__main__":
    while True:
        success, frame = cap.read()  # Load frame

        if success:
            detect_objects(frame)
            cv2.imshow("Current view", frame)
            cap_failure = 0  # Reset failure counter
        else:
            message = "Could not capture frame"
            print("[ERROR] Could not capture frame")

            cap_failure = + 1  # Increment failure counter
            if cap_failure > 5:  # If error occurs more than 5 times, close system
                sys.exit()
            time.sleep(1)  # Rest for a second before reattempting

        # cv2.imshow("Stacked Images", stackedImages)

        cv2.waitKey(1)
