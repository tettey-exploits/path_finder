# Sunday, 09 July 2023
# Visual Assistant for the Visually Impaired

# Import libraries
import sys
import cv2
import time

message = ""
cap = cv2.VideoCapture(0)

classFile = 'resources/coco.names'
configPath = 'resources/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'resources/frozen_inference_graph.pb'

ids_interested_classes = [29, 77, 71, 44, 72, 73]

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load deep neural network
net = cv2.dnn_DetectionModel(weightPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def detect_objects(img, confidence_threshold=0.5):
    results_list = []

    class_ids, confs, bbox = net.detect(img, confThreshold=confidence_threshold)

    if len(class_ids) != 0:
        # detected_class_interest_list is a list of ids of detected classes we are interested
        # in as defined in ids_interested_classes and their corresponding index in class_ids
        detected_class_interest_list = [class_oi for class_oi in enumerate(class_ids) if class_oi[1] in
                                        ids_interested_classes]
        if len(detected_class_interest_list) != 0:
            for index in detected_class_interest_list:
                results_list.append([confs[index[0]], bbox[index[0]].flatten(), index[1]])

    return results_list


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

        cv2.waitKey(1)
