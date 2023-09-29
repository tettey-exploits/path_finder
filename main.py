# Sunday, 09 July 2023
# Visual Assistant for the Visually Impaired

# Import libraries
import cv2
import pyttsx3
from detected_objects_class import DetectedObjectClass
import callibrate_rectify_img as rec_img
from detect_objects import detect_objects as perform_object_detection
import globals
from Timer import Timer
import threading

speech_engine = pyttsx3.init()
speech_engine.setProperty('rate', 125)  # setting up new voice rate
speech_engine.setProperty('volume', 1.0)  # setting up volume level  between 0 and 1
voices = speech_engine.getProperty('voices')  # getting details of current voice
speech_engine.setProperty('voice', voices[15].id)  # changing index, changes voices. 1 for female

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

text_to_read = ""
new_text = False

classFile = 'resources/coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


def annotate_imgs(ssd_image, detected_objects_list, lock_obj):
    global new_text, text_to_read

    for detected_object in detected_objects_list:
        distance = detected_object.detected_object_distance
        confidence = detected_object.detected_object_conf
        box = detected_object.detected_object_bbox
        classId = detected_object.detected_object_name

        text_name = f"Detected object: {classNames[classId - 1]}"
        text_distance = f"Estimated distance: {distance}"

        cv2.rectangle(ssd_image, box, color=(0, 255, 0), thickness=2)
        cv2.putText(ssd_image, text_name, (20, 20),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(ssd_image, text_distance, (20, 45),  # Indicate distance to object
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)
        # cv2.putText(ssd_image, str(round(confidence * 100, 2)), (box[0] + 10, box[1] + 70),
        #             cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)

        if lock.acquire(False) and new_text == False:
            new_text = True
            text_to_read = f"{text_name} and {text_distance} centimeters"

        lock.release()
        print(text_name)
        print(text_distance)


def speech_module(thread_lock):
    global text_to_read, new_text
    while True:
        if new_text:
            speech_engine.say(text_to_read)
            speech_engine.runAndWait()
            with thread_lock:
                new_text = False
                print(f"speech: {new_text}")


if __name__ == "__main__":

    # speech_engine.say("Hello, how are you feeling today?")
    # speech_engine.say("Just ask what you want to find, and I will help you with it")
    # speech_engine.runAndWait()

    lock = threading.Lock()
    speech_thread = threading.Thread(target=speech_module, args=(lock,), daemon=True)
    speech_thread.start()

    begin_object_detection = Timer()

    while True:
        if cap_principal.isOpened() and cap_secondary.isOpened():
            try:
                ret_l, img_principal = cap_principal.read()
                ret_r, img_secondary = cap_secondary.read()
            except Exception:
                print("[ERROR] Could read from camera")
                break
        else:
            print("[ERROR] Camera not initialized.")
            break

        if ret_r and ret_l:  # ensure that image capture was successful

            # Transform left and right images to lie on the same plane
            trans_img_principal, trans_img_secondary = rec_img.rectify_img(img_principal, img_secondary, img_dimensions)

            if begin_object_detection.check_timer(5):
                # Perform image detection on images
                detection_result = perform_object_detection(trans_img_principal, confidence_threshold=0.5)

                detected_objects_object_list = []
                for i in range(len(detection_result)):
                    x, y, w, h = detection_result[i][1]

                    # Crop location of detected object and use it to find a match in the second camera
                    interested_object = trans_img_principal[y - 10:y + h + 10, x - 10:x + w + 10]
                    pt, _ = rec_img.match_template(interested_object, trans_img_secondary, detection_result[i][1])

                    cv2.rectangle(trans_img_secondary, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
                    detected_objects_object_list.append(DetectedObjectClass(detection_result[i], (pt[0], pt[1], w, h)))

                annotate_imgs(trans_img_principal, detected_objects_object_list, lock)  # Compute object distance

            stacked_images = globals.stack_images(([trans_img_principal, trans_img_secondary],), 0.2)
            cv2.imshow("Stacked Images", stacked_images)

            cv2.waitKey(1)
        else:
            print("[Warning]Camera read Failed")

    speech_thread.join()

    cv2.destroyAllWindows()
