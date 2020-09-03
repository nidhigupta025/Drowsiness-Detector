# Importing the required packages
import time
import cv2
import sys
from imutils import face_utils
import dlib
import winsound
from scipy.spatial import distance

Drowsiness_threshold = 0.25
Yawn_threshold = 0.35
blink_threshold = 0.21
# dlib's predefined package to detect faces in the image
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def alarm():
    winsound.PlaySound("alarm.wav", winsound.SND_ASYNC | winsound.SND_ALIAS)


# Function to calculate Eye Aspect Ratio
# It takes the eye coordinates as input and returns the Aspect_Ratio

def EAR(eye):
    # Horizontal Distance b/w two points of the eye
    a = (int((eye[1][0] + eye[2][0]) / 2), int((eye[1][1] + eye[2][1]) / 2))
    b = (int((eye[4][0] + eye[5][0]) / 2), int((eye[4][1] + eye[5][1]) / 2))
    horizontal_distance = distance.euclidean(a, b)
    # Vertical Distance b/w the co-ordinates of side of the eye
    vertical_distance = distance.euclidean(eye[0], eye[3])
    return horizontal_distance / vertical_distance


# Function to calculate Mouth Aspect Ratio
# It takes the coordinates of the inner lip as input and returns the Aspect_Ratio

def MAR(mouth):
    # Horizontal Distances b/w the three points of the mouth detected by the detector
    a = distance.euclidean(mouth[1], mouth[7])
    b = distance.euclidean(mouth[2], mouth[6])
    c = distance.euclidean(mouth[3], mouth[5])
    # Vertical Distance  b/w the coordinates of the side of the eye
    d = distance.euclidean(mouth[0], mouth[4])
    return (a + b + c) / (3 * d)


blinks_counter = 0
blink_detected = False
is_alarm_working = False
mouth_alarm = False
blink_alarm = False
drowsy_eye = False
yawn_detected = False
stressed = False

# Capturing the video from the webcam
cap = cv2.VideoCapture(0)
minute_time = time.time()

# Checking whether VideoCapture object is initialized or not
if not cap.isOpened():
    print("Camera cannot be fetched.")
    sys.exit()

while True:
    if time.time() - minute_time >= 60.0:
        stressed = False
        blinks_counter = 0
        minute_time = time.time()
    # Capturing frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Converting current image frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 0)

    # To loop over the detected faces
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[36:42]
        right_eye = shape[42:48]

        left_eye_ear = EAR(left_eye)
        right_eye_ear = EAR(right_eye)

        average_ear = (left_eye_ear + right_eye_ear) / 2
        if average_ear<=Drowsiness_threshold:
            cv2.putText(frame, "EAR: {:.2f}".format(average_ear), (520, 20),cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "EAR: {:.2f}".format(average_ear), (520, 20),cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)

        if average_ear < Drowsiness_threshold:

            if not drowsy_eye:
                drowsy_eye = True
                drowsy_start_time = time.time()

            if time.time() - drowsy_start_time >= 1.5:

                drowsy_start_time = time.time()
                is_alarm_working = False
                if not is_alarm_working:
                    is_alarm_working = True
                    alarm()

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            drowsy_start_time = time.time()
            is_alarm_working = False

        if average_ear < blink_threshold:
            # cv2.putText(frame, "Blinking", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if not blink_detected:
                blink_detected = True
                blinks_counter += 1

            if blinks_counter == 23 and not is_alarm_working and not mouth_alarm:
                stressed = True
                alarm()

        else:
            blink_detected = False

        if blinks_counter > 22:
            # Printing the blink count in red if it exceeds the per minute average blink threshold
            cv2.putText(frame, "BLINKS: {}".format(blinks_counter), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                        (0, 0, 255), 2)
        else:
            # Otherwise printing the blink count in green if blink count per minute is normal
            cv2.putText(frame, "BLINKS: {}".format(blinks_counter), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                        (0, 255, 0), 2)

        if stressed == True:
            cv2.putText(frame, "YOU'RE STRESSED OUT !! PLEASE TAKE REST !!", (60, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)

        inner_lip = shape[60:68]
        mouth_mar = MAR(inner_lip)
        # print(mouth_mar)
        if mouth_mar>=Yawn_threshold:
            cv2.putText(frame, "MAR: {:.2f}".format(mouth_mar), (520, 40),cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "MAR: {:.2f}".format(mouth_mar), (520, 40),cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)
        if mouth_mar > Yawn_threshold:

            if not yawn_detected:
                yawn_detected = True
                yawn_start_time = time.time()

            if time.time() - yawn_start_time >= 4.0:

                if not is_alarm_working and not mouth_alarm:
                    mouth_alarm = True
                    alarm()

                cv2.putText(frame, "YAWN ALERT!", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            yawn_start_time = time.time()
            mouth_alarm = False

    # Displaying resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# When everything is done, release the VideoCapture object
cv2.destroyAllWindows()
cap.release()