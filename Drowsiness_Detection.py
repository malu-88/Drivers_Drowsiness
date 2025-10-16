import cv2
import dlib
from scipy.spatial import distance
import winsound
import threading
import time

# ==============================
# Function to calculate Eye Aspect Ratio (EAR)
# ==============================
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# ==============================
# Alarm Control using built-in beep
# ==============================
alarm_on = False

def play_alarm():
    """Continuously beep while alarm_on is True"""
    print("[INFO] Alarm thread started")
    while alarm_on:
        winsound.Beep(2500, 500)  # 2500 Hz for 0.5 sec
        time.sleep(0.1)           # short pause to avoid overlap

# ==============================
# Dlib Setup
# ==============================
print("[INFO] Loading facial landmarks predictor...")
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye landmark indexes
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# ==============================
# Parameters
# ==============================
EAR_THRESHOLD = 0.25        # below this = eyes closed
CLOSED_TIME_LIMIT = 3.0     # seconds of eyes closed to trigger alarm
start_closed_time = None

# ==============================
# Start webcam
# ==============================
cap = cv2.VideoCapture(0)
print("[INFO] Press 'a' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect(gray, 0)

    for face in faces:
        shape = predict(gray, face)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Draw eye landmarks
        for (x, y) in leftEye + rightEye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # ==============================
        # Drowsiness Detection Logic
        # ==============================
        if ear < EAR_THRESHOLD:
            if start_closed_time is None:
                start_closed_time = time.time()  # start timer
            elapsed = time.time() - start_closed_time

            cv2.putText(frame, f"Eyes Closed ({elapsed:.1f}s)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if elapsed >= CLOSED_TIME_LIMIT:
                cv2.putText(frame, "********** DROWSY! **********", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

                if not alarm_on:
                    alarm_on = True
                    threading.Thread(target=play_alarm, daemon=True).start()

        else:
            # Eyes open → reset timer and stop alarm
            start_closed_time = None
            cv2.putText(frame, "Eyes Open", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if alarm_on:
                alarm_on = False  # thread will stop automatically

    cv2.imshow("Drowsiness Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("a"):
        break

cap.release()
cv2.destroyAllWindows()
alarm_on = False
