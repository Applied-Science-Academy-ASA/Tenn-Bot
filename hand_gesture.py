# TechVidvan hand Gesture Recognizer

# import necessary packages

import cv2
import numpy as np
import mediapipe as mp
from servo import servo
import tensorflow as tf
from tensorflow.keras.models import load_model
from threading import Thread

import serial
import time
# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

notExit = True

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)
arduino = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=.1)

theHand = 1
theservo = servo()

tolerance = 60
confidence = 30

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=theHand, min_detection_confidence=confidence)
mpDraw = mp.solutions.drawing_utils

# initialize global variableas
className = ''
landmarks = None
detected = False

# Initialize the webcam
cap = cv2.VideoCapture(0)
angle = 90
theEnd = False
def startprogram():
    global landmarks, className, detected, notExit
    while notExit:
        # Read each frame from the webcam
        
        _, frame = cap.read()
        x, y, c = frame.shape

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
        result = hands.process(framergb)

        # print(result)

        
        thecoords = {}
        # post process the result
        if result.multi_hand_landmarks:
            detected = True
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
                thecoords = mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        else:
            detected = False
        for idx, landmark_px in thecoords.items():
            if idx == 9:
                if frame.shape[1]/2 < landmark_px[0]-tolerance and angle < 180:
                    arduino.write(bytes("r", 'utf-8'))
                    time.sleep(0.001)
                elif frame.shape[1]/2 > landmark_px[0]+tolerance and angle >0:
                    arduino.write(bytes("l", 'utf-8'))
                    time.sleep(0.001)
                elif frame.shape[1]/2 <= landmark_px[0]+tolerance and frame.shape[1]/2 >= landmark_px[0]-tolerance and angle >0:
                    if className == "fist":
                        arduino.write(bytes("launch", "utf-8"))
                        time.sleep(2)
                        if className == "fist":
                            arduino.write(bytes("launch", "utf-8"))
                cv2.circle(frame,landmark_px,3,(255,255,0),2)
        # show the prediction on the frame
        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0,0,255), 2, cv2.LINE_AA)
        cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # Show the final output
        cv2.imshow("Output", frame)
        pressedKey = cv2.waitKey(1)
        if pressedKey == ord('q'):
            theExit()
            notExit = False

def predictgesture():
    global landmarks, className, detected, notExit
    while notExit:
        if detected:
            prediction = model.predict([landmarks])
            classID = np.argmax(prediction)
            className = classNames[classID]

t1 = Thread(target=startprogram)
t2 = Thread(target=predictgesture)
t1.start()
t2.start()

def theExit():
    global cap
    cap.release()
    cv2.destroyAllWindows()

# release the webcam and destroy all active windows
