import cv2

import mediapipe as mp
import os
import matplotlib.pyplot as plt
import serial
import time
import math
import numpy as np
from xgb import XGB
from metadata import Metadata
from utils import json2csv
from mediapipe.framework.formats import landmark_pb2
from features import new_features

# Model for mediapipe pose
model_path = 'pose_landmarker_full.task'

#XGBoost Class declaration
mtd = Metadata()

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose

# Setup the Pose function for images - independently for the images standalone processing.
pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.4)

# Setup the Pose function for videos - for video processing.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.4,
                          min_tracking_confidence=0.4)

# New mediapipe syntax: multi pose detection
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_poses = 4) #max = 4 persons

# Initialize mediapipe drawing class - to draw the landmarks points.
mp_drawing = mp.solutions.drawing_utils

# Actualy it is an ESP32, but we call it arduino.
arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=115200, timeout=.1)

#Global Variables
launchcount = 0   # Just launch once FLAG
antilaunch = 8   # Frame imunity TIMER
lantilaunch = False   # Frame imunity FLAG
lockstarttime = 0 # Lock TIMER
lockendtime = 0 # Lock TIMER
personlock = False   # "Lock me" FLAG
block = [0,0]   # Black Rectangle (not used)
lock = False   # Lock pose?
framescaptured= 0
capture = False
startlockcoord = None
afterlockcoord = None

def detectPose(image_pose, pose, draw=False, display=False):
    global launchcount, antilaunch, lantilaunch, block, lockendtime, lockstarttime, personlock, lock, xgb, mtd, framescaptured, capture, startlockcoord
    
    #Coordinates
    rhandcoordx = 0
    rhandcoordy = 0
    rhipcoordx = 0
    rhipcoordy = 0
    rkneecoordx = 0
    rkneecoordy = 0
    rshouldercoordx = 0
    rshouldercoordy = 0
    lshouldercoordx = 0
    lshouldercoordy = 0
    lhipcoordx = 0
    lhipcoordy = 0
    lhandcoordx = 0
    lhandcoordy = 0

    original_image = image_pose.copy()
    
    image_in_RGB = cv2.cvtColor(image_pose, cv2.COLOR_BGR2RGB)

    # New mediapipe syntax: multi pose detection
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_in_RGB.copy())    
    
    #image_in_RGB = cv2.rectangle(image_in_RGB, (block[0], 0), (block[1], image_in_RGB.shape[0]), (0,0,0), -1)
    #resultant = pose.process(image_in_RGB)
    #if resultant.pose_landmarks and draw:
    if True:
        # New mediapipe syntax: multi pose detection
        with PoseLandmarker.create_from_options(options) as landmarker:
            pose_landmarks = landmarker.detect(mp_image)
            landmark_subset = landmark_pb2.NormalizedLandmarkList()
            
            person_count = len(pose_landmarks.pose_landmarks)
            usefulcoords = []
            persondecided = False
            for i in range (person_count):
                addusefulcoords = []
                landmark_subset.landmark.extend(
                    [landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z, visibility=landmark.visibility) for landmark in pose_landmarks.pose_landmarks[i]])
                coords = mp_drawing.draw_landmarks(
                            image=original_image,
                            landmark_list=landmark_subset,
                            connections=mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=3, circle_radius=3),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(49,125,237), thickness=2, circle_radius=2)
                            )
                #for idx, landmark_px in coords.items():
                #if idx == 0:
                pointsCheck = [False,False,False,False,False,False,False,False] # check all required points
                if coords.get(0) != None: #Nose
                    pointsCheck[0] = True 
                if coords.get(11) != None: #Left Shoulder
                    pointsCheck[7] = True
                    lshouldercoordx = coords[11][0]
                    lshouldercoordy = coords[11][1]
                if coords.get(12) != None: #Right Shoulder
                    pointsCheck[1] = True
                    rshouldercoordx = coords[12][0]
                    rshouldercoordy = coords[12][1]
                if coords.get(16) != None: #Right Wrist
                    pointsCheck[2] = True
                    rhandcoordx = coords[16][0]
                    rhandcoordy = coords[16][1]
                if coords.get(24) != None: #Left Hip
                    pointsCheck[3] = True
                    rhipcoordx = coords[24][0]
                    rhipcoordy = coords[24][1]
                if coords.get(26) != None: #Left Knee
                    pointsCheck[4] = True
                    rkneecoordx = coords[26][0]
                    rkneecoordy = coords[26][1]
                if coords.get(15) != None: #Left Wrist
                    pointsCheck[5] = True
                    lhandcoordx = coords[15][0]
                    lhandcoordy = coords[15][1]
                if coords.get(23) != None: #Left Hip
                    pointsCheck[6] = True
                    lhipcoordx = coords[23][0]
                    lhipcoordy = coords[23][1]
                
                if not False in pointsCheck: # check all required points
                    RWrist_RHip = math.sqrt(((rhandcoordx-rhipcoordx)**2)+((rhandcoordy-rhipcoordy)**2)) 
                    RKnee_RShoulder = math.sqrt(((rkneecoordx-rshouldercoordx)**2)+((rkneecoordy-rshouldercoordy)**2))
                    LWrist_LHip = math.sqrt(((lhandcoordx-lhipcoordx)**2)+((lhandcoordy-lhipcoordy)**2))
                    LWrist_RShoulder = math.sqrt(((lhandcoordx-rshouldercoordx)**2)+((lhandcoordy-rshouldercoordy)**2))
                    RWrist_LShoulder = math.sqrt(((rhandcoordx-lshouldercoordx)**2)+((rhandcoordy-lshouldercoordy)**2))
                    addusefulcoords = [lshouldercoordx, lshouldercoordy, rshouldercoordx, rshouldercoordy, rhandcoordx, rhandcoordy, rhipcoordx, rhipcoordy, rkneecoordx, rkneecoordy, lhandcoordx, lhandcoordy, lhipcoordx, lhipcoordy, RWrist_RHip, RKnee_RShoulder, LWrist_LHip, LWrist_RShoulder, RWrist_LShoulder]
                    usefulcoords.append(addusefulcoords)
            
            personindex = 0
            personvalue = 0
            #Deciding Single or XGB mode
            if len(usefulcoords) > 1:
                for i, k in enumerate(usefulcoords):
                    if k[17] <= 30 and k[18] <= 30:
                        if not lock and not persondecided:
                            lockstarttime = time.time()
                            lock = True
                            persondecided = True
                            try:os.remove("metadata.json")
                            except:print("metadata.json not found")
                            try:os.remove("metadata_reg.csv")
                            except:print("metadata_reg.csv not found")
                            try:os.remove("model.pkl")
                            except:print("model.pkl not found")
                        personvalue = k
                        personindex = i
                
                if capture == True:
                    if framescaptured < 10 * len(usefulcoords):
                        for i, k in enumerate(usefulcoords):
                            coordmiddlex = int((k[0] + k[2])/2)
                            coordmiddley = int((k[3] + k[7])/2)
                            #rgb = image_in_RGB[coordmiddley,coordmiddlex]
                            cmiddle = image_in_RGB[coordmiddley-25:coordmiddley+25, coordmiddlex-25:coordmiddlex+25]
                            cside = image_in_RGB[coordmiddley+25:coordmiddley+75, coordmiddlex-75:coordmiddlex-25]
                            mask = cmiddle.copy()
                            mask[:,:] = (255,255,255)
                            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                            mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
                            print(abs(startlockcoord - k[0]))
                            if abs(startlockcoord - k[0]) < 20:
                                mtd.add_features_mask(f"{framescaptured}_1", cmiddle, cside, mask)    
                                startlockcoord = k[0]
                            else:
                                mtd.add_features_mask(f"{framescaptured}_0", cmiddle, cside, mask)
                            framescaptured += 1
                            
                    elif framescaptured >= 10:
                        personlock = True
                        mtd.save_json()
                        json2csv("metadata.json")
                        xgb = XGB(data_path = "metadata_reg.csv")
                        xgb.train(model_path="model.pkl", data_path = "metadata_reg.csv")
                        capture = False

                if personvalue != 0: 
                    if personvalue[17] <= 30 and personvalue[18] <= 30 and lock:
                        lockendtime = time.time()
                        if (lockendtime - lockstarttime) > 3:
                            capture = True
                            startlockcoord = personvalue[0]
                    elif personvalue[17] > 30 or personvalue[18] > 30 and lock:
                        if (lockendtime - lockstarttime) > 5:
                            personlock = False
                            lock = False
                            lockstarttime = 0
                            lockendtime = 0
                print(personlock, lock, lockendtime, lockstarttime, lockendtime - lockstarttime)
                if personlock == True:
                    for k in usefulcoords:
                        coordmiddlex = int((k[0] + k[2])/2)
                        coordmiddley = int((k[3] + k[7])/2)
                        #rgb = image_in_RGB[coordmiddley,coordmiddlex]
                        cmiddle = image_in_RGB[coordmiddley-25:coordmiddley+25, coordmiddlex-25:coordmiddlex+25]
                        cside = image_in_RGB[coordmiddley+25:coordmiddley+75, coordmiddlex-75:coordmiddlex-25]
                        mask = cmiddle.copy()
                        mask[:,:] = (255,255,255)
                        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                        mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
                        features = [str(x) for x in new_features(cmiddle, mask)]
                        features += [str(x) for x in new_features(cside, mask)]
                        similarity = xgb.predict(model_path="model.pkl", data = ' '.join(features))
                        print(similarity)
                        if similarity > 0.7:
                            original_image = cv2.circle(original_image, (coordmiddlex, coordmiddley), 5, (0,255,0), -1)
                            if k[14] > k[15] and launchcount == 0:
                                launchcount += 1
                                time.sleep(1)
                                if antilaunch == 8: 
                                    lantilaunch = True
                            elif k[14] > k[15] and launchcount >= 1:  
                                if antilaunch == 8:
                                    arduino.write(bytes("l", 'utf-8'))
                                    launchcount = 0
                                    lantilaunch = True
                            elif k[16] > k[15] and launchcount == 0:
                                launchcount += 1
                                time.sleep(2)
                                if antilaunch == 8: lantilaunch = True
                            elif k[16] > k[15] and launchcount >= 1:
                                if antilaunch == 8:
                                    arduino.write(bytes("k", 'utf-8'))
                                    launchcount = 0
                                    lantilaunch = True
                            else:
                                arduino.write(bytes(str(coords[0][0]), 'utf-8'))
                                launchcount = 0
                            if lantilaunch:     
                                antilaunch-=1
                                if antilaunch<=0:
                                    lantilaunch = False                    
                                    antilaunch = 8
            elif len(usefulcoords) == 1:
                print("single person")
                if usefulcoords[0][14] > usefulcoords[0][15] and launchcount == 0:
                    launchcount += 1
                    time.sleep(1)
                    if antilaunch == 8: 
                        lantilaunch = True
                elif usefulcoords[0][14] > usefulcoords[0][15] and launchcount >= 1:  
                    if antilaunch == 8:
                        arduino.write(bytes("l", 'utf-8'))
                        launchcount = 0
                        lantilaunch = True
                elif usefulcoords[0][16] > usefulcoords[0][15] and launchcount == 0:
                    launchcount += 1
                    time.sleep(2)
                    if antilaunch == 8: lantilaunch = True
                elif usefulcoords[0][16] > usefulcoords[0][15] and launchcount >= 1:
                    if antilaunch == 8:
                        arduino.write(bytes("k", 'utf-8'))
                        launchcount = 0
                        lantilaunch = True
                else:
                    print("moving")
                    arduino.write(bytes(str(coords[0][0]), 'utf-8'))
                    launchcount = 0
                if lantilaunch:     
                    antilaunch-=1
                    if antilaunch<=0:
                        lantilaunch = False                    
                        antilaunch = 8
  
    if display:
        plt.figure(figsize=[100,100])
        plt.subplot(121);plt.imshow(image_pose[:,:,::-1]);plt.title("Input Image");plt.axis('off');
        plt.subplot(122);plt.imshow(original_image[:,:,::-1]);plt.title("Pose detected Image");plt.axis('off');

    else:
        return original_image


image_path = 'media/sample2.jpg'
output = cv2.VideoCapture(0)
while True: 
    _, frame = output.read()
    oi= detectPose(frame, pose_image, draw=True, display=False)
    cv2.imshow("frame", oi)

    if cv2.waitKey(1) == ord('q'):
        break
