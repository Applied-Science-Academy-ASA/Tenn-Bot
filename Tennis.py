import time
import math

import cv2
import serial
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

from xgb import XGB
from metadata import Metadata
from utils import json2csv

class PoseDetection:
    def __init__(self):
        self.arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=115200, timeout=.1)        

        self.mtd = Metadata()
        self.xgb = XGB()        

        self.startup = True
        self.launchcount = 0
        self.antilaunch = 8
        self.lantilaunch = False
        self.lockstarttime = 0
        self.lockendtime = 0
        self.personlock = False
        self.copycoord = []
        self.block = [0,0]
        self.lock = False
        self.copyr = 0
        self.copyg = 0
        self.copyb = 0
        
        mp_pose = mp.solutions.pose        
        pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.4)
        
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_poses = 4)
        
        mp_drawing = mp.solutions.drawing_utils

    def detectPose(self):
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
        
        tolerance = 60
        
        original_image = image_pose.copy()
        image_in_RGB = cv2.cvtColor(image_pose, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_in_RGB.copy())

        if True:
            with PoseLandmarker.create_from_options(options) as landmarker:
                pose_landmarks = landmarker.detect(mp_image)
                landmark_subset = landmark_pb2.NormalizedLandmarkList()
                for i in range (len(pose_landmarks.pose_landmarks)):
                    landmark_subset.landmark.extend(
                        [landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z, visibility=landmark.visibility) for landmark in pose_landmarks.pose_landmarks[i]])
                    coords = mp_drawing.draw_landmarks(
                                image=original_image,
                                landmark_list=landmark_subset,
                                connections=mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=3, circle_radius=3),
                                connection_drawing_spec=mp_drawing.DrawingSpec(color=(49,125,237), thickness=2, circle_radius=2)
                                )

                    pointsCheck = [False,False,False,False,False,False,False,False]
                    if coords.get(0) != None:
                        pointsCheck[0] = True
                    if coords.get(11) != None:
                        pointsCheck[7] = True
                        lshouldercoordx = coords[11][0]
                        lshouldercoordy = coords[11][1]
                    #if idx == 12:
                    if coords.get(12) != None:
                        pointsCheck[1] = True
                        rshouldercoordx = coords[12][0]
                        rshouldercoordy = coords[12][1]
                    #if idx == 16:
                    if coords.get(16) != None:
                        pointsCheck[2] = True
                        rhandcoordx = coords[16][0]
                        rhandcoordy = coords[16][1]
                    #if idx == 24:
                    if coords.get(24) != None:
                        pointsCheck[3] = True
                        rhipcoordx = coords[24][0]
                        rhipcoordy = coords[24][1]
                    if coords.get(26) != None:
                        pointsCheck[4] = True
                        rkneecoordx = coords[26][0]
                        rkneecoordy = coords[26][1]
                    if coords.get(15) != None:
                        pointsCheck[5] = True
                        lhandcoordx = coords[15][0]
                        lhandcoordy = coords[15][1]
                    if coords.get(23) != None:
                        pointsCheck[6] = True
                        lhipcoordx = coords[23][0]
                        lhipcoordy = coords[23][1]
                    if not False in pointsCheck:
                        distance1 = math.sqrt(((rhandcoordx-rhipcoordx)**2)+((rhandcoordy-rhipcoordy)**2))
                        distance2 = math.sqrt(((rkneecoordx-rshouldercoordx)**2)+((rkneecoordy-rshouldercoordy)**2))
                        distance3 = math.sqrt(((lhandcoordx-lhipcoordx)**2)+((lhandcoordy-lhipcoordy)**2))
                        distance4 = math.sqrt(((lhandcoordx-rshouldercoordx)**2)+((lhandcoordy-rshouldercoordy)**2))
                        distance5 = math.sqrt(((rhandcoordx-lshouldercoordx)**2)+((rhandcoordy-lshouldercoordy)**2))

                        if distance4 <= 15 and distance5 <= 15 and not lock:
                            lockstarttime = time.time()
                            lock = True
                        elif distance4 <= 15 and distance5 <= 15 and lock:
                            lockendtime = time.time()
                            if (lockendtime - lockstarttime) > 3:
                                personlock = True
                                coordmiddlex = int((lshouldercoordx + rshouldercoordx)/2)
                                coordmiddley = int((rshouldercoordy + rhipcoordy)/2)
                                cmiddle = image_in_RGB[coordmiddley-25:coordmiddley+25, coordmiddlex-25:coordmiddlex+25]
                                mask = cmiddle.copy()
                                mask[:,:] = (255,255,255)
                                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                                mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
                                print(mask)
                                cv2.imwrite("test.png", cmiddle)
                                avgrow = np.average(cmiddle, axis = 0)
                                avg = np.average(avgrow, axis = 0)                 
                                r = avg[0]
                                g = avg[1]
                                b = avg[2]
                                copyr = r
                                copyg = g
                                copyb = b
                                mtd.add_features_mask(1, cmiddle, mask)
                                mtd.save_json()
                                json2csv("metadata.json")
                                xgb.train()
                        elif distance4 > 15 or distance5 > 15 and lock:
                            if (lockendtime - lockstarttime) > 5:
                                personlock = False
                                lock = False
                                lockstarttime = 0
                                lockendtime = 0
                        print(personlock, lock, lockendtime, lockstarttime, lockendtime - lockstarttime)
                        if personlock == True:
                            coordmiddlex = int((lshouldercoordx + rshouldercoordx)/2)
                            coordmiddley = int((rshouldercoordy + rhipcoordy)/2)
                            cmiddle = image_in_RGB[coordmiddley-25:coordmiddley+25, coordmiddlex-25:coordmiddlex+25]
                            avgrow = np.average(cmiddle, axis = 0)
                            avg = np.average(avgrow, axis = 0)          
                            original_image = cv2.circle(original_image, (coordmiddlex, coordmiddley), 5, (0,255,0), -1)
                            r = avg[0]
                            g = avg[1]
                            b = avg[2]
                            print(abs(r - copyr) + abs(g - copyg) + abs(b-copyb))
                            if abs(r - copyr) + abs(g - copyg) + abs(b-copyb) < 100:
                                copyr = r
                                copyg = g
                                copyb = b
                                if distance1 > distance2 and launchcount == 0:
                                    launchcount += 1
                                    time.sleep(1)
                                    if antilaunch == 8: lantilaunch = True
                                elif distance1 > distance2 and launchcount >= 1:  
                                    if antilaunch == 8:
                                        arduino.write(bytes("l", 'utf-8'))
                                        launchcount = 0
                                        lantilaunch = True
                                elif distance3 > distance2 and launchcount == 0:
                                    launchcount += 1
                                    time.sleep(2)
                                    if antilaunch == 8: lantilaunch = True
                                elif distance3 > distance2 and launchcount >= 1:
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


