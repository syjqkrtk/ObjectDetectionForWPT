import cv2
import os
import datetime
import numpy as np
import TsaiCalibration as Tsai

# HYPERPARAMETERS DEFINITION
RESOLUTION12 = [1280, 720]
RESOLUTION34 = [1280, 720]

os.system('cls')
print("")
print("Now Loading...")

# WebCam Camera Loading
cap1 = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(3,cv2.CAP_DSHOW)
cap3 = cv2.VideoCapture(4,cv2.CAP_DSHOW)
cap4 = cv2.VideoCapture(1,cv2.CAP_DSHOW)
#cap1 = cv2.VideoCapture(0)
#cap2 = cv2.VideoCapture(3)
#cap3 = cv2.VideoCapture(4)
#cap4 = cv2.VideoCapture(1)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH,RESOLUTION12[0])
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT,RESOLUTION12[1])
cap1.set(cv2.CAP_PROP_FPS,30)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH,RESOLUTION12[0])
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT,RESOLUTION12[1])
cap2.set(cv2.CAP_PROP_FPS,30)
cap3.set(cv2.CAP_PROP_FRAME_WIDTH,RESOLUTION34[0])
cap3.set(cv2.CAP_PROP_FRAME_HEIGHT,RESOLUTION34[1])
cap3.set(cv2.CAP_PROP_FPS,30)
cap4.set(cv2.CAP_PROP_FRAME_WIDTH,RESOLUTION34[0])
cap4.set(cv2.CAP_PROP_FRAME_HEIGHT,RESOLUTION34[1])
cap4.set(cv2.CAP_PROP_FPS,30)

start = datetime.datetime.now()
startTime = start.strftime("%Y-%m-%d %H:%M:%S")

cv2.namedWindow("video")
mode = 1
# Initialization
while True:
    ret1, fram1 = cap1.read()
    ret2, fram2 = cap2.read()        
    ret3, fram3 = cap3.read()
    ret4, fram4 = cap4.read()
    #print(mode)
    
    if mode == 1:
        final = fram1
    elif mode == 2:
        final = fram2
    elif mode == 3:
        final = fram3
    elif mode == 4:
        final = fram4
            
    cv2.imshow('video', final)
    key = cv2.waitKey(1) & 0xFF
    if key == 49:
        mode = 1
        print("1st Cam")
    elif key == 50:
        mode = 2
        print("2nd Cam")
    elif key == 51:
        mode = 3
        print("3rd Cam")
    elif key == 52:
        mode = 4
        print("4th Cam")
            
    
cap1.release()
cv2.destroyAllWindows()