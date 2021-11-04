#import ftplib
import cv2
import datetime
import time
import RSSICollecter as RSSI
import os
import numpy as np
import threading
import ExternalFunction as EF
import TsaiCalibration as Tsai
import GUI as GUI

# HYPERPARAMETERS DEFINITION
IPADDRESS = "143.248.199.32"
#IPADDRESS = "192.168.0.8"
#IPADDRESS = "192.168.137.221"
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
# FTP Server Loading
N_beacon = 4
#ftp = ftplib.FTP()
#ftp.connect(IPADDRESS,3721)
#ftp.login("LOSTARK","s940723")
#ftp.cwd("Documents")
#ftp.cwd("Logs")
# Initialization
filenum = 0
count = 0
Area = 1
rssi = [0,0,0,0]
p = (0,0,0)
ret1, fram1 = cap1.read()
ret2, fram2 = cap2.read()
ret3, fram3 = cap3.read()
ret4, fram4 = cap4.read()
fram1t, fram2t, fram3t, fram4t = fram1, fram2, fram3, fram4
score1, score2, score3, score4 = 0, 0, 0, 0
box1, box2, box3, box4 = 0, 0, 0, 0
class1, class2, class3, class4 = 0, 0, 0, 0
prev_score1, prev_score2, prev_score3, prev_score4 = 0, 0, 0, 0
prev_box1, prev_box2, prev_box3, prev_box4 = 0, 0, 0, 0
prev_class1, prev_class2, prev_class3, prev_class4 = 0, 0, 0, 0

def main():
    global fram1,fram2,fram3,fram4
    global fram1t,fram2t,fram3t,fram4t
    global prev_box1,prev_box2,prev_box3,prev_box4
    global prev_class1,prev_class2,prev_class3,prev_class4
    global prev_score1,prev_score2,prev_score3,prev_score4
    
    count1, count2, count3, count4 = 0, 0, 0, 0
    while True:
        Area = 1
        if Area == 1:
            ret1, fram1t = cap1.read()
            ret2, fram2t = cap2.read()
            
            #fram1 = WF.Zoom(fram1,2)
            #fram2 = WF.Zoom(fram2,2)
            fram1, prev_box1, prev_class1, prev_score1, count1 = EF.Visualize(fram1t, box1, class1, score1, prev_box1, prev_class1, prev_score1, count1)
            fram2, prev_box2, prev_class2, prev_score2, count2 = EF.Visualize(fram2t, box2, class2, score2, prev_box2, prev_class2, prev_score2, count2)
            
            #GUI.camGUI(fram1, fram2, Area)
                
        if Area == 2:
            ret3, fram3t = cap3.read()
            ret4, fram4t = cap4.read()
            
            #fram3 = EF.Zoom(fram3,2)
            #fram4 = EF.Zoom(fram4,2)
            fram3, prev_box3, prev_class3, prev_score3, count3 = EF.Visualize(fram3t, box3, class3, score3, prev_box3, prev_class3, prev_score3, count3)
            fram4, prev_box4, prev_class4, prev_score4, count4 = EF.Visualize(fram4t, box4, class4, score4, prev_box4, prev_class4, prev_score4, count4)

            #GUI.camGUI(fram3, fram4, Area)
    
def getArea():
    while True:
        global Area, filenum, count, rssi
        try:
            Area, filenum, count, rssi = RSSI.getAREA(N_beacon,ftp,startTime,count,filenum,Area,rssi)
        except OSError:
            time.sleep(0.1)
        except ZeroDivisionError:
            time.sleep(0.1)
        
def callGUI():
    while True:
        GUI.mainGUI(startTime, count, rssi, Area, p, fram1, fram2, fram3, fram4)

WebCamThread = threading.Thread(target=main)
WebCamThread.daemon = True
WebCamThread.start()

#FTPThread = threading.Thread(target=getArea)
#FTPThread.daemon = True
#FTPThread.start()

GUIThread = threading.Thread(target=callGUI)
GUIThread.daemon = True
GUIThread.start()

while True:
    if Area == 1:
        box1,class1,score1 = EF.Detection(fram1t)
        box2,class2,score2 = EF.Detection(fram2t)
        if (np.size(box1)-1)*(np.size(box2)-1) != 0:
            p = Tsai.ImgTo3D(box1, box2, Area, RESOLUTION12)
        elif (np.size(prev_box1)-1)*(np.size(prev_box2)-1) != 0:
            p = Tsai.ImgTo3D(prev_box1, prev_box2, Area, RESOLUTION12)
        else:
            p = (0,0,0)
                
    if Area == 2:
        box3,class3,score3 = EF.Detection(fram3t)
        box4,class4,score4 = EF.Detection(fram4t)
        if (np.size(box3)-1)*(np.size(box4)-1) != 0:
            p = Tsai.ImgTo3D(box3, box4, Area, RESOLUTION34)
        elif (np.size(prev_box3)-1)*(np.size(prev_box4)-1) != 0:
            p = Tsai.ImgTo3D(prev_box3, prev_box4, Area, RESOLUTION34)
        else:
            p = (0,0,0)
cap1.release()
cap2.release()
cap3.release()
cap4.release()
cv2.destroyAllWindows()
#ftp.quit()