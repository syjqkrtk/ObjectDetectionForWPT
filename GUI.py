import cv2
import os
import datetime
import numpy as np
import time

#filepath = "C:\\Users\\LOSTARK\\Dropbox\\Development\\WebCam\\"
filepath = "/home/umls/Dropbox/Development/WebCam"
webcam1 = cv2.imread('lib\\Cam1.png')
webcam2 = cv2.imread('lib\\Cam2.png')
webcam1 = cv2.resize(webcam1,(75,496))
webcam2 = cv2.resize(webcam2,(75,496))
region1 = cv2.imread('lib\\region1.png')
region2 = cv2.imread('lib\\region2.png')
region1 = cv2.resize(region1,(378,345))
region2 = cv2.resize(region2,(378,345))
background = cv2.imread('lib\\Background.png')
background = cv2.resize(background,(634,347))
smartphone = cv2.imread('lib\\smartphone.png')
smartphone = cv2.resize(smartphone,(67,69))
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

def mainGUI(startTime, count, RSSI, Area, p, fram1, fram2, fram3, fram4):
    img1 = envGUI(Area)
    img2 = textGUI(startTime, count, RSSI, Area, p)
    if Area == 1:
        cam = camGUI(fram1,fram2)
    else:
        cam = camGUI(fram3,fram4)
    
    try:
        img = cv2.hconcat([img1,img2])
        final = cv2.vconcat([img,cam])
        cv2.imshow('GUI',final)
        if cv2.waitKey(1) == 32:
            str1 = filepath+"WebCam"+str(2*Area-1)
            str2 = filepath+"WebCam"+str(2*Area)
            imgnum1 = np.size(os.listdir(str1))-1
            imgnum2 = np.size(os.listdir(str2))-1
            cv2.imwrite(str1+"\\"+str(imgnum1)+".jpg",fram1)
            cv2.imwrite(str2+"\\"+str(imgnum2)+".jpg",fram2)
            print("Saved")
    except cv2.error:
        time.sleep(0.01)
        

def textGUI(startTime, count, RSSI, Area, p):
    img = np.zeros((531, 944, 3),dtype=np.uint8)
    now = datetime.datetime.now()
    nowTime = now.strftime("%Y-%m-%d %H:%M:%S")
    p = list(p)
    p[0] = round(p[0],2)
    p[1] = round(p[1],2)
    p[2] = round(p[2],2)
    text = "Program is working...\n\nThe program is started at : "+startTime+"\nThe time is now : "+nowTime+"\nThis is "+str(count)+"th scan\n\nRSSI value of the beacon 1 : "+str(RSSI[0])+"\nRSSI value of the beacon 2 : "+str(RSSI[1])+"\nRSSI value of the beacon 3 : "+str(RSSI[2])+"\nRSSI value of the beacon 4 : "+str(RSSI[3])+"\n\nTurn on the "+str(Area*2-1)+" & "+str(Area*2)+"th camera\n3D coordinate of IoT device : "+str(p)
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    text_offset_x = 30
    text_offset_y = 100
    text_diff_y = 25
    for i, line in enumerate(text.split('\n')):
        y = text_offset_y + i*text_diff_y
        cv2.putText(img, line, (text_offset_x, y), font, fontScale=font_scale, color=(255,255,255), thickness=2)
    return img
    
def camGUI(fram1, fram2):
    try:
        fram1s = cv2.resize(fram1,(944,531))
        fram2s = cv2.resize(fram2,(944,531))
        final = cv2.hconcat([fram1s,fram2s])
        return final
    except cv2.error:
        time.sleep(0.01)
        return np.ones((531, 1888, 3),dtype=np.uint8)*255
    
def sumImg(l_img, s_img, p, ratio):
    try:
        y_offset = p[0]
        x_offset = p[1]
        for y in range(s_img.shape[0]):
            for x in range(s_img.shape[1]):
                if np.sum(s_img[y,x,:]) > 0:
                    l_img[y_offset+y,x_offset+x,:] = ratio*s_img[y,x,:]+(1-ratio)*l_img[y_offset+y,x_offset+x,:]
        return l_img
    except IndexError:
        return l_img

tempimg0 = np.ones((531, 944, 3),dtype=np.uint8)*255
tempimg0 = sumImg(tempimg0,background,[93,155],1)

tempimg1 = np.ones((531, 944, 3),dtype=np.uint8)*255
tempimg1 = sumImg(tempimg1,background,[93,155],1)
tempimg1 = sumImg(tempimg1,webcam1,[20,315],1)
tempimg1 = sumImg(tempimg1,region1,[95,155],0.5)

tempimg2 = np.ones((531, 944, 3),dtype=np.uint8)*255
tempimg2 = sumImg(tempimg2,background,[93,155],1)
tempimg2 = sumImg(tempimg2,webcam2,[20,550],1)
tempimg2 = sumImg(tempimg2,region2,[95,411],0.5)

def envGUI(Area):
    if Area == 1:
        img = tempimg1
    elif Area == 2:
        img = tempimg2
    else:
        img = tempimg0
    return img
