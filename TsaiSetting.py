import cv2
import os
import datetime
import numpy as np
import TsaiCalibration as Tsai

# HYPERPARAMETERS DEFINITION
RESOLUTION12 = [1280, 720]
RESOLUTION34 = [1280, 720]

# 시작 전에 앞서, Reference용 사진을 찍고 각각을 WebCam 폴더에 Reference.jpg로 저장한다.
# labelImg 프로그램을 실행해서 6개의 point를 각각 1, 2, 3, 4, 5, 6점으로 잡고 이 프로그램을 실행하면 된다.
N_world = 2
N_Cam = 2
#world = [[0,0,0],[32,0,0],[0,32,0],[0,0,28],[32,32,0],[32,32,28]]
world = [[[60,26,84.5],[60,0,42],[110,0,42],[170,0,0],[110,52,42],[170,52,0]],[[60,26,84.5],[60,0,42],[0,0,42],[-60,0,0],[0,52,42],[-60,52,0]]]
refPt = {}
refPt["refPt1"] = []
refPt["refPt2"] = []
refPt["refPt3"] = []
refPt["refPt4"] = []

os.system('cls')
print("")
print("Now Loading...")

# WebCam Camera Loading
cap1 = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(3,cv2.CAP_DSHOW)
cap3 = cv2.VideoCapture(4,cv2.CAP_DSHOW)
cap4 = cv2.VideoCapture(1,cv2.CAP_DSHOW)
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
filepath = "C:\\Users\\LOSTARK\\Dropbox\\Development\\WebCam\\"

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt1, refPt2, refPt3, refPt4, mode
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		if mode == 1:
			refPt["refPt1"].append([x, y])
		if mode == 2:
			refPt["refPt2"].append([x, y])
		if mode == 3:
			refPt["refPt3"].append([x, y])
		if mode == 4:
			refPt["refPt4"].append([x, y])

cv2.namedWindow("video")
cv2.setMouseCallback("video", click_and_crop)
mode = 0
# Initialization
while True:
    ret1, fram1 = cap1.read()
    ret2, fram2 = cap2.read()        
    ret3, fram3 = cap3.read()
    ret4, fram4 = cap4.read()
    #print(mode)
    
    if mode == 0:
        fram1s = cv2.resize(fram1,(640,360))
        fram2s = cv2.resize(fram2,(640,360))
        final1 = cv2.hconcat([fram1s,fram2s])
        fram3s = cv2.resize(fram3,(640,360))
        fram4s = cv2.resize(fram4,(640,360))
        final2 = cv2.hconcat([fram3s,fram4s])
        final = cv2.vconcat([final1,final2])
    elif mode == 1:
        final = fram1
    elif mode == 2:
        final = fram2
    elif mode == 3:
        final = fram3
    elif mode == 4:
        final = fram4
        
    if mode == 0:
        for i in range(4):
            tempPt = refPt["refPt"+str(i+1)]
            for j in range(int(np.floor(np.size(tempPt)/2))):
                refx = tempPt[j][0]
                refy = tempPt[j][1]
                cv2.rectangle(final,(int(np.floor(refx/2)+640*(i%2)-2),int(np.floor(refy/2)+360*(np.floor(i/2))-2)),(int(np.floor(refx/2)+640*(i%2)+2),int(np.floor(refy/2)+360*(np.floor(i/2))+2)),(0,255,0),1)
    else:
        tempPt = refPt["refPt"+str(mode)]
        for j in range(int(np.floor(np.size(tempPt)/2))):
            refx = tempPt[j][0]
            refy = tempPt[j][1]
            cv2.rectangle(final,(refx-3,refy-3),(refx+3,refy+3),(0,255,0),2)
            
    cv2.imshow('video', final)
    key = cv2.waitKey(1) & 0xFF
    #print(key)
    if key == 32:
        if mode == 0 or mode == 1:
            imgnum1 = np.size(os.listdir("..//WebCam//WebCam1"))-1
            cv2.imwrite(filepath+"WebCam1//Reference.jpg",fram1)
        if mode == 0 or mode == 2:
            imgnum2 = np.size(os.listdir("..//WebCam//WebCam2"))-1
            cv2.imwrite(filepath+"WebCam2//Reference.jpg",fram2)
        if mode == 0 or mode == 3:
            imgnum3 = np.size(os.listdir("..//WebCam//WebCam3"))-1
            cv2.imwrite(filepath+"WebCam3//Reference.jpg",fram3)
        if mode == 0 or mode == 4:
            imgnum4 = np.size(os.listdir("..//WebCam//WebCam4"))-1
            cv2.imwrite(filepath+"WebCam4//Reference.jpg",fram4)
        print("Saved")  
    elif key == 49:
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
    elif key == 48:
        mode = 0
        print("All Cam")
    elif key == 27:
        if (np.size(refPt["refPt1"])==12) and (np.size(refPt["refPt2"])==12) and (np.size(refPt["refPt3"])==12) and (np.size(refPt["refPt4"])==12):
            print("Successfully set K matrix")
            break
        else:
            print("Unsuccessfully set K matrix")
            print("Reset Reference points")
            refPt["refPt1"] = []
            refPt["refPt2"] = []
            refPt["refPt3"] = []
            refPt["refPt4"] = []
            
    
cap1.release()
cap2.release()
cap3.release()
cap4.release()
cv2.destroyAllWindows()
    
K = []

for i in range(N_world):
    for j in range(N_Cam):
        count = 0
        strtemp = '..//WebCam//WebCam'+str(2*i+j+1)+'//Reference.txt'
        file = open(strtemp,'w')
        tempPt = refPt["refPt"+str(2*i+j+1)]
        file.write(str(tempPt))
        file.close()
        K = Tsai.Find_K_Matrix(tempPt, world[i])
        np.save(os.path.join('..//Data','Reference'+str(2*i+j+1)),K)