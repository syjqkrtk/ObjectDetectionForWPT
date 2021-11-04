import os
import datetime
import numpy as np
import TsaiCalibration as Tsai

# HYPERPARAMETERS DEFINITION
RESOLUTION12 = [1280, 720]
RESOLUTION34 = [1280, 720]

# 시작 전에 앞서, Reference용 사진을 찍고 각각을 WebCam 폴더에 Reference.jpg로 저장한다.
# labelImg 프로그램을 실행해서 6개의 point를 각각 1, 2, 3, 4, 5, 6점으로 잡고 이 프로그램을 실행하면 된다.
N_world = 1
N_Cam = 2
#world = [[0,0,0],[32,0,0],[0,32,0],[0,0,28],[32,32,0],[32,32,28]]
world = [[[-60,0,-42],[-60,110,-42],[0,0,0],[0,60,0],[0,110,0],[60,60,0]]]
refPt = {}
refPt["refPt1"] = [[1260,234],[1060,334],[260,687],[760,123],[785,123],[999,443]]
refPt["refPt2"] = [[760,844],[860,583],[840,453],[660,875],[1234,953],[443,999]]

os.system('cls')
print("")
print("Now Loading...")

start = datetime.datetime.now()
startTime = start.strftime("%Y-%m-%d %H:%M:%S")
filepath = "C:\\Users\\LOSTARK\\Dropbox\\Development\\WebCam\\"

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
        
print(Tsai.ImgTo3D([[1260,234,1260,234]], [[760,844,760,844]], 1, [1,1]))