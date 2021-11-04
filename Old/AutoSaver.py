import os
import time
import numpy as np
import cv2

REF1 = cv2.imread("..//WebCam//WebCam1//Reference.jpg",0)
REF2 = cv2.imread("..//WebCam//WebCam2//Reference.jpg",0)
REF3 = cv2.imread("..//WebCam//WebCam3//Reference.jpg",0)
REF4 = cv2.imread("..//WebCam//WebCam4//Reference.jpg",0)

def ClassifyCamera(image):
    res = []
    res.append(np.max(cv2.matchTemplate(REF1, image, cv2.TM_CCOEFF_NORMED)))
    res.append(np.max(cv2.matchTemplate(REF2, image, cv2.TM_CCOEFF_NORMED)))
    res.append(np.max(cv2.matchTemplate(REF3, image, cv2.TM_CCOEFF_NORMED)))
    res.append(np.max(cv2.matchTemplate(REF4, image, cv2.TM_CCOEFF_NORMED)))
    
    return res.index(np.max(res)) + 1

def AutoSave():
    name = os.listdir('..//WebCam')
    time.sleep(0.1)
    if np.size(name) > 4:
        NewImg = cv2.imread("..//WebCam//"+name[0],0)
        WebCamNum = ClassifyCamera(NewImg)
        name2 = os.listdir("..//WebCam//WebCam"+str(WebCamNum))
        PicNum = np.size(name2)-1
        os.rename("..//WebCam//"+name[0],"..//WebCam//WebCam"+str(WebCamNum)+"//"+str(PicNum)+".jpg")
        print("Successfully Classified Image as : Camera "+str(WebCamNum))
        
    return WebCamNum,PicNum