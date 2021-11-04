# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:30:24 2019

@author: user
"""
import socket
import cv2
#import vlc
import threading
import time
import numpy as np
#import ContinuousMove as CM
#import telnetlib

#HOST = "192.168.0.10"
#PORT = "7778"

#telnetObj=telnetlib.Telnet(HOST,PORT)
#print("telnet connected")


running = True
UDP_IP_ADDRESS = ''
UDP_PORT = 8002
BUFFER_SIZE = 1024 # byte 1024 byte = 128byte

frame_width = 640;
frame_height = 480;
frame = cv2.imread('snapshot.png')
cameraimg = cv2.imread('camera.png',-1)
cameraimg = cv2.cvtColor(255-cameraimg[:,:,3],cv2.COLOR_GRAY2RGB)
cameraimg = cv2.resize(cameraimg, None, fx=0.25,fy=0.25,interpolation=cv2.INTER_AREA)
humanimg = cv2.imread('human.png',-1)
humanimg = humanimg[10:190,65:135]
humanimg = cv2.cvtColor(humanimg[:,:,3],cv2.COLOR_GRAY2RGB)

serverSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverSock.bind((UDP_IP_ADDRESS, UDP_PORT))
print('connection started')

x1, y1, x2, y2 = [], [], [], []
num = 0
xy = [640,160]
lasttime = time.time()
renew = 0
addr = 0

def point_in_shape(xs,ys,human_in_danger,xx,yy):
    #print(vector)
    #print(human_in_danger)
    eps_center = [xx, yy]
    eps_width = 100
    eps_height = 100
    rect = [eps_center[0]-eps_width,eps_center[1]-eps_height,eps_center[0]+eps_width,eps_center[1]+eps_height]
    xstart = [xs[0],xs[1],xs[1],xs[0]]
    ystart = [ys[0],ys[0],ys[1],ys[1]]
    xend = [xs[1],xs[1],xs[0],xs[0]]
    yend = [ys[0],ys[1],ys[1],ys[0]]
    for i in range(4):
        x = [xstart[i],xend[i]]
        y = [ystart[i],yend[i]]
        left = lineline(x[0],y[0],x[1],y[1],rect[0],rect[1],rect[0],rect[3])
        right = lineline(x[0],y[0],x[1],y[1],rect[2],rect[1],rect[2],rect[3])
        top = lineline(x[0],y[0],x[1],y[1],rect[0],rect[1],rect[2],rect[1])
        bottom = lineline(x[0],y[0],x[1],y[1],rect[0],rect[3],rect[2],rect[3])
        if left or right or top or bottom:
            return 1, 1
        
        if (rect[0] <= x[0] and rect[2] >= x[0]) and (rect[1] <= y[0] and rect[3] >= y[0]):
            return 1, 1

    rectxstart = [rect[0],rect[2],rect[2],rect[0]]
    rectystart = [rect[1],rect[1],rect[3],rect[3]]
    rectxend = [rect[2],rect[2],rect[0],rect[0]]
    rectyend = [rect[1],rect[3],rect[3],rect[1]]
    for i in range(4):
        x = [rectxstart[i],rectxend[i]]
        y = [rectystart[i],rectyend[i]]
        
        if (xs[0] <= x[0] and xs[1] >= x[0]) and (ys[0] <= y[0] and ys[1] >= y[0]):
            return 1, 1
        
    if human_in_danger:
        return 1, 0

    return 0, 0

def lineline(x1,y1,x2,y2,x3,y3,x4,y4):
    if (y4-y3)*(x2-x1) == (x4-x3)*(y2-y1):
        return 0
    uA = ((x4-x3)*(y1-y3)-(y4-y3)*(x1-x3)) / ((y4-y3)*(x2-x1)-(x4-x3)*(y2-y1))
    uB = ((x2-x1)*(y1-y3)-(y2-y1)*(x1-x3)) / ((y4-y3)*(x2-x1)-(x4-x3)*(y2-y1))
    if (uA >=0 and uA <= 1 and uB >= 0 and uB <= 1):
        return 1
    else:
        return 0


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global xy
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
        if (x < 64) and (y < 64):
            Tx_Message = 'start,@I,end'
            serverSock.sendto(Tx_Message.encode(), addr)
        else:
            xy = [x, y]
      
mode2 = 0

def Display():
    global renew, mode2
    global x1, y1, x2, y2, num

    frame = cv2.imread('snapshot.png')
    cv2.namedWindow('video')
    cv2.setMouseCallback('video', click_and_crop)

    frame = cv2.imread('snapshot.png')
    frame2 = frame.copy()
    frame2 = cv2.resize(frame2, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    while(1):
        if renew:
            frame = cv2.imread('snapshot.png')
            renew = 0
        try:
            frame2 = frame.copy()
            frame2 = cv2.resize(frame2, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        except AttributeError:
            time.sleep(0.01)

        human_in_danger = 0


        if num > 0:
            x1t,y1t,x2t,y2t = x1,y1,x2,y2
            for i in range(num):
                xx1,yy1,xx2,yy2 = x1t[i],y1t[i],x2t[i],y2t[i]

                roi = frame2[yy1:yy2,xx1:xx2]
                humanimg2 = cv2.resize(humanimg, (xx2-xx1,yy2-yy1), interpolation=cv2.INTER_CUBIC)
                humangray = cv2.cvtColor(humanimg2,cv2.COLOR_RGB2GRAY)
                ret, mask = cv2.threshold(humangray, 10, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)
                #print(np.shape(mask))
                #print(np.shape(humanimg2))
                #print(np.shape(roi))
                humanimg2 = cv2.bitwise_and(humanimg2, humanimg2, mask = mask)
                roi = cv2.bitwise_and(roi,roi, mask=mask_inv)
                dst = cv2.add(humanimg2, roi)

                human_in_danger, human_in_danger_now = point_in_shape([xx1,xx2],[yy1,yy2],human_in_danger,xy[0],xy[1])

                if human_in_danger_now:
                    frame2 = cv2.rectangle(frame2, (xx1,yy1), (xx2,yy2), (0,0,255), 3).copy()
                    frame2[yy1:yy2,xx1:xx2,2] = 0.8 * dst[:,:,2] + 0.2 * frame2[yy1:yy2,xx1:xx2,2]
                else:
                    frame2 = cv2.rectangle(frame2, (xx1,yy1), (xx2,yy2), (0,255,0), 3).copy()
                    frame2[yy1:yy2,xx1:xx2,1] = 0.8 * dst[:,:,1] + 0.2 * frame2[yy1:yy2,xx1:xx2,1]
        frame2 = cv2.circle(frame2, (xy[0],xy[1]), 10, (0,0,255),6,8,0).copy()
        frame2 = cv2.rectangle(frame2, (xy[0]-100,xy[1]-100), (xy[0]+100,xy[1]+100), (0,255,255), 3).copy()

        if human_in_danger == 1:
           #winsound.PlaySound("beep.wav", winsound.SND_FILENAME)
           #clientSock.sendto(Tx_Message1.encode(), (UDP_IP_ADDRESS, UDP_PORT))
           message = ("outp off\r\n").encode("utf-8")
           if mode2:
               print("off")
               #telnetObj.write(message)
        else:
           #clientSock.sendto(Tx_Message2.encode(), (UDP_IP_ADDRESS, UDP_PORT))
           message = ("outp on\r\n").encode("utf-8")
           if mode2:
               print("on")
               #telnetObj.write(message)

        if time.time()-lasttime > 0.2:
            x1, y1, x2, y2 = [],[],[],[]
            num = 0
        #print(np.shape(frame2))
        frame2[:64,:64] = cameraimg
        cv2.imshow('video',frame2)
        key = cv2.waitKey(1)&0xFF
        
        if key == ord('m'):
            mode2 = 1 - mode2
            print(mode2)
        if key == ord('q'):
            break

WebCamThread = threading.Thread(target=Display)
WebCamThread.daemon = True
WebCamThread.start()

mode = 0
name = 'temp'
#count = 0

while running:
    #print("waiting for message..")
    data, addr = serverSock.recvfrom(BUFFER_SIZE)
    try:
        New_Data = data.decode()
        print(New_Data)
    except UnicodeDecodeError:
        New_Data = ""
        #print(data)

    #print('Message: ', New_Data) # ì½ì ì¶ë ¥ ?ë©´
    #print('Client IP: %s Port number: %s ' % (addr[0], addr[1])) # client IP, Port (port is randum number)
    #print(New_Data)
    New_Data = New_Data.split(',')
    if (New_Data[0] == 'start') and (New_Data[-1] == 'end'):
        if New_Data[1] == '@A':
            if New_Data[2] == 'move':
                if New_Data[3] == 'left':
                    CM.move_left(float(New_Data[4]))
                elif New_Data[3] == 'right':
                    CM.move_right(float(New_Data[4]))
                elif New_Data[3] == 'up':
                    CM.move_up(float(New_Data[4]))
                elif New_Data[3] == 'down':
                    CM.move_down(float(New_Data[4]))
        elif New_Data[1] == '@D':
            num = int(New_Data[2])
            x1 = []
            y1 = []
            x2 = []
            y2 = []
            for i in range(num):
                name = New_Data[i*5+3]
                x1.append(int(2*float(New_Data[i*5+4])))
                y1.append(int(2*float(New_Data[i*5+5])))
                x2.append(int(2*float(New_Data[i*5+6])))
                y2.append(int(2*float(New_Data[i*5+7])))
                lasttime = time.time()
                #print(name,4*x1,4*y1,4*x2,4*y2)
        elif New_Data[1] == '@F':
            if New_Data[2] == 'fopen':
                mode = 1
                #count = 0
                name = New_Data[3]
                f = open(name,'wb')
                print("Received File : "+name)
            elif New_Data[2] == 'fclose':
                mode = 0
                renew = 1
                f.close()
                print("Download completed")
    elif mode == 1:
        f.write(data)
        #count = count + 1
        #print(count)

    
window.mainloop();
    
    
    
  
