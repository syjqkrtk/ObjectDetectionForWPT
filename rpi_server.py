# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:30:24 2019

@author: user
"""
import socket
import tkinter as tk
import ContinuousMove as CM

running = True
UDP_IP_ADDRESS = ''
UDP_PORT = 8001
BUFFER_SIZE = 1024 # byte 1024 byte = 128byte

serverSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverSock.bind((UDP_IP_ADDRESS, UDP_PORT))
print('connection started')

#window.resizable(width=TRUE, height = TRUE); # 최대화 가능범위 둘 다 FALSE면 작동 불가
window = tk.Tk()
window.title('Result')
window.geometry("700x400")

while running:    
    #print("waiting for message..")
    data, addr = serverSock.recvfrom(BUFFER_SIZE)
    New_Data = data.decode()
    #print('Message: ', New_Data) # 콘솔 출력 화면
    #print('Client IP: %s Port number: %s ' % (addr[0], addr[1])) # client IP, Port (port is randum number)
    print(New_Data)
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
    
window.mainloop();
    
    
    
  
