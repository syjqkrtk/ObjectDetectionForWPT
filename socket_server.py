# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:30:24 2019

@author: user
"""

import socket
import tkinter as tk
#import telnetlib

HOST = "192.168.0.10"
PORT = "7778"

#telnetObj=telnetlib.Telnet(HOST,PORT)
print("telnet connected")

running = True
UDP_IP_ADDRESS = ''
UDP_PORT = 8001
BUFFER_SIZE = 1024 # byte 1024 byte = 128byte

serverSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverSock.bind((UDP_IP_ADDRESS, UDP_PORT))
print("UDP connected")

#window.resizable(width=TRUE, height = TRUE); # 최대화 가능범위 둘 다 FALSE면 작동 불가
window = tk.Tk()
window.title('Result')
window.geometry("700x400")

state = -1

while running:
    #print("waiting for message..")
    data, addr = serverSock.recvfrom(BUFFER_SIZE)
    New_Data = data.decode()
    #print('Message: ', New_Data) # 콘솔 출력 화면
    #print('Client IP: %s Port number: %s ' % (addr[0], addr[1])) # client IP, Port (port is randum number)
    
    if New_Data == '1':
        message = ("outp off\r\n").encode("utf-8")
        #telnetObj.write(message)
        label1 = tk.Label(window, text = 'Danger', font = ("Arial", 150), fg = "red")
        label1.pack()
        label2 = tk.Label(window, text = 'Turn Off', font = ("Arial", 100), fg = "black")
        label2.pack()
    elif New_Data == '0':
        message = ("outp on\r\n").encode("utf-8")
        #telnetObj.write(message)
        label1 = tk.Label(window, text = 'Safe', font = ("Arial", 150), fg = "blue")
        label1.pack()
        label2 = tk.Label(window, text = 'Turn On', font = ("Arial", 100), fg = "black")
        label2.pack()
    
    window.update()
    
    label1.destroy()
    label2.destroy()

#telnetObj.close()
      
window.mainloop();
