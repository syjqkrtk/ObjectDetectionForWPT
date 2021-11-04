import time
import telnetlib

HOST = "192.168.0.10"
PORT = "7778"

telnetObj=telnetlib.Telnet(HOST,PORT)
print("success")
message = ("outp off\r\n").encode("utf-8")
telnetObj.write(message)
telnetObj.close()
