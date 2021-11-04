import numpy as np
import ftplib

def getAREA(N_beacon, ftp, startTime, count, filenum, prevArea, prevRSSI):
	data = []
	ftp.dir(data.append)
	filepath = "C:\\Users\\LOSTARK\\Dropbox\\Development\\Mobile\\"
	RSSI = [0,0,0,0]
	lognum = [0,0,0,0]
	tempnum = np.size(data)

	if tempnum >  filenum + N_beacon:
		count = count + 1
		data.sort()
		for line in data[-(5*N_beacon):]:
			if "2019-" in line:
				filename = line.split()[8]
				#print(filename)
				file = open(filepath + filename,'wb')
				ftp.retrbinary("RETR " + filename, file.write)
				file.close()
				
				file = open(filepath + filename,'r')
				temp = int(file.readline().replace("\n",""))
				ID = file.readline().replace("\n","")
				ID = int(ID.split(",")[1].split("=")[1]) - 10000
				RSSI[ID-1] += temp
				lognum[ID-1] += 1
				file.close()
		for i in range(4):
			if lognum[i] > 0:
				RSSI[i] = RSSI[i]/lognum[i]
			else:
				RSSI[i] = -100
		RSSI[1] = RSSI[1]
		Area = np.where(RSSI == np.max(RSSI[0:2]))[0][0]%2+1

		return Area, tempnum, count, RSSI
	
	return prevArea, filenum, count, prevRSSI