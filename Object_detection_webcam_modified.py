# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 21:52:14 2018

@author: user
"""

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import socket
import time
import threading
import telnetlib

HOST = "192.168.0.10"
PORT = "7778"

telnetObj=telnetlib.Telnet(HOST,PORT)
print("telnet connected")

UDP_IP_ADDRESS =  '143.248.171.95' #'127.0.0.1' # 사람 감지 결과를 보낼 제어 PC IP 주소
UDP_PORT = 6668
server_address = (UDP_IP_ADDRESS, UDP_PORT)

clientSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #AF_INET: 인터넷소캣, SOCK_DGRAM: UDP
print("Socket works")

Message = '4554'

Tx_Message1 = Message.encode()
Tx_Message2 = Message.encode()

Tx_Message1 = '1'
Tx_Message2 = '0'

tf.reset_default_graph()
config = tf.ConfigProto() #cublas error handling
config.gpu_options.allow_growth=True #cublas error handling
session=tf.Session(config=config) #cublas error handling

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils_modified as vis_util

# Name of the directory containing the object detection module we're using
#MODEL_NAME = 'inference_grap   h_resnet101'
#MODEL_NAME = 'inference_graph_resnet50'
MODEL_NAME = 'inference_graph_inception'
#MODEL_NAME = 'inference_graph_mobile'

# Grab path to current working directory
CWD_PATH = 'C:\\Users\\LOSTARK\\Downloads\\out'

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 1
THRESHOLD = 0.8

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')


frame_width = 1280;
frame_height = 720; 
#RESOLUTION12 = [1280, 720]

# Initialize webcam feed
cap1 = cv2.VideoCapture(0,cv2.CAP_DSHOW) # '0' : internal webcam, '1' : external webcam

cap1.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

if cap1.isOpened() == False:
    print("Unable to read camera-1 feed")


xx = 640
yy = 180
key = 0

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global xx, yy
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        xx, yy = x, y


ret1, fram1 = cap1.read()
fram_expanded = np.expand_dims(fram1, axis=0)
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: fram_expanded})
box1 = boxes
class1 = classes
score1 = scores

##iot_location = [np.int32(iot_norm_location[0]*frame_width), np.int32(iot_norm_location[1]*frame_height)]

def main():
    global fram1, key

    mode = 0
    cv2.namedWindow('video')
    cv2.setMouseCallback('video', click_and_crop)
    
    while True:
        ret1, fram1 = cap1.read()
        
        try:
            temp, human_in_danger = vis_util.visualize_boxes_and_labels_on_image_array(
                frame_width,
                frame_height,
                fram1,
                box1,
                class1,
                score1,
                category_index,
                use_normalized_coordinates=True,
                skip_scores=True,
                line_thickness=4,
                min_score_thresh=THRESHOLD, 
                xx=xx, 
                yy=yy)
      
            #if pause_true+pause_true2 == 2:
            if human_in_danger == 1:
               #winsound.PlaySound("beep.wav", winsound.SND_FILENAME)
               #clientSock.sendto(Tx_Message1.encode(), (UDP_IP_ADDRESS, UDP_PORT))
               message = ("outp off\r\n").encode("utf-8")
               if mode:
                   print("off")
                   telnetObj.write(message)
            else:
               #clientSock.sendto(Tx_Message2.encode(), (UDP_IP_ADDRESS, UDP_PORT))
               message = ("outp on\r\n").encode("utf-8")
               if mode:
                   print("on")
                   telnetObj.write(message)
               
            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow('video', fram1)
            #cv2.imshow('Human Detector with Webcam2', frame2)
        
            # Press 'q' to quit
            key = cv2.waitKey(1)
            if key == ord('m'):
                mode = 1 - mode
                print(mode)
            if key == ord('q'):
                break
        except ValueError:
            #print(np.shape(fram1), np.shape(box1))
            time.sleep(0.1)

WebCamThread = threading.Thread(target=main)
WebCamThread.daemon = True
WebCamThread.start()

while True:
    fram_expanded = np.expand_dims(fram1, axis=0)
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: fram_expanded})
    box1 = np.squeeze(boxes)
    class1 = np.squeeze(classes).astype(np.int32)
    score1 = np.squeeze(scores)
    if key == ord('q'):
        break
            
cap1.release()
cv2.destroyAllWindows()
