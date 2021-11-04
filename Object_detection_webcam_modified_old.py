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
video1 = cv2.VideoCapture(0,cv2.CAP_DSHOW) # '0' : internal webcam, '1' : external webcam

video1.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
video1.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

#video2 = cv2.VideoCapture(0) # '0' : internal webcam, '1' : external webcam

#video2.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
#video2.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)



if video1.isOpened() == False:
    print("Unable to read camera-1 feed")
    
#if video2.isOpened() == False:
#    print("Unable to read camera-2 feed")

#video1.set(3, frame_width)
#video1.set(4, frame_height)
#video.set(5, 30)


#video2 = cv2.VideoCapture(2)
#
#if video2.isOpened() == False:
#    print("Unable to read camera-2 feed")
#    
#video2.set(3, frame_width)
#video2.set(4, frame_width)    



tx_norm_location = [0.15, 1, 0.85, 1]; # Tx 위치의 정규화된 좌표계, [0]은 왼쪽 x, [1]은 왼쪽 y, [2]는 오른쪽 x, [3]은 오른쪽 y

#iot_norm_location = [0.82, 0.1, 0.12, 0.9]; # IoT 위치의 정규화된 좌표계 [0]은 center x, [1]은 center y

iot_norm_location = [0.5, 0.2]; # IoT 위치의 정규화된 좌표계 [0]은 center x, [1]은 center y


iot_num = int(len(iot_norm_location)/2) # iot 디바이스 개수

iot_location = [0]*2*iot_num

iot_distance = 5; # 추정된 Tx에서 IoT 사이의 거리

tx_location = [np.int32(tx_norm_location[0]*frame_width), np.int32(tx_norm_location[1]*frame_height), 
               np.int32(tx_norm_location[2]*frame_width), np.int32(tx_norm_location[3]*frame_height)]


for i in range(iot_num):
    iot_location[2*i] = np.int32(iot_norm_location[2*i]*frame_width)
    iot_location[2*i+1] = np.int32(iot_norm_location[2*i+1]*frame_height)
    
##iot_location = [np.int32(iot_norm_location[0]*frame_width), np.int32(iot_norm_location[1]*frame_height)]

while(True):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video1.read()
    
    #frame = cv2.flip(frame,1) # 좌우 반전
    
    frame_expanded = np.expand_dims(frame, axis=0)
    
    #ret2, frame2 = video2.read()
    
    #frame2 = cv2.flip(frame2,1) # 좌우 반전
    
    #frame_expanded2 = np.expand_dims(frame2, axis=0)

    # Perform the actual detection by running the model with the image as input
    now = time.time()
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    
    print(time.time()-now)
    
    #(boxes2, scores2, classes2, num2) = sess.run(
    #    [detection_boxes, detection_scores, detection_classes, num_detections],
    #    feed_dict={image_tensor: frame_expanded2})

    # Draw the results of the detection (aka 'visulaize the results')
    temp, pause_true = vis_util.visualize_boxes_and_labels_on_image_array(
        frame_width,
        frame_height,
        frame,
        tx_norm_location,
        iot_norm_location,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        skip_scores=True,
        line_thickness=4,
        min_score_thresh=0.70)
    #temp, pause_true2 = vis_util.visualize_boxes_and_labels_on_image_array(
    #    frame_width,
    #    frame_height,
    #    frame2,
    #    tx_norm_location,
    #    iot_norm_location,
    #    np.squeeze(boxes2),
    #    np.squeeze(classes2).astype(np.int32),
    #    np.squeeze(scores2),
    #    category_index,
    #    use_normalized_coordinates=True,
    #    skip_scores=True,
    #    line_thickness=4,
    #    min_score_thresh=0.70)
    
  
    #if pause_true+pause_true2 == 2:
    if pause_true == 1:
       #winsound.PlaySound("beep.wav", winsound.SND_FILENAME)
       clientSock.sendto(Tx_Message1.encode(), (UDP_IP_ADDRESS, UDP_PORT))
    else:
       clientSock.sendto(Tx_Message2.encode(), (UDP_IP_ADDRESS, UDP_PORT)) 
    
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Human Detector with Webcam', frame)
    #cv2.imshow('Human Detector with Webcam2', frame2)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video1.release()
#video2.release()
cv2.destroyAllWindows()
