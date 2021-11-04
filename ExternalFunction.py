import cv2
import os
import numpy as np
from utils import label_map_util
from utils import visualization_utils as vis_util
import tensorflow as tf

# HYPERPARAMETERS DEFINITION
THRESHOLD = 0.2
EXTRA = 30

# TensorFlow Model Loading
tf.reset_default_graph()
config = tf.ConfigProto() #cublas error handling
config.gpu_options.allow_growth=True #cublas error handling
session=tf.Session(config=config) #cublas error handling

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
CWD_PATH = 'C:\\Users\\LOSTARK\\Downloads\\out'
#CWD_PATH = '/home/umls/Downloads/out'
# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph_resnet101'
#MODEL_NAME = 'inference_graph_inception'
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'Training_new','label_map.pbtxt')
# Number of classes the object detector can identify
NUM_CLASSES = 2
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

def Zoom(fram, mag):
    try:
        height, width, channels = fram.shape
        centerX,centerY=int(height/2),int(width/2)
        radiusX,radiusY= int(height/(2*mag)),int(width/(2*mag))
        minX,maxX=centerX-radiusX,centerX+radiusX
        minY,maxY=centerY-radiusY,centerY+radiusY
        cropped = fram[minX:maxX, minY:maxY]
        fram = cv2.resize(cropped,(width,height))
        return fram
    except AttributeError:
        return fram

def Visualize(fram, box, Class, score, prev_box, prev_class, prev_score, count):
    if score > THRESHOLD:
        vis_util.visualize_boxes_and_labels_on_image_array(
            fram,
            box,
            Class,
            score,
            category_index,
            use_normalized_coordinates=True,
            skip_scores=True,
            line_thickness=4,
            min_score_thresh=THRESHOLD)
        prev_box = box
        prev_class = Class
        prev_score = score
        count = 0
    else:
        count = count + 1
        if count < EXTRA:
            if prev_score > THRESHOLD:
                vis_util.visualize_boxes_and_labels_on_image_array(
                    fram,
                    prev_box,
                    prev_class,
                    prev_score,
                    category_index,
                    use_normalized_coordinates=True,
                    skip_scores=True,
                    line_thickness=4,
                    min_score_thresh=THRESHOLD)

    return fram, prev_box, prev_class, prev_score, count

def Detection(fram):
    try:
        fram_expanded = np.expand_dims(fram, axis=0)
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: fram_expanded})
        if np.max(scores) > THRESHOLD:
            temp = np.where(scores == np.max(scores))
            box = boxes[temp]
            Class = classes[temp].astype(np.uint8)
            score = scores[temp]        
            return box, Class, score      
        else:
            return 0,0,0
    except TypeError:
        return 0,0,0
    
