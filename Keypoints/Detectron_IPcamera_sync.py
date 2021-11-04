from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import os
import sys
import pprint
import subprocess
from collections import defaultdict
from six.moves import xrange

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable

import _init_paths
import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test import im_detect_all
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as datasets
import utils.misc as misc_utils
import utils.net as net_utils
import utils.vis as vis_utils
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer

import time
import threading
import socket_func
import vlc
import time
import numpy as np
import ContinuousMove1 as CM1
import ContinuousMove2 as CM2

player = {'1' : 0, '2' : 0}
vlc_instance1 = vlc.Instance('--no-snapshot-preview', '--no-osd')
player['1']=vlc_instance1.media_player_new('rtsp://admin:37373737@192.168.0.6:10554/udp/av0_0')
player['1'].play()
vlc_instance2 = vlc.Instance('--no-snapshot-preview', '--no-osd')
player['2']=vlc_instance2.media_player_new('rtsp://admin:37373737@192.168.0.7:10554/udp/av0_0')
player['2'].play()


# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

mode = 2

frame_width = 640;
frame_height = 360;

frame = {'1' : 0, '2' : 0}

frame['1'] = cv2.imread('snapshot1.png')
#frame['1'] = cv2.flip(frame['1'],1)

frame['2'] = cv2.imread('snapshot2.png')
#frame['2'] = cv2.flip(frame['2'],1)

frame_num = {'1' : 0, '2' : 0}
detect_num = {'1' : 0, '2' : 0}

cls_boxes = {'1' : [], '2' : []}
cls_segms = {'1' : [], '2' : []}
cls_keyps = {'1' : [], '2' : []}

refPt = {}
refPt["refPt1"] = [320,90]
refPt["refPt2"] = [320,90]

def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')
    parser.add_argument(
        '--dataset', required=True,
        help='training dataset')

    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file',
        default=[], nargs='+')

    parser.add_argument(
        '--no_cuda', dest='cuda', help='whether use CUDA', action='store_false')

    parser.add_argument('--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--image_dir',
        help='directory to load images for demo')
    parser.add_argument(
        '--images', nargs='+',
        help='images to infer. Must not use with --image_dir')
    parser.add_argument(
        '--output_dir',
        help='directory to save demo results',
        default="infer_outputs")
    parser.add_argument(
        '--merge_pdfs', type=distutils.util.strtobool, default=True)

    args = parser.parse_args()

    return args

if not torch.cuda.is_available():
    sys.exit("Need a CUDA device to run the code.")

args = parse_args()
print('Called with args:')
print(args)

assert args.image_dir or args.images
assert bool(args.image_dir) ^ bool(args.images)

if args.dataset.startswith("coco"):
    dataset = datasets.get_coco_dataset()
    cfg.MODEL.NUM_CLASSES = len(dataset.classes)
elif args.dataset.startswith("keypoints_coco"):
    dataset = datasets.get_coco_dataset()
    cfg.MODEL.NUM_CLASSES = 2
else:
    raise ValueError('Unexpected dataset name: {}'.format(args.dataset))

print('load cfg from file: {}'.format(args.cfg_file))
cfg_from_file(args.cfg_file)

if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
    'Exactly one of --load_ckpt and --load_detectron should be specified.'
cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
assert_and_infer_cfg()

maskRCNN = Generalized_RCNN()

if args.cuda:
    maskRCNN.cuda()

if args.load_ckpt:
    load_name = args.load_ckpt
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
    net_utils.load_ckpt(maskRCNN, checkpoint['model'])

if args.load_detectron:
    print("loading detectron weights %s" % args.load_detectron)
    load_detectron_weight(maskRCNN, args.load_detectron)

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, mode
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        if mode == 1:
            refPt["refPt1"] = [np.floor(x/2), np.floor(y/2)]
        if mode == 2:
            if x < 640:
                refPt["refPt1"] = [np.floor(x/2)+160, np.floor(y/2)]
            else:
                refPt["refPt2"] = [np.floor(x/2)-160, np.floor(y/2)]
            
def WebCam():
    global refPt, mode
    global frame
    global frame_num, detect_num
    global cls_boxes, cls_segms, cls_keyps

    cv2.namedWindow('video')
    cv2.setMouseCallback('video', click_and_crop)

    fig1 = 0
    key = 0
    prev_frame1 = []
    prev_index1 = []
    pprev_frame1 = []
    pprev_index1 = []
    prev_detect1 = 0
    count1 = 0
    ccount1 = 0
    result1 = 0

    prev_frame2 = []
    prev_index2 = []
    pprev_frame2 = []
    pprev_index2 = []
    prev_detect2 = 0
    count2 = 0
    ccount2 = 0
    result2 = 0

    while(1):
        time.sleep(0.1)
        fname1 = 'snapshot1.png'
        player['1'].video_take_snapshot(0, fname1, frame_width, frame_height)
        frame['1'] = cv2.imread(fname1)
        #frame['1'] = cv2.flip(frame['1'],1)
        prev_frame1.append(frame['1'])
        pprev_frame1.append(frame['1'])
        prev_index1.append(frame_num['1'])
        pprev_index1.append(frame_num['1'])
        frame_num['1'] = frame_num['1'] + 1

        fname2 = 'snapshot2.png'
        player['2'].video_take_snapshot(0, fname2, frame_width, frame_height)
        frame['2'] = cv2.imread(fname2)
        #frame['2'] = cv2.flip(frame['2'],1)
        prev_frame2.append(frame['2'])
        pprev_frame2.append(frame['2'])
        prev_index2.append(frame_num['2'])
        pprev_index2.append(frame_num['2'])
        frame_num['2'] = frame_num['2'] + 1

        #print(frame_num['1'],frame_num['2'])

        #temp_frame = pprev_frame[count]
        #print("------")        
        #print(pprev_index[0])
        #print(pprev_index[-1])
        #print(pprev_index)
        #print(detect_num)
        #print(pprev_index[count])
        #print(frame_num)
        count1 = count1 + 1
        count2 = count2 + 1
        #print(count)
        im_name1 = 'video1'
        im_name2 = 'video2'
        #print(im_name)

        if prev_detect1 < detect_num['1']:
            prev_detect1 = detect_num['1']
            pprev_frame1 = prev_frame1[:]
            prev_frame1 = [prev_frame1[-1]]
            pprev_index1 = prev_index1[:]
            prev_index1 = [frame_num['1']-1]
            temp_index1 = pprev_index1.index(detect_num['1'])
            temp_frame1 = pprev_frame1[temp_index1]
            count1 = temp_index1
            tempPt = refPt["refPt1"]
            result1, fig1 = vis_utils.vis_one_image(
                temp_frame1[:,:,::-1],  # BGR -> RGB for visualization
                im_name1,
                args.output_dir,
                cls_boxes['1'],
                cls_segms['1'],
                cls_keyps['1'],
                dataset=dataset,
                box_alpha=0.3,
                show_class=True,
                thresh=0.7,
                kp_thresh=2,
                xx = tempPt[0],
                yy = tempPt[1]
            )
            if mode == 1:
                socket_func.sendmessage(result1)
                fig = cv2.resize(fig1, None, fx = 2, fy =2 , interpolation = cv2.INTER_AREA)
                cv2.imshow('video',fig)
                key = cv2.waitKey(1) & 0xFF
            

        if prev_detect2 < detect_num['2'] and mode == 2:
            prev_detect2 = detect_num['2']
            pprev_frame2 = prev_frame2[:]
            prev_frame2 = [prev_frame2[-1]]
            pprev_index2 = prev_index2[:]
            prev_index2 = [frame_num['2']-1]
            temp_index2 = pprev_index2.index(detect_num['2'])
            temp_frame2 = pprev_frame2[temp_index2]
            count2 = temp_index2
            tempPt = refPt["refPt2"]
            result2, fig2 = vis_utils.vis_one_image(
                temp_frame2[:,:,::-1],  # BGR -> RGB for visualization
                im_name2,
                args.output_dir,
                cls_boxes['2'],
                cls_segms['2'],
                cls_keyps['2'],
                dataset=dataset,
                box_alpha=0.3,
                show_class=True,
                thresh=0.7,
                kp_thresh=2,
                xx = tempPt[0],
                yy = tempPt[1]
            )
            socket_func.sendmessage(result1*result2)
            if np.size(fig1) > 1:
                #print(np.shape(fig1),np.shape(fig2))
                if np.shape(fig1)[1] > 320:
                    fig1 = fig1[:,160:480,:]
                if np.shape(fig2)[1] > 320:
                    fig2 = fig2[:,160:480,:]
                fig = np.hstack((fig1,fig2))
                fig = cv2.resize(fig, None, fx = 2, fy =2 , interpolation = cv2.INTER_AREA)
                cv2.imshow('video',fig)
                key = cv2.waitKey(1) & 0xFF
        
        if key == 49:
            mode = 1
            print("mode 1")
        elif key == 50:
            mode = 2
            print("mode 2")
        elif key == 27:
            mode = 0
            print("exit")
            break
        elif key == 81:
            #print("left")
            if mode == 1:
                CM1.move_left(0.01)
            elif mode == 2:
                CM2.move_left(0.01)
        elif key == 82:
            #print("up")
            if mode == 1:
                CM1.move_up(0.01)
            elif mode == 2:
                CM2.move_up(0.01)
        elif key == 83:
            #print("right")
            if mode == 1:
                CM1.move_right(0.01)
            elif mode == 2:
                CM2.move_right(0.01)
        elif key == 84:
            #print("down")
            if mode == 1:
                CM1.move_down(0.01)
            elif mode == 2:
                CM2.move_down(0.01)
        elif key != 255:
            print(key)
        
        #vis_utils.vis_one_image(
        #    temp_frame[:,:,::-1],  # BGR -> RGB for visualization
        #    "temp",
        #    args.output_dir,
        #    cls_boxes,
        #    cls_segms,
        #    cls_keyps,
        #    dataset=dataset,
        #    box_alpha=0.3,
        #    show_class=True,
        #    thresh=0.7,
        #    kp_thresh=2
        #)

def main():
    """main function"""
    global mode
    global maskRCNN
    global frame_num, detect_num
    global cls_boxes, cls_segms, cls_keyps
    WebCamThread = threading.Thread(target=WebCam)
    WebCamThread.daemon = True
    WebCamThread.start()

    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True, device_ids=[0])  # only support single GPU

    maskRCNN.eval()

    while(1):
        #print(np.shape(frame))
        #assert im is not None
        now = time.time()
        timers = defaultdict(Timer)
        temp_num = frame_num['1']
        cls_boxes['1'], cls_segms['1'], cls_keyps['1'] = im_detect_all(maskRCNN, frame['1'], timers=timers)
        detect_num['1'] = temp_num
        #print(now-time.time())

        if mode == 2:
            temp_num = frame_num['2']
            cls_boxes['2'], cls_segms['2'], cls_keyps['2'] = im_detect_all(maskRCNN, frame['2'], timers=timers)
            detect_num['2'] = temp_num
        
        if mode == 0:
            break
        #cv2.imshow('video',im)


if __name__ == '__main__':
    main()
