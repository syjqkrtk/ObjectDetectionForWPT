# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 01:58:45 2019

@author: Hyomin Kim
"""
import sqlite3
from time import sleep

from onvif import ONVIFCamera


XMAX = 1
XMIN = -1
YMAX = 1
YMIN = -1

mycam = ONVIFCamera('192.168.0.9', 10080, 'admin', '37373737')
# Create media service object
media = mycam.create_media_service()
# Create ptz service object
ptz = mycam.create_ptz_service()

# Get target profile
media_profile = media.GetProfiles()[0];

# Get PTZ configuration options for getting continuous move range
request = ptz.create_type('GetConfigurationOptions')
request.ConfigurationToken = media_profile.PTZConfiguration.token
status = ptz.GetStatus({'ProfileToken': media_profile.token})
ptz_configuration_options = ptz.GetConfigurationOptions(request)

request = ptz.create_type('ContinuousMove')
request.ProfileToken = media_profile.token

ptz.Stop({'ProfileToken': media_profile.token})
status.Position.PanTilt.x = 0.0
status.Position.PanTilt.y = 0.0
request.Velocity = status.Position

# Get range of pan and tilt
# NOTE: X and Y are velocity vector
XMAX = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].XRange.Max
XMIN = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].XRange.Min
YMAX = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].YRange.Max
YMIN = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].YRange.Min

def perform_move(ptz, request, timeout):
    # Start continuous move
    ptz.ContinuousMove(request)
    # Wait a certain time
    sleep(timeout)
    # Stop continuous move
    ptz.Stop({'ProfileToken': request.ProfileToken})

def move_up(timeout=1):
    print('up')
    request.Velocity.PanTilt.x = 0
    request.Velocity.PanTilt.y = YMAX
    perform_move(ptz, request, timeout)

def move_down(timeout=1):
    print('down')
    request.Velocity.PanTilt.x = 0
    request.Velocity.PanTilt.y = YMIN
    perform_move(ptz, request, timeout)

def move_right(timeout=1):
    print('right')
    request.Velocity.PanTilt.x = XMAX
    request.Velocity.PanTilt.y = 0
    perform_move(ptz, request, timeout)

def move_left(timeout=1):
    print('left')
    request.Velocity.PanTilt.x = XMIN
    request.Velocity.PanTilt.y = 0
    perform_move(ptz, request, timeout)
