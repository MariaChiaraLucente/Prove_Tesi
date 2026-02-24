## prova di un nuovo mapping per unity
## messaggio udp stile:

'''
{
  "orthographicSize": 5.0,
  "aspect": 1.7778,
  "frameWidth": 1280,
  "frameHeight": 720
}

'''

import cv2
import math
import socket

# CONFIG ###################
############################
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

# ROI
X1_ROI, Y1_ROI = 233, 29
X2_ROI, Y2_ROI = 1089, 552

############################
# INIT CAMERA
############################
def init_camera():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap

