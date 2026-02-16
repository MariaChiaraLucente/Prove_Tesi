# cd "C:\Users\Odd Alien i9\Desktop\Figure" && python mani_figure.py
import random
import cv2
from cvzone.HandTrackingModule import HandDetector
from ultralytics import YOLO
import math
import numpy as np
import cvzone
import time
import socket

# create UDP socket
#UDP_IP = "192.168.1.39"  # IP address of the client
UDP_IP = "127.0.0.1"  # IP address of the client local
UDP_PORT = 1001  # port number of the client
UDP_PORT2 = 1002  # port number of the client
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


# Webcam
cap = cv2.VideoCapture(0)
# cap.set(3, 1920)
# cap.set(4, 1080)
# cap.set(3, 3840)
# cap.set(4, 2160)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Riconoscimento numero
#model = YOLO ("../Yolo_Weights/best.pt")
model = YOLO ("yolo11n.pt")

classNames = ['Circle', 'Square', 'Triangle']

mask = cv2.imread("mask1080.png")

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Find Function
# x is the raw distance y is the value in cm
# x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
# y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
# coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C

# Game Variables
cx, cy = 850, 625
color = (255, 0, 255)
counter = 0
distanceCM = 0
# score = 0
# timeStart = time.time()
# totalTime = 1000000

#Setup quadrato 1
tl_x = 165
tl_y = 150
br_x = 950
br_y = 150

#Setup quadrato 2
tl_x1 = 0
tl_y1 = 0
br_x1 = 300
br_y1 = 300

#Setup quadrato 3
tl_x2 = 1150
tl_y2 = 830
br_x2 = 1920-340
br_y2 = 1080-72

#Setup quadrato 4
tl_x3 = 0
tl_y3 = 0
br_x3 = 300
br_y3 = 300

#Setup quadrato 5
tl_x4 = 0
tl_y4 = 0
br_x4 = 300
br_y4 = 300

prev_frame_time = 0
new_frame_time = 0

# Loop
while True:
    new_frame_time = time.time()
    success, img_hand = cap.read()
    #img_hand = cv2.flip(img_hand, 1)
    
    success, img_num = cap.read()
    #imgRegion = cv2.bitwise_and(img_num, mask)

    results = model(img_num, stream=True, device=0)

    # if time.time()-timeStart < totalTime:

    hands = detector.findHands(img_hand, draw=False)

    if hands:
        lmList = hands[0]['lmList']
        x, y, k, h = hands[0]['bbox']
        if len(lmList) >= 6:
            x3, y3 = lmList[8][:2]
            x4, y4 = lmList[7][:2]
            
            # distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
            # A, B, C = coff
            # distanceCM = A * distance ** 2 + B * distance + C
            # print(distanceCM, distance)
            
            cv2.line(img_hand, lmList[7][:2], lmList[8][:2], (0, 0, 255), 5)
            cv2.circle(img_hand, lmList[8][:2], 5, (255, 0, 255), cv2.FILLED)
            cv2.circle(img_hand, lmList[7][:2], 5, (255, 0, 255), cv2.FILLED)

            w, _ = detector.findDistance(lmList[8][:2], lmList[5][:2])  # distanza in pixel tra i punti
            W = 2  # distanza in cm tra i punti
            
            # Cercando la distanza della focale della camera
            # d = 60 # distanza approssimativa dalla camera
            # f = (w*d)/W
            # print(f)

            # Cercando la distanza
            #f = 1395  # dimesione della focale
            f = 6277.5
            distanceCM = (W * f) / w  # distanza dalla cam in cm
            print("Distanza mano: ", distanceCM)
        cvzone.putTextRect(img_hand, f'{int(distanceCM)} cm', (x + 100, y + 600))

    # Ciclo per riconoscimento numero
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 200), 3)

            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img_num,(x1,y1,w,h))

            # Confidence
            conf = math.ceil((box.conf[0]*100))/100
            #print(conf)
            #cvzone.putTextRect(img,f'{conf}', (max(0,x1), max(35, y1)))

            # Class Name
            cls = int(box.cls[0])
            # if conf > 0.6 and cls == 9:
            if conf > 0.6:
                cvzone.putTextRect(img_num, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1)
                print(cls)

                message = str(cls).encode()
                sock.sendto(message, (UDP_IP, UDP_PORT))
                #print(message)
                
                # Verifica per la "pressione del bottone"
                if distanceCM > 31:
                    if br_x > x3 > tl_x and br_y > y3 > tl_y:
                        counter = 10
                        color = (0, 255, 0)
                        message = str(counter).encode()
                        sock.sendto(message, (UDP_IP, UDP_PORT2))
                        print(counter)
                    else:
                            counter = 0
                            color = (255, 0, 255)
                #cvzone.putTextRect(img_hand, f'{int(distanceCM)} cm', (x + 5, y - 10))
                cv2.rectangle(img_hand, (tl_x, tl_y), (br_x, br_y), color, 2)

            # send cls via UDP
            #   if cls != prev_cls:
            #   message = str(cls).encode()
            #   sock.sendto(message, (UDP_IP, UDP_PORT))
            #   prev_cls = cls  # update previous cls value
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print("FPS: ", fps)




    cv2.imshow("Image_Hand", img_hand)
    cv2.imshow("Image_Number", img_num)
    #cv2.imshow("ImageRegion", imgRegion)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
