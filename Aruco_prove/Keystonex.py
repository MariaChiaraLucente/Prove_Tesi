################################
# DA PROVARE
################################


# Libs
import cv2
import numpy as np
import cv2.aruco as aruco
####

# Vars
points = []
currentPoint = []
lastPoint = []
####

# Image loading
# fileName = 'test.jpg'
# img = cv2.imread(fileName)
# imgMaster = img.copy()
# imgTemp = img.copy()
####

# al posto di caricare l immagine, detecto la camera e quello che vede
cap = cv2.VideoCapture(1)

#io so che l'immagine della camera è 1280x720
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

#so che la risoluzione del proiettore è
W_PROJ, H_PROJ =  1920, 1200

# se io proietto uno sfondo nero nel proiettore, lo vodero deformato
#quindi mi serve rilevare i 4 estremi deformati, e far si ruotandoli diventino dritti


# Mouse event function: penso sia la funizone in cui seleziono i 4 estremi del proiettore

def mouseCallback(event, x, y, flags, param):

    global lastPoint, currentPoint, imgTemp

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(points) < 4:

            points.append([x, y])
            lastPoint = ([x, y])

            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            imgTemp = img.copy()

    elif event == cv2.EVENT_MOUSEMOVE and len(points) < 4:

        currentPoint = ([x, y])
####




# Drawing loop and current segment code inspired by:
# https://github.com/sampan-s-nayak/manual_polygon_drawer/blob/master/polygon_drawer.py
while True:
    ret, frame = cap.read()
    if not ret: break

    img = frame.copy()
    imgTemp = img.copy()


    if len(points) >= 1 and len(points) <= 2:

        if(currentPoint != lastPoint):
            img = imgTemp.copy()

        cv2.line(img, (lastPoint[0], lastPoint[1]), (currentPoint[0], currentPoint[1]), (255, 0, 0))

    elif len(points) >= 2 and len(points) <= 3:

        if(currentPoint != lastPoint):
            img = imgTemp.copy()

        cv2.line(img, (lastPoint[0], lastPoint[1]),(currentPoint[0], currentPoint[1]), (255, 0, 0))
        cv2.line(img, (points[0][0], points[0][1]),(currentPoint[0], currentPoint[1]), (255, 0, 0))

    cv2.imshow('Image', img)

    key = cv2.waitKey(1) & 0xFF

    # Exit loop if esc
    if key == 27:
        break

    # Keystone transformation
    if len(points) == 4:
        width, height = img.shape[1], img.shape[0]
        
        trueWidth  = np.sqrt( np.abs(points[0][0] - points[1][0]) ** 2 + np.abs(points[0][1] - points[1][1]) ** 2).astype(int)
        trueHeight = np.sqrt( np.abs(points[0][0] - points[3][0]) ** 2 + np.abs(points[0][1] - points[3][1]) ** 2).astype(int)
        
        srcPoints = np.float32(points)
        dstPoints = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        
        matrix = cv2.getPerspectiveTransform(srcPoints, dstPoints)
        
        keystoneImg = cv2.warpPerspective(img, matrix, (width, height))
        adjImg = cv2.resize(keystoneImg, (trueWidth, trueHeight))

        # Finestra nera proiettata
        black = np.zeros_like(img)
        blackKeystone = cv2.warpPerspective(black, matrix, (width, height))
        blackAdj = cv2.resize(blackKeystone, (trueWidth, trueHeight))

        cv2.imshow('Keystone Image', adjImg)
        cv2.imshow('Keystone Projection', blackAdj)
####

cv2.destroyAllWindows()