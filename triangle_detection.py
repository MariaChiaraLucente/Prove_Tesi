#ris camera
#ris crop
#altezza camera 
# conv cm to px
# mappatura posizione
# riconoscimento disco univoco--
        #possibile soluzione: riconoscere un triangolo 
        # riconoscimento tramite colore?
            #---> riconoscimento tramite marker colorato su forma

        # riconoscimento tramite marker personalizzato? 

#riconscimento delle mani, max 2, è possibile distinguere le mani di destra e di sinistra




from turtle import color

from matplotlib.pyplot import hsv
from ultralytics import YOLO
import cv2
import numpy as np


############################
# CONFIG
############################
COLOR_RANGES = {
        'red': ([160, 40, 200], [180, 120, 255]),
        'green': ([50, 40, 50], [90, 255, 255]),
        'blue': ([90, 50, 50], [120, 255, 255]),
        'yellow': ([20, 100, 100], [30, 255, 255]),
        'purple': ([125, 50, 50], [140, 255, 255]),
        'custom': ([80,182,59], [88,222, 77])
    }

# ROI
X1_ROI, Y1_ROI = 233, 29
X2_ROI, Y2_ROI = 1089, 552

def init_models():
    shapes_model = YOLO("best.pt")

    return shapes_model

############################
# INIT CAMERA
############################
def init_camera():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


############################
# PREPROCESS FRAME
############################
def preprocess_frame(frame):
    frame = cv2.flip(frame, -1)
    cropped = frame[Y1_ROI:Y2_ROI, X1_ROI:X2_ROI]
    return cropped


def detect_triangle(frame, model):
    results = model(frame, stream=True, verbose=False)
    #trasformiamo in HSV per creare una maschera di colore
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # setto il range di colore per il verde
    lower_color, upper_color = COLOR_RANGES["green"]

    # creo la maschera per il verde, l'immagina diventa bianca se il pixel è verde, altrimenti è nera
    color_mask = cv2.inRange(hsv, np.array(lower_color), np.array(upper_color))

    # soglia minima di verde all'interno del box per considerarlo un triangolo verde, ed essere rilevato
    min_green_ratio = 0.10 # 20%


    for result in results:
        if result.boxes is None:
            continue
#per ogni box e rilevato da yolo..
        for box in result.boxes:
            #seleziono qurlli dei triangoli con confidenza maggiore di 0.6 
            if box.conf[0] > 0.4 and int(box.cls[0]) == 0:
             #setto le coordinate del box
             x1, y1, x2, y2 = map(int, box.xyxy[0])
                #creo una maschera del box che corrisponde alla maschera di colore verde, quindi 
                # ritaglio solo il pezzo della maschera dentro il box del triangolo e conti quanti pixel bianchi ci sono.
             box_mask = color_mask[y1:y2, x1:x2]
                #calcolo la percentuale di verde all'interno del box, se è superiore alla soglia allora disegno il box del triangolo
             green_ratio = np.count_nonzero(box_mask) / box_mask.size if box_mask.size else 0.0
             if green_ratio >= min_green_ratio:
                 draw_triangle(frame, x1, y1, x2, y2)

    return frame

def draw_triangle(frame, x1, y1, x2, y2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


def run():
    model_shapes = init_models()
    cap = init_camera()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cropped = preprocess_frame(frame)
        triangle_frame = detect_triangle(cropped, model_shapes)
        cv2.imshow("Triangle Detection", triangle_frame)

        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()


#quindi se io faccio il rilevamento di una forma con una percentuale di colore dentro, posso raffinare la rilevazione
# in questo caso ho messo la rilevazione di un cerchio o triangolo, ma il box viene disegnato solo se dentro che  un % di verde
#il verde viene riconosciuto con una maschera di colore, che ha un range in HSV. bianchi ci sono dentro il box del triangolo, se è superiore alla soglia allora disegno il box del triangolo.