## porta udp, con gestione della camera e loop detection

from ultralytics import YOLO
from config import *
import mediapipe as mp
import cv2
import math
import socket
import numpy as np

# Configurazione UDP
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# inizializzazione del modello shapes, oer tracking oggetti con forme
shapes_model_path = "best.pt"
model_shapes = YOLO(shapes_model_path)

#crea la finestra della webcam con il nome Calibration
cv2.namedWindow("Calibration_ball", cv2.WINDOW_NORMAL)
#cv2.setWindowProperty("Calibration_ball",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.namedWindow("Game", cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Game', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) 

#crea la finestra della webcam con il nome Calibration
cv2.namedWindow("Calibration_hands", cv2.WINDOW_NORMAL)
#cv2.setWindowProperty("Calibration_hands",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#cv_vision diventa la webcam, ed apre la finestra creata con la webcam con codice 0
cv_vision= cv2.VideoCapture(1)

#continuo a non capire cosa facciano
cv_vision.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cv_vision.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

################################### COLLEGAMENTO CON UNITY ###################################

def send_ball_to_unity(x1_norm, y1_norm, x2_norm, y2_norm, cx_norm, cy_norm):
    # Formato messaggio: "BALL,0.10,0.20,0.30,0.40,0.25,0.30"
    message = (
        f"BALL, {x1_norm:.4f}, {y1_norm:.4f}, {x2_norm:.4f}, {y2_norm:.4f}, "
        f"{cx_norm:.4f}, {cy_norm:.4f}"
    )
    sock.sendto(message.encode(), (UDP_IP, UDP_PORT))


def send_to_unity(player_id, x_norm, y_norm):
    
    # Formato messaggio: "P1,0.5,0.8"
    message = f"PLAYER{player_id}, {x_norm:.2f}, {y_norm:.2f}"

    sock.sendto(message.encode(), (UDP_IP, UDP_PORT))

#if che controlla se la webcam è stata aperta 
rval = False
if cv_vision.isOpened():
    print("Camera aperta con successo")
    #se la webcam è aperta, legge il primo frame, rval= cv.vision diventa True se la lettura è stata fatta
    #frame è la prima immagine.
    rval, frame = cv_vision.read()


########################### TRACCIAMENTO MANI ##########################
sent_p1 = False
sent_p2 = False

# recuperiamo il modulo hands di mediapipe, che serve per il rilevamento delle mani
mp_hands = mp.solutions.hands

# hands diventa l oggetto di tracking delle mani,
# con i parametri di confidenza e numero massimo di mani da rilevare
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    max_num_hands=2
)

# semplici utils che servono per disegnare i landmarks delle mani, e le connessioni tra i landmarks
mp_drawing = mp.solutions.drawing_utils


# --- Calibrazione distanza camera ---
# W_MPC_WRIST_CM: distanza reale (in cm) tra i landmark MPC e WRIST
# FOCAL_LENGTH_PX: focale equivalente in pixel, da calibrare con una misura nota
W_MPC_WRIST_CM = 4.5

FOCAL_LENGTH_PX = 1079 # calcolo tramite l'angolo del campo visivo della camera e la risoluzione
#######################################################################

#questo non ha senso perche i punti sono fermi lol

'''def distance(frame): 
    print("Calibrating distance...")

    # creo un campo di gioco virtuale, con coordinate in pixel
    cv2.rectangle(frame, (0 + 50, 0 + 50), (1280 - 50, 720 - 50), (255, 255, 255), 2) #campo di gioco virtuale
    
    x_px = 50
    y_px = 50

    x2_px = 1280 - 50 #1230
    y2_px = 50 

    distance_px = math.hypot(x2_px - x_px, y2_px - y_px)
    distance_cm = 41.5 # distanza reale in cm tra i punti (50,50) e (1230,670) calcalcolata empiricamente

    if distance_px > 0:
        # la distanza camera - piano su cui si trovano i due punti di riferimento

        distance = (distance_cm * FOCAL_LENGTH_PX) / distance_px
        print(f"Distanza stimata: {distance:.2f} cm")
    else:
        distance = None

    cv2.putText(frame, f"Distance: {int(distance)} cm", (x_px, y_px), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame'''



def hand_detection(frame):
    global sent_p1, sent_p2

    # questo converte il frame da BGR a RGB, per essere compatibile con mediapipe
    img_hands = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # hands e l oggetto di tracking, eseguendo process cerca di individuare le mani nel frame,
    # e restituisce i risultati in results
    results = hands.process(img_hands)

    # inizializza una lista di posizioni delle mani, con due elementi None (per due mani)
    hands_positions = [None, None]

    # .multi_hand_landmarks e una struttura con cui mediapipe restituisce i landmarks
    # qui dice che entra nell'if se e stata rilevata almeno una mano
    if results.multi_hand_landmarks:

        # prende altezza e larghezza del frame in pixel
        frame_h, frame_w = frame.shape[:2]

        # inizializza due variabili per i dati delle mani,
        # che saranno None finche non vengono rilevate
        p1_data = None
        p2_data = None

        # per ogni mano rilevata..
        for hand_landmarks in results.multi_hand_landmarks:

            # disegna i landmarks nella mano
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # estraiamo il landmark 9 (base del dito medio) come punto di riferimento
            mpc = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

            pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

            # queste sono le coordinate normalizzate del landmark,
            # tra 0 e 1, rispetto alla larghezza e altezza del frame
            # mediapipe non si basa sullo schermo o sulla risoluzione della camera
            # esso si basa sull'immagine che gli viene data (in questo caso frame)
            # quindi sono le coordinate normalizzate rispetto a frame_w e frame_h
            # secondo questa regola, il punto in alto a sinistra e (0,0), il punto in basso a destra e (1,1)
            mpc_x_norm = mpc.x
            mpc_y_norm = mpc.y

            pip_x_norm = pip.x
            pip_y_norm = pip.y


            # convertiamo le coordinate normalizzate in pixel
            mpc_x = int(mpc_x_norm * frame_w)
            mpc_y = int(mpc_y_norm * frame_h)

            pip_x = int(pip_x_norm * frame_w)
            pip_y = int(pip_y_norm * frame_h)   

            ################# CALCOLO DELLA DISTANZA BASATA SULLA DISTANZA TRA DUE PUNTI CONOSCIUTI #################

            # distanza in pixel tra i punti scelti della mano (9 e 10), che uso come punto di riferimento per la distanza dalla camera
            w_px = math.hypot(mpc_x - pip_x, mpc_y - pip_y)

            # distanza stimata dalla camera al punto MPC 
            # distance_cm = (W * f) / w_px, dove W è la distanza reale tra i punti (in cm), f è la focale in pixel, e w_px è la distanza in pixel tra i punti rilevati
            # se w_px è maggiore di 0, calcoliamo la distanza in cm usando la formula di calibrazione
            #maggiore è la distanza in pixel, piu vicina è la mano, piccola è la distanza in cm
            if w_px > 0:
                distance_cm = (W_MPC_WRIST_CM * FOCAL_LENGTH_PX) / w_px
                #print(f"Distanza stimata: {distance_cm:.2f} cm")
            else:
                distance_cm = None

            ##########################################################################################################

            hand_data = (mpc_x_norm, mpc_y_norm, mpc_x, mpc_y)

            if mpc_x_norm < 0.5 and p1_data is None:
                p1_data = hand_data
            elif mpc_x_norm >= 0.5 and p2_data is None:
                p2_data = hand_data

        if p1_data is not None:
            x_norm, y_norm, x, y = p1_data
            hands_positions[0] = (x, y)

            # mando i dati a unity
            send_to_unity(1, x_norm, y_norm)
            sent_p1 = True

            # Disegna indicatore P1 (BLU)
            cv2.rectangle(frame, (x-15, y-15), (x+15, y+15), (255, 0, 0), cv2.FILLED)
            cv2.rectangle(frame, (pip_x-15, pip_y-15), (pip_x+15, pip_y+15), (255, 0, 255), cv2.FILLED)

            cv2.putText(frame, "P1", (x - 10, y - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            # Mostra coordinate normalizzate
            coord_text = f"({mpc_x}, {mpc_y:.2f})"
            cv2.putText(frame, coord_text, (x - 50, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Mostra distanza stimata dalla camera
            if distance_cm is not None:
                cv2.putText(frame, f"{int(distance_cm)} cm", (x - 10, y + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # MANO DESTRA (Player 2)
        if p2_data is not None:
            x_norm, y_norm, x, y = p2_data
            hands_positions[1] = (x, y)

            # mando i dati a unity
            send_to_unity(2, x_norm, y_norm)
            sent_p2 = True

            # Disegna indicatore P2 (ROSSO)
            cv2.rectangle(frame, (x-15, y-15), (x+15, y+15), (0, 0, 255), cv2.FILLED)
            cv2.rectangle(frame, (pip_x-15, pip_y-15), (pip_x+15, pip_y+15), (255, 0, 255), cv2.FILLED)
            cv2.putText(frame, "P2", (x - 10, y - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Mostra coordinate normalizzate
            # coord_text = f"({x_norm:.2f}, {y_norm:.2f})"

            # print(f"x_norm: {x_norm}, y_norm: {y_norm}")

            coord_text = f"({mpc_x}, {mpc_y})"
            cv2.putText(frame, coord_text, (x - 50, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Mostra distanza stimata dalla camera
            if distance_cm is not None:
                cv2.putText(frame, f"{int(distance_cm)} cm", (x - 10, y + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return frame

# Variabile globale per memorizzare la dimensione della pallina
ball_size = None

def sent_resolution_to_unity(width, height):
    message = f"RESOLUTION, {width}, {height}"
    sock.sendto(message.encode(), (UDP_IP, UDP_PORT))

def ball_detection(frame):
    global ball_size

    #inizializzo il il frame della palla per detectarla
    img_ball = frame.copy()
    # i dati della palla sono none, quindi ad ogni frame viene resettato il rilevamento 
    # coordinate in pixel del box (x1,y1,x2,y2) e coordinate del centro (cx,cy)
    ball_data = None
   

    # --- Inferenza Modello Forme (per la pallina) ---
    results_s = model_shapes(img_ball, stream=True, verbose=False)
    

    for r in results_s:
        for box in r.boxes:
                # classe 'circle', restituisce le coordinte in pixel del Box che identifica Yolo
                # in pratica x1,y1 sono le coordinate del punto in alto a sinistra del box, 
                # x2,y2 quelle in basso a destra

            if box.conf[0] > 0.6 and int(box.cls[0]) == 0: # se la confidenza è maggiore di 0.6 e la classe è circle
                    x1_yolo, y1_yolo, x2_yolo, y2_yolo = map(int, box.xyxy[0])

                    # Calcoliamo il centro dalla detection YOLO
                    cx_yolo = (x1_yolo + x2_yolo) // 2
                    cy_yolo = (y1_yolo + y2_yolo) // 2
                    
                    # Salviamo il centro YOLO originale per il cerchio blu
                    cx = cx_yolo
                    cy = cy_yolo
                    
                    # Memorizziamo la dimensione al primo rilevamento
                    if ball_size is None:
                        ball_size = max(1, min(x2_yolo - x1_yolo, y2_yolo - y1_yolo))

############################ calcolo della distanza dalla pallina, basata sualla distanza dei due raggi ###################################
                    # Calcoliamo il box con dimensione fissa basata su ball_size
                    '''
                    # CALCOLO LA DISTANZA TRA IL CENTRO E LA CORNICE DEL BOX, CHE MI SERVE PER CALCOLARE LA DISTANZA DALLA CAMERA
                    r_px= math.hypot(cx - x1_yolo, cy - y1_yolo) #raggio in pixel
                    radius_cm = 5 #raggio reale della pallina in cm

                    if r_px > 0:
                        radius_cm = (radius_cm * FOCAL_LENGTH_PX) / r_px
                        print(f"Distanza stimata: {radius_cm:.2f} cm")
                    else:
                        radius_cm = None
                    '''

                    half = ball_size // 2
                    x1 = cx - half
                    y1 = cy - half
                    x2 = cx + half
                    y2 = cy + half

                    # Verifica se il bounding box tocca i bordi del frame (1200x720) (adesso è croppato) 
                    #quidni se x1 è meno di 0. rimane 0, se x2 è più di 1280 rimane 1280, e così via per y1 e y2
                    frame_h, frame_w = img_ball.shape[:2]

                    if x1 <= 0:
                        x1 = 0
                        cx = half
                        x2 = ball_size
                    if x2 >= frame_w:
                        x2 = frame_w - 1
                        cx = frame_w - 1 - half
                        x1 = cx - half
                    if y1 <= 0:
                        y1 = 0
                        cy = half
                        y2 = ball_size
                    if y2 >= frame_h:
                        y2 = frame_h - 1
                        cy = frame_h - 1 - half
                        y1 = cy - half

                    # i dati salvati della pallina sono:
                    # le coordinate in pixel del box fisso (x1,y1,x2,y2) 
                    # le coordinate del centro fisso (cx,cy)
                    # le coordinate del centro YOLO originale (cx_yolo, cy_yolo)
                    ball_data = (x1, y1, x2, y2, cx, cy, cx_yolo, cy_yolo)

                    if frame_w > 0 and frame_h > 0:
                        x1_norm = x1 / frame_w
                        y1_norm = y1 / frame_h
                        x2_norm = x2 / frame_w
                        y2_norm = y2 / frame_h
                        cx_norm = cx / frame_w
                        cy_norm = cy / frame_h
                        send_ball_to_unity(x1_norm, y1_norm, x2_norm, y2_norm, cx_norm, cy_norm)

                        #print(f"x1_norm: {x1_norm}, y1_norm: {y1_norm}, x2_norm: {x2_norm}, y2_norm: {y2_norm}, cx_norm: {cx_norm}, cy_norm: {cy_norm}")
    
    return ball_data


# Definisci il ROI (Region Of Interest)
x1_roi, y1_roi = 233, 29
x2_roi, y2_roi = 1089, 552

#cv2.moveWindow("Game", 0, 0) # sposta la finestra del gioco a destra della webcam

#loop che entra nella camera finchè ha successo, e quindi finchè i frame vengono letti
#finche rval è true
while rval:
        
    #questo invece legge il frame successivo, e fa la stessa cosa di prima
    rval, frame = cv_vision.read()
    if not rval:
        break
    
    # Specchia orizzontalmente il frame (effetto specchio)
    frame = cv2.flip(frame, - 1)


    # Ritaglia il frame al ROI
    cropped_frame = frame[y1_roi:y2_roi, x1_roi:x2_roi]
    # creo un campo di gioco virtuale, con coordinate in pixel
    # cv2.rectangle(frame, (0 + 50, 0 + 50), (1280 - 50, 720 - 50), (255, 255, 255), 2) #campo di gioco virtuale

    
        # Usa cropped_frame per le detection
    hands_frame = hand_detection(cropped_frame.copy())

    ball_frame = ball_detection(cropped_frame.copy())



    #rilevamento mani
    #hands_frame = hand_detection(frame.copy())
    
    # Rilevamento pallina
    #ball_frame = ball_detection(frame.copy())

    # schermo nero
    black_frame = np.zeros_like(cropped_frame)

    ris_x = black_frame.shape[1]
    ris_y = black_frame.shape[0]
    print(f"Risoluzione del frame: {ris_x}x{ris_y}")

    if ball_frame is not None:
        x1, y1, x2, y2, cx, cy, cx_yolo, cy_yolo = ball_frame
        # Disegna bounding box con le coordinate fisse trovare al primo rilevamento
        cv2.rectangle(cropped_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Disegna centro fisso (verde), quello del box fisso che non puo oltrepassare le pareti
        #cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        
        # Disegna centro YOLO originale (blu) - segue il rilevamento con lo stesso raggio della pallina
        cv2.circle(cropped_frame, (cx_yolo, cy_yolo), 2, (255, 0, 0), cv2.FILLED)

        cv2.putText(cropped_frame, f"Ball ({cx}, {cy})", (cx + 10, cy - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        #qui mostro le coordinate in pixel del box, che sono quelle che mi servono per il gioco
        cv2.putText(cropped_frame, f"Coord: (x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2})", (x1 - 50, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
         # Disegna solo la pallina sul nero
        cv2.rectangle(black_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
       # cv2.circle(black_frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
       # cv2.circle(black_frame, (cx_yolo, cy_yolo), 1, (255, 0, 0), cv2.FILLED)
       # cv2.putText(black_frame, f"Ball ({cx}, {cy})", (cx + 10, cy - 10),
                   # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
    
    #questo mostra i frame letti dalla webcam nelle finestre dedicate
    cv2.imshow("Calibration_hands", hands_frame)
    cv2.imshow("Calibration_ball", cropped_frame)
    cv2.imshow("Game", black_frame)

    #non mi convince, ma fa in modo che ogni 20ms viene catturato 
    #un eventuale tasto premuto, e se è ESC (27) esce dal loop
    #se premo il bottone x, la funzione getWindowProperty ritorna -1, e quindi esce dal loop
    key = cv2.waitKey(1)
    if key == 27 or cv2.getWindowProperty("Calibration_ball", cv2.WND_PROP_VISIBLE) < 1: # codice tasto ESC
        break


#appena esci dal loop distrugge le finestre e rilascia la webcam
try:
    cv2.destroyWindow("Calibration_ball")
    cv2.destroyWindow("Calibration_hands")

except cv2.error:
   pass
cv_vision.release()


#cosa devo fare?
# 1- implemmentare e capire il tracking della mano ok
# trovare i punti in pixel degli estremi del campo visivo. 

#x1:233 ; y1: 29
#x2: 1089 ; y2: 553

# settare come (0,0) il punto in alto a sinistra, il punto (1,1) in basso a destra
#settare (0.1) il punto in basso a sinistra e  (1,0) in alto a destra

