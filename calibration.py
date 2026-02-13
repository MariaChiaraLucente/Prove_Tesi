## porta udp, con gestione della camera e loop detection

from ultralytics import YOLO
from config import *
import mediapipe as mp
import cv2


# inizializzazione del modello shapes, oer tracking oggetti con forme
shapes_model_path = "best.pt"

model_shapes = YOLO(shapes_model_path)


#crea la finestra della webcam con il nome Calibration
cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Calibration",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
#cv_vision diventa la webcam, ed apre la finestra creata con la webcam con codice 0
cv_vision= cv2.VideoCapture(1)

#continuo a non capire cosa facciano
cv_vision.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cv_vision.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



#if che controlla se la webcam è stata aperta 
if cv_vision.isOpened():
        print("Camera aperta con successo")
        #se la webcam è aperta, legge il primo frame, rval= cv.vision diventa True se la lettura è stata fatta
        #frame è la prima immagine.
        rval, frame = cv_vision.read()

#recuperiam il modulo hands di mediapipe, che serve per il rilevamento delle mani
mp_hands = mp.solutions.hands

#hands diventa l oggetto di tracking delle mani,
#  con i parametri di confidenza e numero massimo di mani da rilevare
hands = mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )
# semplici utils che servono per disegnare i landmarks delle mani, e le connessioni tra i landmarks
mp_drawing = mp.solutions.drawing_utils

# Variabile globale per memorizzare la dimensione della pallina
ball_size = None

def ball_detection(frame):
    global ball_size
    #inizializzo il il frame della palla per detectarla
    img_ball = frame.copy()
    ball_data = None


    # --- Inferenza Modello Forme (per la pallina) ---
    results_s = model_shapes(img_ball, stream=True, verbose=False)

    for r in results_s:
        for box in r.boxes:
                # classe 'circle', restituisce le coordinte in pixel del Box che identifica Yolo
                # in pratica x1,y1 sono le coordinate del punto in alto a sinistra del box, 
                # x2,y2 quelle in basso a destra

            if box.conf[0] > 0.6 and ball_data is None:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Qui identifichiamo il centro della pallina in base alle coordinate in pixel
                    cx = x1 + (x2 - x1) // 2
                    cy = y1 + (y2 - y1) // 2
                    
                    
                    # e ne memoriziamo la dimensione per disegnare un quadrato fisso attorno alla pallina
                    if ball_size is None:
                     ball_size = max(1, min(x2 - x1, y2 - y1))

                    # i dati salvati della pallina sono:
                    # le coordinate in pixel del box (x1,y1,x2,y2) 
                    # le coordinate del centro (cx,cy)
                    ball_data = (x1, y1, x2, y2, cx, cy)
    
    return ball_data

#loop che entra nella camera finchè ha successo, e quindi finchè i frame vengono letti
#finche rval è true
while rval:
      #questo invece legge il frame successivo, e fa la stessa cosa di prima
      rval, frame = cv_vision.read()
      if not rval:
         break
      
      # Specchia orizzontalmente il frame (effetto specchio)
      frame = cv2.flip(frame, 1)

      img_ball = frame.copy()
       
      #questo converte il frame da BGR a RGB, per essere compatibile con mediapipe
      img_hands = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      #hands è l oggetto di tracking, eseguendo process cerca di individuare le mani nel frame,
      # e restituisce i risultati in results
      results = hands.process(img_hands)

      # inizializza una lista di posizioni delle mani, con due elementi None (per due mani)
      hands_positions = [None, None]

      # .multi_hand_landmarks è una strutttura con cui mediapipe restituisce i landmarks
      #qui dice che entra nell'if se è stata rilevata almeno una mano
      if results.multi_hand_landmarks:
            
            #prende altezza e larghezza del frame in pixel
            frame_h, frame_w = frame.shape[:2]

            # inizializza due variabili per i dati delle mani,
            #  che saranno None finchè non vengono rilevate
            p1_data = None
            p2_data = None

            #per ogni mano rilevata..
            for hand_landmarks in results.multi_hand_landmarks:

                #disegna i landmarks nella mano
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                #estraiamo il landmark 9 (base del dito medio) come punto di riferimento
                mpc = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

                #queste sono le coordinate normalizzate del landmark, 
                # tra 0 e 1, rispetto alla larghezza e altezza del frame
                # mediapipe non si basa sullo schermo o sulla rissoluzione della camera
                # esso si basa sull'immagine che gli viene data (in questo caso frame) 
                # quindi sono le coordinate normalizzare rispetto a frame_w e frame_h
                #secondo questa regola, il punto in alto a sinistra è (0,0), il punto in basso a destra è (1,1)
                mpc_x_norm = mpc.x
                mpc_y_norm = mpc.y

                #convertiamo le coordinate normalizzate in pixel
                mpc_x = int(mpc_x_norm * frame_w)
                mpc_y = int(mpc_y_norm * frame_h)

                hand_data = (mpc_x_norm, mpc_y_norm, mpc_x, mpc_y)

                if mpc_x_norm < 0.5 and p1_data is None:
                    p1_data = hand_data
                elif mpc_x_norm >= 0.5 and p2_data is None:
                    p2_data = hand_data

            if p1_data is not None:
                x_norm, y_norm, x, y = p1_data
                hands_positions[0] = (x, y)

                # Disegna indicatore P1 (BLU)
                cv2.circle(frame, (x, y), 15, (255, 0, 0), cv2.FILLED)
                cv2.putText(frame, "P1", (x - 10, y - 20), 
                            
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                # Mostra coordinate normalizzate
                coord_text = f"({mpc_x}, {mpc_y:.2f})"
                cv2.putText(frame, coord_text, (x - 50, y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # MANO DESTRA (Player 2)
            if p2_data is not None:
                x_norm, y_norm, x, y = p2_data
                hands_positions[1] = (x, y)
                
                # Disegna indicatore P2 (ROSSO)
                cv2.circle(frame, (x, y), 15, (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, "P2", (x - 10, y - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Mostra coordinate normalizzate
                #coord_text = f"({x_norm:.2f}, {y_norm:.2f})"

                print(f"mpc_x: {mpc_x}, mpc_y: {mpc_y}")
                
                coord_text = f"({mpc_x}, {mpc_y})"
                cv2.putText(frame, coord_text, (x - 50, y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
      
      # Rilevamento pallina
      ball_data = ball_detection(frame)
      if ball_data is not None:
          x1, y1, x2, y2, cx, cy = ball_data
          # Disegna bounding box
          cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
          # Disegna centro
          cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
          cv2.putText(frame, f"Ball ({cx}, {cy})", (cx + 10, cy - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

      #questo mostra il frame letto dalla webcam, l'ultimo che aveva ricevuto
      # e lo tramette nella finestra "Calibration"
      cv2.imshow("Calibration", frame)

      #non mi convince, ma fa in modo che ogni 20ms viene catturato 
      #un eventuale tasto premuto, e se è ESC (27) esce dal loop
      #se premo il bottone x, la funzione getWindowProperty ritorna -1, e quindi esce dal loop
      key = cv2.waitKey(1)
      if key == 27 or cv2.getWindowProperty("Calibration", cv2.WND_PROP_VISIBLE) < 1: # codice tasto ESC
         break







#appena esci dal loop distrugge la finestra "Calibration" e rilascia la webcam
try:
   cv2.destroyWindow("Calibration")
except cv2.error:
   pass
cv_vision.release()


#cosa devo fare?
# 1- implemmentare e capire il tracking della mano
# trovare i punti in pixel degli estremi del campo visivo. 
# settare come (0,0) il punto in alto a sinistra, il punto (1,1) in basso a destra
#settare (0.1) il punto in basso a sinistra e  (1,0) in alto a destra
