## porta udp, con gestione della camera e loop detection

from ultralytics import YOLO
import cv2
import os
from config import *
import mediapipe as mp

class VisionManager:
    def __init__(self):
        # Utilizziamo direttamente MediaPipe per il rilevamento delle mani
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )
        self.mp_drawing = mp.solutions.drawing_utils

        shapes_model_path = "best.pt"

        if not os.path.exists(shapes_model_path):
            print(f"[ERRORE] Modello forme non trovato: {shapes_model_path}")

        self.model_shapes = YOLO(shapes_model_path)

        # inizizzazione della camera

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ball_size = None



    def detect_hands(self, img):
        img_hands = img.copy()
        hands_positions = [None, None]

        img_hands = cv2.cvtColor(img_hands, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_hands)

        if results.multi_hand_landmarks:
            frame_h, frame_w = img.shape[:2]
            hands_data = []

            for hand_landmarks in results.multi_hand_landmarks:
                #disegna i landmarks nella mano
                self.mp_drawing.draw_landmarks(img_hands, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                #estraiamo il landmark 9 (base del dito medio) come punto di riferimento
                mpc = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                mpc_x_norm= mpc.x
                mpc_y_norm = mpc.y

                #convertiamo le coordinate normalizzate in pixel
                mpc_x = int(mpc_x_norm * frame_w)
                mpc_y = int(mpc_y_norm * frame_h)

                hands_data.append((mpc_x_norm, mpc_x, mpc_y))

                hands_data.sort(key=lambda h: h[0])  # ordina per coordinata X normalizzata

                #Player 1 (mano sinistra) e Player 2 (mano destra)

            if len(hands_data) >= 1:
                _, x, y = hands_data[0]
                hands_positions[0] = (x, y)
                
                # Disegna indicatore P1 (BLU)
                cv2.circle(img_hands, (x, y), 15, (255, 0, 0), cv2.FILLED)
                cv2.putText(img_hands, "P1", (x - 10, y - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # MANO DESTRA (Player 2)
            if len(hands_data) >= 2:
                _, x, y = hands_data[1]
                hands_positions[1] = (x, y)
                
                # Disegna indicatore P2 (ROSSO)
                cv2.circle(img_hands, (x, y), 15, (0, 0, 255), cv2.FILLED)
                cv2.putText(img_hands, "P2", (x - 10, y - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return img_hands, hands_positions
    

    def detect_ball(self, img):
        img_ball = img.copy()
        ball_data = None
        ball_out = False

        # --- Inferenza Modello Forme (per la pallina) ---
        results_s = self.model_shapes(img, stream=True, verbose=False)


        for r in results_s:
            for box in r.boxes:
                # classe 'circle'
                if box.conf[0] > 0.6 and ball_data is None:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Qui identifichiamo il centro della pallina
                    cx = x1 + (x2 - x1) // 2
                    cy = y1 + (y2 - y1) // 2
                    
                    # e ne memoriziamo la dimensione per disegnare un quadrato fisso attorno alla pallina
                    if self.ball_size is None:
                        self.ball_size = max(1, min(x2 - x1, y2 - y1))

                    half = self.ball_size // 2
                    x1 = cx - half
                    y1 = cy - half
                    x2 = cx + half
                    y2 = cy + half

                    # Disegna bounding box
                    cv2.rectangle(img_ball, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    radius = max(1, half)
                    ball_data = (cx, cy, radius)

                    if x1< 0 or x2 > CAM_WIDTH:
                        ball_out = True

                    #stampiamo le coordinate della pallina rilevata con timestamp
                    print(f"[{cv2.getTickCount()/cv2.getTickFrequency():.2f}s] Pallina rilevata a: {ball_data}")

        return img_ball, ball_data, ball_out

        
    
    def update(self):
        """
        Metodo principale che aggiorna il rilevamento di mani e pallina.
        
        Returns:
            tuple: (img_hands, img_ball, detections)
                - img_hands: frame con annotazioni delle mani
                - img_ball: frame con annotazioni della pallina
                - detections: dizionario con chiavi "hands", "ball", "ball_out"
        """
        success, img = self.cap.read()
        if not success:
            return None, None, None
        
        # Flip orizzontale per effetto specchio
        img = cv2.flip(img, 1)
        
        # Struttura dati per le detection
        detections = {
            "hands": [None, None],  # [left_hand, right_hand]
            "ball": None,
            "ball_out": False
        }
        
        # --- Rilevamento Mani con MediaPipe ---
        img_hands, hands_positions = self.detect_hands(img)
        detections["hands"] = hands_positions
        
        # --- Rilevamento Pallina con YOLO ---
        img_ball, ball_data, ball_out = self.detect_ball(img)
        detections["ball"] = ball_data
        detections["ball_out"] = ball_out
        
        return img_hands, img_ball, detections
    
    def release(self):
        """Rilascia le risorse (camera e MediaPipe)."""
        if self.cap is not None:
            self.cap.release()
        if self.hands is not None:
            self.hands.close()