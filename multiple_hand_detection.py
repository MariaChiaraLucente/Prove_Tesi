import mediapipe as mp
import cv2
import math
import numpy as np

############################
# CONFIG
############################
X1_ROI, Y1_ROI = 233, 29
X2_ROI, Y2_ROI = 1089, 552
MAX_LOST_FRAMES = 15    # frame prima di resettare un profilo, se la mano sparisce per piu di 15 frame, viene ricalcolata il rilevamento
MAX_ASSIGN_DIST = 300   # Distanza massima in pixel per considerare una mano come "la stessa" del frame precedente.

############################
# STATO TRACKER
# Dizionario globale che mantiene lo stato dei due giocatori.
# Ogni giocatore ha:
#   pos  → ultima posizione nota (x, y), None se non ancora visto
#   lost → quanti frame consecutivi non è stato trovato
#   color→ colore per il disegno
############################
tracker_state = {
    1: {"pos": None, "lost": 0, "color": (255, 80, 0)},
    2: {"pos": None, "lost": 0, "color": (0, 80, 255)},
}

############################
# UPDATE TRACKER
############################
def update_tracker(detected_positions):
    """
    detected_positions: lista di (x, y) delle mani trovate in questo frame.

    Per ogni giocatore cerca la mano rilevata più vicina alla sua
    ultima posizione nota. Se la trova entro MAX_ASSIGN_DIST pixel
    la aggiorna, altrimenti incrementa il contatore 'lost'.
    Se lost supera MAX_LOST_FRAMES il profilo si resetta a None
    e alla prossima apparizione viene riassegnato da capo.

    Restituisce un dict {pid: (x, y)} con le posizioni assegnate.
    """
    assigned  = {}
    available = list(detected_positions)  # copia locale per poter rimuovere elementi

    for pid, state in tracker_state.items():
        if not available:
            break

        if state["pos"] is None:
            # --- PRIMA ASSEGNAZIONE ---
            # Il profilo non ha ancora una posizione nota.
           

            # if mpc.x < 0.5 and p1_data is None:
            #     p1_data = hand_data 
            # elif mpc.x >= 0.5 and p2_data is None:
            #     p2_data = hand_data

            available.sort(key=lambda p: p[0])
            
            # P1 prende indice 0 (più a sinistra), P2 indice -1 (più a destra).
            idx = 0 if pid == 1 else -1
            # Assegna la posizione e la rimuove dalla lista disponibili.
            state["pos"] = available.pop(idx)
            # Aggiorna la posizione e la rimuove dalla lista disponibili.
            state["lost"] = 0
            assigned[pid] = state["pos"]

        else:
            # --- ASSEGNAZIONE PER DISTANZA ---
            # Cerca tra le mani disponibili quella più vicina
            # all'ultima posizione nota di questo giocatore.
            
            best_idx  = None
            best_dist = float("inf")
            '''È un valore speciale più grande di qualsiasi numero. 
            Viene usato qui per inizializzare best_dist con un valore "altissimo", 
            così che qualsiasi distanza reale trovata dopo sarà sicuramente minore 
            e aggiornerà il valore.'''

            # per ogni mano disponibile..
            for i, pos in enumerate(available):
                # calcola la distanza tra questa mano e l'ultima posizione nota della mano rilevata prima
                d = math.hypot(pos[0] - state["pos"][0],
                               pos[1] - state["pos"][1])
                #se questa distanza è inferiore alla soglia..
                if d < best_dist:
                    # Tiene la mano più vicina.
                    best_dist = d
                    #
                    best_idx  = i

            if best_idx is not None and best_dist < MAX_ASSIGN_DIST:
                # trovata una mano abbastanza vicina → aggiorna posizione
                state["pos"]  = available.pop(best_idx)
                state["lost"] = 0
                assigned[pid] = state["pos"]
            else:
                # nessuna mano vicina → giocatore perso per questo frame
                state["lost"] += 1
                if state["lost"] > MAX_LOST_FRAMES:
                    # perso troppo a lungo → reset completo
                    state["pos"] = None
                else:
                    # mantieni l'ultima posizione nota mentre aspetti
                    assigned[pid] = state["pos"]

    return assigned

############################
# INIT
############################
def init_hands():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        max_num_hands=2
    )

def init_camera():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap

############################
# PREPROCESS
############################
def preprocess_frame(frame):
    frame = cv2.flip(frame, -1)
    return frame[Y1_ROI:Y2_ROI, X1_ROI:X2_ROI]

############################
# DETECT + TRACK
############################
def detect_and_track(frame, hands_model):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_model.process(img_rgb)
    h, w    = frame.shape[:2]

    # raccoglie le posizioni (lm[9] = centro palmo) di tutte le mani trovate
    detected = []
    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            lm = hl.landmark
            x  = int(lm[9].x * w)
            y  = int(lm[9].y * h)
            detected.append((x, y))

    # aggiorna il tracker con le posizioni rilevate
    assigned = update_tracker(detected)

    # disegna ogni giocatore
    for pid, (x, y) in assigned.items():
        color = tracker_state[pid]["color"]
        lost  = tracker_state[pid]["lost"]

        cv2.circle(frame, (x, y), 12, color, cv2.FILLED)

        # se lost > 0 significa che stiamo usando l'ultima posizione nota
        label = f"P{pid}" + (" (lost)" if lost > 0 else "")
        cv2.putText(frame, label, (x - 10, y - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame

############################
# MAIN
############################
def run():
    hands_model = init_hands()
    cap         = init_camera()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cropped = preprocess_frame(frame)
        out     = detect_and_track(cropped.copy(), hands_model)

        cv2.imshow("Player Tracker", out)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()




# from ultralytics import YOLO
# import mediapipe as mp
# import cv2
# import math
# import socket
# import numpy as np


# ############################
# # CONFIG
# ############################

# FOCAL_LENGTH_PX = 1079
# W_MPC_WRIST_CM = 4.5

# # ROI
# X1_ROI, Y1_ROI = 233, 29
# X2_ROI, Y2_ROI = 1089, 552

# ############################
# # INIT MODELS
# ############################
# def init_models():
#     shapes_model = YOLO("shape_model.pt")

#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(
#         min_detection_confidence=0.7,
#         min_tracking_confidence=0.5,
#         max_num_hands=2
#     )
#     return shapes_model, hands



# ############################
# # INIT CAMERA
# ############################
# def init_camera():
#     cap = cv2.VideoCapture(1)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#     return cap

# ############################
# # PREPROCESS FRAME
# ############################
# def preprocess_frame(frame):
#     frame = cv2.flip(frame, -1)
#     cropped = frame[Y1_ROI:Y2_ROI, X1_ROI:X2_ROI]
#     return cropped


# ############################
# # HAND DETECTION
# ############################
# def detect_hands(frame, hands):
#     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(img_rgb)

#     if not results.multi_hand_landmarks:
#         return frame

#     h, w = frame.shape[:2]
#     p1_data, p2_data = None, None

#     for hand_landmarks in results.multi_hand_landmarks:

#         #restituisce le coordinate normalizzate (0-1) 
#         mpc = hand_landmarks.landmark[9]
#         wrist = hand_landmarks.landmark[0]


#         # coordinate in pixel
#         mpc_x, mpc_y = int(mpc.x * w), int(mpc.y * h)
#         wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)

#         # distanza mano-camera
#         w_px = math.hypot(mpc_x - wrist_x, mpc_y - wrist_y)
#         distance_cm = (W_MPC_WRIST_CM * FOCAL_LENGTH_PX) / w_px if w_px > 0 else None

#         hand_data = (mpc.x, mpc.y, mpc_x, mpc_y, wrist_x, wrist_y)

#         if mpc.x < 0.5 and p1_data is None:
#             p1_data = hand_data 
#         elif mpc.x >= 0.5 and p2_data is None:
#             p2_data = hand_data

#     draw_hand(frame, p1_data, 1)
#     draw_hand(frame, p2_data, 2)

#     return frame


# def draw_hand(frame, data, player_id):
#     if data is None:
#         return

#     x_norm, y_norm, x, y, wrist_x, wrist_y = data


#     color = (255, 0, 0) if player_id == 1 else (0, 0, 255)

#     cv2.rectangle(frame, (x-15, y-15), (x+15, y+15), color, cv2.FILLED)
#     cv2.rectangle(frame, (wrist_x-15, wrist_y-15), (wrist_x+15, wrist_y+15), color, cv2.FILLED)
#     cv2.putText(frame, f"P{player_id}", (x - 10, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#     cv2.putText(frame, f"P{player_id} wrist", (wrist_x - 10, wrist_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     ############################
# # MAIN LOOP
# ############################
# def run():

#     model_shapes, hands = init_models()
#     cap = init_camera()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         cropped = preprocess_frame(frame)

#         hands_frame = detect_hands(cropped.copy(), hands)
#         cv2.imshow("Calibration_hands", hands_frame)

#         if cv2.waitKey(1) == 27:
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     run()