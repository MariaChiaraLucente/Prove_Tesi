import mediapipe as mp
import cv2
import math
import numpy as np
from filterpy.kalman import KalmanFilter

############################
# CONFIG
############################
X1_ROI, Y1_ROI = 233, 29
X2_ROI, Y2_ROI = 1089, 552
MAX_LOST_FRAMES = 20
MAX_ASSIGN_DIST = 200


############################
# FILTRA DETECTION DUPLICATE
# Se due mani rilevate sono troppo vicine tra loro,
# probabilmente è la stessa mano detectata due volte → tieni solo una
############################
MIN_HAND_DIST = 80   # pixel minimi tra due mani per considerarle distinte


#detected è la lista di tuple delle posizioni rilevate da MediaPipe, e quindi delle mani
def filter_detections(detected):
    # se è minore di due, non c'è rischio di duplicati, restituisci tutto
    if len(detected) < 2:
        return detected
    # calcola la distanza tra le due mani rilevate (distanza tra due punti)
    d = math.hypot(detected[0][0] - detected[1][0],
                   detected[0][1] - detected[1][1])
    
    #Se la distanza è minore di MIN_HAND_DIST (80 pixel):
    #Significa che sono troppo vicine → probabilmente è la stessa mano rilevata due volte
    if d < MIN_HAND_DIST:
        # tieni quella con confidenza implicita (la prima nell'array)
        return [detected[0]]
    return detected

############################
# CREA FILTRO DI KALMAN PER UNA MANO
# stato interno del filtro: [x, y, vx, vy] dove vx e vy è la velocità stimata in pixel/frame
# misura: [x, y]
############################
def make_kalman():
    kf = KalmanFilter(dim_x=4, dim_z=2)

    # F: matrice di transizione — come evolve lo stato
    # x_new = x + vx, y_new = y + vy, vx/vy costanti
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=float)

    # H: matrice di osservazione — cosa misuriamo (solo x, y)
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]], dtype=float)

    # R: rumore di misura — quanto ti fidi del rilevamento MediaPipe
    # valori alti → il filtro si fida meno della misura, più della predizione
    kf.R *= 10

    # P: incertezza iniziale dello stato
    kf.P *= 100

    # Q: rumore di processo — quanto può cambiare la velocità tra un frame e l'altro
    # valori alti → il filtro si adatta più velocemente ai cambi di direzione
    kf.Q *= 0.5

    return kf


############################
# STATO TRACKER
# Ogni giocatore ha:
#   kf   → filtro di Kalman (contiene posizione + velocità stimata)
#   lost → frame consecutivi senza rilevamento
#   color→ colore per il disegno
#   initialized → se il filtro è stato inizializzato con una prima misura
############################
tracker_state = {
    1: {"kf": make_kalman(), "lost": 0, "color": (255, 80, 0),  "initialized": False},
    2: {"kf": make_kalman(), "lost": 0, "color": (0, 80, 255),  "initialized": False},
} #dizionario globale che tiene lo stato del tracker per i due giocatori


# viene chiamata quando si preme 'r' per resettare il tracker,
# ad esempio se i giocatori si sono confusi o sono entrati in scena nuovi giocatori
def reset_tracker():
    for state in tracker_state.values():
        state["kf"]          = make_kalman()
        state["lost"]        = 0
        state["initialized"] = False

############################
# POSIZIONE PREDETTA
# Legge x, y dallo stato interno del filtro
############################
def predicted_pos(state):
    x = float(state["kf"].x[0])
    y = float(state["kf"].x[1])
    return (int(x), int(y))

############################
# PRIMA ASSEGNAZIONE
# Quando uno o entrambi i giocatori non sono ancora inizializzati,
# assegna le mani disponibili per posizione x (sinistra→P1, destra→P2)
############################
def first_assignment(available):

    #itera sui due giocatori pid, e controlla se il loro stato è initialized o no,
    #   filtra solo quelli con "initialized": False
    #   raccoglie gli ID dei giocatori non inizializzati 
    uninitialized = [pid for pid, s in tracker_state.items() if not s["initialized"]]

   # se tutti i giocatori sono già inizializzati o non ci sono mani disponibili, 
   # restituisci le mani disponibili senza assegnare nulla
    if not uninitialized or not available:
        return available

    ''' available = lista di coordinate [(x1, y1), (x2, y2), ...]
        key=lambda p: p[0] = ordina per la coordinata x (asse orizzontale)
        p[0] = coordinata x, p[1] = coordinata y '''
    sorted_hands = sorted(available, key=lambda p: p[0])

    # Il loop assegna una mano a ogni giocatore non inizializzato
    for pid in sorted(uninitialized):
        if not sorted_hands:
            break
        # Assegna la mano corretta al giocatore in base alla posizione x: la mano più a sinistra va a P1, 
        # quella più a destra a P2
        hand = sorted_hands.pop(0) if pid == 1 else sorted_hands.pop(-1)

        # inizializza lo stato del filtro con la prima misura
        # x[0]=x, x[1]=y, x[2]=vx=0, x[3]=vy=0 (velocità iniziale nulla)
        tracker_state[pid]["kf"].x = np.array([[hand[0]],
                                               [hand[1]],
                                               [0.],
                                               [0.]])
        tracker_state[pid]["initialized"] = True
        tracker_state[pid]["lost"]        = 0

    return sorted_hands

def update_tracker(detected_positions):
    assigned  = {}
    available = list(detected_positions)

    # --- PREDICT sempre, per tutti i giocatori inizializzati ---
    for state in tracker_state.values():
        if state["initialized"]:
            state["kf"].predict()

    # --- CASO 0 MANI ---
    if not available:
        for pid, state in tracker_state.items():
            if not state["initialized"]:
                continue
            state["lost"] += 1
            if state["lost"] > MAX_LOST_FRAMES:
                state["kf"]          = make_kalman()
                state["initialized"] = False
            else:
                assigned[pid] = predicted_pos(state)


        print(f"lost: P1={tracker_state[1]['lost']} | P2={tracker_state[2]['lost']}")


        return assigned

    # --- CASO 1 MANO ---
    # associa la mano all'unico giocatore già inizializzato più vicino.
    # Se nessuno è inizializzato, inizializza solo P1.
    # P2 resta in attesa finché non appare una seconda mano distinta.
    if len(available) == 1:
        pos = available[0]
        initialized_pids = [pid for pid, s in tracker_state.items() if s["initialized"]]

        if not initialized_pids:
            # nessuno inizializzato → inizializza P1 e basta
            tracker_state[1]["kf"].x = np.array([[float(pos[0])],
                                                  [float(pos[1])],
                                                  [0.], [0.]])
            tracker_state[1]["initialized"] = True
            tracker_state[1]["lost"]        = 0
            assigned[1] = pos

        else:
            # trova il giocatore inizializzato più vicino alla mano
            best_pid  = min(initialized_pids,
                            key=lambda pid: math.hypot(
                                pos[0] - predicted_pos(tracker_state[pid])[0],
                                pos[1] - predicted_pos(tracker_state[pid])[1]))
            
            # Calcola la distanza effettiva tra la mano rilevata e la predizione 
            # del giocatore più vicino trovato sopra. Serve per decidere
            # se la mano è "sua" o di qualcun altro.
            dist = math.hypot(pos[0] - predicted_pos(tracker_state[best_pid])[0],
                              pos[1] - predicted_pos(tracker_state[best_pid])[1])

            if dist < MAX_ASSIGN_DIST:
                # aggiorna il giocatore più vicino
                # Passa la misura reale al filtro di Kalman. 
                # Il filtro corregge la predizione

                tracker_state[best_pid]["kf"].update(
                    np.array([[float(pos[0])], [float(pos[1])]]))
                tracker_state[best_pid]["lost"] = 0
                assigned[best_pid] = predicted_pos(tracker_state[best_pid])
            else:
                # la mano è troppo lontana da chiunque → potrebbe essere
                # un nuovo giocatore entrato in scena
                # lo sostituisci a chi è sparito da più tempo (lost più alto)

                ''' questo si verifica solo se
                    1 mano rilevata
                    2 giocatori inizializzati
                    la mano è troppo lontana da entrambi (dist > MAX_ASSIGN_DIST)'''
                if len(initialized_pids) > 1:
                    # scegli chi resettare: il giocatore sparito da più tempo
                    best_pid_for_reset = max(initialized_pids, key=lambda pid: tracker_state[pid]["lost"])
                else:
                    # un solo giocatore inizializzato → non resettare, aspetta
                    best_pid_for_reset = None

                if best_pid_for_reset is not None:
                    tracker_state[best_pid_for_reset]["kf"].x = np.array([[float(pos[0])],
                                                                           [float(pos[1])],
                                                                           [0.], [0.]])
                    tracker_state[best_pid_for_reset]["lost"] = 0
                    assigned[best_pid_for_reset] = pos

            # gli altri giocatori inizializzati non visti → lost
            active_pid = best_pid_for_reset if (dist >= MAX_ASSIGN_DIST and best_pid_for_reset is not None) else best_pid
            for pid in initialized_pids:
                if pid == active_pid:
                    continue
                tracker_state[pid]["lost"] += 1
                if tracker_state[pid]["lost"] > MAX_LOST_FRAMES:
                    tracker_state[pid]["kf"]          = make_kalman()
                    tracker_state[pid]["initialized"] = False
                else:
                    assigned[pid] = predicted_pos(tracker_state[pid])

        print(f"lost: P1={tracker_state[1]['lost']} | P2={tracker_state[2]['lost']}")
        return assigned

    # --- CASO 2 MANI ---
    # prima assegnazione se necessario
    available = first_assignment(available)

    # associazione ottimale per distanza dalla posizione predetta
    candidates = []
    for pid, state in tracker_state.items():
        if not state["initialized"]:
            continue
        pred = predicted_pos(state)
        for i, pos in enumerate(available):
            d = math.hypot(pos[0] - pred[0], pos[1] - pred[1])
            candidates.append((d, pid, i))

    candidates.sort(key=lambda c: c[0])

    assigned_pids = set()
    assigned_idxs = set()

    for dist, pid, idx in candidates:
        if pid in assigned_pids or idx in assigned_idxs:
            continue
        if dist > MAX_ASSIGN_DIST:
            continue
        tracker_state[pid]["kf"].update(
            np.array([[float(available[idx][0])],
                      [float(available[idx][1])]]))
        tracker_state[pid]["lost"] = 0
        assigned[pid] = predicted_pos(tracker_state[pid])
        assigned_pids.add(pid)
        assigned_idxs.add(idx)

    # giocatori non associati → lost
    for pid, state in tracker_state.items():
        if pid in assigned_pids or not state["initialized"]:
            continue
        state["lost"] += 1
        if state["lost"] > MAX_LOST_FRAMES:
            state["kf"]          = make_kalman()
            state["initialized"] = False
        else:
            assigned[pid] = predicted_pos(state)
    print(f"lost: P1={tracker_state[1]['lost']} | P2={tracker_state[2]['lost']}")
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

    detected = []
    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            lm = hl.landmark
            x  = int(lm[9].x * w)
            y  = int(lm[9].y * h)
            detected.append((x, y))

    detected = filter_detections(detected)
    assigned = update_tracker(detected)

    for pid, (x, y) in assigned.items():
        color = tracker_state[pid]["color"]
        lost  = tracker_state[pid]["lost"]

        # disegna cerchio pieno sulla posizione filtrata
        cv2.circle(frame, (x, y), 12, color, cv2.FILLED)

        # disegna anche la velocità stimata come freccia
        vx = int(tracker_state[pid]["kf"].x[2] * 5)
        vy = int(tracker_state[pid]["kf"].x[3] * 5)
        cv2.arrowedLine(frame, (x, y), (x + vx, y + vy), color, 2)

        label = f"P{pid}" + (" (lost)" if lost > 0 else "")
        cv2.putText(frame, label, (x - 10, y - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # linea di riferimento: 300 px in basso a sinistra
    cv2.line(frame, (0, h - 20), (200, h - 20), (255, 255, 255), 2)

    return frame

############################
# MAIN
############################
def run():
    hands_model = init_hands()
    cap         = init_camera()

    print("Tasti: [r] reset | [ESC] esci")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cropped = preprocess_frame(frame)
        out     = detect_and_track(cropped.copy(), hands_model)

        cv2.imshow("Kalman Tracker", out)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('r'):
            reset_tracker()
            print("[RESET]")

        

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()