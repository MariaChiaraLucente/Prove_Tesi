from ultralytics import YOLO
import mediapipe as mp
import cv2
import math
import socket
import numpy as np
import time
import torch

############################
# CONFIG
############################
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

# Carica i file generati dagli script precedenti
try:
    H_cam_to_proj = np.load('homography_matrix.npy')
    print("✓ homography_matrix.npy caricata!")
except:
    H_cam_to_proj = None
    print("ATTENZIONE: homography_matrix.npy mancante!")

try:
    crop_data = np.load('crop_params.npy')
    X1_ROI, Y1_ROI, X2_ROI, Y2_ROI = map(int, crop_data)
    print(f"✓ crop_params.npy caricata: ({X1_ROI},{Y1_ROI}) → ({X2_ROI},{Y2_ROI})")
except:
    print("ATTENZIONE: crop_params.npy mancante — uso valori di fallback!")
    X1_ROI, Y1_ROI = 233, 29
    X2_ROI, Y2_ROI = 1089, 552

W_PROJ, H_PROJ = 1920, 1200

COLOR_RANGES = {
    'red': ([160, 40, 200], [180, 120, 255]),
    'green': ([50, 40, 50], [90, 255, 255]),
    'blue': ([90, 50, 50], [120, 255, 255]),
    'yellow': ([20, 100, 100], [30, 255, 255]),
    'purple': ([125, 50, 50], [140, 255, 255]),
    'custom': ([80, 182, 59], [88, 222, 77])
}

# Seleziona GPU se disponibile, altrimenti CPU
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
print(f"✓ Dispositivo inferenza YOLO: {'GPU (CUDA)' if DEVICE == 0 else 'CPU'}")

############################
# UDP
############################

def init_udp():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return sock

def send_to_unity(sock, player_id, x_norm, y_norm):
    message = f"PLAYER{player_id}, {x_norm:.2f}, {y_norm:.2f}"
    sock.sendto(message.encode(), (UDP_IP, UDP_PORT))

def send_ball_to_unity(sock, x1_norm, y1_norm, x2_norm, y2_norm, cx_norm, cy_norm):
    message = (
        f"BALL, {x1_norm:.4f}, {y1_norm:.4f}, {x2_norm:.4f}, {y2_norm:.4f}, "
        f"{cx_norm:.4f}, {cy_norm:.4f}"
    )
    sock.sendto(message.encode(), (UDP_IP, UDP_PORT))


############################
# INIT MODELS
############################
def init_models():
    shapes_model = YOLO("../best.pt")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        model_complexity=0,
        min_tracking_confidence=0.5,
        max_num_hands=2
    )
    return shapes_model, hands


############################
# INIT CAMERA
############################
def init_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    ###controllo fps camera###
    fps_tests = [30, 60, 90, 120]
    max_fps = 0
    for fps in fps_tests:
        cap.set(cv2.CAP_PROP_FPS, fps)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Richiesto {fps} FPS → Ottenuto: {actual_fps} FPS")
        if actual_fps > max_fps:
            max_fps = actual_fps
    
    print(f"✓ FPS massimo supportato: {max_fps}")
    print("============================\n")
    
    # Imposta il massimo rilevato
    cap.set(cv2.CAP_PROP_FPS, max_fps)
    
    return cap


# Calcola PLAY_RECT come percentuale del crop (10% margine per lato)
CROP_W = X2_ROI - X1_ROI
CROP_H = Y2_ROI - Y1_ROI


def apply_projection(cx_crop, cy_crop, H):
    """
    Trasforma le coordinate dal frame 'cropped' ai pixel del proiettore.
    """
    # 1. Riporta le coordinate del crop a quelle del frame intero (1280x720)
    cx_full = cx_crop + X1_ROI
    cy_full = cy_crop + Y1_ROI
    # 2. Applica l'omografia per ottenere le coordinate proiettore
    pt = np.array([[[cx_full, cy_full]]], dtype='float32')
    pt_proj = cv2.perspectiveTransform(pt, H)
    return int(pt_proj[0][0][0]), int(pt_proj[0][0][1])


############################
# HAND DETECTION
############################
def detect_hands(frame, hands, sock):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return frame

    h, w = frame.shape[:2]
    p1_data, p2_data = None, None

    for hand_landmarks in results.multi_hand_landmarks:
        # restituisce le coordinate normalizzate (0-1)
        mpc = hand_landmarks.landmark[9]
        pip = hand_landmarks.landmark[10]

        # coordinate in pixel
        mpc_x, mpc_y = int(mpc.x * w), int(mpc.y * h)
        pip_x, pip_y = int(pip.x * w), int(pip.y * h)

        hand_data = (mpc.x, mpc.y, mpc_x, mpc_y, pip_x, pip_y)

        if mpc.x < 0.5 and p1_data is None:
            p1_data = hand_data
        elif mpc.x >= 0.5 and p2_data is None:
            p2_data = hand_data

    draw_hand(frame, p1_data, 1, sock)
    draw_hand(frame, p2_data, 2, sock)

    return frame


def draw_hand(frame, data, player_id, sock):
    if data is None:
        return

    x_norm, y_norm, x, y, pip_x, pip_y = data
    send_to_unity(sock, player_id, x_norm, y_norm)

    color = (255, 0, 0) if player_id == 1 else (0, 0, 255)

    cv2.rectangle(frame, (x - 15, y - 15), (x + 15, y + 15), color, cv2.FILLED)
    cv2.rectangle(frame, (pip_x - 15, pip_y - 15), (pip_x + 15, pip_y + 15), color, cv2.FILLED)
    cv2.putText(frame, f"P{player_id}", (x - 10, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"P{player_id} pip", (pip_x - 10, pip_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


############################
# BALL DETECTION
############################
ball_size = None
# # Configurazione Kalman Filter per la pallina (x, y, dx, dy)
# kf = cv2.KalmanFilter(4, 2)
# kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
# kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
# kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03  # Fluidità movimento
# kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5  # Fiducia nella rilevazione
# kf_initialized = False
# kf_lost_frames = 0

# Variabili per SMOOTHING Semplice
prev_cx, prev_cy = None, None
velocity_x, velocity_y = 0, 0
frames_lost = 0
SMOOTH_ALPHA = 0.  # 0.1 = molto lento/fluido, 0.9 = molto reattivo/scattoso

def detect_ball(frame, black_frame, model, sock):
    #global ball_size, kf_initialized, kf_lost_frames
    global ball_size, prev_cx, prev_cy, velocity_x, velocity_y, frames_lost


    # trasformiamo in HSV per creare una maschera di colore
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # setto il range di colore per il verde
    lower_color, upper_color = COLOR_RANGES["green"]

    # creo la maschera per il verde
    color_mask = cv2.inRange(hsv, np.array(lower_color), np.array(upper_color))

    # soglia minima di verde all'interno del box
    min_green_ratio = 0.10  # 10%

    # restituisce le dimensioni del frame (altezza, larghezza)
    h, w = frame.shape[:2]

    # # 1. PREDIZIONE: Dove dovrebbe essere la pallina ora?
    # prediction = kf.predict()
    # pred_cx, pred_cy = int(prediction[0, 0]), int(prediction[1, 0])

    # Passiamo il device esplicito (0 = prima GPU, 'cpu' = processore)
    # imgsz=416 invece di default 640 → ~2x più veloce con accuratezza simile
    results = model(frame, stream=True, verbose=False, device=DEVICE, imgsz=416)
    ball_found = False

    for r in results:
        for box in r.boxes:
            # classe circle con confidenza > 0.4
            if box.conf[0] > 0.4 and int(box.cls[0]) == 0:
                # setto le coordinate del box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Controllo colore nel box
                box_mask = color_mask[y1:y2, x1:x2]
                green_ratio = np.count_nonzero(box_mask) / box_mask.size if box_mask.size else 0.0
                
                if green_ratio >= min_green_ratio:
                    ball_found = True

                    # centro della palla ovvero il punto medio del box, in pixel
                    raw_cx = (x1 + x2) // 2
                    raw_cy = (y1 + y2) // 2

                    # # 2. CORREZIONE: Aggiorniamo il filtro con la posizione reale trovata
                    # kf.correct(np.array([[np.float32(raw_cx)], [np.float32(raw_cy)]]))
                    # kf_initialized = True
                    # kf_lost_frames = 0
                    # --- SMOOTHING LOGIC ---
                    if prev_cx is None or frames_lost >= 10:
                        # Primo rilevamento o reset dopo perdita prolungata
                        cx, cy = raw_cx, raw_cy
                        velocity_x, velocity_y = 0, 0
                    else:
                        # Media pesata tra posizione precedente e attuale
                        cx = int(SMOOTH_ALPHA * raw_cx + (1 - SMOOTH_ALPHA) * prev_cx)
                        cy = int(SMOOTH_ALPHA * raw_cy + (1 - SMOOTH_ALPHA) * prev_cy)
                        
                        # Calcolo velocità semplice (per predizione futura)
                        velocity_x = cx - prev_cx
                        velocity_y = cy - prev_cy

                    # # Usiamo le coordinate filtrate (più stabili) invece di quelle raw
                    # cx = int(kf.statePost[0, 0])
                    # cy = int(kf.statePost[1, 0])

                     # Aggiorna stato
                    prev_cx, prev_cy = cx, cy
                    frames_lost = 0

                    # --- LOGICA DI PROIEZIONE ---
                    # Trasformiamo la posizione della palla in pixel del proiettore
                    if H_cam_to_proj is not None:
                        try:
                            px_proj, py_proj = apply_projection(cx, cy, H_cam_to_proj)

                            # Disegna sulla finestra del proiettore
                            if 0 <= px_proj < W_PROJ and 0 <= py_proj < H_PROJ:
                                # Disegniamo un cerchio che "insegue" l'oggetto
                                cv2.circle(black_frame, (px_proj, py_proj), 20, (0, 255, 0), -1)
                        except Exception as e:
                            print(f"Errore proiezione: {e}")

                    # Memorizziamo la dimensione al primo rilevamento
                    if ball_size is None:
                        ball_size = max(1, min(x2 - x1, y2 - y1))

                    half = ball_size // 2
                    x1, y1 = cx - half, cy - half
                    x2, y2 = cx + half, cy + half

                    x1, x2 = max(0, x1), min(w - 1, x2)
                    y1, y2 = max(0, y1), min(h - 1, y2)

                    send_ball_to_unity(
                        sock,
                        x1 / w, y1 / h,
                        x2 / w, y2 / h,
                        cx / w, cy / h
                    )

                    draw_ball(frame, x1, y1, x2, y2, cx, cy)
                    return (x1, y1, x2, y2, cx, cy)

# # 3. GESTIONE PERDITA: Se YOLO non trova nulla, usiamo la predizione per un po'
#     if not ball_found and kf_initialized and kf_lost_frames < 10:
#         kf_lost_frames += 1
#         # Usiamo la predizione del Kalman
#         cx, cy = pred_cx, pred_cy

    # 3. GESTIONE PERDITA: Predizione lineare semplice
    if not ball_found and prev_cx is not None and frames_lost < 10:
        frames_lost += 1
        
        # Applica l'ultima velocità nota (inerzia)
        cx = prev_cx + velocity_x
        cy = prev_cy + velocity_y
        
        # Aggiorna prev per il prossimo frame
        prev_cx, prev_cy = cx, cy

        # Ricostruiamo il box usando l'ultima dimensione nota
        if ball_size is not None:
            half = ball_size // 2
            x1, y1 = cx - half, cy - half
            x2, y2 = cx + half, cy + half

            # Proiezione (Ghost) - Continua a proiettare anche se non vede
            if H_cam_to_proj is not None:
                try:
                    px_proj, py_proj = apply_projection(cx, cy, H_cam_to_proj)
                    if 0 <= px_proj < W_PROJ and 0 <= py_proj < H_PROJ:
                        # Disegniamo il target predetto
                        cv2.rectangle(black_frame, (px_proj - 15, py_proj - 15), (px_proj + 15, py_proj + 15),
                                      (0, 255, 0), -1)
                        cv2.putText(black_frame, "PREDICT", (px_proj + 40, py_proj),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                except:
                    pass

            # Feedback visivo sul monitor (Giallo = Predizione)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 255), cv2.FILLED)
            cv2.putText(frame, "Lost - Predicting...", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1)

            return (x1, y1, x2, y2, cx, cy)

    return None


def draw_ball(frame, x1, y1, x2, y2, cx, cy):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 3, (255, 0, 0), cv2.FILLED)


############################
# MAIN LOOP
############################
def run():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    model_shapes, hands = init_models()
    cap = init_camera()

    # YOLO Warm-up e performance test
    print("\n=== YOLO PERFORMANCE TEST ===")
    dummy = np.zeros((523, 856, 3), dtype=np.uint8)  # Simula cropped frame
    warmup_times = []
    for i in range(10):
        t_start = time.time()
        _ = model_shapes(dummy, device=DEVICE, verbose=False, imgsz=416)
        warmup_times.append((time.time() - t_start) * 1000)
    
    print(f"YOLO Warm-up (10 frames):")
    print(f"  Min: {min(warmup_times):.1f}ms")
    print(f"  Max: {max(warmup_times):.1f}ms")
    print(f"  Avg: {np.mean(warmup_times):.1f}ms")
    print(f"  FPS teorico: {1000/np.mean(warmup_times):.1f}")
    print("============================\n")

    # Creiamo una finestra a tutto schermo per il proiettore
    cv2.namedWindow("PROIEZIONE_PARETE", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("PROIEZIONE_PARETE", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Pre-alloca black_frame una sola volta (performance)
    black_frame = np.zeros((H_PROJ, W_PROJ, 3), dtype=np.uint8)
    
    prev_time = 0
    frame_count = 0
    
    # Statistiche YOLO runtime
    yolo_times = []
    yolo_fps_avg = 0
    YOLO_SKIP = 1  # Processa YOLO ogni N frame (1=sempre, 2=metà, 3=un terzo)

    while True:
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        ret, frame = cap.read()
        if not ret:
            break

        # Pre-process (Specchiamento e ritaglio)
        frame = cv2.flip(frame, -1)
        cropped = frame[Y1_ROI:Y2_ROI, X1_ROI:X2_ROI].copy()

        # Resetta frame proiettore (riutilizzo memoria)
        black_frame.fill(0)

        # Rilevamento mani (sempre)
        hands_frame = detect_hands(cropped.copy(), hands, sock)
        
        # Rilevamento palla (con frame skipping opzionale + timing)
        frame_count += 1
        if frame_count % YOLO_SKIP == 0:
            yolo_t1 = time.time()
            detect_ball(cropped, black_frame, model_shapes, sock)
            yolo_t2 = time.time()
            
            yolo_time_ms = (yolo_t2 - yolo_t1) * 1000
            yolo_times.append(yolo_time_ms)
            
            # Mantieni solo ultimi 30 campioni
            if len(yolo_times) > 30:
                yolo_times.pop(0)
            
            yolo_fps_avg = 1000 / np.mean(yolo_times) if yolo_times else 0

        # Mostra FPS e statistiche YOLO sul frame di debug
        cv2.putText(cropped, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        if yolo_times:
            yolo_info = f"YOLO: {np.mean(yolo_times):.1f}ms ({yolo_fps_avg:.1f} FPS)"
            cv2.putText(cropped, yolo_info, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Calibration_ball", cropped)
        cv2.imshow("PROIEZIONE_PARETE", black_frame)
        cv2.imshow("Hands", hands_frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()