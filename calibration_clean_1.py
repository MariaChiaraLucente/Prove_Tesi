from ultralytics import YOLO
import mediapipe as mp
import cv2
import math
import socket
import numpy as np

############################
# CONFIG
############################
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

FOCAL_LENGTH_PX = 1079
W_MPC_WRIST_CM = 4.5

# ROI
X1_ROI, Y1_ROI = 233, 29
X2_ROI, Y2_ROI = 1089, 552


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
    shapes_model = YOLO("best.pt")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        max_num_hands=2
    )
    return shapes_model, hands


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

        #restituisce le coordinate normalizzate (0-1) 
        mpc = hand_landmarks.landmark[9]
        pip = hand_landmarks.landmark[10]


        # coordinate in pixel
        mpc_x, mpc_y = int(mpc.x * w), int(mpc.y * h)
        pip_x, pip_y = int(pip.x * w), int(pip.y * h)

        # distanza mano-camera
        w_px = math.hypot(mpc_x - pip_x, mpc_y - pip_y)
        distance_cm = (W_MPC_WRIST_CM * FOCAL_LENGTH_PX) / w_px if w_px > 0 else None

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

    cv2.rectangle(frame, (x-15, y-15), (x+15, y+15), color, cv2.FILLED)
    cv2.rectangle(frame, (pip_x-15, pip_y-15), (pip_x+15, pip_y+15), color, cv2.FILLED)
    cv2.putText(frame, f"P{player_id}", (x - 10, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"P{player_id} pip", (pip_x - 10, pip_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


############################
# BALL DETECTION
############################
ball_size = None

def detect_ball(frame, model, sock):
    global ball_size
    # restituisce le dimensioni del frame (altezza, larghezza)
    h, w = frame.shape[:2]

    results = model(frame, stream=True, verbose=False)

    for r in results:
        for box in r.boxes:
            # classe circle con confidenza > 0.6
            if box.conf[0] > 0.6 and int(box.cls[0]) == 0:
                #mappo le coordinare del box che circonda la palla in pixel
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # centro della palla ovvero il punto medio del box, in pixel
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Memorizziamo la dimensione al primo rilevamento
                if ball_size is None:
                    # La dimensione è la media tra larghezza e altezza del box
                    ball_size = max(1, min(x2 - x1, y2 - y1))

                half = ball_size // 2
                x1, y1 = cx - half, cy - half
                x2, y2 = cx + half, cy + half

                x1, x2 = max(0, x1), min(w-1, x2)
                y1, y2 = max(0, y1), min(h-1, y2)

                send_ball_to_unity(
                    sock,
                    x1 / w, y1 / h,
                    x2 / w, y2 / h,
                    cx / w, cy / h
                )

                draw_ball(frame, x1, y1, x2, y2, cx, cy)
                projection(frame, x1, x2, y1, y2, cx, cy)
                return (x1, y1, x2, y2, cx, cy)

    return None


def draw_ball(frame, x1, y1, x2, y2, cx, cy):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 3, (255, 0, 0), cv2.FILLED)
    
    
def projection(frame,x1, x2, y1, y2, cx, cy):
     # Disegna solo la pallina sul nero
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 1, (255, 0, 0), cv2.FILLED)
    cv2.putText(frame, f"Ball ({cx}, {cy})", (cx + 10, cy - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    
############################
# MAIN LOOP
############################
def run():
    sock = init_udp()
    model_shapes, hands = init_models()
    cap = init_camera()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cropped = preprocess_frame(frame)

        hands_frame = detect_hands(cropped.copy(), hands, sock)
        detect_ball(cropped, model_shapes, sock)

        black_frame = np.zeros_like(cropped)
    

        cv2.imshow("Calibration_hands", hands_frame)
        cv2.imshow("Calibration_ball", cropped)
        cv2.imshow("Game", black_frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
