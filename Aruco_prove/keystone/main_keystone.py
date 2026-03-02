from ultralytics import YOLO
import cv2
import socket
import numpy as np

############################
# CONFIG
############################
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
W_PROJ, H_PROJ = 1920, 1200

COLOR_RANGES = {
    'red':    ([160, 40, 200], [180, 120, 255]),
    'green':  ([50,  40,  50], [90,  255, 255]),
    'blue':   ([90,  50,  50], [120, 255, 255]),
    'yellow': ([20, 100, 100], [30,  255, 255]),
    'purple': ([125, 50,  50], [140, 255, 255]),
    'custom': ([80, 182,  59], [88,  222,  77])
}
DETECT_COLOR = "green"
MIN_COLOR_RATIO = 0.10

############################
# CARICA CALIBRAZIONE
############################

# keystone_matrix.npy — prodotta da calibrazione_keystone.py
try:
    H_keystone = np.load('keystone_matrix.npy')
    print("✓ keystone_matrix.npy caricata!")
except:
    H_keystone = None
    print("⚠️  keystone_matrix.npy mancante — nessuna correzione keystone!")

# homography_matrix.npy — prodotta da calibrazione_aruco.py
try:
    H_cam_to_proj = np.load('homography_matrix.npy')
    print("✓ homography_matrix.npy caricata!")
except:
    print("❌ homography_matrix.npy mancante!")
    exit(1)

# crop_params.npy — prodotta da select_crop.py
try:
    crop_data = np.load('crop_params.npy')
    X1_ROI, Y1_ROI, X2_ROI, Y2_ROI = map(int, crop_data)
    print(f"✓ crop_params.npy caricata: ({X1_ROI},{Y1_ROI}) → ({X2_ROI},{Y2_ROI})")
except:
    print("❌ crop_params.npy mancante!")
    exit(1)

CROP_W = X2_ROI - X1_ROI
CROP_H = Y2_ROI - Y1_ROI
print(f"📐 Crop: {CROP_W}x{CROP_H}px")

############################
# UDP
############################
def send_ball(sock, cx_norm, cy_norm):
    """Invia posizione normalizzata della pallina a Unity (0.0 → 1.0)."""
    msg = f"BALL, {cx_norm:.4f}, {cy_norm:.4f}"
    sock.sendto(msg.encode(), (UDP_IP, UDP_PORT))

def send_field_config(sock):
    """Invia le dimensioni del campo a Unity una volta sola all'avvio."""
    msg = f"FIELD, {CROP_W}, {CROP_H}, {CROP_W/CROP_H:.4f}"
    sock.sendto(msg.encode(), (UDP_IP, UDP_PORT))

############################
# FUNZIONI
############################
def apply_projection(x_crop, y_crop):
    """
    Trasforma coordinate del crop → pixel proiettore tramite omografia.
    La keystone viene applicata separatamente su tutto il frame.
    """
    x_full = x_crop + X1_ROI
    y_full = y_crop + Y1_ROI
    pt = np.array([[[x_full, y_full]]], dtype='float32')
    pt_proj = cv2.perspectiveTransform(pt, H_cam_to_proj)
    return int(pt_proj[0][0][0]), int(pt_proj[0][0][1])

def normalize_to_crop(cx, cy):
    """Normalizza le coordinate rispetto all'intero crop (0.0 → 1.0)."""
    rx = cx / CROP_W
    ry = cy / CROP_H
    return max(0.0, min(1.0, rx)), max(0.0, min(1.0, ry))

def project_frame(black_frame):
    """Applica keystone al black_frame prima di proiettarlo."""
    if H_keystone is not None:
        return cv2.warpPerspective(black_frame, H_keystone, (W_PROJ, H_PROJ))
    return black_frame

############################
# BALL DETECTION
############################
ball_size = None

def detect_ball(frame, black_frame, model, sock):
    global ball_size
    h, w = frame.shape[:2]

    # Maschera colore sul frame croppato
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower, upper = COLOR_RANGES[DETECT_COLOR]
    color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

    results = model(frame, stream=True, verbose=False)

    for r in results:
        for box in r.boxes:
            if box.conf[0] < 0.4 or int(box.cls[0]) != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Verifica percentuale di colore dentro il box
            box_mask = color_mask[y1:y2, x1:x2]
            color_ratio = np.count_nonzero(box_mask) / box_mask.size if box_mask.size else 0.0
            if color_ratio < MIN_COLOR_RATIO:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Stabilizza dimensione box
            if ball_size is None:
                ball_size = max(1, min(x2 - x1, y2 - y1))

            half = ball_size // 2
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(w - 1, cx + half)
            y2 = min(h - 1, cy + half)

            # Normalizza rispetto al crop intero e invia a Unity
            rx, ry = normalize_to_crop(cx, cy)
            send_ball(sock, rx, ry)

            # Proietta sul black_frame (keystone applicata dopo)
            px_proj, py_proj = apply_projection(cx, cy)
            if 0 <= px_proj < W_PROJ and 0 <= py_proj < H_PROJ:
                cv2.circle(black_frame, (px_proj, py_proj), 18, (0, 255, 0), -1)

            # Debug sul frame camera
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), cv2.FILLED)
            cv2.putText(frame, f"({rx:.2f}, {ry:.2f})", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            return (cx, cy, rx, ry)

    return None

############################
# MAIN LOOP
############################
def run():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    model = YOLO("../best.pt")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Manda config campo a Unity una volta sola
    send_field_config(sock)

    cv2.namedWindow("PROIEZIONE_PARETE", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("PROIEZIONE_PARETE", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, -1)

        # Crop alla zona della proiezione
        cropped = frame[Y1_ROI:Y2_ROI, X1_ROI:X2_ROI].copy()

        # Black frame su cui disegniamo tutto (in coordinate proiettore)
        black_frame = np.zeros((H_PROJ, W_PROJ, 3), dtype=np.uint8)

        # Rileva pallina e aggiorna black_frame
        detect_ball(cropped, black_frame, model, sock)

        # Applica keystone e mostra sul proiettore
        cv2.imshow("PROIEZIONE_PARETE", project_frame(black_frame))
        cv2.imshow("Calibration_ball", cropped)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()