import cv2
import numpy as np

############################
# CONFIG
############################
W_PROJ, H_PROJ = 1920, 1200
W_CAM,  H_CAM  = 1280, 720
CAM_INDEX = 0

############################
# VARS
############################
points_cam = []      # punti cliccati nella finestra camera (coord frame intero)
current_point = [0, 0]
last_point    = [0, 0]
cam_display   = None
cam_temp      = None

############################
# MOUSE CALLBACK
# Si attacca alla finestra della camera (frame intero)
############################
def mouse_callback(event, x, y, flags, param):
    global last_point, current_point, cam_temp, cam_display

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points_cam) < 4:
            points_cam.append([x, y])
            last_point = [x, y]
            cv2.circle(cam_display, (x, y), 10, (0, 255, 0), -1)
            cv2.putText(cam_display, str(len(points_cam)), (x + 14, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cam_temp = cam_display.copy()
            print(f"  Punto {len(points_cam)}: cam({x}, {y})")

    elif event == cv2.EVENT_MOUSEMOVE:
        current_point = [x, y]

############################
# CREA FRAME PROIETTORE GUIDA
# Cerchi bianchi agli angoli per agevolare il clic preciso
############################
def make_proj_guide():
    frame = np.full((H_PROJ, W_PROJ, 3), 20, dtype=np.uint8)

    # Bordo bianco
    cv2.rectangle(frame, (2, 2), (W_PROJ-2, H_PROJ-2), (255, 255, 255), 4)

    # Griglia leggera
    for i in range(1, 4):
        cv2.line(frame, (W_PROJ*i//4, 0), (W_PROJ*i//4, H_PROJ), (60, 60, 60), 1)
        cv2.line(frame, (0, H_PROJ*i//4), (W_PROJ, H_PROJ*i//4), (60, 60, 60), 1)

    # Cerchi bianchi agli angoli — facili da identificare nella ripresa camera
    margin = 60
    corners = [
        (margin,          margin,          "1"),
        (W_PROJ - margin, margin,          "2"),
        (W_PROJ - margin, H_PROJ - margin, "3"),
        (margin,          H_PROJ - margin, "4"),
    ]
    for cx, cy, label in corners:
        cv2.circle(frame, (cx, cy), 40, (255, 255, 255), -1)
        cv2.circle(frame, (cx, cy), 42, (200, 200, 200), 2)
        cv2.putText(frame, label, (cx - 12, cy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

    cv2.putText(frame, "Clicca i 4 cerchi bianchi nella finestra camera",
                (W_PROJ//2 - 420, H_PROJ//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (160, 160, 160), 2)

    return frame

############################
# FRAME CAM → frame intero (nessun crop)
############################
def frame_to_display(frame):
    """Restituisce il frame intero ridimensionato a W_CAM x H_CAM se necessario."""
    h, w = frame.shape[:2]
    if (w, h) != (W_CAM, H_CAM):
        return cv2.resize(frame, (W_CAM, H_CAM))
    return frame.copy()

############################
# INIT CAMERA
############################
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W_CAM)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H_CAM)

ret, test_frame = cap.read()
if not ret:
    print("ERRORE: impossibile aprire la camera.")
    exit(1)

############################
# INIT FINESTRE
############################
# Finestra proiettore (schermo secondario, fullscreen)
cv2.namedWindow("PROIEZIONE_KEYSTONE", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("PROIEZIONE_KEYSTONE", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

proj_guide = make_proj_guide()
cv2.imshow("PROIEZIONE_KEYSTONE", proj_guide)

# Finestra camera — frame intero 1280×720
cv2.namedWindow("CAMERA_VIEW", cv2.WINDOW_NORMAL)
cv2.resizeWindow("CAMERA_VIEW", W_CAM, H_CAM)

ret, frame = cap.read()
cam_display = frame_to_display(frame)
cam_temp    = cam_display.copy()

cv2.imshow("CAMERA_VIEW", cam_display)
cv2.setMouseCallback("CAMERA_VIEW", mouse_callback)

print("=" * 60)
print("  CALIBRAZIONE KEYSTONE — Clic sulla finestra camera")
print("=" * 60)
print()
print("Il proiettore mostra 4 cerchi bianchi agli angoli.")
print("La camera mostra il frame INTERO (nessun crop).")
print()
print("Clicca i 4 cerchi bianchi nell'ordine:")
print("   1 → Alto-Sinistra")
print("   2 → Alto-Destra")
print("   3 → Basso-Destra")
print("   4 → Basso-Sinistra")
print()
print("S = salva   R = ricomincia   ESC = esci")
print()

############################
# LOOP
############################
while True:
    ret, frame = cap.read()
    if ret:
        cam_display = frame_to_display(frame)
        # Ridisegna i punti già cliccati
        for i, (px, py) in enumerate(points_cam):
            cv2.circle(cam_display, (px, py), 10, (0, 255, 0), -1)
            cv2.putText(cam_display, str(i + 1), (px + 14, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cam_temp = cam_display.copy()

    draw = cam_temp.copy()

    # Linee dinamiche
    if 1 <= len(points_cam) < 4:
        cv2.line(draw, tuple(last_point), tuple(current_point), (255, 165, 0), 2)
    if len(points_cam) >= 2:
        cv2.line(draw, tuple(points_cam[0]), tuple(current_point), (255, 165, 0), 2)
    if len(points_cam) == 4:
        cv2.line(draw, tuple(points_cam[3]), tuple(points_cam[0]), (0, 255, 0), 2)
        cv2.putText(draw, "S = salva   R = ricomincia",
                    (W_CAM//2 - 220, H_CAM - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("CAMERA_VIEW", draw)

    key = cv2.waitKey(16) & 0xFF

    if key == ord('r'):
        points_cam.clear()
        last_point = [0, 0]
        print("↺ Reset.")

    elif key == ord('s') and len(points_cam) == 4:

        pts_cam = np.float32(points_cam)

        # I 4 angoli corrispondenti nel proiettore (coord assolute proiettore)
        margin = 60
        pts_proj_corners = np.float32([
            [margin,          margin          ],   # Alto-SX
            [W_PROJ - margin, margin          ],   # Alto-DX
            [W_PROJ - margin, H_PROJ - margin ],   # Basso-DX
            [margin,          H_PROJ - margin ],   # Basso-SX
        ])

        # H_cam_to_proj: mappa coordinate camera → coordinate proiettore
        H_cam_to_proj = cv2.getPerspectiveTransform(pts_cam, pts_proj_corners)
        np.save("H_cam_to_proj.npy", H_cam_to_proj)
        print("✓ H_cam_to_proj.npy salvata!")

        # Converti i punti camera in coordinate proiettore
        pts_cam_h   = pts_cam.reshape(-1, 1, 2)
        pts_proj_h  = cv2.perspectiveTransform(pts_cam_h, H_cam_to_proj)
        pts_proj    = pts_proj_h.reshape(-1, 2)

        print("\nPunti convertiti in coordinate proiettore:")
        labels = ["Alto-SX", "Alto-DX", "Basso-DX", "Basso-SX"]
        for i, (px, py) in enumerate(pts_proj):
            print(f"  {labels[i]}: ({px:.0f}, {py:.0f})")

        # Rettangolo di destinazione — tutto il frame proiettore
        dst = np.float32([
            [0,      0      ],
            [W_PROJ, 0      ],
            [W_PROJ, H_PROJ ],
            [0,      H_PROJ ],
        ])

        # H_keystone: pre-distorce il contenuto proiettato
        H_keystone = cv2.getPerspectiveTransform(pts_proj, dst)
        np.save("keystone_matrix.npy", H_keystone)
        print("✓ keystone_matrix.npy salvata!")

        # Test visivo
        test = np.zeros((H_PROJ, W_PROJ, 3), dtype=np.uint8)
        cv2.rectangle(test, (50, 50), (W_PROJ-50, H_PROJ-50), (0, 255, 0), 6)
        cv2.line(test, (W_PROJ//2, 0), (W_PROJ//2, H_PROJ), (0, 100, 255), 2)
        cv2.line(test, (0, H_PROJ//2), (W_PROJ, H_PROJ//2), (0, 100, 255), 2)
        cv2.putText(test, "DEVE APPARIRE RETTANGOLARE",
                    (W_PROJ//2 - 350, H_PROJ//2 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        test_warped = cv2.warpPerspective(test, H_keystone, (W_PROJ, H_PROJ))
        cv2.imshow("PROIEZIONE_KEYSTONE", test_warped)

        print("\nTest proiettato — il rettangolo deve apparire dritto sul tavolo.")
        print("Premi un tasto per uscire.")
        cv2.waitKey(0)
        break

    elif key == 27:
        print("Uscita senza salvare.")
        break

cap.release()
cv2.destroyAllWindows()
print()
print("✓ Fatto. Esegui main.py")