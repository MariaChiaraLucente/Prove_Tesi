#prova con calibrazione keystone

import cv2
import numpy as np
import cv2.aruco as aruco
import time

############################
# CONFIG
############################
W_PROJ, H_PROJ = 1920, 1200

############################
# CARICA KEYSTONE
############################
try:
    H_keystone = np.load('keystone_matrix.npy')
    H_keystone_inv = np.linalg.inv(H_keystone)
    print("✓ keystone_matrix.npy caricata!")
except:
    print("❌ keystone_matrix.npy mancante — esegui prima calibrazione_keystone.py!")
    exit(1)

############################
# CARICA CROP
############################
try:
    crop_data = np.load('crop_params.npy')
    X1_ROI, Y1_ROI, X2_ROI, Y2_ROI = map(int, crop_data)
    print(f"✓ crop_params.npy caricata: ({X1_ROI},{Y1_ROI}) → ({X2_ROI},{Y2_ROI})")
except:
    print("❌ crop_params.npy mancante — esegui prima select_crop.py!")
    exit(1)

############################
# CALCOLA pts_projector
# dalla keystone inversa
############################
# Gli angoli del rettangolo visibile in coordinate proiettore
# non sono più (0,0)-(1920,1200) perché il keystone ha distorto tutto.
# Usiamo la keystone inversa per trovare i pixel proiettore
# che corrispondono agli angoli del rettangolo corretto.
corners_rect = np.array([
    [[0,      0      ]],
    [[W_PROJ, 0      ]],
    [[W_PROJ, H_PROJ ]],
    [[0,      H_PROJ ]]
], dtype='float32')

corners_proj = cv2.perspectiveTransform(corners_rect, H_keystone_inv)
pts_projector = corners_proj.reshape(4, 2).astype(np.float32)
print(f"✓ pts_projector calcolati dalla keystone inversa:")
print(f"  TL: ({pts_projector[0][0]:.1f}, {pts_projector[0][1]:.1f})")
print(f"  TR: ({pts_projector[1][0]:.1f}, {pts_projector[1][1]:.1f})")
print(f"  BR: ({pts_projector[2][0]:.1f}, {pts_projector[2][1]:.1f})")
print(f"  BL: ({pts_projector[3][0]:.1f}, {pts_projector[3][1]:.1f})")

############################
# ARUCO CONFIG
############################
print("\n📍 Configurazione ArUco...")
MARKER_SIZE = 150
MARGIN = 60

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
print("✓ Dizionario ArUco: DICT_6X6_250")

############################
# CREA FRAME MARKER
############################
def make_marker_frame():
    """Crea il frame con i 4 marker ArUco agli angoli, poi applica keystone."""
    frame = np.zeros((H_PROJ, W_PROJ, 3), dtype=np.uint8)

    positions = [
        (MARGIN, MARGIN),                                              # ID 0 — TL
        (W_PROJ - MARGIN - MARKER_SIZE, MARGIN),                      # ID 1 — TR
        (W_PROJ - MARGIN - MARKER_SIZE, H_PROJ - MARGIN - MARKER_SIZE), # ID 2 — BR
        (MARGIN, H_PROJ - MARGIN - MARKER_SIZE)                       # ID 3 — BL
    ]

    for i, (px, py) in enumerate(positions):
        marker_img = aruco.generateImageMarker(aruco_dict, i, MARKER_SIZE)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        frame[py:py + MARKER_SIZE, px:px + MARKER_SIZE] = marker_bgr
        cv2.putText(frame, f"ID{i}", (px, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)

    # Applica keystone — così i marker appaiono negli angoli
    # del rettangolo corretto sulla parete
    return cv2.warpPerspective(frame, H_keystone, (W_PROJ, H_PROJ))

############################
# INIT CAMERA
############################
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ Camera non disponibile!")
    exit(1)

CAMERA_WIDTH  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
CAMERA_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"✓ Camera aperta: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")

############################
# FINESTRA PROIETTORE
############################
cv2.namedWindow("PROIEZIONE_ARUCO", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("PROIEZIONE_ARUCO", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("PROIEZIONE_ARUCO", make_marker_frame())

print("\n📍 Marker proiettati sul rettangolo corretto.")
print("Inquadra la parete. Premi 'S' per salvare, 'Q' per uscire.\n")

start_time = time.time()
pts_camera = None

############################
# LOOP
############################
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, -1)
    display = frame.copy()

    # Rileva marker nel frame intero (non croppato —
    # il crop verrà usato solo da main.py)
    corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids is not None:
        detected_ids = sorted(ids.flatten().tolist())
        aruco.drawDetectedMarkers(display, corners, ids)

        if len(ids) == 4 and all(i in detected_ids for i in [0, 1, 2, 3]):
            pts_camera = np.zeros((4, 2), dtype=np.float32)
            for i in range(len(ids)):
                idx = ids[i][0]
                if idx < 4:
                    pts_camera[idx] = np.mean(corners[i][0], axis=0)
                    cx, cy = int(pts_camera[idx][0]), int(pts_camera[idx][1])
                    cv2.circle(display, (cx, cy), 8, (0, 255, 0), -1)
                    cv2.putText(display, f"ID{idx} ({cx},{cy})", (cx + 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            cv2.putText(display, "✓ Tutti i 4 marker — premi S per salvare",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(display, f"Marker rilevati: {detected_ids} — servono 0,1,2,3",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    else:
        cv2.putText(display, "Nessun marker rilevato",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Mostra anche il crop per riferimento
    cv2.rectangle(display, (X1_ROI, Y1_ROI), (X2_ROI, Y2_ROI), (0, 165, 255), 2)

    cv2.imshow("CAMERA — Calibrazione ArUco", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if pts_camera is not None:
            print("\n📍 Calcolo omografia...")

            # Omografia: da coordinate camera → coordinate proiettore
            # pts_camera  = dove la camera vede i marker
            # pts_projector = dove si trovano quei marker sul proiettore
            #                 (calcolati dalla keystone inversa, non hardcoded)
            H, _ = cv2.findHomography(pts_camera, pts_projector)
            np.save("homography_matrix.npy", H)
            print(f"✓ homography_matrix.npy salvata!")
            print(f"  Determinante: {np.linalg.det(H):.6f}")

            # NON riscriviamo crop_params.npy — è già stato calcolato
            # da select_crop.py in modo indipendente dalla omografia

            print(f"\n✓ Calibrazione completata in {time.time() - start_time:.1f}s")
            print("Prossimo passo: esegui select_play_rect.py")
            break
        else:
            print("❌ Non tutti i 4 marker sono visibili. Riprova.")

    elif key == ord('q'):
        print("Uscita senza salvare.")
        break

cap.release()
cv2.destroyAllWindows()
print("\n✓ Programma terminato")