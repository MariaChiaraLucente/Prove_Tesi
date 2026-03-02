import cv2
import numpy as np
import cv2.aruco as aruco
import time

# 1. Carica i parametri della camera (precedentemente salvati)
# non so se sono perfetti perche ho provato con quelli della scacchiera da telefono e le foto avevano distanza dalla camera diversa

# print("📍 Caricamento parametri camera...")
# data = np.load('camera_params.npz')
# mtx, dist = data['mtx'], data['dist']
# print(f"✓ PSarametri camera caricati: mtx shape={mtx.shape}, dist shape={dist.shape}")
# print(f"  Centro principale: ({mtx[0,2]:.1f}, {mtx[1,2]:.1f})")

# 2. Configurazione proiettore e ArUco
print("📍 Configurazione ArUco...")
W_PROJ, H_PROJ =  1920, 1200

# CONFIGURAZIONE GRIGLIA (Deve matchare ArucoCalib_pattern.py)
ROWS = 3
COLS = 4
MARKER_SIZE = 150
MARGIN_X = 100
MARGIN_Y = 80

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
print(f"✓ Dizionario ArUco caricato: DICT_6X6_250")
parameters = aruco.DetectorParameters()
print(f"✓ Parametri detector: OK")

# Genera mappa {ID: (x, y)} dei CENTRI dei marker sul proiettore
MARKER_MAP = {}
step_x = (W_PROJ - 2 * MARGIN_X - MARKER_SIZE) / (COLS - 1) if COLS > 1 else 0
step_y = (H_PROJ - 2 * MARGIN_Y - MARKER_SIZE) / (ROWS - 1) if ROWS > 1 else 0
mid_offset = MARKER_SIZE / 2.0

marker_count = 0
for r in range(ROWS):
    for c in range(COLS):
        x = MARGIN_X + c * step_x
        y = MARGIN_Y + r * step_y
        MARKER_MAP[marker_count] = (x + mid_offset, y + mid_offset)
        marker_count += 1

print(f"✓ Configurazione attesa: {len(MARKER_MAP)} marker in griglia {ROWS}x{COLS}")

cap = cv2.VideoCapture(0)

print("📍 Apertura camera...")
if not cap.isOpened():
    print("❌ Camera 1 non disponibile, provo camera 0...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ Nessuna camera disponibile!")
        exit(1)

# Imposta risoluzione camera a 1280x720
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Get camera properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"✓ Camera aperta: {frame_width}x{frame_height} @ {fps} FPS")
print(f"  (Risoluzione richiesta: {CAMERA_WIDTH}x{CAMERA_HEIGHT})")

print("\n📍 Inizio acquisizione...")
print("Inquadra la proiezione. Premi 's' per salvare l'omografia o 'q' per uscire.\n")

frame_count = 0
start_time = time.time()
markers_found_count = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Flip frame (-1 = flip sia orizzontale che verticale)
    frame = cv2.flip(frame, -1)

    # Rimuovi distorsione dal frame della camera per precisione
    # h, w = frame.shape[:2]
    # new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0)
    # frame_undistorted = cv2.undistort(frame, mtx, dist, None, new_mtx)

    # Rileva i marker nella camera
    corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    
    # Log sulla rilevazione
    if ids is not None:
        detected_count = len(ids)
        # detected_ids = sorted(ids.flatten().tolist())
        # print(f"  ✓ Marker rilevati: {detected_ids} ({detected_count}/{len(MARKER_MAP)})")
        if detected_count >= 4:
            markers_found_count += 1
    else:
        if frame_count % 60 != 0:
            print(f"  ✗ Nessun marker rilevato")

    # Raccogli i punti validi
    valid_pts_camera = []
    valid_pts_projector = []

    if ids is not None:
        for i in range(len(ids)):
            mid = ids[i][0]
            if mid in MARKER_MAP:
                center = np.mean(corners[i][0], axis=0)
                valid_pts_camera.append(center)
                valid_pts_projector.append(MARKER_MAP[mid])
        
    if len(valid_pts_camera) >= 4:
        # Disegna per feedback visivo
        aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.putText(frame, f"Marker validi: {len(valid_pts_camera)}/{len(MARKER_MAP)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Calibrazione Sistema', frame)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):
        if len(valid_pts_camera) >= 4:
            print(f"\n📍 Calcolo omografia con {len(valid_pts_camera)} punti...")
            
            src_pts = np.array(valid_pts_camera, dtype=np.float32)
            dst_pts = np.array(valid_pts_projector, dtype=np.float32)

            # CALCOLO OMOGRAFIA: Da Camera a Proiettore
            # Usa RANSAC per robustezza (ignora outlier)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            # --- VERIFICA PRECISIONE (Errore di Riproiezione) ---
            # Proiettiamo i punti camera usando H e vediamo quanto distano dai punti reali del proiettore
            pts_proj_estimated = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
            error = np.mean(np.linalg.norm(dst_pts - pts_proj_estimated, axis=1))
            
            print(f"✓ Matrice di Omografia salvata: homography_matrix.npy")
            print(f"  ★ Errore medio di riproiezione: {error:.2f} pixel")
            if error < 3.0: print("    -> Calibrazione ECCELLENTE.")
            elif error < 6.0: print("    -> Calibrazione BUONA.")
            else: print("    -> ATTENZIONE: Errore alto. Verifica distorsione lente o planarità superficie.")

            np.save("homography_matrix.npy", H)
            print(f"  Punti usati: {np.sum(mask)}/{len(src_pts)}")
            print(f"\nCalibrazione completata in {time.time() - start_time:.1f}s")

            # ── CALCOLO CROP tramite omografia inversa ───────────────────

            corners_proj = np.array([
                [[0,       0      ]],   # Alto-SX
                [[W_PROJ,  0      ]],   # Alto-DX
                [[W_PROJ,  H_PROJ ]],   # Basso-DX
                [[0,       H_PROJ ]]    # Basso-SX
            ], dtype='float32')

            H_inv = np.linalg.inv(H)
            corners_cam = cv2.perspectiveTransform(corners_proj, H_inv)

            X1 = int(np.min(corners_cam[:, 0, 0]))
            Y1 = int(np.min(corners_cam[:, 0, 1]))
            X2 = int(np.max(corners_cam[:, 0, 0]))
            Y2 = int(np.max(corners_cam[:, 0, 1]))

            # Clamp ai limiti del frame
            X1 = max(0, X1)
            Y1 = max(0, Y1)
            X2 = min(frame_width,  X2)
            Y2 = min(frame_height, Y2)

            np.save("crop_params.npy", np.array([X1, Y1, X2, Y2]))
            print(f"✓ crop_params.npy salvata")
            print(f"  X1={X1}, Y1={Y1}, X2={X2}, Y2={Y2}")
            print(f"  Dimensioni crop: {X2-X1} x {Y2-Y1} px")

            print(f"\n✓ Calibrazione completata in {time.time()-start_time:.1f}s")
            break
        else:
            print(f"\n❌ Trovati solo {len(valid_pts_camera)} marker. Ne servono almeno 4.")
            
    elif key & 0xFF == ord('q'):
        print("\n📍 Uscita...")
        break

cap.release()
cv2.destroyAllWindows()
print("\n✓ Programma terminato")