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

size = 200
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
print(f"✓ Dizionario ArUco caricato: DICT_6X6_250")
parameters = aruco.DetectorParameters()
print(f"✓ Parametri detector: OK")

# Coordinate SORGENTE (i pixel esatti sul proiettore)
# Ordine: ID 0, 1, 2, 3 (Alto-SX, Alto-DX, Basso-DX, Basso-SX)
pts_projector = np.array([
    [20, 20],
    [W_PROJ - 21, 20],
    [W_PROJ - 21, H_PROJ - 21],
    [20, H_PROJ - 21]
], dtype=np.float32)

cap = cv2.VideoCapture(1)

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
        detected_ids = sorted(ids.flatten().tolist())
        print(f"  ✓ Marker rilevati: {detected_ids} ({detected_count}/4)")
        if detected_count == 4:
            markers_found_count += 1
    else:
        if frame_count % 60 != 0:
            print(f"  ✗ Nessun marker rilevato")

    if ids is not None and len(ids) == 4:
        # Ordiniamo i punti rilevati per matchare pts_projector (ID 0,1,2,3)
        pts_camera = np.zeros((4, 2), dtype=np.float32)
        for i in range(len(ids)):
            idx = ids[i][0]
            if idx < 4:
                # Usiamo il centro del marker come punto di riferimento
                pts_camera[idx] = np.mean(corners[i][0], axis=0)
                print(f"    - ID {idx}: ({pts_camera[idx][0]:.1f}, {pts_camera[idx][1]:.1f})")
        
        # Disegna per feedback visivo
        aruco.drawDetectedMarkers(frame, corners, ids)
        print("  ★ Tutti i 4 marker rilevati - puoi premere 's' per salvare")

    cv2.imshow('Calibrazione Sistema', frame)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):
        if ids is not None and len(ids) == 4:
            print("\n📍 Calcolo omografia...")
            # CALCOLO OMOGRAFIA: Da Camera a Proiettore
            H, _ = cv2.findHomography(pts_camera, pts_projector)
            np.save("homography_matrix.npy", H)
            print(f"✓ Matrice di Omografia salvata: homography_matrix.npy")
            print(f"  Shape: {H.shape}, Determinante: {np.linalg.det(H):.6f}")
            print(f"\nCalibrazione completata in {time.time() - start_time:.1f}s")

            # ── CALCOLO CROP tramite omografia inversa ───────────────────
            # I bordi reali della proiezione in pixel proiettore sono (0,0)-(W,H).
            # Usiamo H_inv per mapparli nel frame camera → crop esatto.

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
            print("\n❌ Non tutti i 4 marker sono visibili, riprova.")
            break
    elif key & 0xFF == ord('q'):
        print("\n📍 Uscita...")
        break

cap.release()
cv2.destroyAllWindows()
print("\n✓ Programma terminato")