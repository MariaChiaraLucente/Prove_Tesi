import cv2
import numpy as np
import cv2.aruco as aruco

# 1. Carica i parametri della camera (precedentemente salvati)
data = np.load('camera_params.npz')
mtx, dist = data['mtx'], data['dist']

# 2. Configurazione proiettore e ArUco
W_PROJ, H_PROJ = 1920, 1080
size = 200
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# Coordinate SORGENTE (i pixel esatti sul proiettore)
# Ordine: ID 0, 1, 2, 3 (Alto-SX, Alto-DX, Basso-DX, Basso-SX)
pts_projector = np.array([
    [0, 0], 
    [W_PROJ - 1, 0], 
    [W_PROJ - 1, H_PROJ - 1], 
    [0, H_PROJ - 1]
], dtype=np.float32)

cap = cv2.VideoCapture(0)

print("Inquadra la proiezione. Premi 's' per salvare l'omografia o 'q' per uscire.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Rimuovi distorsione dal frame della camera per precisione
    h, w = frame.shape[:2]
    new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0)
    frame_undistorted = cv2.undistort(frame, mtx, dist, None, new_mtx)

    # Rileva i marker nella camera
    corners, ids, _ = aruco.detectMarkers(frame_undistorted, aruco_dict, parameters=parameters)

    if ids is not None and len(ids) == 4:
        # Ordiniamo i punti rilevati per matchare pts_projector (ID 0,1,2,3)
        pts_camera = np.zeros((4, 2), dtype=np.float32)
        for i in range(len(ids)):
            idx = ids[i][0]
            if idx < 4:
                # Usiamo il centro del marker come punto di riferimento
                pts_camera[idx] = np.mean(corners[i][0], axis=0)
        
        # Disegna per feedback visivo
        aruco.drawDetectedMarkers(frame_undistorted, corners, ids)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            # CALCOLO OMOGRAFIA: Da Camera a Proiettore
            H, _ = cv2.findHomography(pts_camera, pts_projector)
            np.save("homography_matrix.npy", H)
            print("Matrice di Omografia salvata con successo!")
            break

    cv2.imshow('Calibrazione Sistema', frame_undistorted)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()