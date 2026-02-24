import cv2
import numpy as np

# 1. Carica i parametri della calibrazione
try:
    with np.load('camera_params.npz') as data:
        mtx = data['mtx']
        dist = data['dist']
    print("Parametri caricati con successo!")
except FileNotFoundError:
    print("Errore: file 'camera_params.npz' non trovato. Calibra prima la camera!")
    exit()

# 2. Avvia la webcam (0 è solitamente la camera integrata)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Errore: Impossibile aprire la webcam.")
    exit()

print("Webcam avviata. Premi 'q' per uscire.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # 3. Ottieni la nuova matrice per ottimizzare la visualizzazione
    # alpha=0 taglia i bordi neri, alpha=1 mostra tutta l'immagine stirata
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

    # 4. Applica l'undistort al frame attuale
    dst = cv2.undistort(frame, mtx, dist, None, new_camera_mtx)

    # (Opzionale) Ritaglio basato sulla ROI per pulire i bordi
    x, y, w_roi, h_roi = roi
    if w_roi > 0 and h_roi > 0:
        dst = dst[y:y+h_roi, x:x+w_roi]

    # 5. Mostra i risultati
    # Mettiamo i due video a confronto (opzionale, se hanno dimensioni diverse fallirà il concat)
    cv2.imshow('Webcam Originale', frame)
    cv2.imshow('Webcam Corretta (Undistorted)', dst)

    # Esci se premi il tasto 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()