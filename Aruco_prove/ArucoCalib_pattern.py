import cv2
import numpy as np
import cv2.aruco as aruco

# Risoluzione del tuo proiettore (cambiala se diversa)
W_PROJ, H_PROJ = 1920, 1200

# CONFIGURAZIONE GRIGLIA
ROWS = 3       # Numero di righe
COLS = 4       # Numero di colonne
MARKER_SIZE = 150
MARGIN_X = 100
MARGIN_Y = 80

img = np.ones((H_PROJ, W_PROJ, 3), dtype=np.uint8) * 255
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)  # DEVE CORRISPONDERE a ArucoCreation.py!

# Calcola spaziatura
step_x = (W_PROJ - 2 * MARGIN_X - MARKER_SIZE) / (COLS - 1) if COLS > 1 else 0
step_y = (H_PROJ - 2 * MARGIN_Y - MARKER_SIZE) / (ROWS - 1) if ROWS > 1 else 0

marker_id = 0
for r in range(ROWS):
    for c in range(COLS):
        # Calcola posizione top-left
        x = int(MARGIN_X + c * step_x)
        y = int(MARGIN_Y + r * step_y)
        
        marker_img = aruco.generateImageMarker(aruco_dict, marker_id, MARKER_SIZE)
        marker_rgb = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        img[y:y+MARKER_SIZE, x:x+MARKER_SIZE] = marker_rgb
        
        # Scrivi ID per debug
        cv2.putText(img, str(marker_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50,50,50), 2)
        marker_id += 1

cv2.imwrite("calib_pattern_grid.png", img)
print(f"Immagine 'calib_pattern_grid.png' creata con {ROWS*COLS} marker.")