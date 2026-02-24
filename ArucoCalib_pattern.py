import cv2
import numpy as np
import cv2.aruco as aruco

# Risoluzione del tuo proiettore (cambiala se diversa)
W_PROJ, H_PROJ = 1920, 1080

# Crea un'immagine bianca
img = np.ones((H_PROJ, W_PROJ, 3), dtype=np.uint8) * 255
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Inseriamo 4 marker agli angoli (ID 0, 1, 2, 3)
# Dimensione marker: 200px
size = 200
# Angoli: alto-sx, alto-dx, basso-dx, basso-sx
positions = [(0,0), (W_PROJ-size, 0), (W_PROJ-size, H_PROJ-size), (0, H_PROJ-size)]

for i, pos in enumerate(positions):
    marker_img = aruco.generateImageMarker(aruco_dict, i, size)
    marker_rgb = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
    img[pos[1]:pos[1]+size, pos[0]:pos[0]+size] = marker_rgb

cv2.imwrite("calib_pattern.png", img)
print("Immagine 'calib_pattern.png' creata. Proiettala a tutto schermo.")