import cv2
from cv2 import aruco

# Scegli il dizionario
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# Crea la board: 7 quadrati orizzontali, 5 verticali
# Dimensioni esempio: quadrato 0.04m (4cm), marker 0.02m (2cm)
board = aruco.CharucoBoard((7, 5), 0.04, 0.02, aruco_dict)

# Disegna e salva l'immagine per la stampa
img = board.generateImage((2000, 1500))
cv2.imwrite("mia_charuco_board.png", img)

