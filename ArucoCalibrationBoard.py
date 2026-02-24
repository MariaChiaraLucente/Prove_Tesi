import cv2
import cv2.aruco as aruco
import numpy as np
import os

# --- 1. CONFIGURAZIONE DELLA TUA BOARD ---
# Modifica questi parametri in base alla tua board fisica
CHARUCO_BOARD_SHAPE = (7, 5)  # Numero di quadrati (X, Y) - CORRETTO dalle immagini!
SQUARE_LENGTH = 0.01        # Dimensione lato quadrato in metri (es. 4cm)
MARKER_LENGTH = 0.006         # Dimensione lato marker ArUco in metri (es. 2cm)
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# Cartella contenente le tue foto (es. .jpg o .png)
IMAGE_FOLDER = 'C:\\Users\\OddXeon\\Desktop\\Progetto_tesi_MCL\\clone\\Prove_Tesi\\output_images'

# Crea l'oggetto Board
board = aruco.CharucoBoard(CHARUCO_BOARD_SHAPE, SQUARE_LENGTH, MARKER_LENGTH, ARUCO_DICT)
detector = aruco.CharucoDetector(board)

all_charuco_corners = []
all_charuco_ids = []
image_size = None

# --- 2. RILEVAMENTO MARKER NELLE IMMAGINI ---
images = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]

print(f"Inizio elaborazione di {len(images)} immagini...")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if image_size is None:
        image_size = gray.shape[::-1] # (larghezza, altezza)

    # Rileva scacchiera Charuco
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    # DEBUG: aggiungiamo informazioni dettagliate
    num_markers = len(marker_ids) if marker_ids is not None else 0
    num_charuco = len(charuco_ids) if charuco_ids is not None else 0
    
    print(f"📁 {os.path.basename(fname)}:")
    print(f"   ArUco marker: {num_markers}")
    print(f"   ChArUco angoli: {num_charuco}")
    
    if marker_ids is not None:
        print(f"   Marker IDs: {marker_ids.flatten()}")
    if charuco_ids is not None:
        print(f"   ChArUco IDs: {charuco_ids.flatten()}")

    # Se troviamo abbastanza angoli (almeno 4), li aggiungiamo per la calibrazione
    if charuco_ids is not None and len(charuco_ids) > 4:
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        print(f"✅ Immagine accettata!")
    else:
        print(f"❌ Immagine scartata (serve >4 angoli ChArUco)")
    print()

# --- 3. CALIBRAZIONE EFFETTIVA ---
if len(all_charuco_corners) > 0:
    print("\nCalibrazione in corso...")
    
    # Calibra la camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    print("\n--- RISULTATI CALIBRAZIONE ---")
    print(f"Errore di riproiezione: {ret}")
    print("\nMatrice della Camera (Parametri Intrinseci):")
    print(camera_matrix)
    print("\nCoefficienti di Distorsione:")
    print(dist_coeffs)

    # SALVATAGGIO DEI PARAMETRI
    # È fondamentale salvarli per usarli nei tuoi progetti futuri
    np.savez("camera_params.npz", mtx=camera_matrix, dist=dist_coeffs)
    print("\nParametri salvati in 'camera_params.npz'")

else:
    print("Errore: Non sono stati trovati abbastanza marker nelle immagini per procedere.")