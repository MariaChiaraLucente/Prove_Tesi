import cv2
import cv2.aruco as aruco
import numpy as np

# Carica la prima immagine per vedere cosa contiene
img_path = 'C:\\Users\\OddXeon\\Desktop\\Progetto_tesi_MCL\\clone\\Prove_Tesi\\output_images\\frame_0.png'
img = cv2.imread(img_path)

if img is None:
    print("❌ Impossibile caricare l'immagine")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("🔍 Analizzando contenuto immagine...")
print(f"Dimensioni: {img.shape}")

# Test diversi tipi di rilevamento
print("\n=== TEST 1: ArUco marker ===")
dictionaries_to_test = [
    ("DICT_4X4_50", aruco.DICT_4X4_50),
    ("DICT_5X5_100", aruco.DICT_5X5_100), 
    ("DICT_6X6_250", aruco.DICT_6X6_250),
    ("DICT_7X7_250", aruco.DICT_7X7_250)
]

for name, dict_type in dictionaries_to_test:
    aruco_dict = aruco.getPredefinedDictionary(dict_type)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    
    corners, ids, rejected = detector.detectMarkers(gray)
    num_markers = len(ids) if ids is not None else 0
    
    print(f"  {name}: {num_markers} marker")
    
    if num_markers > 0:
        print(f"    IDs trovati: {ids.flatten() if ids is not None else 'None'}")

print("\n=== TEST 2: Scacchiera normale ===")
# Test per scacchiera normale (senza ArUco)
for size in [(7, 5), (8, 6), (9, 7), (10, 8), (6, 4)]:
    ret, corners = cv2.findChessboardCorners(gray, size, None)
    if ret:
        print(f"✅ Trovata scacchiera {size[0]}x{size[1]} - {len(corners)} angoli")
    else:
        print(f"❌ Nessuna scacchiera {size[0]}x{size[1]}")

print("\n=== VISUALIZZAZIONE ===")
# Mostra l'immagine per ispezione visiva
cv2.imshow("Immagine da analizzare", img)
cv2.imshow("Immagine in grigio", gray)

print("Premi un tasto per chiudere le finestre...")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n=== CONCLUSIONI ===")
print("Analizza l'immagine mostrata per capire cosa contiene.")
print("Se vedi una scacchiera normale -> usa calibrazione con cv2.calibrateCamera")
print("Se vedi marker ArUco -> verifica quale dizionario funziona")