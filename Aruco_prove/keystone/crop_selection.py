#da fare prima di tutte coe
#seleziono l'aria da cropare, e salvo i parametri in un file .npy
# quest'arai deve contenere la proiezione anche se distorta


import cv2
import numpy as np

############################
# INIT CAMERA
############################
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Camera non disponibile!")
    exit(1)

frame = cv2.flip(frame, -1)

############################
# SELEZIONE ROI
############################
print("=" * 50)
print("  SELEZIONE CROP — Zona proiezione")
print("=" * 50)
print()
print("Trascina un rettangolo attorno alla zona proiettata.")
print("SPAZIO o INVIO per confermare, C per annullare.")
print()

rect = cv2.selectROI("Seleziona zona proiezione", frame, fromCenter=False)
cv2.destroyAllWindows()

x, y, w, h = rect

if w == 0 or h == 0:
    print("❌ Selezione annullata.")
    exit(1)

np.save("crop_params.npy", np.array([x, y, x + w, y + h]))

print(f"✓ crop_params.npy salvata!")
print(f"  Angolo TL: ({x}, {y})")
print(f"  Angolo BR: ({x+w}, {y+h})")
print(f"  Dimensioni: {w} x {h} px")
print()
print("Prossimo passo: esegui calibrazione_keystone.py")