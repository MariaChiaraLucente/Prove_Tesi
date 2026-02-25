#ris camera
#ris crop
#altezza camera 
# conv cm to px
# mappatura posizione
# riconoscimento disco univoco--
        #possibile soluzione: riconoscere un triangolo 
        # riconoscimento tramite colore?
            #---> riconoscimento tramite marker colorato su forma

        # riconoscimento tramite marker personalizzato? 

#riconscimento delle mani, max 2, è possibile distinguere le mani di destra e di sinistra





from matplotlib.pyplot import hsv
from ultralytics import YOLO
import cv2
import numpy as np


def init_models():
    shapes_model = YOLO("../best.pt")
    return shapes_model

############################
# INIT CAMERA
############################
def init_camera():
    cap = cv2.VideoCapture(1)

    return cap

def detect_circle(frame, model, last_count=-1, target_reached=False):
    results = model(frame, stream=True, verbose=False)
    
    circles_found = 0
    detections = []

    for result in results:
        if result.boxes is None:
            continue
        # Per ogni box rilevato da yolo
        for box in result.boxes:
            # Seleziono quelli dei cerchi con confidenza maggiore di 0.6 
            if box.conf[0] > 0.6 and int(box.cls[0]) == 0:
                # Setto le coordinate del box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                circles_found += 1
                detections.append({
                    'id': circles_found,
                    'x1': x1, 'y1': y1,
                    'x2': x2, 'y2': y2,
                    'conf': confidence
                })
                
                draw_square(frame, x1, y1, x2, y2)
    
    # Stampa report solo se il conteggio è cambiato E non abbiamo ancora raggiunto il target
    if circles_found > 0 and circles_found != last_count and not target_reached:
        print(f"\n{'='*60}")
        print(f"[DETECTION REPORT] Cerchi rilevati: {circles_found}")
        print(f"{'='*60}")
        for det in detections:
            center_x = (det['x1'] + det['x2']) // 2
            center_y = (det['y1'] + det['y2']) // 2
            width = det['x2'] - det['x1']
            height = det['y2'] - det['y1']
            
            print(f"  Cerchio #{det['id']}:")
            print(f"    └─ Top-Left:     ({det['x1']:4d}, {det['y1']:4d})")
            print(f"    └─ Bottom-Right: ({det['x2']:4d}, {det['y2']:4d})")
            print(f"    └─ Centro:       ({center_x:4d}, {center_y:4d})")
            print(f"    └─ Dimensioni:   {width}x{height} px")
            print(f"    └─ Confidenza:   {det['conf']:.2%}")
            print()
    
    # Mostra counter sul frame
    color = (0, 255, 0) if circles_found < 4 else (0, 255, 255)  # Giallo quando raggiunge 4
    cv2.putText(frame, f"Circles: {circles_found}/4", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return frame, circles_found

def draw_square(frame, x1, y1, x2, y2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


def run():
    model_shapes = init_models()
    cap = init_camera()
    
    last_count = -1  # Per tracciare l'ultimo conteggio e evitare stampe ripetute
    target_reached = False  # Flag per sapere se abbiamo raggiunto 4 forme

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        square_frame, circles_found = detect_circle(frame, model_shapes, last_count, target_reached)
        cv2.imshow("Circle Detection", square_frame)
        
        # Aggiorna il conteggio e controlla se abbiamo raggiunto 4
        if circles_found != last_count:
            last_count = circles_found
            if circles_found == 4 and not target_reached:
                target_reached = True
                print(f"\n{'🎯'*30}")
                print("TARGET RAGGIUNTO! 4 cerchi rilevati!")
                print(f"{'🎯'*30}\n")

        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()


#quindi se io faccio il rilevamento di una forma con una percentuale di colore dentro, posso raffinare la rilevazione
# in questo caso ho messo la rilevazione di un cerchio o triangolo, ma il box viene disegnato solo se dentro che  un % di verde
#il verde viene riconosciuto con una maschera di colore, che ha un range in HSV. bianchi ci sono dentro il box del triangolo, se è superiore alla soglia allora disegno il box del triangolo.