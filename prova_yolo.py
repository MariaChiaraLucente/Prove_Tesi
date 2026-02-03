import cv2
from ultralytics import YOLO
from pythonosc import udp_client

# Configurazione OSC
client = udp_client.SimpleUDPClient("127.0.0.1", 7000)

# Modello YOLO
model = YOLO('yolo11n.pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Ottengo le dimensioni per la normalizzazione
    h_video, w_video, _ = frame.shape

    # PREDICT: Cerco SOLO persone (classes=[0]) per velocità
    results = model.predict(frame, stream=True, verbose=False, classes=[0])

    # Flag per sapere se abbiamo trovato qualcuno in questo frame
    persona_trovata = False

    for result in results:
        # Se ci sono box rilevati
        if len(result.boxes) > 0:
            # Prendiamo SOLO IL PRIMO box per evitare sfarfallii se ci sono 2 persone
            box = result.boxes[0] 
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Calcolo centro
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Disegno debug
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

            # --- NORMALIZZAZIONE ---
            norm_x = center_x / w_video
            norm_y = center_y / h_video

            # Invio OSC
            client.send_message("/user/pos", [norm_x, norm_y])
            print(f"OSC: {norm_x:.2f}, {norm_y:.2f}")

            persona_trovata = True
            
            # Importante: Interrompiamo il ciclo dei risultati. 
            # Tracciamo solo una persona alla volta per ora.
            break 
    
    # Opzionale: Se non trovo nessuno, potrei mandare un segnale per nascondere il cursore
    if not persona_trovata:
        # Esempio: manda coordinate negative o un messaggio di stato
        pass 

    cv2.imshow('YOLO Tracking', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()