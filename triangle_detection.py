from ultralytics import YOLO
import cv2

# ROI
X1_ROI, Y1_ROI = 233, 29
X2_ROI, Y2_ROI = 1089, 552

def init_models():
    shapes_model = YOLO("best.pt")

    return shapes_model

############################
# INIT CAMERA
############################
def init_camera():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


############################
# PREPROCESS FRAME
############################
def preprocess_frame(frame):
    frame = cv2.flip(frame, -1)
    cropped = frame[Y1_ROI:Y2_ROI, X1_ROI:X2_ROI]
    return cropped


def detect_triangle(frame, model):
    results = model(frame, stream=True, verbose=False)

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            if box.conf[0] > 0.6 and int(box.cls[0]) == 2:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                draw_triangle(frame, x1, y1, x2, y2)
                return frame

    return frame

def draw_triangle(frame, x1, y1, x2, y2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # cv2.circle(frame, (cx, cy), 3, (255, 0, 0), cv2.FILLED)

def run():
    model_shapes = init_models()
    cap = init_camera()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cropped = preprocess_frame(frame)
        triangle_frame = detect_triangle(cropped, model_shapes)
        cv2.imshow("Triangle Detection", triangle_frame)

        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()