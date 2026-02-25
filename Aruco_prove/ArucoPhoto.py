import cv2
import os

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Cannot open camera")
    
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

img_counter = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    cv2.imshow('Webcam', frame)
    
    k = cv2.waitKey(1)
    
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == ord('s'):
        # 's' pressed
        img_name = os.path.join(output_dir, "frame_{}.png".format(img_counter))
        cv2.imwrite(img_name, frame)
        print(f"{img_name} saved!")
        img_counter += 1

cap.release()
cv2.destroyAllWindows()

