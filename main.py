from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)

    fire_detected = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]

            if name == "fire":
                fire_detected = True
                print("🔥 FIRE DETECTED !!!")

    cv2.imshow("Fire Detection", results[0].plot())

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
