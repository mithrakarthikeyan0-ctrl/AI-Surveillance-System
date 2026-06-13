import torch
import cv2
import datetime
import os
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
TARGET_OBJECTS = ["person"]
if not os.path.exists("detections"):
    os.makedirs("detections")
def log_event(event):
    with open("events.log", "a") as file:
        file.write(f"{datetime.datetime.now()} - {event}\n")
cap = cv2.VideoCapture(0)

print("AI Smart Surveillance System Started")
print("Press ESC to exit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to access camera")
        break
    results = model(frame)
    detections = results.pandas().xyxy[0]

    for index, row in detections.iterrows():

        label = row['name']
        confidence = row['confidence']
        if label in TARGET_OBJECTS and confidence > 0.5:

            message = f"{label} detected with confidence {confidence:.2f}"
            print(message)
            log_event(message)
            filename = f"detections/{label}_{int(datetime.datetime.now().timestamp())}.jpg"
            cv2.imwrite(filename, frame)
    results.render()
    cv2.imshow("AI Smart Surveillance", frame)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()

print("System Stopped")
