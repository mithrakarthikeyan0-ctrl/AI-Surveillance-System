import torch
import cv2
import datetime
import os

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Objects we want to detect
TARGET_OBJECTS = ["person"]

# Create folder to save detections
if not os.path.exists("detections"):
    os.makedirs("detections")


# Function to log events
def log_event(event):
    with open("events.log", "a") as file:
        file.write(f"{datetime.datetime.now()} - {event}\n")


# Start webcam
cap = cv2.VideoCapture(0)

print("AI Smart Surveillance System Started")
print("Press ESC to exit")

while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to access camera")
        break

    # Run YOLO detection
    results = model(frame)

    # Convert results to dataframe
    detections = results.pandas().xyxy[0]

    for index, row in detections.iterrows():

        label = row['name']
        confidence = row['confidence']

        # Check if object is in our target list
        if label in TARGET_OBJECTS and confidence > 0.5:

            message = f"{label} detected with confidence {confidence:.2f}"
            print(message)

            # Log event
            log_event(message)

            # Save screenshot
            filename = f"detections/{label}_{int(datetime.datetime.now().timestamp())}.jpg"
            cv2.imwrite(filename, frame)

    # Draw detection boxes
    results.render()

    # Display video
    cv2.imshow("AI Smart Surveillance", frame)

    # Exit if ESC pressed
    if cv2.waitKey(1) == 27:
        break


# Release camera
cap.release()
cv2.destroyAllWindows()

print("System Stopped")
