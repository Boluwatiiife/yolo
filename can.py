import cv2
import urllib.request
import numpy as np
import torch
import requests

url = 'http://192.168.0.3/cam-hi.jpg'

# Initialize YOLOv5 model with best.pt
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# ESP32 URL to send detection signals
esp_url = 'http://192.168.0.3/detect'

# Create a VideoCapture object
cap = cv2.VideoCapture(url)

# Check if the IP camera stream is opened successfully
if not cap.isOpened():
    print("Failed to open the IP camera stream")
    exit()

# Read and display video frames
while True:
    # Read a frame from the video stream
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    im = cv2.imdecode(imgnp, -1)

    # Perform object detection on the frame
    results = model(im)

    # Filter results based on confidence threshold and class name
    detected_cans = []
    for *box, conf, cls in results.xyxy[0]:
        if conf > 0.6 and model.names[int(cls)] == "can":
            detected_cans.append((box, conf, cls))
            # Draw the bounding box
            cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            # Put the label with the confidence score
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.putText(im, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Check if any "can" object is detected with confidence above 0.8
    if len(detected_cans) > 0:
        # Send a signal to the ESP32
        requests.get(esp_url)

    # Display the results
    cv2.imshow('live Cam Testing', im)
    key = cv2.waitKey(5)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
