import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for built-in webcam, 1 for external cam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Convert frame to YOLO input format
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    # Process detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]  # Object class scores
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Adjust threshold as needed
                # Get object bounding box
                center_x, center_y, w, h = (obj[:4] * [width, height, width, height]).astype(int)
                x, y = center_x - w // 2, center_y - h // 2

                # Draw bounding box
                color = (0, 255, 0)  # Green for detected objects
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Display class label
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show frame
    cv2.imshow("Real-Time Object Detection", frame)

    # Move the exit condition **outside the detection loops**
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
