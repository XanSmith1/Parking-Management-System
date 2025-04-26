
from ultralytics import YOLO
import cv2

# Load your trained YOLO model
model = YOLO("best.pt")  # Ensure this is the correct model path

# Test on an image
image_path = "/Users/xandersmith/Desktop/test_vid/test2.png"  # Replace with an image from your dataset
image = cv2.imread(image_path)

# Run YOLO detection
results = model.predict(image, conf=0.25)

# Draw bounding boxes on the image
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls)  # Class index
        label = "Vacant" if cls == 0 else "Occupied"
        color = (0, 255, 0) if cls == 0 else (0, 0, 255)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Save the processed image with detections
output_path = "/Users/xandersmith/Desktop/test_vid/test2.png"
cv2.imwrite(output_path, image)
print(f"✅ Detection complete! Saved as '{output_path}'")

# Display the detected image (optional)
cv2.imshow("YOLO Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
