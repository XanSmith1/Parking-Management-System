import cv2
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO("best.pt")  # Ensure best.pt is in the same directory or provide the full path

# Path to the test image
image_path = "/Users/xandersmith/Desktop/test_vid/2012-09-14_12_11_19_.jpg"  # Replace with your actual image file path
output_path = "/Users/xandersmith/Desktop/test_vid/2012-09-14_12_11_19_.jpg"  # Path to save the annotated image

# Read the image
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("❌ Error: Image not found. Check the file path.")

# Resize image to match YOLO model input size (optional)
image_resized = cv2.resize(image, (640, 640))

# Run YOLOv8 inference
results = model.predict(image_resized, conf=0.25)

# Draw bounding boxes and labels
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
        cls = int(box.cls)  # Class index (0 for vacant, 1 for occupied)
        label = "Vacant" if cls == 0 else "Occupied"
        color = (0, 255, 0) if cls == 0 else (0, 0, 255)  # Green for vacant, Red for occupied

        # Draw bounding box and label
        cv2.rectangle(image_resized, (x1, y1), (x2, y2), color, 2)


# Save the processed image
cv2.imwrite(output_path, image_resized)
print(f"✅ Detection complete! Processed image saved as '{output_path}'.")

# Show the processed image (optional)
cv2.imshow("Parking Lot Detection", image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
