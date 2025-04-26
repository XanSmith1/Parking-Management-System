from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load YOLO model

# Start training with the corrected dataset
model.train(
    data=r'/Users/xandersmith/Desktop/UTL(ParkMngmnt)FDS/full_DS/DS.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device='cuda',
    name="parking_model"
)
