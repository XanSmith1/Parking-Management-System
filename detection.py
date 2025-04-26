# detection.py

from CustomParkingManagement import CustomParkingManagement
import cv2
import numpy as np

k = 0.75
device = 'cpu'

management = CustomParkingManagement(
    model_path="yolov8n.pt",
    json_file="bounding_boxes.json",
    bg_color=(255, 255, 255),
    occ=(0, 0, 255),
    acr=(0, 255, 0)
)

cap = cv2.VideoCapture("new_video.MOV")

def run_detection_once():
    ret, im0 = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return None, None

    im0 = cv2.resize(im0, (int(im0.shape[1]*k), int(im0.shape[0]*k)))

    results = management.model.track(im0, persist=True, show=False,
                                     imgsz=im0.shape[:2], classes=[2, 7],
                                     conf=0.2, iou=0.1, device=device)

    if results and results[0].boxes:
        management.boxes = results[0].boxes.xyxy.cpu().numpy()
        management.clss = results[0].boxes.cls.cpu().numpy()
    else:
        management.boxes = []
        management.clss = []

    frame = management.process_data(im0)

    total = len(management.json)
    occupied = management.pr_info.get("Occupancy", 0)
    available = management.pr_info.get("Available", 0)

    slots = []
    for i in range(total):
        status = "occupied" if i < occupied else "available"
        slots.append({"id": i+1, "status": status})

    status_json = {
        "occupied": occupied,
        "available": available,
        "Space": slots
    }

    return frame, status_json
def generate_detected_frames():
    while True:
        ret, im0 = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        im0 = cv2.resize(im0, (int(im0.shape[1] * k), int(im0.shape[0] * k)))

        results = management.model.track(im0, persist=True, show=False,
                                         imgsz=im0.shape[:2], classes=[2, 7],
                                         conf=0.2, iou=0.1, device=device)

        if results and results[0].boxes:
            management.boxes = results[0].boxes.xyxy.cpu().numpy()
            management.clss = results[0].boxes.cls.cpu().numpy()
        else:
            management.boxes = []
            management.clss = []

        frame = management.process_data(im0)

        # Encode frame to JPEG
        success, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

