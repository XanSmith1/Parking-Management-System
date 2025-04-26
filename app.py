from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

from ultralytics.solutions import ParkingManagement
import threading
import time
import json
import cv2
import torch
import numpy as np
import logging

# Force CPU to avoid CUDA + torchvision::nms compatibility issues
device = 'cpu'
k = 0.75

logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

class CustomParkingManagement(ParkingManagement):
    def process_data(self, im0):
        self.extract_tracks(im0)
        total = len(self.json)
        occupied = 0
        slots = []

        for i, region in enumerate(self.json):
            pts_array = np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))
            rg_occupied = False

            for box, cls in zip(self.boxes, self.clss):
                xc, yc = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                dist = cv2.pointPolygonTest(pts_array, (xc, yc), False)
                if dist >= 0:
                    rg_occupied = True
                    break

            if rg_occupied:
                color = (0, 0, 255)
                occupied += 1
            else:
                color = (0, 255, 0)

            slots.append({
                "id": i + 1,
                "status": "occupied" if rg_occupied else "available"
            })

            cv2.polylines(im0, [pts_array], isClosed=True, color=color, thickness=4)
            region_center = np.mean(pts_array.reshape(-1, 2), axis=0).astype(int)
            cv2.putText(im0, f"Slot {i+1}", tuple(region_center),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        available = total - occupied
        self.pr_info["Occupancy"], self.pr_info["Available"] = occupied, available

        cv2.rectangle(im0, (10, 10), (340, 80), (40, 40, 40), -1, cv2.LINE_AA)
        cv2.rectangle(im0, (10, 10), (340, 80), (100, 100, 100), 2, cv2.LINE_AA)
        cv2.circle(im0, (25, 45), 10, (0, 255, 0) if available >= total / 2 else (0, 0, 255), -1)

        status_text = f"Occupied: {occupied} | Available: {available}"
        cv2.putText(im0, status_text, (50, 55), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        return im0, {
            "occupied": occupied,
            "available": available,
            "Space": slots
        }


management = CustomParkingManagement(
    model_path="yolov8n.pt",
    json_file="bounding_boxes.json",
    bg_color=(255, 255, 255),
    occ=(0, 0, 255),
    acr=(0, 255, 0)
)

cap = cv2.VideoCapture("new_video.MOV")

def generate_detected_frames():
    while cap.isOpened():
        ret, im0 = cap.read()
        if not ret:
            break

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

        processed_frame, status_data = management.process_data(im0)
        cv2.imwrite("static/snapshot.jpg", processed_frame)

        with open("status.json", "w") as f:
            json.dump(status_data, f)

        _, jpeg = cv2.imencode('.jpg', processed_frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.05)

@app.get("/", response_class=HTMLResponse)
def security_page(request: Request):
    return templates.TemplateResponse("security.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/status")
def status():
    with open("status.json") as f:
        return JSONResponse(content=json.load(f))

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_detected_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
