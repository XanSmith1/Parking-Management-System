from ultralytics import solutions
from ultralytics.solutions import ParkingManagement
import cv2
import torch
import numpy as np
import time

k = 0.75
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class CustomParkingManagement(solutions.ParkingManagement):
    def process_data(self, im0):
        self.extract_tracks(im0)
        total = len(self.json)
        occupied = 0

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
                color = (0, 0, 255)  # Red
                occupied += 1
            else:
                color = (0, 255, 0)  # Green

            cv2.polylines(im0, [pts_array], isClosed=True, color=color, thickness=4)

            # OPTIONAL: label each region
            region_center = np.mean(pts_array.reshape(-1, 2), axis=0).astype(int)
            cv2.putText(im0, f"Space {i+1}", tuple(region_center),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        available = total - occupied
        self.pr_info["Occupancy"], self.pr_info["Available"] = occupied, available

        # ======= Visual Counter Banner =======
        banner_color = (40, 40, 40)
        text_color = (255, 255, 255)
        status_icon_color = (0, 255, 0) if available >= total / 2 else (0, 0, 255)

        # Banner background
        cv2.rectangle(im0, (10, 10), (340, 80), banner_color, -1, cv2.LINE_AA)
        cv2.rectangle(im0, (10, 10), (340, 80), (100, 100, 100), 2, cv2.LINE_AA)

        # Status icon
        cv2.circle(im0, (25, 45), 10, status_icon_color, -1)

        # Status text
        status_text = f"Occupied: {occupied} | Available: {available}"
        cv2.putText(im0, status_text, (50, 55),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, text_color, 2, cv2.LINE_AA)

        self.display_output(im0)
        return im0


management = CustomParkingManagement(
    model_path="yolov8n.pt",
    json_file="bounding_boxes.json",
    bg_color=(255, 255, 255),
    occ=(0, 0, 255),
    acr=(0, 255, 0)
)
management.model.to('cpu')

def check_parking():
    cap = cv2.VideoCapture("new_video.MOV")
    while cap.isOpened():
        ret, im0 = cap.read()
        if not ret:
            break

        im0 = cv2.resize(im0, (int(im0.shape[1] * k), int(im0.shape[0] * k)))

        results = management.model.track(im0, persist=True, show=False,
                                         imgsz=im0.shape[:2], classes=[2, 7],
                                         conf=0.2, iou=0.1, )

        if results and results[0].boxes:
            management.boxes = results[0].boxes.xyxy.cpu().numpy()
            management.clss = results[0].boxes.cls.cpu().numpy()
        else:
            management.boxes = []
            management.clss = []

        management.process_data(im0)

        cv2.imshow("Parking Management", im0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    check_parking()


if __name__ == '__main__':
    main()
