import cv2
from ultralytics.solutions.parking_management import ParkingManagement

# Ensure you have the correct model path and polygon JSON
management = ParkingManagement(model_path="best.pt")
polygon_json_path = "/Users/xandersmith/PycharmProjects/Ultralytics/ultravideo/.venv/bounding_boxes.json"

cap = cv2.VideoCapture("/Users/xandersmith/Desktop/UTL(ParkMngmnt)FDS/full_DS/valid/images/2012-10-31_12_38_16_")
while cap.isOpened():
    ret, im0 = cap.read()
    if not ret:
        break

    json_data = management.parking_regions_extraction(polygon_json_path)
    results = management.model.track(im0, persist=True, show=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()
        management.process_data(json_data, im0, boxes, clss)

    management.display_frames(im0)

cap.release()
cv2.destroyAllWindows()
