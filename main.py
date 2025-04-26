from CustomParkingManagement import CustomParkingManagement
import cv2
import time

management = CustomParkingManagement(
    model_path="yolov8n.pt",
    json_file="bounding_boxes.json",
    bg_color=(255, 255, 255),
    occ=(0, 0, 255),
    acr=(0, 0, 255)
)

def check_parking():
    cap = cv2.VideoCapture("new_video.MOV")


    while cap.isOpened():
        ret, im0 = cap.read()
        if not ret:
            break

        im0 = cv2.resize(im0, (int(im0.shape[1]*0.75), int(im0.shape[0]*0.75)))
        management.model.track(im0, persist=True, show=False, classes=[2, 7], conf=0.2, iou=0.1, device='cpu')
        management.process_data(im0)

        cv2.imshow("Parking Management", im0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    check_parking()
