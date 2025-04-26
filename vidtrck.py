from ultralytics import solutions
from ultralytics.solutions import ParkingManagement
import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics.utils.plotting import Annotator
k = 0.75


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_path = {'Nano': 'yolov8n.pt', 'Small': 'yolov8s.pt', 'Medium': 'yolov8m.pt'}
input_src = 'Record.MOV'


class CustomParkingManagement(solutions.ParkingManagement):
    def process_data(self, im0):
        """
        Processes the model data for parking lot management.

        This function analyzes the input image, extracts tracks, and determines the occupancy status of parking
        regions defined in the JSON file. It annotates the image with occupied and available parking spots,
        and updates the parking information.

        Args:
            im0 (np.ndarray): The input inference image.

        Examples:
            >>> parking_manager = ParkingManagement(json_file="parking_regions.json")
            >>> image = cv2.imread("parking_lot.jpg")
            >>> parking_manager.process_data(image)
        """
        self.extract_tracks(im0)  # extract tracks from im0
        es, fs = len(self.json), 0  # empty slots, filled slots
        annotator = Annotator(im0, self.line_width)  # init annotator

        for region in self.json:
            # Convert points to a NumPy array with the correct dtype and reshape properly
            pts_array = np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))
            rg_occupied = False  # occupied region initialization
            for box, cls in zip(self.boxes, self.clss):
                xc, yc = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                dist = cv2.pointPolygonTest(pts_array, (xc, yc), False)
                if dist >= 0:
                    # cv2.circle(im0, (xc, yc), radius=self.line_width * 4, color=self.dc, thickness=-1)
                    # annotator.display_objects_labels(
                    #     im0, self.model.names[int(cls)], (104, 31, 17), (255, 255, 255), xc, yc, 10
                    # )
                    rg_occupied = True
                    break
            fs, es = (fs + 1, es - 1) if rg_occupied else (fs, es)
            # Plotting Parking regions
            self.arc, self.occ = self.arc, (0, 0, 255)
            cv2.polylines(im0, [pts_array], isClosed=True, color=self.occ if rg_occupied else self.arc, thickness=5)

        self.pr_info["Occupancy"], self.pr_info["Available"] = fs, es

        annotator.display_analytics(im0, self.pr_info, (104, 31, 17), (255, 255, 255), 10)
        self.display_output(im0)  # display output with base class function
        return im0  # return output image for more usage


management = CustomParkingManagement(
    model_path="yolov8n.pt",
    # txt_color=(0, 0, 0),
    bg_color=(255, 255, 255),
    occ=(0, 0, 255),
    acr=(0, 0, 255),
    json_file="bounding_boxes.json",
)


def mark_lots(state):

    if state:
        cap = cv2.VideoCapture("Record.MOV")

        assert cap.isOpened(), "Error reading video file"
        r, c, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        while cap.isOpened():
            ret, im0 = cap.read()
            break

        r, c, ch = im0.shape
        img = cv2.resize(im0, (int(c*k), int(r*k)))
        cv2.imwrite('sample.jpg', img)

        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # solutions.ParkingPtsSelection()

# ===========================================================================?


def check_parking():

    # Ensure you have the correct model path and polygon JSON
    # management = ParkingManagement(model_path="yolov8n.pt")

    cap = cv2.VideoCapture("final.MOV")
    while cap.isOpened():
        ret, im0 = cap.read()
        if not ret:
            break
        sz = list(im0.shape)
        sz[:2] = [int(v * k) for v in sz[:2]]

        im0 = cv2.resize(im0, (sz[1], sz[0]))

        results = management.model.track(im0, persist=True, show=False, imgsz=sz[:2], classes=[2, 7], conf=0.2, iou=0.1,
                                         device='cpu')
        management.process_data(im0)

        # Render the results on the image
        rendered_image = results[0].plot(labels=False, boxes=False)
        # image = Image.fromarray(rendered_image)

        # return img, management.pr_info["Available"], management.pr_info["Occupancy"]
        # return image, management.pr_info["Available"], management.pr_info["Occupancy"]

        # json_data = management.parking_regions_extraction(polygon_json_path)
        # results = management.model.track(im0, persist=True, show=False, imgsz=sz[:2], classes=[2, 7], conf=0.001, iou=0.1, device='cuda:0')
        #
        # if results[0] and results[0].boxes:
        #     boxes = results[0].boxes.xyxy.cpu().tolist()
        #     clss = results[0].boxes.cls.cpu().tolist()
        #     management.process_data(json_data, im0, boxes, clss)
        #
        management.display_output(im0)
        time.sleep(0.001)

        # Display the output
        cv2.imshow("Parking Management", im0)  # Display the processed frame
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    mark_lots(True)
    check_parking()


if __name__ == '__main__':
    main()