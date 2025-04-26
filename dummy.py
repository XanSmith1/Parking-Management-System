# Create a dummy snapshot.jpg so the app doesn't crash
import numpy as np
import cv2

dummy = np.zeros((480, 640, 3), dtype=np.uint8)  # black image
cv2.imwrite("static/snapshot.jpg", dummy)
