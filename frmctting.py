import cv2
import os

# Input video path
video_path = '/Users/xandersmith/Desktop/UTL(ParkMngmnt)/tst_vid.MOV'  # Replace with your video file path
output_folder = '/Users/xandersmith/Desktop/UTL(ParkMngmnt)/extracted_frames'  # Folder to save the frames

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()  # Read the next frame
    if not ret:  # If no frame is returned, we've reached the end of the video
        break

    # Save the frame as an image file
    frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)
    print(f"Saved: {frame_filename}")
    frame_count += 1

# Release the video capture object
cap.release()
print(f"Done! Extracted {frame_count} frames to the folder: {output_folder}")
