import os
import shutil

# Base dataset path
base_path = '/Users/xandersmith/Desktop/UTL(ParkMngmnt)'

# Define the path for the new 'full_DS' folder
full_ds_folder = f'/Users/xandersmith/Desktop/UTL(ParkMngmnt)/full_DS'
os.makedirs(full_ds_folder, exist_ok=True)

# Folders to process: 'test', 'train', 'valid'
source_folders = ['train', 'test', 'valid']

# Move all .jpg and .txt files from source folders into 'full_DS'
for folder in source_folders:
    # Full path to the current folder
    folder_path = os.path.join(base_path, folder)

    # Process image files in the main folder
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and file.endswith('.jpg'):  # Move images
            shutil.move(file_path, os.path.join(full_ds_folder, file))

    # Process annotation files inside the 'labels' subfolder
    labels_folder = os.path.join(folder_path, 'labels')
    if os.path.exists(labels_folder):  # Check if 'labels' subfolder exists
        for file in os.listdir(labels_folder):
            file_path = os.path.join(labels_folder, file)
            if os.path.isfile(file_path) and file.endswith('.txt'):  # Move annotations
                shutil.move(file_path, os.path.join(full_ds_folder, file))

print(f"All images and annotations moved into: {full_ds_folder}")

