import os
import shutil

# Define directories
base_dir = '/Users/xandersmith/Desktop/UTL(ParkMngmnt)/full_DS'
all_annotations_dir = os.path.join(base_dir, 'all_annotations')

# Create a directory to consolidate annotations
os.makedirs(all_annotations_dir, exist_ok=True)

# Consolidate annotations
label_folders = ['/Users/xandersmith/Desktop/UTL(ParkMngmnt)/full_DS/test/labels',
                 '/Users/xandersmith/Desktop/UTL(ParkMngmnt)/full_DS/train/labels',
                 '/Users/xandersmith/Desktop/UTL(ParkMngmnt)/full_DS/valid/labels']
for label_folder in label_folders:
    full_path = os.path.join(base_dir, label_folder)
    for annotation in os.listdir(full_path):
        if annotation.endswith('.txt'):  # Assuming annotations are .txt files
            shutil.move(os.path.join(full_path, annotation), os.path.join(all_annotations_dir, annotation))

# Match annotations with images
image_folders = ['/Users/xandersmith/Desktop/UTL(ParkMngmnt)/full_DS/test/images',
                 '/Users/xandersmith/Desktop/UTL(ParkMngmnt)/full_DS/train/images',
                 '/Users/xandersmith/Desktop/UTL(ParkMngmnt)/full_DS/valid/images']
for image_folder in image_folders:
    label_folder = image_folder.replace('images', 'labels')
    os.makedirs(os.path.join(base_dir, label_folder), exist_ok=True)

    full_image_path = os.path.join(base_dir, image_folder)
    full_label_path = os.path.join(base_dir, label_folder)

    for image in os.listdir(full_image_path):
        image_name, _ = os.path.splitext(image)
        annotation_file = f"{image_name}.txt"
        annotation_path = os.path.join(all_annotations_dir, annotation_file)

        if os.path.exists(annotation_path):
            shutil.move(annotation_path, os.path.join(full_label_path, annotation_file))
        else:
            print(f"Warning: No annotation found for {image}")

# Check for unmatched annotations
remaining_annotations = os.listdir(all_annotations_dir)
if remaining_annotations:
    print("Unmatched annotations found:")
    for annotation in remaining_annotations:
        print(annotation)
