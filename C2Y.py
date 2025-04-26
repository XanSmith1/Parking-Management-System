import json
import os


# Define the function to convert COCO to YOLO format
def convert_coco_to_yolo(coco_json_path, output_dir, image_folder):
    # Load the COCO JSON file
    with open(coco_json_path) as f:
        data = json.load(f)

    # Get the categories and images
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    images = {img['id']: dict(file_name=img['file_name'], width=img['width'], height=img['height']) for img in
              data['images']}

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert each annotation to YOLO format
    for ann in data['annotations']:
        img_id = ann['image_id']
        img_info = images[img_id]
        img_file = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        category = ann['category_id'] - 1  # YOLO class index starts from 0
        bbox = ann['bbox']

        # Shorten the file name for both the image and the annotation
        img_file = img_file.split('_jpg')[0] + '.jpg'

        # Convert COCO bbox (x, y, width, height) to YOLO (x_center, y_center, width, height)
        x_center = bbox[0] + bbox[2] / 2
        y_center = bbox[1] + bbox[3] / 2
        width = bbox[2]
        height = bbox[3]

        # Normalize the coordinates
        yolo_bbox = [
            x_center / img_width,
            y_center / img_height,
            width / img_width,
            height / img_height
        ]

        # Write to YOLO .txt file
        label_file = os.path.join(output_dir, img_file.replace('.jpg', '.txt'))
        with open(label_file, 'a') as f:
            f.write(f"{category} " + " ".join(map(str, yolo_bbox)) + "\n")

    print(f"COCO to YOLO conversion complete for: {coco_json_path}")


# Shorten file names of images and labels
def shorten_file_names(folder, file_extension):
    for filename in os.listdir(folder):
        if filename.endswith(file_extension):
            new_name = filename.split('_jpg')[0] + file_extension
            os.rename(
                os.path.join(folder, filename),
                os.path.join(folder, new_name)
            )
    print(f"File names shortened in: {folder} ({file_extension} files)")


# File paths
train_coco_json = '/Users/xandersmith/Desktop/dataset3/train/_annotations.coco.json'
valid_coco_json = '/Users/xandersmith/Desktop/dataset3/valid/_annotations.coco.json'
test_coco_json = '/Users/xandersmith/Desktop/dataset3/test/_annotations.coco.json'

train_output = '/Users/xandersmith/Desktop/dataset3/train/labels'
valid_output = '/Users/xandersmith/Desktop/dataset3/valid/labels'
test_output = '/Users/xandersmith/Desktop/dataset3/test/labels'

train_images = '/Users/xandersmith/Desktop/dataset3/train'
valid_images = '/Users/xandersmith/Desktop/dataset3/valid'
test_images = '/Users/xandersmith/Desktop/dataset3/test'

# Convert COCO to YOLO and shorten file names
convert_coco_to_yolo(train_coco_json, train_output, train_images)
shorten_file_names(train_output, '.txt')
shorten_file_names(train_images, '.jpg')

convert_coco_to_yolo(valid_coco_json, valid_output, valid_images)
shorten_file_names(valid_output, '.txt')
shorten_file_names(valid_images, '.jpg')

convert_coco_to_yolo(test_coco_json, test_output, test_images)
shorten_file_names(test_output, '.txt')
shorten_file_names(test_images, '.jpg')

