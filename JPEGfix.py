import os

# Path to the folder with incorrectly recognized files
image_folder = '/Users/xandersmith/Desktop/UTL(ParkMngmnt)/allimages_dataset3'

for filename in os.listdir(image_folder):
    if filename.endswith('.jpeg'):
        # Rename to .jpg
        new_name = filename.replace('.jpeg', '.jpg')
        os.rename(
            os.path.join(image_folder, filename),
            os.path.join(image_folder, new_name)
        )

print("All .jpeg files have been renamed to .jpg.")
