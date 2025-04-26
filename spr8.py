import os
import random
import shutil

# Path to the folder containing all files
full_ds_folder = '/Users/xandersmith/Desktop/UTL(ParkMngmnt)/full_DS'

# Output folders for train, test, and valid
output_folders = {
    'train': '/Users/xandersmith/Desktop/UTL(ParkMngmnt)/train',
    'test':  '/Users/xandersmith/Desktop/UTL(ParkMngmnt)/test',
    'valid': '/Users/xandersmith/Desktop/UTL(ParkMngmnt)/valid'
}

# Create the output folders if they don't exist
for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

# Get all files in the full_ds_folder
all_files = [f for f in os.listdir(full_ds_folder) if os.path.isfile(os.path.join(full_ds_folder, f))]

# Shuffle files for random distribution
random.shuffle(all_files)

# Split files into train (70%), test (15%), and valid (15%)
total_files = len(all_files)
train_split = int(total_files * 0.7)
valid_split = int(total_files * 0.15)

train_files = all_files[:train_split]
valid_files = all_files[train_split:train_split + valid_split]
test_files = all_files[train_split + valid_split:]

# Function to move files into the respective folders
def move_files(files, target_folder):
    for file in files:
        shutil.move(os.path.join(full_ds_folder, file), os.path.join(target_folder, file))

# Move the files into train, test, and valid folders
move_files(train_files, output_folders['train'])
move_files(valid_files, output_folders['valid'])
move_files(test_files, output_folders['test'])

# Create classes.txt in each folder
for folder in output_folders.values():
    classes_file = os.path.join(folder, 'classes.txt')
    with open(classes_file, 'w') as f:
        f.write('Empty\nFilled\n')

print("Files split and classes.txt created successfully!")
