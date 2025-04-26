import os

# Path to the folder containing all files
full_ds_folder = '/Users/xandersmith/Desktop/UTL(ParkMngmnt)/full_DS'

# Shorten the file names
for file in os.listdir(full_ds_folder):
    file_path = os.path.join(full_ds_folder, file)
    if os.path.isfile(file_path):  # Ensure it's a file
        # Shorten the name by keeping only the first 10 characters before the extension
        name, ext = os.path.splitext(file)  # Split the file name and extension
        new_name = name[:20] + ext  # Keep only the first 10 characters
        new_file_path = os.path.join(full_ds_folder, new_name)

        # Rename the file
        os.rename(file_path, new_file_path)
        print(f"Renamed: {file} -> {new_name}")

print("File names shortened successfully!")
