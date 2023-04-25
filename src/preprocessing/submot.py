import os
import random
import shutil

# Define the path to the folder containing the subfolders
path_to_folder = "/home/raphael/work/datasets/MOTSynth/frames"

# Define the name and path of the folder where the random files will be saved
new_folder_name = "random_files"
new_folder_path = os.path.join(path_to_folder, new_folder_name)

# Create the new folder if it doesn't already exist
if not os.path.exists(new_folder_path):
    os.mkdir(new_folder_path)

# Loop through each subfolder
for subfolder_name in os.listdir(path_to_folder):
    subfolder_path = os.path.join(path_to_folder, subfolder_name , "rgb")
    if os.path.isdir(subfolder_path):
        # Create a list of all the files in the subfolder
        file_list = os.listdir(subfolder_path)
        # Shuffle the file list randomly
        random.shuffle(file_list)
        # Take the first 10 files from the shuffled list
        chosen_files = file_list[:10]
        # Copy each chosen file to the new folder
        for chosen_file in chosen_files:
            chosen_file_path = os.path.join(subfolder_path, chosen_file)
            new_file_path = os.path.join(new_folder_path, chosen_file)
            shutil.copyfile(chosen_file_path, subfolder_name+new_file_path)
