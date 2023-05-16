import os
import shutil
import os.path as osp
import json
import numpy as np


def subset_MOTSynth(fulldataset_folder_path, new_folder_path, num_frame_per_sequence=10, delay=3):

    # Full dataset path
    fulldataset_folder_frame_path = osp.join(fulldataset_folder_path, "frames")
    fulldataset_folder_annot_path = osp.join(fulldataset_folder_path, "coco annot")

    # New folder path
    new_folder_frame_path = osp.join(new_folder_path, "frames")
    new_folder_annot_path = osp.join(new_folder_path, "coco annot")

    # Create the new folder if it doesn't already exist
    os.makedirs(new_folder_annot_path, exist_ok=True)
    os.makedirs(new_folder_frame_path, exist_ok=True)


    # For each subfolder of rgb frames
    for subfolder_name in np.sort(os.listdir(fulldataset_folder_frame_path)):

        coco_annot_path = f"{fulldataset_folder_annot_path}/{subfolder_name}.json"

        if not os.path.exists(coco_annot_path):
            print(f"Disgarding sequence {subfolder_name} because no annotations were found")
            continue

        if not os.path.exists(osp.join(fulldataset_folder_frame_path, subfolder_name, "rgb")):
            print(f"Disgarding sequence {subfolder_name} because no rgb subfolder was found")
            continue

        print(f"Subsetting sequence {subfolder_name}")
        subfolder_path = os.path.join(fulldataset_folder_frame_path, subfolder_name, "rgb")
        file_list = os.listdir(subfolder_path)

        # Load the coco annotations
        try:
            with open(coco_annot_path) as f:
                data = json.load(f)

            # Where to save
            os.makedirs(os.path.join(new_folder_frame_path, subfolder_name, "rgb"), exist_ok=True)

            # Create new annotations
            new_images_beforecopy = data["images"][::len(data["images"])//num_frame_per_sequence]

            # Copy the subset of images
            included_images_idx = []
            for i, img in enumerate(new_images_beforecopy):
                # Copy each chosen file to the new folder
                chosen_file_path = os.path.join(subfolder_path, img["file_name"].split("/")[-1])
                new_file_path = os.path.join(new_folder_path, img["file_name"])

                if os.path.exists(chosen_file_path):
                    shutil.copyfile(chosen_file_path, new_file_path)
                    included_images_idx.append(i)
                else:
                    print(f"File {chosen_file_path} does not exist. Disgarding it.")

            new_images = [img for i, img in enumerate(new_images_beforecopy) if i in included_images_idx]
            new_annots = [annot for annot in data["annotations"] if annot["image_id"] in [img["id"]+delay for img in new_images]]
            new_data = {"images": new_images, "annotations": new_annots, "categories": data["categories"],
                        "info": data["info"], "licenses": data["licenses"]}
            with open(os.path.join(new_folder_annot_path, f"{subfolder_name}.json"), "w") as f:
                json.dump(new_data, f)

        except:
            print(f"Error while loading annotations {coco_annot_path}. Disgarding sequence.")


if __name__ == "__main__":
    # Parameters
    NUM_FRAME_PER_SEQUENCE = 10
    DELAY = 3
    fulldataset_folder_path = "/home/raphael/work/datasets/MOTSynth/"
    new_folder_path = "/home/raphael/work/datasets/MOTSynth_small"
    subset_MOTSynth(fulldataset_folder_path, new_folder_path, num_frame_per_sequence=NUM_FRAME_PER_SEQUENCE, delay=DELAY)