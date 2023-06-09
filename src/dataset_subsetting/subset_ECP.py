import os
import shutil
import os.path as osp


def subset_ECP(fulldataset_folder_path, newdataset_folder_path, num_frame_per_sequence=10):

    for luminosity in ["day", "night"]:
        for subfolder_name in os.listdir(osp.join(fulldataset_folder_path, luminosity, "img", "val")):

            # Print and create destination folder
            print(luminosity, subfolder_name)
            os.makedirs(osp.join(newdataset_folder_path, luminosity, "img", "val", subfolder_name), exist_ok=True)
            os.makedirs(osp.join(newdataset_folder_path, luminosity, "labels", "val", subfolder_name), exist_ok=True)

            # Get the list of images and labels
            img_name_list = os.listdir(osp.join(fulldataset_folder_path, luminosity, "img", "val", subfolder_name))
            new_img_name_list = img_name_list[::len(img_name_list)//num_frame_per_sequence]
            new_labels_name_list = [x.replace(".png", ".json") for x in new_img_name_list]

            # Copy the subset of images and labels
            for new_img_name, new_label_name in zip(new_img_name_list, new_labels_name_list):

                current_img_path = osp.join(fulldataset_folder_path, luminosity, "img", "val", subfolder_name, new_img_name)
                current_label_path = osp.join(fulldataset_folder_path, luminosity, "labels", "val", subfolder_name, new_label_name)

                new_img_path = osp.join(newdataset_folder_path, luminosity, "img", "val", subfolder_name, new_img_name)
                new_label_path = osp.join(newdataset_folder_path, luminosity, "labels", "val", subfolder_name, new_label_name)

                shutil.copyfile(current_img_path, new_img_path)
                shutil.copyfile(current_label_path, new_label_path)


if __name__ == "__main__":

    # %% Sub ECP at 10th subsampling
    NUM_FRAME_PER_SEQUENCE = 10
    # Full dataset path
    fulldataset_folder_path = "/media/raphael/Projects/datasets/EuroCityPerson/ECP"
    newdataset_folder_path = "/home/raphael/work/datasets/PedestrianDetectionSensitivityDatasets/ecp_small"

    # Subset ECP
    subset_ECP(fulldataset_folder_path, newdataset_folder_path, num_frame_per_sequence=10)

