from src.utils import target_2_json, target_2_torch
import glob
import matplotlib.image as mpimg
import json
import numpy as np
import torch
import os.path as osp
import pandas as pd
import os
from .processing import DatasetProcessing
from .twincity_preprocessing2 import get_twincity_boxes, find_duplicate_indices, target_2_torch, creation_date
from configs_path import ROOT_DIR

class TwincityProcessing(DatasetProcessing):
    """
    Class that handles the preprocessing of (extracted) ECP Dataset in order to get a standardized dataset format.
    """

    def __init__(self, root, max_samples=100):
        self.dataset_name = "twincity"
        super().__init__(root, max_samples)
        os.makedirs(self.saves_dir, exist_ok=True)


    def get_annotations_and_imagepaths(self):

        root = self.root
        max_samples_per_seq = self.max_samples
        folders = glob.glob(osp.join(root, "*"))

        targets = {}
        df_gtbbox_metadata_list = []
        df_frame_metadata_list = []
        df_sequence_metadata_list = []

        for folder in folders:
            print(f"Reading folder {folder}")
            _, targets_folder, metadatas_folder = self.get_dataset_from_folder(folder, max_samples_per_seq)
            df_gtbbox_metadata_folder, df_frame_metadata_folder, df_sequence_metadata_folder = metadatas_folder

            # add outputs to their respective structures
            targets.update(targets_folder)
            df_frame_metadata_folder["seq_name"] = folder.split("/")[-1]
            df_gtbbox_metadata_list.append(df_gtbbox_metadata_folder)
            df_frame_metadata_list.append(df_frame_metadata_folder)
            df_sequence_metadata_list.append(df_sequence_metadata_folder)

        df_gtbbox_metadata = pd.concat(df_gtbbox_metadata_list, axis=0)
        df_frame_metadata = pd.concat(df_frame_metadata_list, axis=0)

        # todo do better gtbbox id (suffix of frame_id ?)
        df_gtbbox_metadata["id"] = list(range(len(df_gtbbox_metadata)))

        df_gtbbox_metadata = df_gtbbox_metadata.set_index(["frame_id", "id"])
        if df_frame_metadata.index.name != "frame_id":
            df_frame_metadata = df_frame_metadata.set_index("frame_id")

        return targets, df_gtbbox_metadata, df_frame_metadata, None



    def get_dataset_from_folder(self, folder, max_samples_per_seq=100):

        #todo factorize this
        root = self.root
        save_folder = self.saves_dir

        # save_folder = f"/home/raphael/work/code/pedestrian-detection-sensitivity-analysis/src/demos/data/preprocessing/motsynth/twincity/{version}/{folder_name}"
        os.makedirs(save_folder, exist_ok=True)

        path_df_gtbbox_metadata = osp.join(save_folder, f"df_gtbbox_{max_samples_per_seq}.csv")
        path_df_frame_metadata = osp.join(save_folder, f"df_frame_{max_samples_per_seq}.csv")
        path_df_sequence_metadata = osp.join(save_folder, f"df_sequence_{max_samples_per_seq}.csv")
        path_target = osp.join(save_folder, f"targets_{max_samples_per_seq}.json")

        try:
            print("Try")
            df_gtbbox_metadata = pd.read_csv(path_df_gtbbox_metadata)
            df_frame_metadata = pd.read_csv(path_df_frame_metadata)
            df_sequence_metadata = pd.read_csv(path_df_sequence_metadata)
            with open(path_target) as jsonFile:
                targets = target_2_torch(json.load(jsonFile))
            print("End Try")

        except:
            metadata_path = glob.glob(osp.join(folder, "Metadata*"))[0]
            images_path_list = glob.glob(osp.join(folder, "*.png"))
            with open(metadata_path) as file:
                metadata = json.load(file)

            ordering = np.argsort([creation_date(x) for x in images_path_list])
            img_annot_path_list = np.array(images_path_list)[ordering[::2]]
            img_rgb_path_list = np.array(images_path_list)[ordering[1::2]]

            df_gtbbox_list = []
            targets = {}
            frame_id_list = []
            df_frame_list = []

            # For all images in the sequence
            for i, img_annot_path in enumerate(img_annot_path_list):

                if i > max_samples_per_seq:
                    break

                img_annot = mpimg.imread(img_annot_path)
                bboxes, df = get_twincity_boxes(img_annot, metadata)

                # todo hack to set the anomalies as ignore regions
                frame_id = img_annot_path.split("/Snapshot-2023-")[-1].split(".png")[0]
                df["frame_id"] = frame_id  # todo frame_id should be the one of the rgb
                df["id"] = frame_id  # todo have to choose a denomination

                target = [
                    dict(
                        boxes=
                        bboxes)
                ]
                target[0]["labels"] = torch.tensor([0] * len(target[0]["boxes"]))
                targets[frame_id] = target
                frame_id_list.append(frame_id)

                # %% Check if everything OK
                """
                from torchvision.utils import draw_bounding_boxes
                img_rgb = mpimg.imread(img_rgb_path_list[0])
                img_rgb_torch = torch.tensor(img_rgb * 255, dtype=torch.uint8)[:, :, :3]
                img_rgb_torch = torch.swapaxes(img_rgb_torch, 0, 1)
                img_rgb_torch = torch.swapaxes(img_rgb_torch, 0, 2)
                drawn_boxes = draw_bounding_boxes(torch.tensor(img_rgb_torch), bboxes, colors="red")
                show(drawn_boxes)
                plt.show()
                """


                dict_frame_metadata = {
                    "weather": metadata["weather"],
                    "id": frame_id,
                    "file_name": img_rgb_path_list[i].split("v5/")[1],
                }
                df_frame_metadata = pd.DataFrame(dict_frame_metadata, index=[frame_id])
                for col in df.mean(numeric_only=True).keys():
                    df_frame_metadata[col] = df[col].mean()
                df_frame_metadata.index.name = "frame_id"

                df_gtbbox_list.append(df)
                df_frame_list.append(df_frame_metadata)

            df_frame_metadata = pd.concat(df_frame_list, axis=0)

            df_frame_metadata["pitch"] = metadata["cameraRotation"]["pitch"]
            df_frame_metadata["num_peds"] = len(metadata["peds"])
            df_frame_metadata["num_vehicles"] = metadata["vehiclesNb"]
            df_frame_metadata["hour"] = metadata["hour"]
            df_frame_metadata["is_night"] = metadata["hour"] > 21 or metadata[
                "hour"] < 6  # todo pas ouf, dépend du jour de l'année ...

            df_gtbbox_metadata = pd.concat(df_gtbbox_list, axis=0)
            df_sequence_metadata = pd.DataFrame()  # todo for now

            # %% Save folder files
            df_gtbbox_metadata.to_csv(path_df_gtbbox_metadata)
            df_frame_metadata.to_csv(path_df_frame_metadata)
            df_sequence_metadata.to_csv(path_df_sequence_metadata)
            with open(path_target, 'w') as f:
                json.dump(target_2_json(targets), f)

        metadatas = df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata

        return root, targets, metadatas

