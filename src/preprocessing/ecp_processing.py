import json
import numpy as np
import torch
import os.path as osp
import pandas as pd
import os
from .processing import DatasetProcessing

"""
Check more info https://eurocity-dataset.tudelft.nl/
For example : ignore regions, snow ...
"""

def syntax_occl_ECP(x):
    # keep only occluded
    x = [x for x in x if "occluded" in x]
    if x and "occluded" in x[0]:
        return int(x[0].replace("occluded>", "")) / 100
    elif x:
        print(x)
    else:
        return 0

def syntax_truncated_ECP(x):
    # keep only truncated
    x = [x for x in x if "truncated" in x]
    if x and "truncated" in x[0]:
        return int(x[0].replace("truncated>", "")) / 100
    elif x:
        print(x)
    else:
        return 0


def syntax_criteria_ECP(x, criteria):
    # "sitting-lying"
    # keep only truncated
    x = [x for x in x if criteria in x]
    if x and criteria in x[0]:
        return 1
    elif x:
        print(x, criteria)
    else:
        return 0



class ECPProcessing(DatasetProcessing):
    """
    Class that handles the preprocessing of (extracted) ECP Dataset in order to get a standardized dataset format.
    """

    def __init__(self, root, max_samples=100): #todo max_samples here is different than from MoTSynth

        super().__init__(root, max_samples)

        self.dataset_name = "ecp"
        self.saves_dir = f"data/preprocessing/{self.dataset_name}"
        os.makedirs(self.saves_dir, exist_ok=True)

    def get_dataset(self):
        targets, metadatas, frame_id_list, img_path_list = self.get_annotations_and_imagepaths()
        return self.root, targets, metadatas, frame_id_list, img_path_list

    def get_ECP_annotations_and_imagepaths_folder(self, time, set, city):

        # Here ECP specific
        root = f"{self.root}/{time}/labels/{set}/{city}"





        total_frame_ids = [x.split(".json")[0] for x in os.listdir(root) if ".json" in x]

        # Init dicts for bboxes annotations and metadata
        targets = {}
        targets_metadata = {}
        for i, frame_id in enumerate(total_frame_ids):

            # set max samples #todo
            if i > self.max_samples:
                break

            # Load ECP annotations
            json_path = f"{self.root}/{time}/labels/{set}/{city}/{frame_id}.json"
            with open(json_path) as jsonFile:
                annot_ECP = json.load(jsonFile)

                target = [
                    dict(
                        boxes=torch.tensor(
                            [(c["x0"], c["y0"], c["x1"], c["y1"]) for c in annot_ECP["children"] if
                             c["identity"] in ["pedestrian", "rider"]]  # todo might not be the thing todo
                        ),
                    )
                ]

                target[0]["labels"] = torch.tensor([0] * len(target[0]["boxes"]))

                # Keep only if at least 1 pedestrian
                if len(target[0]["boxes"]) > 0:
                    targets[frame_id] = target

                    tags = [c["tags"] for c in annot_ECP["children"] if c["identity"] in ["pedestrian", "rider"]]
                    areas = [(c["x1"] - c["x0"]) * (c["y1"] - c["y0"]) for c in annot_ECP["children"] if
                             c["identity"] in ["pedestrian", "rider"]]

                    heights = [(c["y1"] - c["y0"]) for c in annot_ECP["children"] if
                             c["identity"] in ["pedestrian", "rider"]]

                    widths = [(c["x1"] - c["x0"]) for c in annot_ECP["children"] if
                             c["identity"] in ["pedestrian", "rider"]]

                    aspect_ratio = [h/w for h,w in zip(heights, widths)]

                    iscrowd = [1 * ("group" in c["identity"]) for c in annot_ECP["children"] if
                               c["identity"] in ["pedestrian", "rider"]]

                    targets_metadata[frame_id] = (annot_ECP["tags"], tags,
                                                  areas, heights, widths, aspect_ratio, iscrowd)

        frame_id_list = list(targets.keys())
        img_path_list = []
        for frame_id in frame_id_list:
            # print(frame_id)
            img_path = f"{time}/img/val/{city}/{frame_id}.png"
            img_path_list.append(img_path)



        return targets, targets_metadata, frame_id_list, img_path_list


    def get_annotations_and_imagepaths(self):

        """
        try:
            print("Try")
            df_gtbbox_metadata = pd.read_csv(path_df_gtbbox_metadata).set_index(["image_id", "id"])
            df_frame_metadata = pd.read_csv(path_df_frame_metadata).set_index("Unnamed: 0")
            df_sequence_metadata = pd.read_csv(path_df_sequence_metadata).set_index("Unnamed: 0")


            with open(path_target) as jsonFile:
                targets = target_2_torch(json.load(jsonFile))
            print("End Try")
        """


        targets = {}
        frame_id_list = []
        img_path_list = []
        df_frame_metadata = pd.DataFrame()
        df_gtbbox_metadata = pd.DataFrame()

        key_tags = []

        for luminosity in ["day", "night"]:
            for chosen_set in ["val"]:
                for city in os.listdir(
                        f"{self.root}/{luminosity}/img/{chosen_set}"):
                    if city not in ["berlin_small"]:
                        print(luminosity, city)

                        targets_folder, targets_metadata_folder, frame_id_list_folder, img_path_list_folder = self.get_ECP_annotations_and_imagepaths_folder(
                            luminosity, chosen_set, city)

                        # Df frame                 # print(np.unique([val[0] for key,val in targets_metadata.items()]))
                        categories = ["motionBlur", "rainy", "wiper", "lenseFlare", "constructionSite"]
                        df_frames_metadata_folder = pd.DataFrame(
                            {key: [cat in val[0] for cat in categories] for key, val in
                             targets_metadata_folder.items()}).T
                        df_frames_metadata_folder.columns = categories
                        df_frames_metadata_folder["is_night"] = luminosity == "night"
                        df_frames_metadata_folder["city"] = city
                        df_frames_metadata_folder["path"] = img_path_list_folder
                        df_frames_metadata_folder["frame_id"] = frame_id_list_folder
                        df_frames_metadata_folder["adverse_weather"] = 1*df_frames_metadata_folder["rainy"]
                        df_frames_metadata_folder["file_name"] = img_path_list_folder
                        df_frames_metadata_folder["id"] = frame_id_list_folder
                        df_frames_metadata_folder["seq_name"] = city + " "+luminosity

                        df_frames_metadata_folder["weather"] = "dry"
                        df_frames_metadata_folder[df_frames_metadata_folder["rainy"]]["weather"] = "rainy"

                        # df_frames_metadata_folder["file_name"] = img_path_list_folder
                        # media/raphael/Projects/datasets/EuroCityPerson/ECP/

                        # Df bbox_gt
                        # {key: {key: v for v in val[1]} for key,val in targets_metadata.items()}
                        categories_gtbbox = [f"occluded>{i}0" for i in range(1, 10)] + ["depiction"]
                        df_gt_bbox_folder = pd.DataFrame()
                        for key, val in targets_metadata_folder.items():

                            key_tags.append(val[1])

                            df_gt_bbox_frame = pd.DataFrame(val[1:]).T
                            df_gt_bbox_frame["frame_id"] = key
                            for cat in categories_gtbbox:
                                try:
                                    df_gt_bbox_frame[cat] = cat in val[0]
                                except:
                                    print("coucou")
                            df_gt_bbox_frame.drop(columns=[0], inplace=True)
                            df_gt_bbox_frame["occlusion_rate"] = [syntax_occl_ECP(x) for x in val[1]]
                            df_gt_bbox_frame["truncation_rate"] = [syntax_truncated_ECP(x) for x in val[1]]

                            for criteria in ['sitting-lying', 'behind-glass', 'unsure_orientation']:
                                df_gt_bbox_frame[criteria] = [syntax_criteria_ECP(x, criteria) for x in val[1]]

                            df_gt_bbox_frame.rename(columns={1: "area",
                                                             2: "height",
                                                             3: "width",
                                                             4: "aspect_ratio",
                                                             5: "is_crowd"}, inplace=True)

                            df_gt_bbox_folder = pd.concat([df_gt_bbox_folder, df_gt_bbox_frame])

                        # Add the folder
                        targets.update(targets_folder)
                        frame_id_list += frame_id_list_folder
                        img_path_list += img_path_list_folder
                        df_frame_metadata = pd.concat([df_frame_metadata, df_frames_metadata_folder])
                        df_gtbbox_metadata = pd.concat([df_gtbbox_metadata, df_gt_bbox_folder])

        df_gtbbox_metadata["id_in_frame"] = df_gtbbox_metadata.groupby("frame_id").apply(
            lambda x: pd.Series(list(range(0, len(x))))).values
        df_gtbbox_metadata["id"] = df_gtbbox_metadata["frame_id"] + "_" + df_gtbbox_metadata["id_in_frame"].astype(str)
        df_gtbbox_metadata = df_gtbbox_metadata.set_index(["frame_id", "id"])

        return targets, df_gtbbox_metadata, df_frame_metadata, None
