import json
import torch
import pandas as pd
from .processing import DatasetProcessing
import os
import numpy as np
import os.path as osp

def subset_dict(dictionary, keys):
    return {key: dictionary[key] for key in keys if key in dictionary}

def bbox_2_x0y0x1y1(dict):
    dict["x0"] = dict["bbox"][0]
    dict["y0"] = dict["bbox"][1]
    dict["x1"] = dict["bbox"][2] # + dict["bbox"][2]
    dict["y1"] = dict["bbox"][3] # + dict["bbox"][3]  # todo check that
    return dict


def get_bbox(df):
    return df[["x0", "y0", "x1", "y1"]].values

class COCOProcessing(DatasetProcessing):
    """
    Class that handles the preprocessing of (extracted) ECP Dataset in order to get a standardized dataset format.
    """

    def __init__(self, root, dataset_name, max_samples_per_sequence=100, task="pedestrian_detection"): #todo max_samples here is different than from MoTSynth

        self.dataset_name = dataset_name
        super().__init__(root, max_samples_per_sequence)
        #self.saves_dir = f"data/preprocessing/{self.dataset_name}"
        os.makedirs(self.saves_dir, exist_ok=True)

        self.coco_json_path = osp.join(self.root, "coco.json")

        assert task in ["pedestrian_detection"]

    def get_dataset(self, force_recompute=False):

        #todo if needed save and format as the other datasets.
        #todo 2 : fix with sequence_id assumed to be 0 for now
        sequence_id = 0

        # Load coco dataset
        with open(self.coco_json_path) as f:
            coco_json = json.load(f)

        # WHich features to keep
        gtbbox_features = ['id', 'image_id', 'category_id', 'iscrowd', 'area', "x0", "y0", "x1", "y1"]
        frame_features = ['file_name', 'height', "width"]

        # Dataframes
        df_sequence_metadata = pd.DataFrame({"sequence_id": sequence_id}, index=[0])
        df_frame_metadata = pd.concat([pd.DataFrame(img, index=[str(img["id"])])[frame_features] for img in coco_json["images"]])
        df_gtbbox_metadata = pd.concat([pd.DataFrame(subset_dict(bbox_2_x0y0x1y1(annot), gtbbox_features), index=[annot["id"]])[gtbbox_features]
                                        for annot in coco_json["annotations"]])
        df_gtbbox_metadata.rename(columns={"image_id": "frame_id"}, inplace=True)
        df_gtbbox_metadata["frame_id"] = df_gtbbox_metadata["frame_id"].astype(str)
        df_gtbbox_metadata["id"] = df_gtbbox_metadata["id"].astype(str)
        df_gtbbox_metadata = df_gtbbox_metadata.set_index(["frame_id","id"])
        df_gtbbox_metadata["width"] = df_gtbbox_metadata["x1"] - df_gtbbox_metadata["x0"]
        df_gtbbox_metadata["height"] = df_gtbbox_metadata["y1"] - df_gtbbox_metadata["y0"]
        df_gtbbox_metadata["aspect_ratio"] = df_gtbbox_metadata["width"] / df_gtbbox_metadata["height"]
        mu = 0.4185
        std = 0.12016
        #df_gtbbox_metadata["aspect_ratio_is_typical"] = np.logical_and(df_gtbbox_metadata["aspect_ratio"] < mu + std,
        #                                                               df_gtbbox_metadata["aspect_ratio"] > mu - std)
        df_gtbbox_metadata["sequence_id"] = sequence_id

        df_frame_metadata.rename_axis('frame_id')
        df_frame_metadata["seq_name"] = 0
        df_frame_metadata["id"] = df_frame_metadata.index
        df_frame_metadata["frame_id"] = df_frame_metadata.index
        df_frame_metadata["sequence_id"] = sequence_id
        df_frame_metadata["num_pedestrian"] = df_gtbbox_metadata.groupby("frame_id").apply(len).loc[
            df_frame_metadata.index]

        # Assumptions
        df_frame_metadata["is_night"] = 0
        df_frame_metadata["weather"] = "dry"

        # Targets in torch format
        targets = {}
        for key, val in df_gtbbox_metadata.groupby("frame_id"):
            frame_bboxes = np.stack(val.apply(get_bbox, axis=1).values)
            target = [dict(boxes=torch.tensor(frame_bboxes))]
            target[0]["labels"] = torch.tensor([0] * len(target[0]["boxes"]))
            targets[key] = target

        # todo limit with max_samples

        return self.root, targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata