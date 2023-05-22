import json
import numpy as np
import torch
import os.path as osp
import pandas as pd
import os
from src.utils import target_2_json, target_2_torch
from src.plot_utils import xywh2xyxy
from .processing import DatasetProcessing
from .coco_processing import subset_dict


def dftarget_2_torch(df):
    targets = df.groupby("frame_id").apply(
        lambda x: x[["x0", "y0", "x1", "y1"]].values).to_dict()
    targets_torch = {}
    for key, val in targets.items():
        # Target and labels
        target = [dict(boxes=torch.tensor(val))]
        target[0]["labels"] = torch.tensor([0] * len(target[0]["boxes"]))
        targets_torch[key] = target
    return targets_torch

def get_scalar_dict(dict, dict_frame_to_scalar):
    scalar_dict = {}
    for key, values in dict.items():
        if key in dict_frame_to_scalar.keys():
            for i, value in enumerate(values):
                scalar_dict[f"{dict_frame_to_scalar[key][i]}"] = value
        else:
            scalar_dict[key] = values
    return scalar_dict


def flatten_list(l):
    return [item for sublist in l for item in sublist]

class MotsynthProcessing(DatasetProcessing):

    def __init__(self, root, max_samples=200, sequence_ids=None):

        self.dataset_name = "motsynth"
        super().__init__(root, max_samples)

        self.delay = 3
        self.frames_dir = f"{root}/frames"
        self.annot_dir = f"{root}/coco annot"
        os.makedirs(self.saves_dir, exist_ok=True)


        self.sequence_ids = sequence_ids
        self.sequence_ids = self.get_usable_sequence_ids()
        self.num_sequences = len(self.sequence_ids)

        #if max_samples is not None:
        #    self.max_num_sample_per_sequence = int(max_samples / self.num_sequences)
        #else:
        self.max_num_sample_per_sequence = 1000  # todo max to 1000 arbirterary
        # self.max_samples_per_sequence = self.max_samples // len(self.get_usable_sequence_ids())

        # Additional info
        self.RESOLUTION = (1920, 1080)
        self.NUM_KEYPOINTS = 22

    # To load from multiple sequences

    def get_usable_sequence_ids(self):
        """
        Get usable sequence ids.
        :return:
        """
        if self.sequence_ids is None:
            exclude_ids_frames = set(["060", "081", "026", "132", "136", "102", "099", "174", "140"])
            sequence_ids_frames = set(np.sort(os.listdir(self.frames_dir)).tolist())
            sequence_ids_json = set([i.replace(".json", "") for i in
                                  os.listdir(self.annot_dir)]) - exclude_ids_frames
            sequence_ids = list(np.sort(list(set.intersection(sequence_ids_frames, sequence_ids_json))))

            if self.max_samples is not None:
                if self.max_samples < len(sequence_ids):
                    sequence_ids = np.random.choice(sequence_ids, self.max_samples, replace=False)
        else:
            sequence_ids = self.sequence_ids
        return sequence_ids



    #todo here should be shared with coco style datasets
    def preprocess_motsynth_sequence(self, sequence_id="004"):

        # Open annotation file
        json_path = f"{self.annot_dir}/{sequence_id}.json"
        with open(json_path) as jsonFile:
            annot_motsynth = json.load(jsonFile)

        for img in annot_motsynth["images"]:
            img["id"] += self.delay

        # If there is subsampling
        if self.max_num_sample_per_sequence < len(annot_motsynth["images"]):
            images_list = annot_motsynth["images"][::len(annot_motsynth["images"]) // self.max_num_sample_per_sequence]
            annot_list = [x for x in annot_motsynth["annotations"] if x["image_id"] in [i["id"] for i in images_list]]
        else:
            images_list = annot_motsynth["images"]
            annot_list = annot_motsynth["annotations"]


        frame_keys_to_keep = ['file_name', 'id', 'frame_n', 'cam_world_pos', 'cam_world_rot', 'height', 'width'] #'ignore_mask' #todo renaming ?
        annot_keys_to_keep = ['id', 'image_id', 'category_id', 'area', 'bbox', 'iscrowd',
                              'num_keypoints', 'ped_id', 'model_id', 'attributes', 'is_blurred', 'keypoints'] # , 'keypoints_3d' 'segmentation',

        # image_types = {key: type(val) for key, val in images_list[0].items()} #todo later

        new_images, new_annots = [], []

        for image in images_list:
            subset_image = subset_dict(image, frame_keys_to_keep)
            subset_image["sequence_id"] = sequence_id
            new_images.append(subset_image)

        for annot in annot_list:
            subset_annot = subset_dict(annot, annot_keys_to_keep)
            subset_annot["sequence_id"] = sequence_id
            new_annots.append(subset_annot)

        infos = annot_motsynth["info"]
        infos["sequence_id"] = sequence_id

        #todo filter when too many annots because of delay
        image_id_annot = set(np.unique([img["image_id"] for img in new_annots]))
        image_id_image = set([img["id"] for img in new_images])
        intersection_id = set.intersection(image_id_annot, image_id_image)
        new_annots = [annot for annot in new_annots if annot["image_id"] in intersection_id]
        new_images = [img for img in new_images if img["id"] in intersection_id]

        return infos, new_images, new_annots


    #todo a part of this can be factorized
    def preprocess(self):
        """
        Get a full coco style Dataset
        :return:
        """

        infos, new_images, new_annots = [], [], []
        for sequence in self.sequence_ids:
            print(f"Checking sequence {sequence}")
            infos_sequence, new_images_sequence, new_annots_sequence = self.preprocess_motsynth_sequence(sequence_id=sequence)

            new_images.append(new_images_sequence)
            new_annots.append(new_annots_sequence)
            infos.append(infos_sequence)

        new_images = [item for sublist in new_images for item in sublist]
        new_annots = [item for sublist in new_annots for item in sublist]

        #%% Now how to transform to dataframe ? To a scalar that can be in a dataframe
        dict_frame_to_scalar = {
            "cam_world_rot": ("yaw", "pitch", "roll"), # todo Has to be checked
            "cam_world_pos": ("x", "y", "z"),
        }

        dict_annot_to_scalar = {
            "attributes": [f"attribute_{i}" for i in range(11)],
            "bbox": ("x0", "y0", "x1", "y1"),
            "keypoints": flatten_list([[f"x_{i}", f"y_{i}", f"o_{i}"] for i in range(22)]),
        }

        df_frame_metadata = pd.DataFrame([get_scalar_dict(img, dict_frame_to_scalar) for img in new_images])
        df_gtbbox_metadata = pd.DataFrame([get_scalar_dict(annot, dict_annot_to_scalar) for annot in new_annots])
        df_sequence_metadata = pd.DataFrame(infos)

        #%% Renaming
        df_sequence_metadata = df_sequence_metadata.set_index("sequence_id")
        df_frame_metadata["id"] = df_frame_metadata["id"].astype(str)
        df_gtbbox_metadata[['image_id', 'id']] = df_gtbbox_metadata[['image_id', 'id']].astype(str)
        df_frame_metadata = df_frame_metadata.rename(columns={"id": "frame_id"}).set_index("frame_id")
        df_gtbbox_metadata = df_gtbbox_metadata.rename(columns={"image_id": "frame_id"}).set_index(["frame_id", "id"])

        # Targets
        targets = dftarget_2_torch(df_gtbbox_metadata)

        #todo temporary adding sequence information to
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Add Sequence information to Frame

        # Add sequence information
        sequence_columns = ['is_night', "weather"]
        for sequence_id in df_sequence_metadata.index:
            df_frame_metadata.loc[df_frame_metadata["sequence_id"] == sequence_id, sequence_columns] = df_sequence_metadata.loc[sequence_id, sequence_columns].values

        # Set Weather variables
        """
        We want : 
        - Dry : CLEAR, OVERCAST, EXTRASUNNY, CLOUDS
        - Rain : RAIN, THUNDER
        - Reduced Visibility : SMOG, FOGGY, BLIZZARD
        """
        dry_cats = ["CLEAR", "OVERCAST", "EXTRASUNNY", "CLOUDS"]
        rainy_cats = ["RAIN", "THUNDER"]
        reduced_visibility_cats = ["SMOG", "FOGGY", "BLIZZARD"]
        df_frame_metadata["weather_original"] = df_frame_metadata["weather"]
        df_frame_metadata["weather"] = 0
        df_frame_metadata.loc[np.isin(df_frame_metadata["weather_original"], dry_cats), "weather"] = "dry"
        df_frame_metadata.loc[np.isin(df_frame_metadata["weather_original"], rainy_cats), "rainy_cats"] = "rainy"
        df_frame_metadata.loc[np.isin(df_frame_metadata["weather_original"], reduced_visibility_cats), "weather"] = "reduced visibility"

        # Occlusions
        keypoints_label_names = [f"o_{i}" for i in range(self.NUM_KEYPOINTS)]
        df_gtbbox_metadata["occlusion_rate"] = df_gtbbox_metadata[keypoints_label_names].apply(lambda x: (2 - x)).mean(axis=1)

        return targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata