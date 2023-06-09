import os
from src.utils import target_2_json, target_2_torch
import numpy as np
from configs_path import ROOT_DIR
import json
import pandas as pd
import os.path as osp

from .preprocessing_utils import *


def df_index_int2str(df):
    if df.index.name is not None:
        if df.index.dtype == int:
            df.index = df.index.astype(str)
    elif df.index.names is not None:
        names = df.index.names
        df = df.reset_index()
        for name in names:
            df[name] = df[name].astype(str)
        df = df.set_index(names)
    else:
        raise ValueError("Dataframe should have an index or index names")
    return df

class DatasetProcessing:
    def __init__(self, root, max_samples_per_sequence, task):
        self.root = root
        self.max_samples_per_sequence = max_samples_per_sequence
        np.random.seed(0)
        self.saves_dir = f"{ROOT_DIR}/cache/preprocessing/{self.dataset_name}"
        os.makedirs(self.saves_dir, exist_ok=True)
        self.task = task

    def __str__(self):
        return self.dataset_name

    def get_dataset(self, force_recompute=False):
        targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = self.load_or_preprocess(force_recompute)

        # Common post-processing
        df_gtbbox_metadata = self.format_gtbbox_metadata(df_gtbbox_metadata)
        df_frame_metadata = self.format_frame_metadata(df_frame_metadata, df_gtbbox_metadata)

        return self.root, targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata


    def preprocess_specific(self, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata):
        print("Dummy preprocess specific")
        return df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata

    def load_or_preprocess(self, force_recompute=False):

        # Set the paths
        path_df_gtbbox_metadata = osp.join(self.saves_dir, f"df_gtbbox_{self.max_samples_per_sequence}.csv")
        path_df_frame_metadata = osp.join(self.saves_dir, f"df_frame_{self.max_samples_per_sequence}.csv")
        path_df_sequence_metadata = osp.join(self.saves_dir, f"df_sequence_{self.max_samples_per_sequence}.csv")
        path_target = osp.join(self.saves_dir, f"targets_{self.max_samples_per_sequence}.json")

        # Check if all files exist
        exist_all_paths = True
        file_paths = [path_df_gtbbox_metadata, path_df_frame_metadata, path_df_sequence_metadata, path_target]
        for file_path in file_paths:
            if not os.path.exists(file_path):
                exist_all_paths = False

        # If all files exist, load them
        if exist_all_paths and not force_recompute:
            print("Loading previously computed dataset")
            # Load it
            df_gtbbox_metadata = pd.read_csv(path_df_gtbbox_metadata, index_col=["frame_id", "id"])
            df_frame_metadata = pd.read_csv(path_df_frame_metadata, index_col=["frame_id"])
            df_sequence_metadata = pd.read_csv(path_df_sequence_metadata, index_col=["sequence_id"])
            with open(path_target) as jsonFile:
                targets = target_2_torch(json.load(jsonFile))

            df_frame_metadata = df_index_int2str(df_frame_metadata)
            df_gtbbox_metadata = df_index_int2str(df_gtbbox_metadata)


        # Else, compute them
        else:
            targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = self.preprocess(force_recompute=force_recompute)
            df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = self.preprocess_specific(df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata)

            # Save dataframes
            df_gtbbox_metadata.to_csv(path_df_gtbbox_metadata)
            df_frame_metadata.to_csv(path_df_frame_metadata)
            df_sequence_metadata.to_csv(path_df_sequence_metadata)
            with open(path_target, 'w') as f:
                json.dump(target_2_json(targets), f)

        return targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata



    def get_sequence_dict(self):
        raise NotImplementedError

    def preprocess_sequence(self, sequence_id, img_sequence_dir, annot_sequence_dir, force_recompute=False):
        raise NotImplementedError

    #todo a part of this can be factorized
    def preprocess(self, force_recompute=False):
        """
        Get a full coco style Dataset
        :return:
        """

        infos, new_images, new_annots = [], [], []
        for sequence_id, (img_sequence_dir, annot_sequence_dir) in self.get_sequence_dict().items():

            infos_sequence, new_images_sequence, new_annots_sequence = self.preprocess_sequence(sequence_id,
                                                                                                img_sequence_dir,
                                                                                                annot_sequence_dir,
                                                                                                force_recompute=force_recompute)

            # Limit the number of samples per sequence
            if self.max_samples_per_sequence is not None:
                new_images_sequence = new_images_sequence[:self.max_samples_per_sequence]
                new_annots_sequence = [annot for annot in new_annots_sequence if annot["image_id"] in [img["id"] for img in new_images_sequence]]

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

        #todo temporary adding sequence information to (specific MoTSynth)
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Add Sequence information to Frame


        return targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata


    def add_dummy_columns(self):
        pass

    def format_gtbbox_metadata(self, df_gtbbox_metadata):

        # Bounding Box Height, Width & Aspect Ratios
        df_gtbbox_metadata["height"] = df_gtbbox_metadata["y1"] - df_gtbbox_metadata["y0"]
        df_gtbbox_metadata["width"] = df_gtbbox_metadata["x1"] - df_gtbbox_metadata["x0"]

        df_gtbbox_metadata["height"] = df_gtbbox_metadata["height"].astype(int)
        df_gtbbox_metadata["width"] = df_gtbbox_metadata["width"].astype(int)
        df_gtbbox_metadata["aspect_ratio"] = df_gtbbox_metadata["width"] / df_gtbbox_metadata["height"]
        mu = 0.4185
        std = 0.12016
        df_gtbbox_metadata["aspect_ratio_is_typical"] = np.logical_and(df_gtbbox_metadata["aspect_ratio"] < mu + std,
                                                                       df_gtbbox_metadata["aspect_ratio"] > mu - std)

        if "ignore_region" not in df_gtbbox_metadata.columns:
            df_gtbbox_metadata["ignore_region"] = 0

        return df_gtbbox_metadata

    def format_frame_metadata(self, df_frame_metadata, df_gtbbox_metadata):
        df_frame_metadata["num_pedestrian"] = df_gtbbox_metadata.groupby("frame_id").apply(len).loc[
            df_frame_metadata.index]

        if df_frame_metadata["num_pedestrian"].min() == 0:
            raise ValueError("No pedestrian in at least one frame of the dataset")

        df_frame_metadata["is_night"] = 1 * df_frame_metadata["is_night"]

        return df_frame_metadata

    @staticmethod
    def add_weather_cats(df):
        weather_cats_renaming = {
            # Dry Weather
            "sunny": "clear",
            "clear": "clear",
            "clouds": "clear",
            "overcast": "clear",
            # Rainy Weather
            "rain": "rain",
            "thunder": "rain",
            # Reduced Visibility
            "smog": "reduced visibility",
            "foggy": "reduced visibility",
            "snow": "reduced visibility",
        }

        df["weather_cats"] = df["weather"].replace(weather_cats_renaming)
        return df



