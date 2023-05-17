import os
from src.utils import target_2_json, target_2_torch
import numpy as np
from configs_path import ROOT_DIR
import json
import pandas as pd
import os.path as osp

class DatasetProcessing:
    def __init__(self, root, max_samples):
        self.root = root
        self.max_samples = max_samples
        np.random.seed(0)

        self.saves_dir = f"{ROOT_DIR}/data/preprocessing/{self.dataset_name}"
        os.makedirs(self.saves_dir, exist_ok=True)

    def __str__(self):
        return self.dataset_name

    def get_dataset(self):
        targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = self.get_annotations_and_imagepaths()

        # Common post-processing
        df_gtbbox_metadata = self.format_gtbbox_metadata(df_gtbbox_metadata)
        df_frame_metadata = self.format_frame_metadata(df_frame_metadata, df_gtbbox_metadata)

        return self.root, targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata



    def load_or_preprocess(self, force_recompute=False):

        # Set the paths
        path_df_gtbbox_metadata = osp.join(self.saves_dir, f"df_gtbbox_{self.max_samples}.csv")
        path_df_frame_metadata = osp.join(self.saves_dir, f"df_frame_{self.max_samples}.csv")
        path_df_sequence_metadata = osp.join(self.saves_dir, f"df_sequence_{self.max_samples}.csv")
        path_target = osp.join(self.saves_dir, f"targets_{self.max_samples}.json")

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
            df_gtbbox_metadata = pd.read_csv(path_df_gtbbox_metadata, index_col=["image_id", "id"])
            df_frame_metadata = pd.read_csv(path_df_frame_metadata, index_col=["frame_id"])
            df_sequence_metadata = pd.read_csv(path_df_sequence_metadata, index_col=["sequence_id"])
            with open(path_target) as jsonFile:
                targets = target_2_torch(json.load(jsonFile))

        # Else, compute them
        else:
            targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = self.preprocess_motsynth()

            # Save dataframes
            df_gtbbox_metadata.to_csv(path_df_gtbbox_metadata)
            df_frame_metadata.to_csv(path_df_frame_metadata)
            df_sequence_metadata.to_csv(path_df_sequence_metadata)
            with open(path_target, 'w') as f:
                json.dump(target_2_json(targets), f)

        return targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata

    def add_dummy_columns(self):
        pass

    def format_gtbbox_metadata(self, df_gtbbox_metadata):

        # Bounding Box Height, Width & Aspect Ratios
        df_gtbbox_metadata["height"] = df_gtbbox_metadata["height"].astype(int)
        df_gtbbox_metadata["width"] = df_gtbbox_metadata["width"].astype(int)
        df_gtbbox_metadata["aspect_ratio"] = df_gtbbox_metadata["width"] / df_gtbbox_metadata["height"]
        mu = 0.4185
        std = 0.12016
        df_gtbbox_metadata["aspect_ratio_is_typical"] = np.logical_and(df_gtbbox_metadata["aspect_ratio"] < mu + std,
                                                                       df_gtbbox_metadata["aspect_ratio"] > mu - std)

        if "ignore-region" not in df_gtbbox_metadata.columns:
            df_gtbbox_metadata["ignore-region"] = 0

        return df_gtbbox_metadata

    def format_frame_metadata(self, df_frame_metadata, df_gtbbox_metadata):
        df_frame_metadata["num_pedestrian"] = df_gtbbox_metadata.groupby("frame_id").apply(len).loc[
            df_frame_metadata.index]

        df_frame_metadata["is_night"] = 1 * df_frame_metadata["is_night"]

        return df_frame_metadata



