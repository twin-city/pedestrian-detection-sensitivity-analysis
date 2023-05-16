import os

import numpy as np
from configs_path import ROOT_DIR

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



