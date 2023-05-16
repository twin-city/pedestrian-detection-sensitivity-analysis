import numpy as np
from configs_path import ROOT_DIR

class DatasetProcessing:
    def __init__(self, root, max_samples):
        self.root = root
        self.max_samples = max_samples
        np.random.seed(0)

        self.saves_dir = f"{ROOT_DIR}/data/preprocessing/{self.dataset_name}"

    def __str__(self):
        return self.dataset_name

    def get_dataset(self):
        """
        Get all the annotations and image file paths from the original dataset.
        :return:
        """
        raise NotImplementedError

    def add_dummy_columns(self):
        pass

    def format_gtbbox_metadata(self, df_gtbbox_metadata):
        mu = 0.4185
        std = 0.12016
        df_gtbbox_metadata["aspect_ratio_is_typical"] = np.logical_and(df_gtbbox_metadata["aspect_ratio"] < mu + std,
                                                                       df_gtbbox_metadata["aspect_ratio"] > mu - std)
        return df_gtbbox_metadata

    def format_frame_metadata(self, df_frame_metadata, df_gtbbox_metadata):
        df_frame_metadata["num_pedestrian"] = df_gtbbox_metadata.groupby("frame_id").apply(len).loc[
            df_frame_metadata.index]
        return df_frame_metadata



