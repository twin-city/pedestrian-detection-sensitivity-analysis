import os

from src.dataset.dataset_factory import DatasetFactory
import unittest
import pandas as pd
from src.utils import subset_dataframe
from configs_path import ROOT_DIR
import os.path as osp

class TestOutput(unittest.TestCase):

    def test_load_datasets(self):


        # Folder should be created though
        output_path = osp.join(ROOT_DIR, "tests/output_3_datasets")
        os.makedirs(output_path, exist_ok=True)

        # Delete if exists
        for file_extension in [".csv", ".md"]:
            if os.path.exists(osp.join(ROOT_DIR, f"tests/output_3_datasets/df_descr{file_extension}")):
                os.remove(osp.join(ROOT_DIR, f"tests/output_3_datasets/df_descr{file_extension}"))

        dataset_names = ["twincity", "motsynth", "EuroCityPerson"]
        max_samples = [50, 600, 30]
        for dataset_name, max_sample in zip(dataset_names, max_samples):
            dataset = DatasetFactory.get_dataset(dataset_name, max_sample)
            dataset.create_markdown_description_table(output_path)
            # dataset_tuple = dataset.get_dataset_as_tuple()
            # root, targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = dataset_tuple


        df_descr_gt = pd.read_csv(osp.join(output_path, "df_descr_gt.csv")).set_index("characteristics")
        df_descr = pd.read_csv(osp.join(output_path, "df_descr.csv")).set_index("characteristics")

        self.assertEqual(df_descr_gt.shape, df_descr.shape)
        pd.testing.assert_frame_equal(df_descr_gt, df_descr)