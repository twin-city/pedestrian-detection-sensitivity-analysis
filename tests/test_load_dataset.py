import os

from src.dataset.dataset_factory import DatasetFactory
import unittest
import pandas as pd
from src.utils import subset_dataframe
from configs_path import ROOT_DIR
import os.path as osp

class TestOutput(unittest.TestCase):


    def test_load_ecp_dataset(self):

        # Folder should be created though
        output_path = osp.join(ROOT_DIR, "tests/output_ecp_dataset")
        os.makedirs(output_path, exist_ok=True)

        # Delete if exists
        for file_extension in [".csv", ".md"]:
            if os.path.exists(osp.join(output_path, f"df_descr{file_extension}")):
                os.remove(osp.join(output_path, f"df_descr{file_extension}"))

        dataset_names = ["ecp_small"]
        DATASET_ROOT = "/home/raphael/work/datasets/PedestrianDetectionSensitivityDatasets"
        roots = [osp.join(DATASET_ROOT, x) for x in ["ecp_small"]]
        max_samples = [50, None, None]
        for root, dataset_name, max_sample in zip(roots, dataset_names, max_samples):

            dataset = DatasetFactory.get_dataset(dataset_name, max_sample, root, force_recompute=True)
            dataset.create_markdown_description_table(output_path)

        # Assert we have the same dataset of description
        df_descr_gt = pd.read_csv(osp.join(output_path, "df_descr_gt.csv")).set_index("characteristics")
        df_descr = pd.read_csv(osp.join(output_path, "df_descr.csv")).set_index("characteristics")
        self.assertEqual(df_descr_gt.shape, df_descr.shape)
        pd.testing.assert_frame_equal(df_descr_gt, df_descr)


    def test_load_motsynth_dataset(self):

        # Folder should be created though
        output_path = osp.join(ROOT_DIR, "tests/output_motsynth_dataset")
        os.makedirs(output_path, exist_ok=True)

        # Delete if exists
        for file_extension in [".csv", ".md"]:
            if os.path.exists(osp.join(output_path, f"df_descr{file_extension}")):
                os.remove(osp.join(output_path, f"df_descr{file_extension}"))

        dataset_names = ["motsynth_small"]
        DATASET_ROOT = "/home/raphael/work/datasets/PedestrianDetectionSensitivityDatasets"
        roots = [osp.join(DATASET_ROOT, x) for x in ["motsynth_small"]]
        max_samples = [50, None, None]
        for root, dataset_name, max_sample in zip(roots, dataset_names, max_samples):

            dataset = DatasetFactory.get_dataset(dataset_name, max_sample, root, force_recompute=True)
            dataset.create_markdown_description_table(output_path)
            # dataset_tuple = dataset.get_dataset_as_tuple()
            # root, targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = dataset_tuple

        # Assert we have the same dataset of description
        df_descr_gt = pd.read_csv(osp.join(output_path, "df_descr_gt.csv")).set_index("characteristics")
        df_descr = pd.read_csv(osp.join(output_path, "df_descr.csv")).set_index("characteristics")
        self.assertEqual(df_descr_gt.shape, df_descr.shape)
        pd.testing.assert_frame_equal(df_descr_gt, df_descr)


    def test_maxsample_dataset(self):
        dataset_names = ["ecp_small"]
        DATASET_ROOT = "/home/raphael/work/datasets/PedestrianDetectionSensitivityDatasets"
        output_path = osp.join(ROOT_DIR, "tests/dataset_building/max_sample")
        os.makedirs(output_path, exist_ok=True)
        roots = [osp.join(DATASET_ROOT, x) for x in ["ecp_small"]]
        max_samples = [1, None, None]
        for root, dataset_name, max_sample in zip(roots, dataset_names, max_samples):
            dataset = DatasetFactory.get_dataset(dataset_name, max_sample, root, force_recompute=True)
            dataset.create_markdown_description_table(output_path)


    """


    def test_load_datasets(self):

        # Folder should be created though
        output_path = osp.join(ROOT_DIR, "tests/output_3_datasets")
        os.makedirs(output_path, exist_ok=True)

        # Delete if exists
        for file_extension in [".csv", ".md"]:
            if os.path.exists(osp.join(ROOT_DIR, f"tests/output_3_datasets/df_descr{file_extension}")):
                os.remove(osp.join(ROOT_DIR, f"tests/output_3_datasets/df_descr{file_extension}"))


        dataset_names = ["twincity", "motsynth_small", "ecp_small"]
        DATASET_ROOT = "/home/raphael/work/datasets/PedestrianDetectionSensitivityDatasets"
        roots = [osp.join(DATASET_ROOT, x) for x in ["twincity-Unreal/v5", "motsynth_small", "ecp_small"]]
        max_samples = [50, None, None]
        for root, dataset_name, max_sample in zip(roots, dataset_names, max_samples):

            dataset = DatasetFactory.get_dataset(dataset_name, max_sample, root)
            dataset.create_markdown_description_table(output_path)
            # dataset_tuple = dataset.get_dataset_as_tuple()
            # root, targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = dataset_tuple

        # Assert we have the same dataset of description
        df_descr_gt = pd.read_csv(osp.join(output_path, "df_descr_gt.csv")).set_index("characteristics")
        df_descr = pd.read_csv(osp.join(output_path, "df_descr.csv")).set_index("characteristics")
        self.assertEqual(df_descr_gt.shape, df_descr.shape)
        pd.testing.assert_frame_equal(df_descr_gt, df_descr)
    """