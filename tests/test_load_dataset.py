import os
import unittest
import pandas as pd
import os.path as osp
from src.dataset.dataset_factory import DatasetFactory
from src.utils import subset_dataframe
from configs_path import ROOT_DIR

class TestOutput(unittest.TestCase):
    def create_output_folder(self, output_path):
        os.makedirs(output_path, exist_ok=True)

    def delete_existing_files(self, output_path):
        file_extensions = [".csv", ".md"]
        for file_extension in file_extensions:
            file_path = osp.join(output_path, f"df_descr{file_extension}")
            if os.path.exists(file_path):
                os.remove(file_path)


    def perform_test_load_dataset(self, dataset_name, root, max_samples):

        output_path = osp.join(ROOT_DIR, f"tests/outputs/test_load_dataset/{dataset_name}")
        self.create_output_folder(output_path)
        self.delete_existing_files(output_path)

        dataset = DatasetFactory.get_dataset(dataset_name, max_samples, root, force_recompute=True)
        dataset.create_markdown_description_table(output_path)

        df_descr_gt = pd.read_csv(osp.join(output_path, "df_descr_gt.csv")).set_index("characteristics")
        df_descr = pd.read_csv(osp.join(output_path, "df_descr.csv")).set_index("characteristics")
        self.assertEqual(df_descr_gt.shape, df_descr.shape)
        pd.testing.assert_frame_equal(df_descr_gt, df_descr)


    def test_load_twincity_dataset(self):
        dataset_name = "twincity"
        root = "/home/raphael/work/datasets/PedestrianDetectionSensitivityDatasets/twincity-Unreal/v5"
        max_samples = 1
        self.perform_test_load_dataset(dataset_name, root, max_samples)

    def test_load_ecp_dataset(self):
        dataset_name = "ecp_small"
        DATASET_ROOT = "/home/raphael/work/datasets/PedestrianDetectionSensitivityDatasets"
        root = osp.join(DATASET_ROOT, dataset_name)
        max_samples = 50
        self.perform_test_load_dataset(dataset_name, root, max_samples)

    def test_load_motsynth_dataset(self):
        dataset_name = "motsynth_small"
        DATASET_ROOT = "/home/raphael/work/datasets/PedestrianDetectionSensitivityDatasets"
        root = osp.join(DATASET_ROOT, dataset_name)
        max_samples = 50
        self.perform_test_load_dataset(dataset_name, root, max_samples)


    #todo factorize
    def test_load_coco_datasets(self):
        import os
        from configs_path import ROOT_DIR
        import os.path as osp
        from src.preprocessing.coco_processing import COCOProcessing
        from src.dataset.dataset import Dataset

        output_path = osp.join(ROOT_DIR, "tests/outputs/test_load_dataset/coco")
        os.makedirs(output_path, exist_ok=True)
        dataset_name = "Fudan"
        max_samples = 200
        root = "/home/raphael/work/datasets/PennFudanPed"
        coco_json_path = "/home/raphael/work/datasets/PedestrianDetectionSensitivityDatasets/PennFudanPed/coco.json"
        cocoprocess = COCOProcessing(root, coco_json_path, dataset_name, max_samples).get_dataset()
        dataset = Dataset(dataset_name, max_samples, *cocoprocess)
        dataset.create_markdown_description_table(output_path)
