import unittest
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint
from src.detection.metrics import compute_fp_missratio
import os
import os.path as osp

class testPreprocessTwincity(unittest.TestCase):

    def test_preprocess_twincity(self):
        print("coucou")

        DATASET_DIR = "/home/raphael/work/datasets/PedestrianDetectionSensitivityDatasets/"

        force_recompute = True

        benchmark_params = [
            {"dataset_name": "Twincity-Unreal-v8", "max_samples": 1},
        ]

        # Compute the descriptive markdown table
        from src.dataset.dataset_factory import DatasetFactory
        for param in benchmark_params:
            dataset_name, max_samples = param.values()
            print(dataset_name, max_samples)
            root = osp.join(DATASET_DIR, dataset_name)
            dataset = DatasetFactory.get_dataset(dataset_name, max_samples, root, force_recompute=force_recompute)
