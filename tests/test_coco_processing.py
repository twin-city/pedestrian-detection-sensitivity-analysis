import os

from src.dataset.dataset_factory import DatasetFactory
import unittest
import pandas as pd
from src.utils import subset_dataframe
from configs_path import ROOT_DIR
import os.path as osp
from src.preprocessing.coco_processing import COCOProcessing
from src.dataset.dataset import Dataset

class TestOutput(unittest.TestCase):

    def test_load_coco_datasets(self):

        output_path = osp.join(ROOT_DIR, "tests/output_anycoco_dataset")
        os.makedirs(output_path, exist_ok=True)

        dataset_name = "Fudan"
        max_samples = 200
        root = "/home/raphael/work/datasets/PennFudanPed"
        coco_json_path = "/home/raphael/work/datasets/PennFudanPed/coco.json"

        cocoprocess = COCOProcessing(root, coco_json_path, dataset_name, max_samples).get_dataset()
        dataset = Dataset(dataset_name, max_samples, *cocoprocess)
        dataset.create_markdown_description_table(output_path)



