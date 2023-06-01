import unittest
import matplotlib.pyplot as plt
import os.path as osp
from src.detection.detector import Detector
from src.detection.metrics import detection_metric
from src.demos.configs import ODD_limit, ODD_criterias, param_heatmap_metrics, metrics, \
    gtbbox_filtering_all, dict_filter_frames, gtbbox_filtering_cats
from src.plot_utils import plot_results_img
from src.dataset.dataset_factory import DatasetFactory


class testPlotImg(unittest.TestCase):

    def test_plot_img(self):

        # Parameters for results generation
        model_idx = 0
        threshold = 0.5
        frame_idx = 4
        title = "coucou"

        # Parameters to factorize
        model_names = ["faster-rcnn_cityscapes", "mask-rcnn_coco"]
        DATASET_DIR = "/home/raphael/work/datasets/PedestrianDetectionSensitivityDatasets/" #todo change this



        benchmark_params = [
            {"dataset_name": "Twincity-Unreal-v8", "max_samples": 20},
            # {"dataset_name": "ecp_small", "max_samples": 30},
            # {"dataset_name": "motsynth_small", "max_samples": 30},
            # {"dataset_name": "PennFudanPed", "max_samples": 200},
        ]

        # Compute the descriptive markdown table
        param = benchmark_params[0]
        dataset_name, max_samples = param.values()
        model_name = model_names[model_idx]
        root = osp.join(DATASET_DIR, dataset_name)
        dataset = DatasetFactory.get_dataset(dataset_name, max_samples, root=root, force_recompute=False)
        root, targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = dataset.get_dataset_as_tuple()

        # Perform detection and compute metrics
        detector = Detector(model_name, device="cuda")
        preds = detector.get_preds_from_files(dataset_name, root, df_frame_metadata)
        gtbbox_filtering = gtbbox_filtering_all
        metric = detection_metric(gtbbox_filtering)
        df_mr_fppi, df_gt_bbox = metric.compute(dataset_name, model_name, preds, targets, df_gtbbox_metadata,
                                                gtbbox_filtering)

        # Compute the metrics
        img_path, frame_id = osp.join(root, df_frame_metadata.iloc[frame_idx]["file_name"]), df_frame_metadata.index[frame_idx]

        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        plot_results_img(img_path, frame_id, preds=preds, targets=targets,
                         df_gt_bbox=df_gt_bbox, threshold=threshold, ax=ax,
                         title=title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


