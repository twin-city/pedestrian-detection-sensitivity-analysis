import os

import matplotlib.pyplot as plt
import os.path as osp
from src.detection.detector import Detector
from src.detection.metrics import detection_metric
from src.demos.configs import gtbbox_filtering_all
from src.plot_utils import plot_results_img
from src.dataset.dataset_factory import DatasetFactory
from src.detection.detector_factory import DetectorFactory

# Parameters for results generation
model_idx = 0
threshold = 0.5
frame_idx = 0
force_recompute = False

plot_gtbbox = True
plot_pred = False

# Which to plot ??
from src.utils import subset_dataframe
filter_frame = {"weather": "Rain"}
filter_frame = {"is_night": 0, "weather": "Rain", "pitch": {">": -10}}

filter_frame = {"is_night": 0, "weather": "Rain", "pitch": -30.0}
filter_frame = {"pitch": {">": -60, "<": -20}, "weather": "Rain", "is_night": 0}

# Parameters to factorize
model_names = ["faster-rcnn_cityscapes", "mask-rcnn_coco"]
DATASET_DIR = "/home/raphael/work/datasets/PedestrianDetectionSensitivityDatasets/"  # todo change this

benchmark_params = [
    {"dataset_name": "Twincity-Unreal-v9", "max_samples": 1},
    # {"dataset_name": "ecp_small", "max_samples": 30},
    # {"dataset_name": "motsynth_small", "max_samples": 30},
    # {"dataset_name": "PennFudanPed", "max_samples": 200},
]

gtbbox_filtering = {'ignore_region': 0,
                    'occlusion_rate': {'<': 0.99},
                    'height': {'>': 25}}

# Compute the descriptive markdown table
param = benchmark_params[0]
dataset_name, max_samples = param.values()
model_name = model_names[model_idx]
root = osp.join(DATASET_DIR, dataset_name)
dataset = DatasetFactory.get_dataset(dataset_name, max_samples, root=root, force_recompute=force_recompute)
root, targets, df_gtbbox_metadata, df_frame_metadata_0, df_sequence_metadata = dataset.get_dataset_as_tuple()

os.makedirs(f"../../results/plots_twincity/", exist_ok=True)

for is_night in [0,1]:
    for weather in ["clear", "rain", "snow"]:
        for pitch in [0.0, -30.0, -70.0]: #Careful, set as float


            #is_night = 1
            #weather = "Clear Sky"
            #pitch = -70.0

            #is_night = 1
            #weather = "Clear Sky"

            # weather = "Clear Sky"




            print(is_night, pitch, weather)

            filter_frame = {"pitch": float(pitch), "weather": weather, "is_night": is_night}

            import numpy as np
            #if pitch == 0 and is_night == 0 and weather=="Rain":
            #    continue
            #if pitch == -30 and is_night == 0 and weather=="Rain":
            #    continue
            #if pitch == -30 and is_night == 1 and weather=="Rain":
            #    continue


            title = f"Pitch: {pitch}, Weather: {weather}, is_night: {is_night}"


            df_frame_metadata = subset_dataframe(df_frame_metadata_0.copy(), filter_frame) #todo perform the filtering



            # Perform detection and compute metrics
            detector = DetectorFactory.get_detector(model_name, device="cuda")
            #detector = Detector(model_name, device="cuda")
            preds = detector.get_preds_from_files(dataset_name, root, df_frame_metadata)
            metric = detection_metric(gtbbox_filtering)
            df_mr_fppi, df_gt_bbox = metric.compute(dataset_name, model_name, preds, targets, df_gtbbox_metadata,
                                                    gtbbox_filtering)

            # Adapt the plot
            if plot_pred:
                pass
            elif plot_gtbbox:
                pass
            else:
                pass

            # name = "Snapshot-2023-06-05_120320-8297400"
            #df_gtbbox_metadata.loc['Snapshot-2023-06-05_120133-6876815', "ignore_region"]
            #df_gt_bbox.loc['Snapshot-2023-06-05_120133-6876815', "matched"].min()

            # Compute the metrics
            img_path, frame_id = osp.join(root, df_frame_metadata.iloc[frame_idx]["file_name"]), df_frame_metadata.index[frame_idx]


            # Plot gt
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            plot_results_img(img_path, frame_id, preds=None, targets=targets,
                             df_gt_bbox=df_gt_bbox, threshold=threshold, ax=ax,
                             title=title)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"results/plots_twincity/{is_night}_{weather}_{pitch}_GT.png")
            plt.show()


            # Plot preds
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            plot_results_img(img_path, frame_id, preds=preds, targets=None,
                             df_gt_bbox=df_gt_bbox, threshold=threshold, ax=ax,
                             title=title)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"results/plots_twincity/{is_night}_{weather}_{pitch}_Preds.png")
            plt.show()

            # Plot preds + gt
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            plot_results_img(img_path, frame_id, preds=preds, targets=targets,
                             df_gt_bbox=df_gt_bbox, threshold=threshold, ax=ax,
                             title=title)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"results/plots_twincity/{is_night}_{weather}_{pitch}_Preds+GT.png")
            plt.show()