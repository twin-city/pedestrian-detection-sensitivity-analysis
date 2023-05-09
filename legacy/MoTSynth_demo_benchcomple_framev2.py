import os
import pandas as pd
import setuptools.errors
import numpy as np
import os.path as osp
import os
from src.detection.metrics import compute_model_metrics_on_dataset
import matplotlib.pyplot as plt
import os
#%% params of input dataset

# Parameters for results generation
from src.demos.configs import ODD_limit, ODD_criterias, param_heatmap_metrics, metrics, occl_thresh, height_thresh, gtbbox_filtering_all, dict_filter_frames

dict_filter_frames = {
    "Overall": [{}],
    "Day / Night": ({"is_night": 0}, {"is_night": 1}),
    #"Adverse Weather": ({"weather": ["Partially cloudy"]}, {"weather": ["Foggy"]}),
    "Camera Angle": ({"pitch": {"<": -10}}, {"pitch": {">": -10}}),
}

# Which models to study
model_names = ["faster-rcnn_cityscapes", "mask-rcnn_coco"]

# Dataset
dataset_name = "motsynth"
root = "/home/raphael/work/datasets/MOTSynth/"
max_sample = 600  # Uniform sampled in dataset



#%% Get the dataset
from src.preprocessing.motsynth_processing import MotsynthProcessing
motsynth_processor = MotsynthProcessing(root, max_samples=max_sample, video_ids=None)
dataset = motsynth_processor.get_dataset() #todo as class
root, targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = dataset


#todo in abstract dataset class
resolution = (1920, 1080)
results_dir = osp.join("../", "results", dataset_name, f"{dataset_name}{max_sample}")
os.makedirs(results_dir, exist_ok=True)
seq_cofactors = ["is_night"] #todo virrer d'ici

#%% See Dataset Characteristics ==============================

#%% Correlations

from src.utils import compute_correlations
from src.plot_utils import plot_correlations
corr_matrix, p_matrix = compute_correlations(df_frame_metadata.groupby("seq_name").apply(lambda x: x.mean()), seq_cofactors)
plot_correlations(corr_matrix, p_matrix, title="Correlations between metadatas at sequence level")


#%% Height & Occlusion
#todo set dataset plot in dataset object
from src.plot_utils import plot_dataset_statistics
plot_dataset_statistics(df_gtbbox_metadata, results_dir)


#%% Compute metrics and plot them ==============================

#%% What cases do we study ?
from src.utils import compute_models_metrics_from_gtbbox_criteria
gtbbox_filtering_cats = {}
gtbbox_filtering_cats.update(gtbbox_filtering_all)
df_analysis = compute_models_metrics_from_gtbbox_criteria(dataset_name, dataset, df_frame_metadata, gtbbox_filtering_cats, model_names)

#%% Model performance :  Plots MR vs FPPI on frame filtering
from src.plot_utils import plot_fppi_mr_vs_frame_cofactor
plot_fppi_mr_vs_frame_cofactor(df_analysis, dict_filter_frames, ODD_criterias, results_dir="")

#%% do the plot value difference (simplified, each metric)
from src.plot_utils import plot_heatmap_metrics
thresholds = [0.5, 0.9, 0.99]
df_analysis_heatmap = df_analysis[np.isin(df_analysis["threshold"], thresholds)]
plot_heatmap_metrics(df_analysis_heatmap, model_names, metrics, ODD_limit,
                     param_heatmap_metrics=param_heatmap_metrics, results_dir=results_dir)

#%% Study a Use-case ==============================

# Plot an image in particular
from src.plot_utils import plot_image_with_detections
frame_idx = 0
model_name = model_names[0]
plot_thresholds = [0, 0.5, 0.9, 0.9999]
gtbbox_filtering = gtbbox_filtering_all["Overall"]
plot_image_with_detections(dataset, dataset_name, model_name, plot_thresholds, gtbbox_filtering, i=frame_idx)