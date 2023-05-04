import os
import pandas as pd
import setuptools.errors
import numpy as np
import os.path as osp


#%% params of input dataset

dataset_name = "motsynth"
root_motsynth = "/home/raphael/work/datasets/MOTSynth/"
max_sample = 600  # Uniform sampled in dataset


metrics = ["MR", "FPPI"]
ODD_criterias = {
    "MR": 0.5,
    "FPPI": 5,
}

occl_thresh = [0.35, 0.8]
height_thresh = [20, 50, 120]
resolution = (1920, 1080)

#%% Get the dataset
from src.preprocessing.motsynth_processing import MotsynthProcessing
motsynth_processor = MotsynthProcessing(root_motsynth, max_samples=max_sample, video_ids=None)
dataset = motsynth_processor.get_dataset() #todo as class
root, targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = dataset

#todo in processing
df_gtbbox_metadata["aspect_ratio"] = 1/df_gtbbox_metadata["aspect_ratio"]
mu = 0.4185
std = 0.12016
df_gtbbox_metadata["aspect_ratio_is_typical"] = np.logical_and(df_gtbbox_metadata["aspect_ratio"] < mu+std,  df_gtbbox_metadata["aspect_ratio"] > mu-std)
df_frame_metadata["num_person"] = df_gtbbox_metadata.groupby("frame_id").apply(len).loc[df_frame_metadata.index]


#%% Multiple plots
from src.utils import compute_correlations, plot_correlations
#corr_matrix, p_matrix = compute_correlations(df_frame_metadata.groupby("seq_name").apply(lambda x: x.mean()), seq_cofactors)
#plot_correlations(corr_matrix, p_matrix, title="Correlations between metadatas at sequence level")


#%% Now plot the multiple cases !!!!!!
from src.detection.metrics import compute_model_metrics_on_dataset
model_names = ["faster-rcnn_cityscapes", "mask-rcnn_coco"]


#%% What cases do we study ?


gtbbox_filtering_all = {
    "Overall": {
        "occlusion_rate": (0.99, "max"),  # Not fully occluded
        "height": (height_thresh[1], "min"),
    },
}

gtbbox_filtering_height_cats = {
    "No occlusion": {
        "occlusion_rate": (0.01, "max"),  # Unoccluded
        "height": (height_thresh[1], "min"),
    },
    "Partial occlusion": {
        "occlusion_rate": (0.01, "min"),
        "occlusion_rate": (occl_thresh[0], "max"),
        "height": (height_thresh[1], "min"),
    },
    "Heavy occlusion": {
        "occlusion_rate": (0.01, "max"),  # Unoccluded
        "occlusion_rate": (occl_thresh[0], "min"),
        "occlusion_rate": (occl_thresh[1], "max"),
        "height": (height_thresh[1], "min"),
    },
}

gtbbox_filtering_aspectratio_cats = {
    "Typical aspect ratios": {
        "aspect_ratio_is_typical": 1,  #
        "occlusion_rate": (0.01, "max"),  # Unoccluded
    },
    "Atypical aspect ratios": {
        "aspect_ratio_is_typical": 0,  #
        "occlusion_rate": (0.01, "max"),  # Unoccluded
    },
}

gtbbox_filtering_occlusion_cats = {
    "near": {
        "occlusion_rate": (0.01, "max"),  # Unoccluded
        "height": (height_thresh[2], "min"),
    },
    "medium": {
        "occlusion_rate": (0.01, "max"),  # Unoccluded
        "height": (height_thresh[2], "max"),
        "height": (height_thresh[1], "min"),
    },
    "far": {
        "occlusion_rate": (0.01, "max"),  # Unoccluded
        "height": (height_thresh[1], "max"),
        "height": (height_thresh[0], "min"),
    },
}

gtbbox_filtering_cats = {}
gtbbox_filtering_cats.update(gtbbox_filtering_all)
gtbbox_filtering_cats.update(gtbbox_filtering_height_cats)
gtbbox_filtering_cats.update(gtbbox_filtering_aspectratio_cats)
gtbbox_filtering_cats.update(gtbbox_filtering_occlusion_cats)



#%% Compute for multiple criterias

df_metrics_criteria_list = []
for key, val in gtbbox_filtering_cats.items():
    df_results_aspectratio = pd.concat(
        [compute_model_metrics_on_dataset(model_name, dataset_name, dataset, val, device="cuda")[0] for
         model_name in model_names])
    df_results_aspectratio["gtbbox_filtering_cat"] = key
    df_metrics_criteria_list.append(df_results_aspectratio)
df_metrics_criteria = pd.concat(df_metrics_criteria_list, axis=0)

#%%
import matplotlib.pyplot as plt
from src.utils import plot_ffpi_mr_on_ax

fig, ax = plt.subplots(3, 3, figsize=(10,10), sharey=True)
plot_ffpi_mr_on_ax(df_metrics_criteria, "Overall", ax[0,0], odd=ODD_criterias)
plot_ffpi_mr_on_ax(df_metrics_criteria, "Typical aspect ratios", ax[0,1])
plot_ffpi_mr_on_ax(df_metrics_criteria, "Atypical aspect ratios", ax[0,2])
plot_ffpi_mr_on_ax(df_metrics_criteria, "near", ax[1,0])
plot_ffpi_mr_on_ax(df_metrics_criteria, "medium", ax[1,1])
plot_ffpi_mr_on_ax(df_metrics_criteria, "far", ax[1,2])
plot_ffpi_mr_on_ax(df_metrics_criteria, "No occlusion", ax[2,0])
plot_ffpi_mr_on_ax(df_metrics_criteria, "Partial occlusion", ax[2,1])
plot_ffpi_mr_on_ax(df_metrics_criteria, "Heavy occlusion", ax[2,2])
plt.tight_layout()
plt.show()


#%% After bench, do the plot value difference (simplified, each metric)











