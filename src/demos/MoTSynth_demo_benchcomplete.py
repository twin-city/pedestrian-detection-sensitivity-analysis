import os
import pandas as pd
import setuptools.errors
import numpy as np
import os.path as osp

#%% params of input dataset

dataset_name = "motsynth"
root_motsynth = "/home/raphael/work/datasets/MOTSynth/"
max_sample = 600  # Uniform sampled in dataset

seq_cofactors = ["adverse_weather", "is_night", "pitch"]
bbox_cofactors = ["height", "aspect_ratio", "is_crowd", "occlusion_rate"]

metrics = ["MR", "FPPI"]
ODD_criterias = {
    "MR": 0.5,
    "FPPI": 5,
}



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
corr_matrix, p_matrix = compute_correlations(df_frame_metadata.groupby("seq_name").apply(lambda x: x.mean()), seq_cofactors)
plot_correlations(corr_matrix, p_matrix, title="Correlations between metadatas at sequence level")


#%% Now plot the multiple cases !!!!!!

from src.detection.metrics import compute_model_metrics_on_dataset
model_names = ["faster-rcnn_cityscapes", "mask-rcnn_coco"]

#%% Height

occl_thresh = [0.35, 0.8]
height_thresh = [20, 50, 120]

import matplotlib.pyplot as plt

df_gtbbox_metadata.hist("height", bins=200)
plt.xlim(0, 300)
plt.axvline(height_thresh[0], c="red")
plt.axvline(height_thresh[1], c="red")
plt.axvline(height_thresh[2], c="red")
plt.show()

#%% occlusion


criteria = "occlusion_criteria"
occl_thresh = [0.35, 0.8]

gtbbox_filtering_height_cats = {
    "unoccluded": {
        "occlusion_rate": (0.01, "max"),  # Unoccluded
        "height": (height_thresh[1], "min"),
    },
    "partial": {
        "occlusion_rate": (0.01, "min"),
        "occlusion_rate": (occl_thresh[0], "max"),
        "height": (height_thresh[1], "min"),
    },
    "heavy": {
        "occlusion_rate": (0.01, "max"),  # Unoccluded
        "occlusion_rate": (occl_thresh[0], "min"),
        "occlusion_rate": (occl_thresh[1], "max"),
        "height": (height_thresh[1], "min"),
    },
}

df_metrics_height_list = []
for key, val in gtbbox_filtering_height_cats.items():
    df_results_criteria = pd.concat(
        [compute_model_metrics_on_dataset(model_name, dataset_name, dataset, val, device="cuda")[0] for
         model_name in model_names])
    df_results_criteria[criteria] = key
    df_metrics_height_list.append(df_results_criteria)
df_metrics_heights = pd.concat(df_metrics_height_list, axis=0)


#%% Compute for multiple criterias
#todo add one nominal (eg max occlusion and min height) >25px & image borders & (100% occlusion ? does not seem so in benchmark, normal, does not exist in dataset...)

occl_thresh = [0.35, 0.8]
height_thresh = [25, 45, 120]
resolution = (1920, 1080)

"""
gtbbox_filtering_height_cats = {
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

df_metrics_height_list = []

for key, val in gtbbox_filtering_height_cats.items():
    df_results_height = pd.concat(
        [compute_model_metrics_on_dataset(model_name, dataset_name, dataset, val, device="cuda")[0] for
         model_name in model_names])
    df_results_height["height_criteria"] = key
    df_metrics_height_list.append(df_results_height)
df_metrics_heights = pd.concat(df_metrics_height_list, axis=0)


#%%

df_study = df_metrics_heights

from src.utils import subset_dataframe
import matplotlib.pyplot as plt


min_x, max_x = 0.5, 20
min_y, max_y = 0.1, 1

#n_col = max([len(val) for _, val in gtbbox_filtering_height_cats.items()])
# n_row = len(dict_filter_frames)

n_col = len(gtbbox_filtering_height_cats)

fig, ax = plt.subplots(1, n_col, figsize=(10,6), sharey=True)

i = 0
for j, (height_cat, df_study_heightcat) in enumerate(df_study.groupby("height_criteria")):
    for model, df_analysis_model in df_study_heightcat.groupby("model_name"):
        metrics_model = df_analysis_model.groupby("threshold").apply(lambda x: x.mean(numeric_only=True))
        ax[j].plot(metrics_model["FPPI"], metrics_model["MR"], label=model)
        ax[j].scatter(metrics_model["FPPI"], metrics_model["MR"])
    ax[j].set_xscale('log')
    ax[j].set_yscale('log')
    #ax[j].set_ylim(min_y, max_y)
    #ax[j].set_xlim(min_x, max_x)
    ax[j].set_title(height_cat)
    ax[j].legend()

    import matplotlib.patches as patches
    x = min_x
    y = min_y
    width = ODD_criterias["FPPI"] - min_x
    height = ODD_criterias["MR"] - min_y
    # Add the grey square patch to the axes
    grey_square = patches.Rectangle((x, y), width, height, facecolor='grey', alpha=0.5)
    #ax[j].add_patch(grey_square)
    #ax[j].text(min_x+width/2/10, min_y+height/2/10, s="ODD")

plt.tight_layout()
plt.show()
"""

#todo test is also for aspect ratio and occlusions --> show it



#%% aspect ratios

occl_thresh = [0.35, 0.8]
height_thresh = [25, 45, 120]
resolution = (1920, 1080)

gtbbox_filtering_aspectratio_cats = {
    "normal_aspect_ratio": {
        "aspect_ratio_is_typical": 1,  #
        "occlusion_rate": (0.01, "max"),  # Unoccluded
    },
    "abnormal_aspect_ratio": {
        "aspect_ratio_is_typical": 0,  #
        "occlusion_rate": (0.01, "max"),  # Unoccluded
    },
}

df_metrics_height_list = []

for key, val in gtbbox_filtering_aspectratio_cats.items():
    df_results_aspectratio = pd.concat(
        [compute_model_metrics_on_dataset(model_name, dataset_name, dataset, val, device="cuda")[0] for
         model_name in model_names])
    df_results_aspectratio["aspectratio_criteria"] = key
    df_metrics_height_list.append(df_results_aspectratio)
df_metrics_heights = pd.concat(df_metrics_height_list, axis=0)


#%%


df_study = df_metrics_heights
gtbbox_filtering_cats = gtbbox_filtering_height_cats
criteria = "occlusion_criteria"

from src.utils import subset_dataframe
import matplotlib.pyplot as plt


min_x, max_x = 0.5, 20
min_y, max_y = 0.01, 1

#n_col = max([len(val) for _, val in gtbbox_filtering_height_cats.items()])
# n_row = len(dict_filter_frames)

n_col = len(gtbbox_filtering_cats)

fig, ax = plt.subplots(1, n_col, figsize=(10,6), sharey=True)

i = 0
for j, (height_cat, df_study_heightcat) in enumerate(df_study.groupby(criteria)):
    for model, df_analysis_model in df_study_heightcat.groupby("model_name"):
        metrics_model = df_analysis_model.groupby("threshold").apply(lambda x: x.mean(numeric_only=True))
        ax[j].plot(metrics_model["FPPI"], metrics_model["MR"], label=model)
        ax[j].scatter(metrics_model["FPPI"], metrics_model["MR"])
    ax[j].set_xscale('log')
    ax[j].set_yscale('log')
    ax[j].set_ylim(min_y, max_y)
    ax[j].set_xlim(min_x, max_x)
    ax[j].set_title(height_cat)
    ax[j].legend()

    import matplotlib.patches as patches
    x = min_x
    y = min_y
    width = ODD_criterias["FPPI"] - min_x
    height = ODD_criterias["MR"] - min_y
    # Add the grey square patch to the axes
    grey_square = patches.Rectangle((x, y), width, height, facecolor='grey', alpha=0.5)
    #ax[j].add_patch(grey_square)
    #ax[j].text(min_x+width/2/10, min_y+height/2/10, s="ODD")

plt.tight_layout()
plt.show()
