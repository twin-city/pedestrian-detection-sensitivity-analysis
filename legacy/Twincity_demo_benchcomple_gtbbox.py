import pandas as pd
import numpy as np

#%% params of input dataset

dataset_name = "twincity"
root = "/home/raphael/work/datasets/twincity-Unreal/v5"


metrics = ["MR", "FPPI"]
ODD_criterias = {
    "MR": 0.5,
    "FPPI": 5,
}

occl_thresh = [0.35, 0.8]
height_thresh = [20, 50, 120]
resolution = (1920, 1080)

#%% Get the dataset
from legacy.twincity_preprocessing2 import get_twincity_dataset
dataset = get_twincity_dataset(root, 50)
root, targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = dataset


#root, targets, metadatas, frame_id_list, img_path_list = dataset
#df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = metadatas

#todo in processing
#df_gtbbox_metadata["aspect_ratio"] = 1/df_gtbbox_metadata["aspect_ratio"]
mu = 0.4185
std = 0.12016
df_gtbbox_metadata["aspect_ratio_is_typical"] = np.logical_and(df_gtbbox_metadata["aspect_ratio"] < mu+std,  df_gtbbox_metadata["aspect_ratio"] > mu-std)
#df_frame_metadata["num_pedestrian"] = df_gtbbox_metadata.groupby("frame_id").apply(len).loc[df_frame_metadata.index]

#%% Plot example

df_frame_metadata[df_frame_metadata["is_night"]==1]["file_name"]

#%% Multiple plots
#corr_matrix, p_matrix = compute_correlations(df_frame_metadata.groupby("seq_name").apply(lambda x: x.mean()), seq_cofactors)
#plot_correlations(corr_matrix, p_matrix, title="Correlations between metadatas at sequence level")


#%% Now plot the multiple cases !!!!!!
from src.detection.metrics import compute_model_metrics_on_dataset
model_names = ["faster-rcnn_cityscapes", "mask-rcnn_coco"]

#%% Height

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,2)

df_gtbbox_metadata.hist("height", bins=200, ax=ax[0])
ax[0].set_xlim(0, 300)
ax[0].axvline(height_thresh[0], c="red")
ax[0].axvline(height_thresh[1], c="red")
ax[0].axvline(height_thresh[2], c="red")


df_gtbbox_metadata.hist("area", bins=100, ax=ax[1])
#ax[0].set_xlim(0, 300)
#ax[1].axvline(occl_thresh[0], c="red")
#ax[1].axvline(occl_thresh[1], c="red")
#ax[1].set_xlim(500)
#ax[0].axvline(height_thresh[2], c="red")

plt.show()

#%% What cases do we study ?


gtbbox_filtering_all = {
    "Overall": {
        #"occlusion_rate": (0.99, "max"),  # Not fully occluded
        "height": (height_thresh[1], "min"),
    },
}



gtbbox_filtering_aspectratio_cats = {
    "Typical aspect ratios": {
        "aspect_ratio_is_typical": 1,  #
        #"occlusion_rate": (0.01, "max"),  # Unoccluded
    },
    "Atypical aspect ratios": {
        "aspect_ratio_is_typical": 0,  #
        #"occlusion_rate": (0.01, "max"),  # Unoccluded
    },
}

gtbbox_filtering_height_cats = {
    "near": {
        #"occlusion_rate": (0.01, "max"),  # Unoccluded
        "height": (height_thresh[2], "min"),
    },
    "medium": {
        #"occlusion_rate": (0.01, "max"),  # Unoccluded
        "height": (height_thresh[2], "max"),
        "height": (height_thresh[1], "min"),
    },
    "far": {
        #"occlusion_rate": (0.01, "max"),  # Unoccluded
        "height": (height_thresh[1], "max"),
        "height": (height_thresh[0], "min"),
    },
}

gtbbox_filtering_cats = {}
gtbbox_filtering_cats.update(gtbbox_filtering_all)
gtbbox_filtering_cats.update(gtbbox_filtering_height_cats)
gtbbox_filtering_cats.update(gtbbox_filtering_aspectratio_cats)



#%% Do we have biases toward people ??? Compute which bounding box were successfully classified as box !!!!
# function of other parameters ...

model_name = model_names[0]
gt_bbox_filtering = gtbbox_filtering_cats["Overall"]
threshold = 0.5

df_metrics_frame, df_metrics_gtbbox = compute_model_metrics_on_dataset(model_name, dataset_name, dataset, gt_bbox_filtering, device="cuda")

df_metrics_gtbbox = df_metrics_gtbbox[df_metrics_gtbbox["threshold"]==threshold]


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

from src.utils import plot_ffpi_mr_on_ax

fig, ax = plt.subplots(3, 3, figsize=(10,10), sharey=True)
plot_ffpi_mr_on_ax(df_metrics_criteria, "Overall", ax[0,0], odd=ODD_criterias)
plot_ffpi_mr_on_ax(df_metrics_criteria, "Typical aspect ratios", ax[0,1])
plot_ffpi_mr_on_ax(df_metrics_criteria, "Atypical aspect ratios", ax[0,2])
plot_ffpi_mr_on_ax(df_metrics_criteria, "near", ax[1,0])
plot_ffpi_mr_on_ax(df_metrics_criteria, "medium", ax[1,1])
plot_ffpi_mr_on_ax(df_metrics_criteria, "far", ax[1,2])
#plot_ffpi_mr_on_ax(df_metrics_criteria, "No occlusion", ax[2,0])
#plot_ffpi_mr_on_ax(df_metrics_criteria, "Partial occlusion", ax[2,1])
#plot_ffpi_mr_on_ax(df_metrics_criteria, "Heavy occlusion", ax[2,2])
plt.tight_layout()
plt.show()