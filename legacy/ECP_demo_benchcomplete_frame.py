import os

import pandas as pd
# from src.utils import filter_gt_bboxes, plot_results_img, compute_ffpi_against_fp2
import os.path as osp
import numpy as np
from src.utils import plot_heatmap_metrics
import matplotlib.pyplot as plt

#%% params

dataset_name = "EuroCityPerson"
model_name = "faster-rcnn_cityscapes"
max_sample = 30 # Uniform sampled in dataset

model_name = "faster-rcnn_cityscapes"
seq_cofactors = ["adverse_weather", "is_night"]# , "pitch"]
metrics = ["MR", "FPPI"]

gtbbox_filtering = {
    "occlusion_rate": (0.9, "max"),# At least 1 keypoint visible
    "truncation_rate": (0.9, "max"),
    "area": (20, "min")
}

ODD_criterias = {
    "MR": 0.5,
    "FPPI": 5,
}

model_names = ["faster-rcnn_cityscapes", "mask-rcnn_coco"]

occl_thresh = [0.35, 0.8]
height_thresh = [20, 50, 120]
resolution = (1920, 1024)

param_heatmap_metrics = {
    "MR": {
        "vmin": -0.15,
        "vmax": 0.2,
        "center": 0.,
        "cmap": "RdYlGn_r",
    },
    "FPPI": {
        "vmin": -0.15,
        "vmax": 0.2,
        "center": 0.,
        "cmap": "RdYlGn_r",
    },
}

metrics = ["MR", "FPPI"]

ODD_limit = [
    ({"is_night": 1}, "Night"),
    ({"adverse_weather": 1}, "Adverse Weather"),
    ({"pitch": {"<":-10}}, "High-angle shot"),
]

import os
results_dir = osp.join("../", "results", "ECP", f"ECP_{max_sample}")
os.makedirs(results_dir, exist_ok=True)

#%%
from legacy.ecp_processing_legacy import ECPProcessing

root_ecp = "/media/raphael/Projects/datasets/EuroCityPerson/ECP/"
ecp_processor = ECPProcessing(root_ecp, max_samples=max_sample)
dataset = ecp_processor.get_dataset()
root, targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = dataset



#%% Multiple plots
#corr_matrix, p_matrix = compute_correlations(df_frame_metadata.groupby("seq_name").apply(lambda x: x.mean()), seq_cofactors)
#plot_correlations(corr_matrix, p_matrix, title="Correlations between metadatas at sequence level")

#%%

df_gtbbox_metadata["height"] = df_gtbbox_metadata["height"].astype(int)

fig, ax = plt.subplots(1,2)

df_gtbbox_metadata.hist("height", bins=200, ax=ax[0])
ax[0].set_xlim(0, 300)
ax[0].axvline(height_thresh[0], c="red")
ax[0].axvline(height_thresh[1], c="red")
ax[0].axvline(height_thresh[2], c="red")

df_gtbbox_metadata.hist("occlusion_rate", bins=22, ax=ax[1])
#ax[0].set_xlim(0, 300)
ax[1].axvline(occl_thresh[0], c="red")
ax[1].axvline(occl_thresh[1], c="red")
ax[1].set_xlim(0,1)
#ax[0].axvline(height_thresh[2], c="red")

plt.show()

#%% Now plot the multiple cases !!!!!!
from src.detection.metrics import compute_model_metrics_on_dataset




#%% What cases do we study ?

#todo filter truncated in MoTSynth those out of the image --> exluded in Caltech for boundary effects

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
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(10,10), sharey=True)
plot_ffpi_mr_on_ax(df_metrics_criteria, "Overall", ax, odd=ODD_criterias)
plt.tight_layout()
plt.show()


#%%

df_analysis = df_analysis = pd.merge(df_metrics_criteria.reset_index(), df_frame_metadata, on="frame_id")

from src.utils import subset_dataframe
import matplotlib.pyplot as plt

dict_filter_frames = {
    "Overall": [{}],
    "Day / Night": ({"is_night": 0}, {"is_night": 1}),
    "Adverse Weather": ({"adverse_weather": 0}, {"adverse_weather": 1}),
    #"pitch": ({"pitch": {"<": -10}}, {"pitch": {">": -10}}),
}

min_x, max_x = 0.01, 100  # 0.01 false positive per image to 100
min_y, max_y = 0.05, 1  # 5% to 100% Missing Rate

n_col = max([len(val) for _, val in dict_filter_frames.items()])
n_row = len(dict_filter_frames)

fig, ax = plt.subplots(n_row, n_col, figsize=(8,14))
for i, (key, filter_frames) in enumerate(dict_filter_frames.items()):

    ax[i, 0].set_ylabel(key, fontsize=20)

    for j, filter_frame in enumerate(filter_frames):

        for model, df_analysis_model in df_analysis.groupby("model_name"):
            df_analysis_subset = subset_dataframe(df_analysis_model, filter_frame)
            metrics_model = df_analysis_subset.groupby("threshold").apply(lambda x: x.mean(numeric_only=True))
            ax[i, j].plot(metrics_model["FPPI"], metrics_model["MR"], label=model)
            ax[i, j].scatter(metrics_model["FPPI"], metrics_model["MR"])

        ax[i,j].set_xscale('log')
        ax[i,j].set_yscale('log')
        ax[i,j].set_ylim(min_y, max_y)
        ax[i,j].set_xlim(min_x, max_x)
        ax[i,j].set_title(filter_frame)
        ax[i, j].legend()

        import matplotlib.patches as patches
        x = min_x
        y = min_y
        width = ODD_criterias["FPPI"] - min_x
        height = ODD_criterias["MR"] - min_y
        # Add the grey square patch to the axes
        grey_square = patches.Rectangle((x, y), width, height, facecolor='grey', alpha=0.5)
        ax[i,j].add_patch(grey_square)
        ax[i,j].text(min_x+width/2/10, min_y+height/2/10, s="ODD")

plt.tight_layout()
plt.show()


#%% Plot the heatmap

thresholds = [0.5, 0.9, 0.99]
df_analysis_heatmap = df_analysis[np.isin(df_analysis["threshold"], thresholds)]
plot_heatmap_metrics(df_analysis_heatmap, model_names, metrics, ODD_limit,
                     param_heatmap_metrics=param_heatmap_metrics, results_dir=results_dir)
