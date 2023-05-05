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

#%% Height

import matplotlib.pyplot as plt

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
#ax[0].axvline(height_thresh[2], c="red")

plt.show()

#%% What cases do we study ?

#todo filter truncated in MoTSynth those out of the image --> exluded in Caltech for boundary effects

gtbbox_filtering_all = {
    "Overall": {
        "occlusion_rate": (0.99, "max"),  # Not fully occluded
        "height": (height_thresh[1], "min"),
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





#%% After bench, do the plot value difference (simplified, each metric)

#%% Model performance :  Plots MR vs FPPI on frame filtering


df_analysis = df_analysis = pd.merge(df_metrics_criteria.reset_index(), df_frame_metadata, on="frame_id")

from src.utils import subset_dataframe
import matplotlib.pyplot as plt

dict_filter_frames = {
    "Overall": [{}],
    "Day / Night": ({"is_night": 0}, {"is_night": 1}),
    "Adverse Weather": ({"adverse_weather": 0}, {"adverse_weather": 1}),
    "Camera Angle": ({"pitch": {"<": -10}}, {"pitch": {">": -10}}),
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


#%%


import seaborn as sns
from matplotlib.colors import BoundaryNorm

threshold = 0.9

ODD_limit = [
    {"is_night": 1},
    {"adverse_weather": 1},
    # {"pitch": {"<":-10}},
]
ax_y_labels = ["night", "bad weather"]  # , "high-angle shot"]
df_analysis_50 = df_analysis[df_analysis["threshold"] == threshold]

for metric in ["MR", "FPPI"]:

    mean_metric_values = df_analysis_50.groupby("model_name").apply(lambda x: x[metric].mean())

    df_odd_model_list = []
    for model_name in model_names:
        perc_increase_list = []
        for limit in ODD_limit:
            condition = {}
            condition.update({"model_name": model_name})
            condition.update(limit)
            df_subset = subset_dataframe(df_analysis_50, condition)
            df_subset = df_subset[df_subset["model_name"] == model_name]
            perc_increase_list.append(df_subset[metric].mean()-mean_metric_values.loc[model_name])
        df_odd_model_list.append(pd.DataFrame(perc_increase_list, index=ODD_limit, columns=[model_name]))

    df_odd_model = pd.concat(df_odd_model_list, axis=1)
    df_odd_model.index = ax_y_labels



    # Define the boundaries of each zone
    bounds = [0, 0.1, 0.2, 0.5]
    # Define a unique color for each zone
    colors = ['green', 'yellow', 'red']
    # Create a colormap with discrete colors
    cmap = sns.color_palette(colors, n_colors=len(bounds)-1).as_hex()
    # Create a BoundaryNorm object to define the colormap
    norm = BoundaryNorm(bounds, len(cmap))

    #cmap = "YlOrRd"
    cmap = "RdYlGn_r"

    fig, ax = plt.subplots(1,1)
    sns.heatmap(df_odd_model, annot=True,
                cmap=cmap, center=0,
                ax=ax, fmt=".2f", cbar_kws={'format': '%.2f'})
    ax.collections[0].colorbar.set_label('Decrease in performance')
    plt.title(f"Impact of parameters on {metric}")
    plt.tight_layout()
    plt.show()



#%%
i = 30

img_path = osp.join(root, df_frame_metadata["file_name"].iloc[i])
frame_id = df_frame_metadata.index[i]

gtbbox_filtering_all = {
    "Overall": {
        "occlusion_rate": (0.99, "max"),  # Not fully occluded
        "height": (height_thresh[1], "min"),
    },
}
from src.detection.metrics import filter_gt_bboxes

gtbbox_filtering = gtbbox_filtering_all["Overall"]

if len(pd.DataFrame(df_gtbbox_metadata.loc[frame_id]).T) == 1:
    df_gtbbox_metadata_frame = pd.DataFrame(df_gtbbox_metadata.loc[frame_id]).T.reset_index()
else:
    df_gtbbox_metadata_frame = df_gtbbox_metadata.loc[frame_id].reset_index()
excluded_gt = filter_gt_bboxes(df_gtbbox_metadata_frame, gtbbox_filtering)


from src.detection.detector import Detector

detector = Detector(model_name, device="cpu")
preds = detector.get_preds_from_files(dataset_name, root, df_frame_metadata)

from src.detection.metrics import detection_metric
metric = detection_metric(gtbbox_filtering)
df_mr_fppi, df_gt_bbox = metric.compute(dataset_name, model_name, preds, targets, df_gtbbox_metadata,
                                        gtbbox_filtering)
#todo add threshold to 1 in data
from src.utils import plot_results_img
plot_results_img(img_path, frame_id, preds=preds, targets=targets,
             df_gt_bbox=df_gt_bbox, threshold=0.9999) #todo seems there is a bug, woman in middle should be in red and guy should be red. No sense of all this.

#%%
plot_results_img(img_path, frame_id, preds=preds, targets=None,
             df_gt_bbox=df_gt_bbox, threshold=0.99) #todo seems there is a bug, woman in middle should be in red and guy should be red. No sense of all this.

#%%
plot_results_img(img_path, frame_id, preds=None, targets=targets,
             df_gt_bbox=df_gt_bbox, threshold=0.99) #todo seems there is a bug, woman in middle should be in red and guy should be red. No sense of all this.

