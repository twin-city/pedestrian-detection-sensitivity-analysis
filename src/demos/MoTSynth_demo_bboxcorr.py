import os
import pandas as pd
import setuptools.errors
import numpy as np
import os.path as osp


"""
Only to study per bbox recall/precision vs bbox metadata
"""

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
df_metrics_frame["gtbbox_filtering_cat"] = "Overall"


#%% Plot one plot to check it

from src.utils import plot_ffpi_mr_on_ax
fig, ax = plt.subplots(1, 1, figsize=(10,10), sharey=True)
plot_ffpi_mr_on_ax(df_metrics_frame, "Overall", ax, odd=ODD_criterias)
plt.tight_layout()
plt.show()





#df_metrics_frame, df_metrics_gtbbox = compute_model_metrics_on_dataset(model_name, dataset_name, dataset, gt_bbox_filtering, device="cuda")
df_metrics_gtbbox = df_metrics_gtbbox[df_metrics_gtbbox["threshold"]==threshold]



#%% Check if 1 0 is linked with params

df_metrics_gtbbox_study = df_metrics_gtbbox[np.isin(df_metrics_gtbbox, [0,1])]
df_metrics_gtbbox_study = df_metrics_gtbbox_study[df_metrics_gtbbox_study["threshold"]==0.5]

features_bbox = ['area','height', 'width', 'aspect_ratio', 'is_crowd', 'is_blurred', "occlusion_rate"]
attributes = ['attributes_0', 'attributes_1', 'attributes_2',
       'attributes_3', 'attributes_4', 'attributes_5', 'attributes_6',
       'attributes_7', 'attributes_8', 'attributes_9', 'attributes_10']

df_metrics_gtbbox_study.loc[:,features_bbox+attributes] = df_gtbbox_metadata.loc[df_metrics_gtbbox_study.index, features_bbox+attributes]


criteria = "occlusion_rate"

fig, ax = plt.subplots(1,1)
ax.hist(df_metrics_gtbbox_study[df_metrics_gtbbox_study["matched"]==1][criteria],
        density=True, alpha=0.5, bins=200, label="matched")
ax.hist(df_metrics_gtbbox_study[df_metrics_gtbbox_study["matched"]==0][criteria],
        density=True, alpha=0.5, bins=200, label="unmatched")
#ax.set_xlim(0,400)
plt.legend()
plt.show()


#%%

att_list = []

for i in range(11):
    df_att = pd.get_dummies(df_metrics_gtbbox_study[f"attributes_{i}"], prefix=f"att{i}")
    df_metrics_gtbbox_study[df_att.columns] = df_att
    att_list += list(df_att.columns)

#%%

import seaborn as sns
corr_matrix = df_metrics_gtbbox_study.corr()
fig, ax = plt.subplots(figsize=(3,12))
sns.heatmap(pd.DataFrame(corr_matrix["matched"][att_list]), center=0, cmap="PiYG")
plt.tight_layout()
plt.show()
#%%

df_analysis = pd.merge(df_metrics_frame.reset_index(), df_frame_metadata, on="frame_id")
df_analysis = df_analysis[df_analysis["threshold"]==0.5]

mean_occl_rate = df_metrics_gtbbox_study.groupby("frame_id").apply(lambda x: x.mean())["occlusion_rate"].reset_index()

df_frame_mean = pd.merge(df_analysis, mean_occl_rate, on="frame_id")
