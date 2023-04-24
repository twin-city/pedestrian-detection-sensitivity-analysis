import os
import matplotlib.pyplot as plt
import pandas as pd
import setuptools.errors
from utils import filter_gt_bboxes, plot_results_img, compute_ffpi_against_fp2
import numpy as np
import os.path as osp
import json
import torch

#%% params
dataset_name = "EuroCityPerson"
model_name = "faster-rcnn_cityscapes"
max_sample = 30 # Uniform sampled in dataset

#%%
from src.preprocessing.ecp_processing import ECPProcessing

root_ecp = "/media/raphael/Projects/datasets/EuroCityPerson/ECP/"
ecp_processor = ECPProcessing(root_ecp, max_samples=max_sample)
targets, df_gtbbox_metadata, df_frame_metadata, _ = ecp_processor.get_dataset()

img_path_list = [osp.join(root_ecp, x) for x in df_frame_metadata["file_name"]]
frame_id_list = list(df_frame_metadata["id"].values.astype(str))

#%%
from src.detection.detector import Detector
detector = Detector(model_name, device="cuda")
preds = detector.get_preds_from_files(dataset_name, frame_id_list, img_path_list)


#%%Plot example
gtbbox_filtering = {"occlusion_rate": (0.3, "max"),
                    "area": (380, "min")}

i = 40
frame_id = frame_id_list[i]
img_path = img_path_list[i]

df_gtbbox_metadata_frame = df_gtbbox_metadata.loc[(frame_id)] #todo delay
excluded_gt = filter_gt_bboxes(df_gtbbox_metadata_frame, gtbbox_filtering)
occlusions_ids = [i for i, idx in enumerate(df_gtbbox_metadata_frame.index) if idx in excluded_gt]


plot_results_img(img_path, frame_id, preds, targets, occlusions_ids)
print(img_path_list)
print(frame_id_list)

# df_gtbbox_metadata_frame[["area", "occlusion_rate", "truncation_rate"]]

#%% Compute the metrics
gtbbox_filtering = {}

gtbbox_filtering = {"occlusion_rate": (0.9, "max"),
                    "truncation_rate": (0.9, "max"),
                    "area": (40, "min")}

df_mr_fppi, _ = compute_ffpi_against_fp2(dataset_name, model_name, preds, targets, df_gtbbox_metadata, gtbbox_filtering)
#todo matched bbox

#%% Concat results and metadata
df_analysis = pd.merge(df_mr_fppi, df_frame_metadata, on="frame_id")
df_analysis_frame = df_analysis.groupby("frame_id").apply(lambda x: x.mean(numeric_only=True))

#%% study correlations
import matplotlib.pyplot as plt
frame_cofactors = ["rainy", "is_night"]
metrics = ["MR", "FPPI"]
from scipy.stats import pearsonr
corr_matrix = df_analysis_frame[metrics+frame_cofactors].corr(method=lambda x, y: pearsonr(x, y)[0])
p_matrix = df_analysis_frame[metrics+frame_cofactors].corr(method=lambda x, y: pearsonr(x, y)[1])

print(p_matrix)
import seaborn as sns
sns.heatmap(corr_matrix, annot=True)
plt.show()

sns.heatmap(p_matrix, annot=True)
plt.show()

#%% Day 'n Night
import matplotlib.pyplot as plt


#%% Day 'n Night example

night_ids = df_analysis_frame[df_analysis_frame["is_night"]==1].index.to_list()
day_ids = df_analysis_frame[df_analysis_frame["is_night"]==0].index.to_list()

metrics_day = df_mr_fppi.loc[day_ids].groupby("threshold").apply(lambda x: x.mean(numeric_only=True))
metrics_night = df_mr_fppi.loc[night_ids].groupby("threshold").apply(lambda x: x.mean(numeric_only=True))

fig, ax = plt.subplots(1,1)
ax.plot(metrics_day["FPPI"], metrics_day["MR"], label="day")
ax.plot(metrics_night["FPPI"], metrics_night["MR"], label="night")

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_ylim(0.1, 1)
ax.set_xlim(0.1, 20)
plt.legend()
plt.show()


#%%
df_analysis_50 = pd.merge(df_mr_fppi[df_mr_fppi.index.get_level_values('threshold') == 0.5], df_frame_metadata, on="frame_id")
df_gtbbox_metadata_frame = df_gtbbox_metadata.groupby("frame_id").apply(lambda x: x.mean(numeric_only=True))
df_analysis_50_gtbbox = pd.merge(df_gtbbox_metadata_frame, df_analysis_50, on="frame_id")

import matplotlib.pyplot as plt
seq_cofactors = ["is_night", "adverse_weather", "occlusion_rate"]
metrics = ["MR", "FPPI"]
from scipy.stats import pearsonr
corr_matrix = df_analysis_50_gtbbox[metrics+seq_cofactors].corr(method=lambda x, y: pearsonr(x, y)[0])
p_matrix = df_analysis_50_gtbbox[metrics+seq_cofactors].corr(method=lambda x, y: pearsonr(x, y)[1])

print(p_matrix)
import seaborn as sns
sns.heatmap(corr_matrix, annot=True)
plt.show()

sns.heatmap(p_matrix, annot=True)
plt.show()

sns.heatmap(corr_matrix[p_matrix<0.05].loc[:,["MR", "FPPI"]], annot=True)
plt.tight_layout()
plt.show()