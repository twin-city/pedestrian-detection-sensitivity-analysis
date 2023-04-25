import os
import pandas as pd
import setuptools.errors
import numpy as np
import os.path as osp

#%% params
model_name = "faster-rcnn_cityscapes"

dataset_name = "motsynth"
root_motsynth = "/home/raphael/work/datasets/MOTSynth/"
max_sample = 400  # Uniform sampled in dataset

#%% Get the dataset
from src.preprocessing.motsynth_processing import MotsynthProcessing
motsynth_processor = MotsynthProcessing(root_motsynth, max_samples=max_sample, video_ids=None)
targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = motsynth_processor.get_dataset()

#%% Detections
from src.detection.detector import Detector
detector = Detector(model_name, device="cuda")
preds = detector.get_preds_from_files(dataset_name, root_motsynth, df_frame_metadata)

#########################################   Peform Tests   ############################################################

#%% Compute the metrics
from src.detection.metrics import detection_metric

# Define the ground truth bounding box filtering
gtbbox_filtering = {
    "occlusion_rate": (0.96, "max"),# At least 1 keypoint visible
    "area": (200, "min")
}

# Compute the metric
metric = detection_metric(gtbbox_filtering)
df_mr_fppi, df_matched_gtbbox = metric.compute(dataset_name, model_name, preds, targets, df_gtbbox_metadata, gtbbox_filtering)


#%% Concat results and metadata
df_analysis = pd.merge(df_mr_fppi, df_frame_metadata, on="frame_id")
df_analysis_frame = df_analysis.groupby("frame_id").apply(lambda x: x.mean(numeric_only=True))

#%% Additional foo

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def compute_correlations(df, features):
    corr_matrix = df[features].corr(
        method=lambda x, y: pearsonr(x, y)[0])
    p_matrix = df[features].corr(
        method=lambda x, y: pearsonr(x, y)[1])
    return corr_matrix, p_matrix

def plot_correlations(corr_matrix, p_matrix, title=""):
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    sns.heatmap(corr_matrix[p_matrix<0.05], annot=True, ax=ax[0])
    sns.heatmap(p_matrix, annot=True, ax=ax[1])
    if title:
        ax[1].set_title(title)
    plt.tight_layout()
    plt.show()

#%% Correlations

seq_cofactors = ["adverse_weather", "is_night", "pitch"]
metrics = ["MR", "FPPI"]
df_analysis_seq = df_analysis_frame.groupby("seq_name").apply(lambda x: x.mean())
features = metrics + seq_cofactors

corr_matrix, p_matrix = compute_correlations(df_analysis_seq, features)
plot_correlations(corr_matrix, p_matrix, title="per_sequence")

corr_matrix, p_matrix = compute_correlations(df_analysis_frame, features)
plot_correlations(corr_matrix, p_matrix, title="per_frame")


#%% MR vs FPPI on varying criterias

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
