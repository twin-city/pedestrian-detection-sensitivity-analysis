import os
import pandas as pd
import setuptools.errors
import numpy as np
import os.path as osp

#%% params of input dataset

dataset_name = "motsynth"
root_motsynth = "/home/raphael/work/datasets/MOTSynth/"
max_sample = 400  # Uniform sampled in dataset

#%% Get the dataset
from src.preprocessing.motsynth_processing import MotsynthProcessing
motsynth_processor = MotsynthProcessing(root_motsynth, max_samples=max_sample, video_ids=None)
dataset = motsynth_processor.get_dataset() #todo as class
root, targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = dataset


#%% Params of detection


from src.detection.detector import Detector
from src.detection.metrics import detection_metric

def compute_model_metrics_on_dataset(model_name, dataset_name, dataset, gtbbox_filtering, device="cuda"):

    # dataset info
    root, targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = dataset

    detector = Detector(model_name, device=device)
    preds = detector.get_preds_from_files(dataset_name, root, df_frame_metadata)

    metric = detection_metric(gtbbox_filtering)
    metric_results = metric.compute(dataset_name, model_name, preds, targets, df_gtbbox_metadata,
                                    gtbbox_filtering)

    return metric_results

gtbbox_filtering = {
    "occlusion_rate": (0.96, "max"),# At least 1 keypoint visible
    "area": (200, "min")
}

model_name = "faster-rcnn_cityscapes"
df_mr_fppi_model1, _ = compute_model_metrics_on_dataset(model_name, dataset_name, dataset, gtbbox_filtering, device="cuda")

model_name = "mask-rcnn_coco"
df_mr_fppi_model2, _ = compute_model_metrics_on_dataset(model_name, dataset_name, dataset, gtbbox_filtering, device="cuda")


#%% Concat results and metadata
df_analysis_model1 = pd.merge(df_mr_fppi_model1.reset_index(), df_frame_metadata, on="frame_id")
df_analysis_frame_model_1 = df_analysis_model1.groupby("frame_id").apply(lambda x: x.mean(numeric_only=True))


df_analysis_model2 = pd.merge(df_mr_fppi_model2.reset_index(), df_frame_metadata, on="frame_id")
df_analysis_frame_model_2 = df_analysis_model2.groupby("frame_id").apply(lambda x: x.mean(numeric_only=True))

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

df = df_analysis_frame_model_2

seq_cofactors = ["adverse_weather", "is_night", "pitch"]
metrics = ["MR", "FPPI"]
df_analysis_seq = df.groupby("seq_name").apply(lambda x: x.mean())
features = metrics + seq_cofactors

corr_matrix, p_matrix = compute_correlations(df_analysis_seq, features)
plot_correlations(corr_matrix, p_matrix, title="per_sequence")

corr_matrix, p_matrix = compute_correlations(df, features)
plot_correlations(corr_matrix, p_matrix, title="per_frame")


#%%

#todo do a plot on this, also give intuition with some plot of the given criterias. + extreme plots ???

from src.utils import subset_dataframe

filter_frames = [
    {},
    {"is_night": 0},
    {"is_night": 1},
    {"adverse_weather": 0},
    {"adverse_weather": 1},
    {"pitch": {"<": -10}},
    {"pitch": {">": -10, "<": 10}},
    {"pitch": {">": 10}}
]

for filter_frame in filter_frames:

    # Filter use cases
    df_mr_fppi_model1_subset = subset_dataframe(df_analysis_model1, filter_frame)
    df_mr_fppi_model2_subset = subset_dataframe(df_analysis_model2, filter_frame)

    metrics_model1 = df_mr_fppi_model1_subset.groupby("threshold").apply(lambda x: x.mean(numeric_only=True))
    metrics_model2 = df_mr_fppi_model2_subset.groupby("threshold").apply(lambda x: x.mean(numeric_only=True))

    fig, ax = plt.subplots(1,1)
    ax.plot(metrics_model1["FPPI"], metrics_model1["MR"], label="model_1")
    ax.plot(metrics_model2["FPPI"], metrics_model2["MR"], label="model_2")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(0.1, 1)
    ax.set_xlim(0.1, 20)
    plt.legend()
    plt.title(filter_frame)
    plt.show()



