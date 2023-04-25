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
    df_mr_fppi, df_gt_bbox = metric.compute(dataset_name, model_name, preds, targets, df_gtbbox_metadata,
                                    gtbbox_filtering)

    df_mr_fppi["model_name"] = model_name
    df_mr_fppi["dataset_name"] = dataset_name
    df_mr_fppi["gtbbox_filtering"] = str(gtbbox_filtering)

    return df_mr_fppi, df_gt_bbox

gtbbox_filtering = {
    "occlusion_rate": (0.96, "max"),# At least 1 keypoint visible
    "area": (200, "min")
}

model_names = ["faster-rcnn_cityscapes", "mask-rcnn_coco"]


df_metrics = pd.concat([compute_model_metrics_on_dataset(model_name, dataset_name, dataset, gtbbox_filtering, device="cuda")[0]
for model_name in model_names])

df_analysis = pd.merge(df_metrics.reset_index(), df_frame_metadata, on="frame_id")



"""
df_mr_fppi_model1, _ = compute_model_metrics_on_dataset(model_name, dataset_name, dataset, gtbbox_filtering, device="cuda")

df_mr_fppi_model2, _ = compute_model_metrics_on_dataset(model_name, dataset_name, dataset, gtbbox_filtering, device="cuda")

#%% Concat results and metadata
df_analysis_model1 = pd.merge(df_mr_fppi_model1.reset_index(), df_frame_metadata, on="frame_id")
df_analysis_frame_model_1 = df_analysis_model1.groupby("frame_id").apply(lambda x: x.mean(numeric_only=True))


df_analysis_model2 = pd.merge(df_mr_fppi_model2.reset_index(), df_frame_metadata, on="frame_id")
df_analysis_frame_model_2 = df_analysis_model2.groupby("frame_id").apply(lambda x: x.mean(numeric_only=True))
"""



#%% Additional foo




###########################################################################################
#%% Compare the models
###########################################################################################

#todo do a plot on this, also give intuition with some plot of the given criterias. + extreme plots ???

from src.utils import subset_dataframe
import matplotlib.pyplot as plt

dict_filter_frames = {
    "all": [{}],
    "is_night": ({"is_night": 0}, {"is_night": 1}),
    "adverse_weather": ({"adverse_weather": 0}, {"adverse_weather": 1}),
    "pitch": ({"pitch": {"<": -10}}, {"pitch": {">": -10}}),
}

n_col = max([len(val) for _, val in dict_filter_frames.items()])
n_row = len(dict_filter_frames)

fig, ax = plt.subplots(n_row, n_col, figsize=(6,12))
for i, (key, filter_frames) in enumerate(dict_filter_frames.items()):
    for j, filter_frame in enumerate(filter_frames):

        for model, df_analysis_model in df_analysis.groupby("model_name"):
            df_analysis_subset = subset_dataframe(df_analysis_model, filter_frame)
            metrics_model = df_analysis_subset.groupby("threshold").apply(lambda x: x.mean(numeric_only=True))
            ax[i, j].plot(metrics_model["FPPI"], metrics_model["MR"], label="model")

        ax[i,j].set_xscale('log')
        ax[i,j].set_yscale('log')
        ax[i,j].set_ylim(0.1, 1)
        ax[i,j].set_xlim(0.5, 20)
        ax[i,j].set_title(filter_frame)
        ax[i, j].legend()

plt.tight_layout()
plt.show()



###########################################################################################
#%% Zoom on one model
###########################################################################################

#%% Correlations
from src.utils import compute_correlations, plot_correlations

model_name = "faster-rcnn_cityscapes"

df = df_analysis[df_analysis["model_name"] == model_name].groupby("frame_id").apply(lambda x: x.mean(numeric_only=True))

seq_cofactors = ["adverse_weather", "is_night", "pitch"]
metrics = ["MR", "FPPI"]
df_analysis_seq = df.groupby("seq_name").apply(lambda x: x.mean())
features = metrics + seq_cofactors

corr_matrix, p_matrix = compute_correlations(df_analysis_seq, features)
plot_correlations(corr_matrix, p_matrix, title="per_sequence")

corr_matrix, p_matrix = compute_correlations(df, features)
plot_correlations(corr_matrix, p_matrix, title="per_frame")



