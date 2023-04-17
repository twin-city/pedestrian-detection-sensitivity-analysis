import os
import pandas as pd
import setuptools.errors
import numpy as np
from utils import filter_gt_bboxes, plot_results_img, compute_ffpi_against_fp2
import os.path as osp

#%% params
model_name = "faster-rcnn_cityscapes"
dataset_name = "motsynth"
max_sample = 550 # Uniform sampled in dataset

# Dataset #todo add statistical comparison between datasets
from src.preprocessing.motsynth_processing import MotsynthProcessing
motsynth_processor = MotsynthProcessing(max_samples=max_sample, video_ids=None)
targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = motsynth_processor.get_MoTSynth_annotations_and_imagepaths()
df_frame_metadata.index.rename("frame_id", inplace=True)
df_frame_metadata = df_frame_metadata.reset_index()
df_frame_metadata["frame_id"] = df_frame_metadata["frame_id"].astype(str)
df_frame_metadata = df_frame_metadata.set_index("frame_id")

df_gtbbox_metadata = df_gtbbox_metadata.reset_index()
df_gtbbox_metadata["id"] = df_gtbbox_metadata["id"].astype(str)
df_gtbbox_metadata["image_id"] = df_gtbbox_metadata["image_id"].astype(str)
df_gtbbox_metadata = df_gtbbox_metadata.set_index(["image_id","id"])

print(targets.keys())

# Frame id list
root_frame = "/home/raphael/work/datasets/MOTSynth/"
img_path_list = [osp.join(root_frame, x) for x in df_frame_metadata["file_name"]]
frame_id_list = list(df_frame_metadata["id"].values.astype(str))

#%%

# Detections
from src.detection.detector import Detector
detector = Detector(model_name, device="cuda")
preds = detector.get_preds_from_files(dataset_name, frame_id_list, img_path_list)

#########################################   Peform Tests   ############################################################

#%% Analyze results on an image

#todo a test --> is it the image we saved ?

gtbbox_filtering = {"occlusion_rate": (0.9, "max"),
                    "area": (20, "min")}


# plot
i = 1
frame_id = frame_id_list[i]
img_path = img_path_list[i]
df_gtbbox_metadata_frame = df_gtbbox_metadata.loc[frame_id] #todo delay
excluded_gt = filter_gt_bboxes(df_gtbbox_metadata_frame, gtbbox_filtering)
occlusions_ids = [i for i, idx in enumerate(df_gtbbox_metadata_frame.index) if idx in excluded_gt]
plot_results_img(img_path, frame_id, preds, targets, occlusions_ids)

#%% As in ECP

df_mr_fppi = compute_ffpi_against_fp2(dataset_name, model_name, preds, targets, df_gtbbox_metadata, gtbbox_filtering)


#%% Concat results and metadata
df_analysis = pd.merge(df_mr_fppi, df_frame_metadata, on="frame_id")
df_analysis_frame = df_analysis.groupby("frame_id").apply(lambda x: x.mean())

#%% study correlations
frame_cofactors = ["adverse_weather", "is_night"]
metrics = ["MR", "FPPI"]
from scipy.stats import pearsonr
corr_matrix = df_analysis_frame[metrics+frame_cofactors].corr(method=lambda x, y: pearsonr(x, y)[0])
p_matrix = df_analysis_frame[metrics+frame_cofactors].corr(method=lambda x, y: pearsonr(x, y)[1])


#%% Day 'n Night
import matplotlib.pyplot as plt

night_ids = df_analysis_frame[df_analysis_frame["is_night"]==1].index.to_list()
day_ids = df_analysis_frame[df_analysis_frame["is_night"]==0].index.to_list()

metrics_day = df_mr_fppi.loc[day_ids].groupby("threshold").apply(lambda x: x.mean(numeric_only=True))
metrics_night = df_mr_fppi.loc[night_ids].groupby("threshold").apply(lambda x: x.mean(numeric_only=True))

fig, ax = plt.subplots(1,1)
ax.plot(metrics_day["MR"], metrics_day["FPPI"], label="day")
ax.plot(metrics_night["MR"], metrics_night["FPPI"], label="night")

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_ylim(0.1, 1)
ax.set_xlim(0.1, 20)
plt.legend()
plt.show()