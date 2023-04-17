import os
import pandas as pd
import setuptools.errors
from utils import filter_gt_bboxes, plot_results_img, compute_ffpi_against_fp2
import numpy as np
import os.path as osp

#%% params
model_name = "faster-rcnn_cityscapes"
max_sample = 500 # Uniform sampled in dataset

# Dataset #todo add statistical comparison between datasets
from src.preprocessing.motsynth_processing import MotsynthProcessing
motsynth_processor = MotsynthProcessing(max_samples=max_sample, video_ids=None)
targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = motsynth_processor.get_MoTSynth_annotations_and_imagepaths()

print(targets.keys())

# Frame id list
root_frame = "/home/raphael/work/datasets/MOTSynth/"
img_path_list = [osp.join(root_frame, x) for x in df_frame_metadata["file_name"]]
frame_id_list = list(df_frame_metadata["id"].values.astype(str))

#%%

# Detections
from src.detection.detector import Detector
detector = Detector(model_name, device="cuda")
preds = detector.get_preds_from_files(frame_id_list, img_path_list)

#########################################   Peform Tests   ############################################################

#%% Analyze results on an image

#todo a test --> is it the image we saved ?

gtbbox_filtering = {"occlusion_rate": (0.9, "max"),
                    "area": (20, "min")}


# plot
i = 1
frame_id = frame_id_list[i]
img_path = img_path_list[i]
df_gtbbox_metadata_frame = df_gtbbox_metadata.loc[int(frame_id)] #todo delay
excluded_gt = filter_gt_bboxes(df_gtbbox_metadata_frame, gtbbox_filtering)
occlusions_ids = [i for i, idx in enumerate(df_gtbbox_metadata_frame.index) if idx  in excluded_gt]
plot_results_img(img_path, frame_id, preds, targets, occlusions_ids)