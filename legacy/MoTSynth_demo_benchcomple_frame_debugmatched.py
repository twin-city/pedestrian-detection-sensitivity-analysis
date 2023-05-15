import os

import matplotlib.pyplot as plt
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
df_frame_metadata["num_pedestrian"] = df_gtbbox_metadata.groupby("frame_id").apply(len).loc[df_frame_metadata.index]


#%% Multiple plots
from src.utils import compute_correlations, plot_correlations
#corr_matrix, p_matrix = compute_correlations(df_frame_metadata.groupby("seq_name").apply(lambda x: x.mean()), seq_cofactors)
#plot_correlations(corr_matrix, p_matrix, title="Correlations between metadatas at sequence level")


#%% Now plot the multiple cases !!!!!!
from src.detection.metrics import compute_model_metrics_on_dataset
model_names = ["faster-rcnn_cityscapes", "mask-rcnn_coco"]

model_name = model_names[0]



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

detector = Detector(model_name, device="cuda")
preds = detector.get_preds_from_files(dataset_name, root, df_frame_metadata)

from src.detection.metrics import detection_metric
metric = detection_metric(gtbbox_filtering)
_, df_gt_bbox = metric.compute(dataset_name, model_name, preds, targets, df_gtbbox_metadata,
                                        gtbbox_filtering)
#todo add threshold to 1 in data
from src.utils import plot_results_img
#plot_results_img(img_path, frame_id, preds=preds, targets=targets,
#             df_gt_bbox=df_gt_bbox, threshold=0.9999) #todo seems there is a bug, woman in middle should be in red and guy should be red. No sense of all this.

#%%

i = 500
img_path = osp.join(root, df_frame_metadata["file_name"].iloc[i])
frame_id = df_frame_metadata.index[i]

plot_results_img(img_path, frame_id, preds=preds, targets=targets,
             df_gt_bbox=df_gt_bbox, threshold=0.99) #todo seems there is a bug, woman in middle should be in red and guy should be red. No sense of all this.


#%%

plot_results_img(img_path, frame_id, preds=preds, targets=targets,
             df_gt_bbox=None, threshold=0.99) #todo seems there is a bug, woman in middle should be in red and guy should be red. No sense of all this.

#%%
plot_results_img(img_path, frame_id, preds=None, targets=targets,
             df_gt_bbox=df_gt_bbox, threshold=0.99) #todo seems there is a bug, woman in middle should be in red and guy should be red. No sense of all this.



#%% Seems there is a bug on matching --> Need to debug that --> plot for each box ...

threshold = 0.99

from src.detection.metrics import compute_fp_missratio

results = compute_fp_missratio(preds[frame_id], targets[frame_id], threshold=threshold, excluded_gt=excluded_gt)
