import os
import numpy as np
import pandas as pd
import torch
import json
from mmcv.ops import nms
import cv2
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from mmdet.apis import init_detector, inference_detector
import pandas as pd
import numpy as np
import torchvision
from utils import *
import os.path as osp
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os.path as osp

#%%
get_motsynth_day_night_video_ids(max_iter=50, force=False)

#%%

# Parameters data
video_id = "187"
max_sample = 20




#%% params

# model
model_name = "cityscapes"
device = "cpu"

if model_name == "cityscapes":
    checkpoint_root = "/home/raphael/work/checkpoints/detection"
    configs_root = "/home/raphael/work/code/pedestrian-detection-sensitivity-analysis/configs"
    faster_rcnn_cityscapes_pth = "faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth"
    faster_rcnn_cityscapes_cfg = "models/faster_rcnn/faster_rcnn_cityscapes.py"
    config_file = osp.join(configs_root, faster_rcnn_cityscapes_cfg)
    checkpoint_file = osp.join(checkpoint_root, faster_rcnn_cityscapes_pth)
else:
    raise ValueError(f"Model name {model_name} not known")


#%% Compute predictions
video_id = "004"
targets_day, targets_metadata, frame_id_list_day, img_path_list_day = get_MoTSynth_annotations_and_imagepaths(video_id=video_id, max_samples=max_sample)
preds_day = get_preds_from_files(config_file, checkpoint_file, frame_id_list_day, img_path_list_day, device=device)

video_id = "170"
targets_night, targets_metadata_night, frame_id_list_night, img_path_night = get_MoTSynth_annotations_and_imagepaths(video_id=video_id, max_samples=max_sample)
preds_night = get_preds_from_files(config_file, checkpoint_file, frame_id_list_night, img_path_night, device=device)


#%% Analyze results on an image

threshold = 1

# choose image
i = 0
frame_id = frame_id_list_day[i]
img_path = img_path_list_day[i]

# plot
plot_results_img(img_path, frame_id, preds_day, targets_day)

# Compute metrics from image
pred_bbox, target_bbox = preds_day[frame_id], targets_day[frame_id]
compute_fp_missratio2(pred_bbox, target_bbox, threshold=threshold)

# plot metrics
plot_fp_fn_img(frame_id_list_day, img_path_list_day, preds_day, targets_day, index_frame=i, threshold=threshold)

#todo accord the i and frame_id

#%% Compute miss ratio


avrg_fp_list_day, avrg_missrate_list_day = compute_ffpi_against_fp(preds_day, targets_day)
avrg_fp_list_night, avrg_missrate_list_night = compute_ffpi_against_fp(preds_night, targets_night)


#%%
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(avrg_fp_list_day, avrg_missrate_list_day, c="green", label="Day test set")
ax.scatter(avrg_fp_list_day, avrg_missrate_list_day, c="green")

ax.plot(avrg_fp_list_night, avrg_missrate_list_night, c="purple", label="Night test set")
ax.scatter(avrg_fp_list_night, avrg_missrate_list_night, c="purple")

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_ylim(0.1,1)
ax.set_xlim(0.001, 20)

plt.legend()
plt.show()
