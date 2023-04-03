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
video_ids = ["004", "170","130", "033", "103", "107", "145"]
max_sample = 50

#todo bug 140, 174


#%% params

# model
model_name = "cityscapes"
device = "cuda"

if model_name == "cityscapes":
    checkpoint_root = "/home/raphael/work/checkpoints/detection"
    configs_root = "/home/raphael/work/code/pedestrian-detection-sensitivity-analysis/configs"
    faster_rcnn_cityscapes_pth = "faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth"
    faster_rcnn_cityscapes_cfg = "models/faster_rcnn/faster_rcnn_cityscapes.py"
    config_file = osp.join(configs_root, faster_rcnn_cityscapes_cfg)
    checkpoint_file = osp.join(checkpoint_root, faster_rcnn_cityscapes_pth)
else:
    raise ValueError(f"Model name {model_name} not known")


#%%


targets, targets_metadata, frames_metadata, frame_id_list, img_path_list = get_MoTSynth_annotations_and_imagepaths(video_ids=video_ids, max_samples=max_sample)
preds = get_preds_from_files(config_file, checkpoint_file, frame_id_list, img_path_list, device=device)


#%% Analyze results on an image

threshold = 0.6

# choose image
i = 3
frame_id = frame_id_list[i]
img_path = img_path_list[i]

target_metadata = targets_metadata[frame_id]
occlusions = [(x-1).mean() for x in target_metadata["keypoints"]]
occlusions_ids = list(np.where(np.array(occlusions) > 0.0)[0])
occlusions_ids = []

# plot
plot_results_img(img_path, frame_id, preds, targets)

# Compute metrics from image
pred_bbox, target_bbox = preds[frame_id], targets[frame_id]

#compute_fp_missratio2(pred_bbox, target_bbox, threshold=threshold, excluded_gt=occlusions_ids)

#%%

# plot metrics
plot_fp_fn_img(frame_id_list, img_path_list, preds, targets, index_frame=i, threshold=threshold)

#todo accord the i and frame_id

#%% Compute miss ratio

ids_night = [key for key,val in frames_metadata.items() if val["is_night"]]
ids_day = [key for key,val in frames_metadata.items() if not val["is_night"]]

avrg_fp_list_1, avrg_missrate_list_1 = compute_ffpi_against_fp(preds, targets, targets_metadata, ids_day)
avrg_fp_list_2, avrg_missrate_list_2 = compute_ffpi_against_fp(preds, targets, targets_metadata, ids_night)


#%%
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(avrg_fp_list_1, avrg_missrate_list_1, c="green", label="Day test set")
ax.scatter(avrg_fp_list_1, avrg_missrate_list_1, c="green")

ax.plot(avrg_fp_list_2, avrg_missrate_list_2, c="purple", label="Night test set")
ax.scatter(avrg_fp_list_2, avrg_missrate_list_2, c="purple")

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_ylim(0.1, 1)
ax.set_xlim(0.1, 20)

plt.legend()
plt.show()


#%% Add the keypoints

visual_check_motsynth_annotations(video_num="004", img_file_name="0005.jpg", shift=3)


#%% Analyze results on an image

target_metadata = targets_metadata[frame_id]

occlusions = [(x-1).mean() for x in target_metadata["keypoints"]]


threshold = 0.6

# choose image
i = 3
frame_id = frame_id_list[i]
img_path = img_path_list[i]

# plot
plot_results_img(img_path, frame_id, preds, targets, excl_gt_indices=[3])


#%%

video_num="004"
img_file_name="0005.jpg"
shift=3

json_path = f"/home/raphael/work/datasets/MOTSynth/coco annot/{video_num}.json"
with open(json_path) as jsonFile:
    annot_motsynth = json.load(jsonFile)

img_id = [(x["id"]) for x in annot_motsynth["images"] if img_file_name in x["file_name"]][0]
bboxes = [xywh2xyxy(x["bbox"]) for x in annot_motsynth["annotations"] if x["image_id"] == img_id + shift]

img_path = f"/home/raphael/work/datasets/MOTSynth/frames/{video_num}/rgb/{img_file_name}"
img = plt.imread(img_path)
img = add_bboxes_to_img(img, bboxes, c=(0, 255, 0), s=6)

ped_id = 16

# keypoints = np.array(annot_motsynth["annotations"][ped_id]["keypoints"]).reshape((22,3))

annots = [x for x in annot_motsynth["annotations"] if x["image_id"] == img_id + shift]

keypoints = [(np.array(x["keypoints"])).reshape((22,3)) for x in annot_motsynth["annotations"] if x["image_id"] == img_id + shift]
plt.scatter(keypoints[ped_id][:,0], keypoints[ped_id][:,1], c=keypoints[ped_id][:,2])

plt.imshow(img)
plt.show()

#%%
img_path = f"/home/raphael/work/datasets/MOTSynth/frames/{video_num}/rgb/{img_file_name}"
img = plt.imread(img_path)
plt.imshow(img)
plt.show()