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


#todo : occluded body joints
#todo city and time in metadata also, then set filters when we want to compare ? But apriori compare on all, no need per city ?
#todo balance may be needed though ...

import json
#%%

#todo take subset of all the images nto be not too long ?

def get_ECP_annotations_and_imagepaths(video_id="004", max_samples=100000):


    json_path = f"/home/raphael/work/datasets/MOTSynth/coco annot/{video_id}.json"

    with open(json_path) as jsonFile:
        annot_motsynth = json.load(jsonFile)

    targets_metadata = {}
    targets = {}
    j = 0
    i = 0
    for image in annot_motsynth["images"]:

        if j > max_samples:
            break
        j += 1


        frame_id = image["id"]


        annots_img = []
        while annot_motsynth["annotations"][i]["image_id"] == image["id"]: #todo bug limit
            bbox_xywh = annot_motsynth["annotations"][i]["bbox"]
            x, y, w, h = bbox_xywh
            bbox_xywh = x, y, x+w, y+h
            annots_img.append(bbox_xywh)
            i += 1


        target = [
            dict(
                boxes=torch.tensor(
                    annots_img)
                )]

        target[0]["labels"] = torch.tensor([0] * len(target[0]["boxes"]))

        # Keep only if at least 1 pedestrian
        if len(target[0]["boxes"]) > 0:
            targets[frame_id] = target
            targets_metadata[frame_id] = annot_motsynth["info"]
            # targets_metadata[frame_id] = (annot_ECP["tags"], [ann["tags"] for ann in annot_ECP["children"]])

    frame_id_list = list(targets.keys())
    img_path_list = [osp.join("/home/raphael/work/datasets/MOTSynth", image["file_name"]) for image in
                     annot_motsynth["images"][:len(frame_id_list)]]

    return targets, targets_metadata, frame_id_list, img_path_list

#%%

import os.path as osp


#todo assumes idex and frame_id are same same

#%%
targets, targets_metadata, frame_id_list, img_path_list = get_ECP_annotations_and_imagepaths(video_id="004", max_samples=100)

#%%



# params
model_name = "cityscapes"
time = "day"
city = "budapest"

if model_name == "cityscapes":
    checkpoint_root = "/home/raphael/work/checkpoints/detection"
    configs_root = "/home/raphael/work/code/pedestrian-detection-sensitivity-analysis/configs"
    faster_rcnn_cityscapes_pth = "faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth"
    faster_rcnn_cityscapes_cfg = "models/faster_rcnn/faster_rcnn_cityscapes.py"
    config_file = osp.join(configs_root, faster_rcnn_cityscapes_cfg)
    checkpoint_file = osp.join(checkpoint_root, faster_rcnn_cityscapes_pth)
else:
    raise ValueError(f"Model name {model_name} not known")


preds = get_preds_from_files(config_file, checkpoint_file, frame_id_list, img_path_list)


#%% Get that are night
import mmcv
video_id = "004"
import json

if os.path.exists("/home/raphael/work/datasets/MOTSynth/coco_infos.json"):
    with open("/home/raphael/work/datasets/MOTSynth/coco_infos.json") as jsonFile:
        video_info = json.load(jsonFile)
else:
    video_info = {}



for i, video_file in enumerate(mmcv.scandir("/home/raphael/work/datasets/MOTSynth/coco annot/")):

    print(video_file)

    if video_file.replace(".json", "") not in video_info.keys():
        try:
            json_path = f"/home/raphael/work/datasets/MOTSynth/coco annot/{video_file}"

            with open(json_path) as jsonFile:
                annot_motsynth = json.load(jsonFile)


            is_night = annot_motsynth["info"]["is_night"]
            print(video_file, is_night)

            video_info[video_file.replace(".json", "")] = annot_motsynth["info"]
        except:
            print(f"Did not work for {video_file}")

    if i > 50:
        break

with open("/home/raphael/work/datasets/MOTSynth/coco_infos.json", 'w') as f:
    json.dump(video_info, f)
night = []
day = []

#%%

print("night",
      [key for key, value in video_info.items() if value["is_night"] and os.path.exists(f"/home/raphael/work/datasets/MOTSynth/frames/{key}")])


print("day",
      [key for key, value in video_info.items() if not value["is_night"] and os.path.exists(f"/home/raphael/work/datasets/MOTSynth/frames/{key}")])



#%%



targets_day, targets_metadata, frame_id_list, img_path_list = get_ECP_annotations_and_imagepaths(
    video_id="006", max_samples=100)
preds_day = get_preds_from_files(config_file, checkpoint_file, frame_id_list, img_path_list)

targets_night, targets_metadata, frame_id_list, img_path_list = get_ECP_annotations_and_imagepaths(video_id="170", max_samples=100)
preds_night = get_preds_from_files(config_file, checkpoint_file, frame_id_list, img_path_list)



#%% Plot first one


"""
Here brief analysis : probably because of groups of people, or NMS, too many boxes
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2


i = 0


frame_id = frame_id_list[i]
img_path = img_path_list[i]

# load img and plot results
def add_bboxes_to_img(img, bboxes, c=(0,0,255), s=1):
    for bbox in bboxes:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        img = cv2.rectangle(img, (x1, y1), (x2, y2), c, s)
    return img

def plot_results_img(img_path, frame_id, preds, targets, s=1):
    img = plt.imread(img_path)
    img = add_bboxes_to_img(img, preds[frame_id][0]["boxes"], c=(0, 0, 255), s=1
                            )
    img = add_bboxes_to_img(img, targets[frame_id][0]["boxes"], c=(0, 255, 0), s=6)
    plt.imshow(img)
    plt.show()

plot_results_img(img_path, frame_id, preds_day, targets_day)


#%%

#%% Check how the bboxes were attributed
import matplotlib.pyplot as plt
import cv2
import torch

#targets_day, targets_metadata, frame_id_list, img_path_list = get_ECP_annotations_and_imagepaths(video_id="107", max_samples=100)
#preds_day = get_preds_from_files(config_file, checkpoint_file, frame_id_list, img_path_list)


#todo get the index of false positives

threshold = 0.1
preds = preds_day
targets = targets_day
frame_id = frame_id_list[i]
img_path = img_path_list[i]

results = {}
results[frame_id] = compute_fp_missratio(preds[frame_id], targets[frame_id], threshold=threshold)


img = plt.imread(img_path)
#img = add_bboxes_to_img(img, preds[frame_id][0]["boxes"], c=(0, 0, 255))

index_matched = torch.tensor(results[frame_id][2])
index_missed = torch.tensor(results[frame_id][3])
index_fp = torch.tensor(results[frame_id][4])

if len(index_matched):
    img = add_bboxes_to_img(img, targets[frame_id][0]["boxes"][index_matched], c=(0, 255, 0), s=6)
if len(index_missed):
    img = add_bboxes_to_img(img, targets[frame_id][0]["boxes"][index_missed], c=(255, 0, 0), s=6)
if len(index_fp):
    img = add_bboxes_to_img(img, preds[frame_id][0]["boxes"][index_fp], c=(0, 255, 255), s=6)


plot_box = preds[frame_id][0]["boxes"][preds[frame_id][0]["scores"] > threshold]
img = add_bboxes_to_img(img, plot_box, c=(0, 0, 255), s=3)

plt.imshow(img)
plt.show()

#%%

img = plt.imread(img_path)
img = add_bboxes_to_img(img, [bbox], c=(0, 0, 255), s=5)
plt.imshow(img)
plt.show()


#%% Debug the false positive

pred_bbox, target_bbox, threshold = preds[frame_id], targets[frame_id], 0


score_sorted = np.argsort(pred_bbox[0]["scores"].numpy())[::-1]

possible_target_bboxs = [target_bbox for target_bbox in target_bbox[0]["boxes"]]
possible_target_bboxs_ids = list(range(len(target_bbox[0]["boxes"])))
matched_target_bbox_list = []
unmatched_preds = []

for i in score_sorted:

    if len(possible_target_bboxs) == 0 or pred_bbox[0]["scores"][i] < threshold:
        break

    bbox = pred_bbox[0]["boxes"][i]

    # Compute all IoU
    IoUs = [torchvision.ops.box_iou(bbox.unsqueeze(0), target_bbox.unsqueeze(0)) for
            target_bbox in possible_target_bboxs]
    IoUs_index = [i for i,IoU in enumerate(IoUs) if IoU > 0.5]
    if len(IoUs_index) == 0:
        unmatched_preds.append(i)
    else:
        # Match it best with existing boxes
        matched_target_bbox = np.argmax(IoUs)
        matched_target_bbox_list.append(possible_target_bboxs_ids[matched_target_bbox])

        # Remove
        possible_target_bboxs.pop(matched_target_bbox)
        possible_target_bboxs_ids.pop(matched_target_bbox)


    img = plt.imread(img_path)
    img = add_bboxes_to_img(img, [bbox], c=(0, 0, 255), s=5)
    plt.imshow(img)
    plt.show()

    img = plt.imread(img_path)
    matched_target_bbox = 12
    img = add_bboxes_to_img(img,
                            [possible_target_bboxs[possible_target_bboxs_ids[matched_target_bbox]]], c=(0, 255, 0), s=5)
    plt.imshow(img)
    plt.show()

# Compute the False Positives
target_bbox_missed = np.setdiff1d(list(range(len(target_bbox[0]["boxes"]))), matched_target_bbox_list).tolist()

# Number of predictions above threshold - Number of matched target_bboxs
#fp_image = max(0, (pred_bbox[0]["scores"] > threshold).numpy().sum() - len(matched_target_bbox_list))
fp_image = len(unmatched_preds)

# False negatives
fn_image = max(0, len(target_bbox[0]["boxes"]) - len(matched_target_bbox_list))
miss_ratio_image = fn_image / len(target_bbox[0]["boxes"])

