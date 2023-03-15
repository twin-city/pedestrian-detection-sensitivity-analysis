




"""
For now inference detector is launched multiple times


- How to save ?

- Will be accelerated later with

./tools/dist_test.sh \
    configs/mask_rcnn_r50_fpn_1x_coco.py \
    checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth \
    8 \
    --out results.pkl \
    --eval bbox segm
"""


import os
import numpy as np
import torch
import json

#%% Which images to apply on ? And annotations For now Caltech Pedestrian


# Load Caltech annotations
json_path = "/home/raphael/work/code/dataset_conversion_utils/caltech-pedestrian-dataset-converter/data/annotations.json"
with open(json_path) as jsonFile:
    annot_caltech = json.load(jsonFile)


# All targets in set01/V000 of Caltech Pedestrian to torch format
targets = {}
for frame_id, frame in annot_caltech['set01']["V000"]["frames"].items():
    target = [
      dict(
        boxes=torch.tensor([(b["pos"][0], b["pos"][1], b["pos"][0] + b["pos"][2], b["pos"][1] + b["pos"][3])
     for b in annot_caltech['set01']["V000"]["frames"][frame_id]]),
        labels=torch.tensor([0]*len(annot_caltech['set01']["V000"]["frames"][frame_id])),
      )
    ]
    targets[frame_id] = target

# Which images to work on ?
frame_id_list = list(annot_caltech['set01']["V000"]["frames"].keys())[:100]

#%% Perform the inferences

from mmdet.apis import init_detector, inference_detector
import mmcv

# Specify the path to model config and checkpoint file
config_file = '../mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = "/home/raphael/work/checkpoints/detection/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cpu')

os.makedirs("out/caltech_pedestrian", exist_ok=True)

for frame_id in frame_id_list:
    print(frame_id)

    out_file = f"out/caltech_pedestrian/results_{frame_id}.npy"

    if not os.path.exists(out_file):
        img_path = "/home/raphael/work/code/dataset_conversion_utils/caltech-pedestrian-dataset-converter/data/images/set01_V000_{}.png".format(
            frame_id)

        # test a single image and show the results
        result = inference_detector(model, img_path)

        # How to save ?
        np.save(out_file, result[0])



#%% Load the results and convert to pytorch

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

preds = {}
for frame_id in frame_id_list:
    bboxes_people = np.load(f"out/caltech_pedestrian/results_{frame_id}.npy")
    pred = [
      dict(
        boxes=torch.stack([torch.tensor(bbox[:4]) for bbox in bboxes_people], axis=0),
        scores=torch.stack([torch.tensor(bbox[4]) for bbox in bboxes_people], axis=0),
        labels=torch.tensor([0]* len(bboxes_people)),
      )
    ]
    preds[frame_id] = pred


#%% Compute False Positives and FFPI

import numpy as np
import torchvision


def compute_fp_missratio(pred_bbox, target_bbox):
    score_sorted = np.argsort(pred_bbox[0]["scores"].numpy())[::-1]

    possible_target_bboxs = [target_bbox for target_bbox in target_bbox[0]["boxes"]]
    possible_target_bboxs_ids = list(range(len(target_bbox[0]["boxes"])))
    matched_target_bbox_list = []

    for i in score_sorted:

        if len(possible_target_bboxs) == 0 or pred_bbox[0]["scores"][i] < 0.5:
            break

        bbox = pred_bbox[0]["boxes"][i]

        # Compute all IoU
        IoUs = [torchvision.ops.box_iou(bbox.unsqueeze(0), target_bbox.unsqueeze(0)) for
                target_bbox in possible_target_bboxs]

        # Match it best with existing boxes
        matched_target_bbox = np.argmax(IoUs)
        matched_target_bbox_list.append(possible_target_bboxs_ids[matched_target_bbox])

        # Remove
        possible_target_bboxs.pop(matched_target_bbox)
        possible_target_bboxs_ids.pop(matched_target_bbox)

    # %% Compute the False Positives

    # Number of predictions above threshold - Number of matched target_bboxs
    fp_image = max(0, (pred_bbox[0]["scores"] > 0.5).numpy().sum() - len(matched_target_bbox_list))

    # False negatives
    fn_image = max(0, len(target_bbox[0]["boxes"]) - len(matched_target_bbox_list))
    miss_ratio_image = fn_image / len(target_bbox[0]["boxes"])

    return fp_image, miss_ratio_image



#%% foo to compute the results


results = {}
for frame_id in frame_id_list:
    results[frame_id] = compute_fp_missratio(preds[frame_id], targets[frame_id])
print(results)



#%% Check si cohérent avec les images !!!!!!
