import matplotlib.pyplot as plt
from configs.paths_cfg import *

#%% Load Dataset

import json
json_path = "/home/raphael/work/code/dataset_conversion_utils/caltech-pedestrian-dataset-converter/data/annotations.json"
with open(json_path) as jsonFile:
    annot_caltech = json.load(jsonFile)


#%% Show results on images / bboxes
from mmdet.apis import init_detector, inference_detector
import mmcv

# Specify the path to model config and checkpoint file
config_file = '../mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
#checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
checkpoint_file = "/home/raphael/work/checkpoints/detection/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cpu')

# test a single image and show the results

img_path = "/home/raphael/work/code/dataset_conversion_utils/caltech-pedestrian-dataset-converter/data/images/set01_V000_1680.png"
result = inference_detector(model, img_path)
bboxes_people = result[0]
#model.show_result(img_path, result, out_file='result.jpg')

#%% Plot model

import cv2
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt

img = plt.imread(img_path)
for bbox in bboxes_people:
    x1, y1, x2, y2, p = [int(v) for v in bbox]
    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, int(255)), 1)
plt.imshow(img)
plt.show()



#%% Plot an image


for frame_id, frame in annot_caltech['set01']["V000"]["frames"].items():
    for bbox in frame:
        print(frame_id, bbox["occl"], bbox["lbl"])

img_num = "1680"
path_img = "/home/raphael/work/code/dataset_conversion_utils/caltech-pedestrian-dataset-converter/data/images/set01_V000_{}.png".format(img_num)
img = plt.imread(path_img)

for bbox in annot_caltech['set01']["V000"]["frames"][img_num]:
    print(bbox["occl"])
    x, y, w, h = [int(v) for v in bbox["pos"]]
    col = (0, 0, 255)
    img = cv2.rectangle(img, (x, y), (x + w, y + h), col, 1)
    print(w,h)

    x, y, w, h = [int(v) for v in bbox["posv"]]
    col = (0, 255, 0)
    img = cv2.rectangle(img, (x, y), (x + w, y + h), col, 1)
    print(w,h)

plt.imshow(img)
plt.show()


#%% Use torchmetrics

"""
https://github.com/Cartucho/mAP/issues/43



for bbox in annot_caltech['set01']["V000"]["frames"][img_num]:
    x, y, w, h = [int(v) for v in bbox["pos"]]
    x, y, w, h = bbox["pos"]
"""






#%%
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

preds = [
  dict(
    boxes=torch.stack([torch.tensor(bbox[:4]) for bbox in bboxes_people], axis=0),
    scores=torch.stack([torch.tensor(bbox[4]) for bbox in bboxes_people], axis=0),
    labels=torch.tensor([0]* len(bboxes_people)),
  )
]

target = [
  dict(
    boxes=torch.tensor([(b["pos"][0], b["pos"][1], b["pos"][0] + b["pos"][2], b["pos"][1] + b["pos"][3])
 for b in annot_caltech['set01']["V000"]["frames"][img_num]]),
    labels=torch.tensor([0]*len(annot_caltech['set01']["V000"]["frames"][img_num])),
  )
]

metric = MeanAveragePrecision()
metric.update(preds, target)
from pprint import pprint
pprint(metric.compute())

#%% Missing Rate and FPPI
import numpy as np
import torchvision
score_sorted = np.argsort(preds[0]["scores"].numpy())

possible_targets = [target_bbox for target_bbox in target[0]["boxes"]]
matched_target_list = []

for i in score_sorted:
    bbox = preds[0]["boxes"][i]

    # Compute all IoU
    IoUs = [torchvision.ops.box_iou(bbox.unsqueeze(0), target_bbox.unsqueeze(0)) for
            target_bbox in possible_targets]

    # Match it best with existing boxes
    matched_target = np.argmax(IoUs)
    matched_target_list.append(matched_target) #todo bug because we removed IDs

    # Remove
    possible_targets.pop(matched_target)

    if len(possible_targets)==0:
        break






#%% Compute miss rate + False positives

"""
Greedy method (à confirmer plus tard)
"""





#%%








#%%
"""
KAIST Pedestrian Detection Benchmark
"""


#%% Now that I have tje bboxes what do I do ?

"""
https://gist.github.com/tarlen5/008809c3decf19313de216b9208f3734


- IoU on synscapes


On : a benchmark --> Miss rate
 
 This is preferred to precision recall curves for certain tasks, e.g. automotive applications, as typically there is an upper limit on the acceptable false positives perimage rate independent of pedestrian density.

"""




#%% Correlate result metrics to bbox properties / image property