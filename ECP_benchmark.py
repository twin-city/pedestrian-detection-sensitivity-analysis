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


#todo city and time in metadata also, then set filters when we want to compare ? But apriori compare on all, no need per city ?
#todo balance may be needed though ...


def get_ECP_annotations_and_imagepaths(time, set, city, max_samples=100000):

    # Here ECP specific
    root = f"/media/raphael/Projects/datasets/EuroCityPerson/ECP/{time}/labels/{set}/{city}"
    total_frame_ids = [x.split("_")[1].split(".json")[0] for x in os.listdir(root) if ".json" in x]

    # Init dicts for bboxes annotations and metadata
    targets = {}
    targets_metadata = {}
    for i, frame_id in enumerate(total_frame_ids):

        # set max samples #todo
        if i>max_samples:
            break

        # Load ECP annotations
        json_path = f"/media/raphael/Projects/datasets/EuroCityPerson/ECP/{time}/labels/{set}/{city}/{city}_{frame_id}.json"
        with open(json_path) as jsonFile:
            annot_ECP = json.load(jsonFile)

            target = [
              dict(
                boxes=torch.tensor(
                    [(c["x0"], c["y0"], c["x1"], c["y1"]) for c in annot_ECP["children"] if c["identity"] == "pedestrian"] #todo might not be the thing todo
                ),
              )
            ]

            target[0]["labels"] = torch.tensor([0]*len(target[0]["boxes"]))

            # Keep only if at least 1 pedestrian
            if len(target[0]["boxes"]) > 0:
                targets[frame_id] = target
                targets_metadata[frame_id] = (annot_ECP["tags"], [ann["tags"] for ann in annot_ECP["children"]])

    frame_id_list = list(targets.keys())

    img_path_list = []
    for frame_id in frame_id_list:
        print(frame_id)
        img_path = f"/media/raphael/Projects/datasets/EuroCityPerson/ECP/{time}/img/val/{city}/{city}_{frame_id}.png"
        img_path_list.append(img_path)

    return targets, targets_metadata, frame_id_list, img_path_list



def get_preds_from_files(config_file, checkpoint_file, frame_id_list, file_list, nms=False):

    preds = {}

    model = init_detector(config_file, checkpoint_file, device='cuda')

    for frame_id, img_path in zip(frame_id_list, file_list):

        # test a single image and show the results
        result = inference_detector(model, img_path)
        bboxes_people = result[0]

        if nms:
            bboxes_people, _ = nms(
                bboxes_people[:, :4],
                bboxes_people[:, 4],
                0.25,
                score_threshold=0.25)

        pred = [
            dict(
                boxes=torch.tensor(bboxes_people[:, :4]),
                scores=torch.tensor(bboxes_people[:, 4]),
                labels=torch.tensor([0] * len(bboxes_people)),
            )
        ]
        preds[frame_id] = pred

    return preds





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



targets, targets_metadata, frame_id_list, img_path_list = get_ECP_annotations_and_imagepaths(time, "val", city, max_samples=100)
preds = get_preds_from_files(config_file, checkpoint_file, frame_id_list, img_path_list)


#%%

city = "roma"

time = "day"
targets_day, targets_metadata_day, frame_id_list, img_path_list = get_ECP_annotations_and_imagepaths(time, "val", city, max_samples=100)
preds_day = get_preds_from_files(config_file, checkpoint_file, frame_id_list, img_path_list)

time = "night"
targets_night, targets_metadata_night, frame_id_list, img_path_list = get_ECP_annotations_and_imagepaths(time, "val", city, max_samples=100)
preds_night = get_preds_from_files(config_file, checkpoint_file, frame_id_list, img_path_list)


def compute_mAP(preds, targets):
    metric = MeanAveragePrecision()
    metric.update([pred[0] for pred in preds.values()],
                  [target[0] for target in targets.values()])
    computed_metrics = metric.compute()
    return computed_metrics

#from pprint import pprint
#pprint(metric.compute())


metrics_day = compute_mAP(preds_day, targets_day)
metrics_night = compute_mAP(preds_night, targets_night)


df_mAP = pd.DataFrame([metrics_day, metrics_night])
df_mAP

#%% Compute False Positives and FFPI



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

ax.set_ylim(0,1)
ax.set_xlim(0.001, 20)

plt.legend()
plt.show()