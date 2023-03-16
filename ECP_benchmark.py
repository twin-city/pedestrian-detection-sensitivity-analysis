

import os
import numpy as np
import torch
import json
from mmcv.ops import nms


#%% Load the ECP images
time = "day"
set = "val"
city = "roma"

for city in ["koeln", "leipzig"]:


    root = f"/media/raphael/Projects/datasets/EuroCityPerson/ECP/{time}/labels/{set}/{city}"
    son_files = [x for x in os.listdir(root) if ".json" in x]
    total_frame_ids = [x.split("_")[1].split(".json")[0] for x in os.listdir(root) if ".json" in x]





    targets = {}
    for frame_id in total_frame_ids:

        # Load ECP annotations
        json_path = f"/media/raphael/Projects/datasets/EuroCityPerson/ECP/{time}/labels/{set}/{city}/{city}_{frame_id}.json"
        with open(json_path) as jsonFile:
            annot_ECP = json.load(jsonFile)

            target = [
              dict(
                boxes=torch.tensor(
                    [(c["x0"], c["y0"], c["x1"], c["y1"]) for c in annot_ECP["children"] if c["identity"]=="pedestrian"]
                ),
              )
            ]

            target[0]["labels"] = torch.tensor([0]*len(target[0]["boxes"]))

            # Keep only if at least 1 pedestrian
            if len(target[0]["boxes"]) > 0:
                targets[frame_id] = target

    #%% Which images to apply on ? And annotations For now Caltech Pedestrian
    frame_id_list = list(targets.keys())[:200]




    #%% Perform the inferences


    from mmdet.apis import init_detector, inference_detector
    import mmcv

    # Specify the path to model config and checkpoint file
    config_file = '../mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = "/home/raphael/work/checkpoints/detection/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda')

    os.makedirs("out/ECP", exist_ok=True)

    for frame_id in frame_id_list:
        print(frame_id)

        out_file = f"out/ECP/results_{frame_id}.npy"

        if not os.path.exists(out_file):
            img_path = f"/media/raphael/Projects/datasets/EuroCityPerson/ECP/{time}/img/{set}/{city}/{city}_{frame_id}.png"

            # test a single image and show the results
            result = inference_detector(model, img_path)

            # How to save ?
            np.save(out_file, result[0])



    #%% Load the results and convert to pytorch

    import torch
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    preds = {}
    for frame_id in frame_id_list:
        bboxes_people = np.load(f"out/ECP/results_{frame_id}.npy")

        """
        bboxes_people_nms, _ = nms(
            bboxes_people[:, :4],
            bboxes_people[:, 4],
            0.25,
            score_threshold=0.25)
        """

        pred = [
          dict(
            boxes=torch.tensor(bboxes_people[:,:4]),
            scores=torch.tensor(bboxes_people[:,4]),
            labels=torch.tensor([0]* len(bboxes_people)),
          )
        ]
        preds[frame_id] = pred


    #%% Compute False Positives and FFPI

    import numpy as np
    import torchvision


    from utils import *


    #%% Compute miss ratio and fp depending on threshold
    avrg_fp_list, avrg_missrate_list = compute_ffpi_against_fp(preds, targets)
    #%%

    if city == "koeln":
        avrg_fp_list_night, avrg_missrate_list_night = avrg_fp_list, avrg_missrate_list
    elif city == "leipzig":
        avrg_fp_list_day, avrg_missrate_list_day = avrg_fp_list, avrg_missrate_list
    else:
        print("Problem wwith time")

#%%
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(avrg_fp_list_day, avrg_missrate_list_day, c="green")
ax.scatter(avrg_fp_list_day, avrg_missrate_list_day, c="green")

ax.plot(avrg_fp_list_night, avrg_missrate_list_night, c="purple")
ax.scatter(avrg_fp_list_night, avrg_missrate_list_night, c="purple")

ax.set_xscale('log')
ax.set_yscale('log')
plt.show()


#%%

frame_id = frame_id_list[0]

preds[frame_id]

det_bboxes, _ = nms(
    preds[frame_id][0]["boxes"],
    preds[frame_id][0]["scores"],
    0.5,
    score_threshold=0.5)


img_path = f"/media/raphael/Projects/datasets/EuroCityPerson/ECP/{time}/img/val/{city}/{city}_{frame_id}.png"
img = plt.imread(img_path)
img = add_bboxes_to_img(img, det_bboxes[:, :4], c=(0, 0, 255), s=1)
plt.imshow(img)
plt.show()

#%% Check si cohérent avec les images !!!!!!


"""
Here brief analysis : probably because of groups of people, or NMS, too many boxes
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

results = {}
for frame_id in preds.keys():
    results[frame_id] = compute_fp_missratio(preds[frame_id], targets[frame_id])

frame_id_max_missrate = np.argmax([x[1] for x in results.values()])
frame_id_max_fp = np.argmax([x[0] for x in results.values()])


frame_id = frame_id_list[frame_id_max_fp]
frame_id = frame_id_list[frame_id_max_missrate] #todo pas ouf annot ici

# load img and plot results
def add_bboxes_to_img(img, bboxes, c=(0,0,255), s=1):
    for bbox in bboxes:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        img = cv2.rectangle(img, (x1, y1), (x2, y2), c, s)
    return img

def plot_results_img(frame_id, preds, targets, s=1):
    img_path = f"/media/raphael/Projects/datasets/EuroCityPerson/ECP/{time}/img/val/{city}/{city}_{frame_id}.png"
    img = plt.imread(img_path)
    img = add_bboxes_to_img(img, preds[frame_id][0]["boxes"], c=(0, 0, 255), s=s)
    img = add_bboxes_to_img(img, targets[frame_id][0]["boxes"], c=(0, 255, 0), s=s)
    plt.imshow(img)
    plt.show()

plot_results_img(frame_id, preds, targets)


#%% Check how the bboxes were attributed
import matplotlib.pyplot as plt
import cv2
import torch
frame_id = frame_id_list[0]

img_path = f"/media/raphael/Projects/datasets/EuroCityPerson/ECP/{time}/img/val/{city}/{city}_{frame_id}.png"
img = plt.imread(img_path)
#img = add_bboxes_to_img(img, preds[frame_id][0]["boxes"], c=(0, 0, 255))

index_matched = torch.tensor(results[frame_id][2])
index_missed = torch.tensor(results[frame_id][3])

if len(index_matched):
    img = add_bboxes_to_img(img, targets[frame_id][0]["boxes"][index_matched], c=(0, 255, 0), s=2)
if len(index_missed):
    img = add_bboxes_to_img(img, targets[frame_id][0]["boxes"][index_missed], c=(255, 0, 0), s=2)
img = add_bboxes_to_img(img, preds[frame_id][0]["boxes"], c=(0, 0, 255), s=1)

plt.imshow(img)
plt.show()

# todo Debug the ECP Benchmark ici !!! et choper des métriques déjà calculées pourcheck