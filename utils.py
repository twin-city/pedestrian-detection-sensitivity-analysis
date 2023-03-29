import cv2
import torch
from mmdet.apis import init_detector, inference_detector
import numpy as np
import torchvision
import os
import matplotlib.pyplot as plt
import mmcv
import json
import os.path as osp
from tqdm import tqdm
from utils import *


def xywh2xyxy(bbox):
    x, y, w, h = bbox
    return x, y, x + w, y + h



def compute_fp_missratio(pred_bbox, target_bbox, threshold=0.5):
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
        IoUs_index = [IoU for IoU in IoUs if IoU > 0.5]
        if len(IoUs_index) == 0:
            unmatched_preds.append(i)
        else:
            # Match it best with existing boxes
            matched_target_bbox = np.argmax(IoUs)
            matched_target_bbox_list.append(possible_target_bboxs_ids[matched_target_bbox])

            # Remove
            possible_target_bboxs.pop(matched_target_bbox)
            possible_target_bboxs_ids.pop(matched_target_bbox)

    # Compute the False Positives
    target_bbox_missed = np.setdiff1d(list(range(len(target_bbox[0]["boxes"]))), matched_target_bbox_list).tolist()

    # Number of predictions above threshold - Number of matched target_bboxs
    fp_image = max(0, (pred_bbox[0]["scores"] > threshold).numpy().sum() - len(matched_target_bbox_list))

    # False negatives
    fn_image = max(0, len(target_bbox[0]["boxes"]) - len(matched_target_bbox_list))
    miss_ratio_image = fn_image / len(target_bbox[0]["boxes"])

    return fp_image, miss_ratio_image, matched_target_bbox_list, target_bbox_missed, unmatched_preds


def compute_fp_missratio2(pred_bbox, target_bbox, threshold=0.5):

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

        # If no target bbox overlap with IoU>=0.5, set as false positive
        if len(IoUs_index) == 0:
            unmatched_preds.append(i)
        # If there exist an overlap
        else:
            # Match it best with target boxes still unmatched
            matched_target_bbox = np.argmax(IoUs)
            matched_target_bbox_list.append(possible_target_bboxs_ids[matched_target_bbox])

            # Remove
            possible_target_bboxs.pop(matched_target_bbox)
            possible_target_bboxs_ids.pop(matched_target_bbox)

    # Compute the False Positives
    target_bbox_missed = np.setdiff1d(list(range(len(target_bbox[0]["boxes"]))), matched_target_bbox_list).tolist()

    # Number of predictions above threshold - Number of matched target_bboxs
    #fp_image = max(0, (pred_bbox[0]["scores"] > threshold).numpy().sum() - len(matched_target_bbox_list))
    fp_image = len(unmatched_preds)

    # False negatives
    fn_image = max(0, len(target_bbox[0]["boxes"]) - len(matched_target_bbox_list))
    miss_ratio_image = fn_image / len(target_bbox[0]["boxes"])

    return fp_image, miss_ratio_image, matched_target_bbox_list, target_bbox_missed, unmatched_preds


def compute_ffpi_against_fp(preds, targets):
    """
    On preds keys.
    :param preds:
    :param targets:
    :return:
    """

    thresholds = list(np.arange(0, 1, 0.1))+[0.99]#+list(np.arange(0.9, 1, 0.3))

    avrg_fp_list = []
    avrg_missrate_list = []

    for threshold in thresholds:
        results = {}
        for frame_id in preds.keys():
            results[frame_id] = compute_fp_missratio2(preds[frame_id], targets[frame_id], threshold=threshold)
        avrg_fp = np.mean([x[0] for x in results.values()])
        avrg_missrate = np.mean([x[1] for x in results.values()])

        avrg_fp_list.append(avrg_fp)
        avrg_missrate_list.append(avrg_missrate)

    return avrg_fp_list, avrg_missrate_list




def get_preds_from_files(config_file, checkpoint_file, frame_id_list, file_list, nms=False, device="cuda"):

    preds = {}

    model = init_detector(config_file, checkpoint_file, device=device)

    #for frame_id, img_path in zip(frame_id_list, file_list):

    for i in tqdm(range(len(file_list))):

        frame_id = frame_id_list[i]
        img_path = file_list[i]

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

#%% Plot utils

def add_bboxes_to_img(img, bboxes, c=(0,0,255), s=1):
    for bbox in bboxes:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        img = cv2.rectangle(img, (x1, y1), (x2, y2), c, s)
    return img

def plot_results_img(img_path, frame_id, preds=None, targets=None):
    img = plt.imread(img_path)
    if preds is not None:
        img = add_bboxes_to_img(img, preds[frame_id][0]["boxes"], c=(0, 0, 255), s=3)
    if targets is not None:
        img = add_bboxes_to_img(img, targets[frame_id][0]["boxes"], c=(0, 255, 0), s=6)
    plt.imshow(img)
    plt.show()


def plot_fp_fn_img(frame_id_list, img_path_list, preds, targets, index_frame, threshold=0.5):
    preds = preds
    targets = targets
    frame_id = frame_id_list[index_frame]
    img_path = img_path_list[index_frame]

    results = {}
    results[frame_id] = compute_fp_missratio(preds[frame_id], targets[frame_id], threshold=threshold)

    img = plt.imread(img_path)
    # img = add_bboxes_to_img(img, preds[frame_id][0]["boxes"], c=(0, 0, 255))

    index_matched = torch.tensor(results[frame_id][2])
    index_missed = torch.tensor(results[frame_id][3])
    index_fp = torch.tensor(results[frame_id][4])

    # Predictions
    plot_box = preds[frame_id][0]["boxes"][preds[frame_id][0]["scores"] > threshold]
    img = add_bboxes_to_img(img, plot_box, c=(0, 0, 255), s=3)

    if len(index_missed):
        img = add_bboxes_to_img(img, targets[frame_id][0]["boxes"][index_missed], c=(255, 0, 0), s=6)
    if len(index_fp):
        img = add_bboxes_to_img(img, preds[frame_id][0]["boxes"][index_fp], c=(0, 255, 255), s=6)
    if len(index_matched):
        img = add_bboxes_to_img(img, targets[frame_id][0]["boxes"][index_matched], c=(0, 255, 0), s=6)

    plt.imshow(img)
    plt.show()


#%% Utils functions for MoTSynth

def visual_check_motsynth_annotations(video_num="0.04", img_file_name="0200.jpg", shift=3):


    json_path = f"/home/raphael/work/datasets/MOTSynth/coco annot/{video_num}.json"
    with open(json_path) as jsonFile:
        annot_motsynth = json.load(jsonFile)


    img_id = [(x["id"]) for x in annot_motsynth["images"] if img_file_name in x["file_name"]][0]
    bboxes = [xywh2xyxy(x["bbox"]) for x in annot_motsynth["annotations"] if x["image_id"] == img_id+shift]

    img_path = f"/home/raphael/work/datasets/MOTSynth/frames/{video_num}/rgb/{img_file_name}"
    img = plt.imread(img_path)
    img = add_bboxes_to_img(img, bboxes, c=(0, 255, 0), s=6)
    plt.imshow(img)
    plt.show()



def get_motsynth_day_night_video_ids(max_iter=50, force=False):


    # Save
    if os.path.exists("/home/raphael/work/datasets/MOTSynth/coco_infos.json") or not force:
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

        if i > max_iter:
            break

    with open("/home/raphael/work/datasets/MOTSynth/coco_infos.json", 'w') as f:
        json.dump(video_info, f)
    night = []
    day = []

    day_index = [key for key, value in video_info.items() if
           not value["is_night"] and os.path.exists(f"/home/raphael/work/datasets/MOTSynth/frames/{key}")]
    night_index = [key for key, value in video_info.items() if
           value["is_night"] and os.path.exists(f"/home/raphael/work/datasets/MOTSynth/frames/{key}")]

    print("night", night_index)
    print("day", day_index)

    return day, night


def get_MoTSynth_annotations_and_imagepaths(video_id="004", max_samples=100000):


    json_path = f"/home/raphael/work/datasets/MOTSynth/coco annot/{video_id}.json"

    with open(json_path) as jsonFile:
        annot_motsynth = json.load(jsonFile)

    targets_metadata = {}
    targets = {}
    j = 0
    i = 0

    # begin at image 3 due to delay
    delay = 3
    #for image in annot_motsynth["images"][delay:]:

    for image in annot_motsynth["images"][delay:delay+max_samples]:

        #image = annot_motsynth["images"][delay:][i]


        if j > max_samples:
            break
        j += 1


        frame_id = image["id"]
        bboxes = [xywh2xyxy(x["bbox"]) for x in annot_motsynth["annotations"] if x["image_id"] == frame_id]

        """
        annots_img = []
        while annot_motsynth["annotations"][i]["image_id"] == image["id"]: #todo bug limit
            bbox_xywh = annot_motsynth["annotations"][i]["bbox"]
            x, y, w, h = bbox_xywh
            bbox_xywh = x, y, x+w, y+h
            annots_img.append(bbox_xywh)
            i += 1"""


        target = [
            dict(
                boxes=torch.tensor(
                    bboxes)
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

