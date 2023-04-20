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
import pandas as pd


def xywh2xyxy(bbox):
    x, y, w, h = bbox
    return x, y, x + w, y + h


def compute_fp_missratio2(pred_bbox, target_bbox, threshold=0.5, excluded_gt=[]):

    score_sorted = np.argsort(pred_bbox[0]["scores"].numpy())[::-1]

    # All target bboxes
    possible_target_bboxs = [target_bbox for target_bbox in target_bbox[0]["boxes"]]
    possible_target_bboxs_ids = list(range(len(target_bbox[0]["boxes"])))

    # Included
    incl_possible_target_bboxs_ids = np.setdiff1d(possible_target_bboxs_ids, excluded_gt)
    #incl_possible_target_bboxs = [possible_target_bboxs[i] for i in incl_possible_target_bboxs_ids]

    # Excluded
    #excl_possible_target_bboxs = [possible_target_bboxs[i] for i in excluded_gt]
    #excl_possible_target_bboxs_ids = excluded_gt

    matched_target_bbox_list = []
    unmatched_preds = []

    # For each pred bbox in decreasing probability score order
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

        # All matches are to excluded bboxes --> nothing happens
        elif np.all([x in excluded_gt for x in IoUs_index]):
            pass
        # Else there exist at least an overlap with an included bounding box
        else:
            # Match it best with target boxes, included and still unmatched
            matched_target_bbox = np.intersect1d(torch.stack(IoUs).reshape(-1).numpy().argsort(), incl_possible_target_bboxs_ids)[-1]
            matched_target_bbox_list.append(possible_target_bboxs_ids[matched_target_bbox])

            # Remove
            possible_target_bboxs.pop(matched_target_bbox)
            possible_target_bboxs_ids.pop(matched_target_bbox)

    # Compute the False Positives
    target_bbox_missed = np.setdiff1d(list(range(len(target_bbox[0]["boxes"]))), matched_target_bbox_list).tolist()

    # Number of predictions above threshold - Number of matched target_bboxs
    fp_image = len(unmatched_preds)

    # False negatives
    # fn_image = max(0, len(target_bbox[0]["boxes"]) - len(matched_target_bbox_list))
    fn_image = max(0, len(incl_possible_target_bboxs_ids) - len(matched_target_bbox_list))
    miss_ratio_image = fn_image / len(target_bbox[0]["boxes"])

    return fp_image, miss_ratio_image, matched_target_bbox_list, target_bbox_missed, unmatched_preds


def filter_gt_bboxes(df_gtbbox_metadata_frame, gtbbox_filtering):
    if gtbbox_filtering is not {}:
        # todo use a set
        excluded = set()
        for key, val in gtbbox_filtering.items():
            if val[1] == "min":
                excluded |= set(df_gtbbox_metadata_frame[df_gtbbox_metadata_frame[key] < val[0]].index)
            elif val[1] == "max":
                excluded |= set(df_gtbbox_metadata_frame[df_gtbbox_metadata_frame[key] > val[0]].index)
            else:
                raise ValueError("Nor minimal nor maximal filtering proposed.")
        excluded_gt = list(excluded)
    else:
        excluded_gt = []

    return excluded_gt


def compute_ffpi_against_fp2(dataset_name, model_name, preds, targets, df_gtbbox_metadata, gtbbox_filtering={}):
    """
    On preds keys.
    :param preds:
    :param targets:
    :return:
    """

    thresholds = list(np.arange(0, 1, 0.1))+[0.99]#+list(np.arange(0.9, 1, 0.3))


    df_root = f"data/preds/{dataset_name}_{model_name}"
    os.makedirs(df_root, exist_ok=True)
    df_file = f"{df_root}/metrics-{json.dumps(gtbbox_filtering)}.json"

    # If exist load it
    if os.path.isfile(df_file):
        df_mr_fppi = pd.read_csv(df_file, index_col="frame_id").reset_index()
        df_mr_fppi["frame_id"] = df_mr_fppi["frame_id"].astype(str)
        df_mr_fppi = df_mr_fppi.set_index(["frame_id", "threshold"])
    else:
        df_mr_fppi = pd.DataFrame(columns=["frame_id", "threshold", "MR", "FPPI"]).set_index(["frame_id", "threshold"])



    df_mr_fppi_list = []
    frame_ids = preds.keys() #todo all for now

    # maybe do a set here ?

    for i, frame_id in enumerate(frame_ids):

        # If image not already parsed
        if str(frame_id)  in df_mr_fppi.index:
            print(f"{frame_id}  {gtbbox_filtering} Already done")
        else:
            print(f"{frame_id}  {gtbbox_filtering} Not already done")

            results = {}
            for threshold in thresholds:

                    #todo handle only 1 ???
                    if len(pd.DataFrame(df_gtbbox_metadata.loc[frame_id]).T) == 1:
                        df_gtbbox_metadata_frame = pd.DataFrame(df_gtbbox_metadata.loc[frame_id]).T.reset_index()
                    else:
                        df_gtbbox_metadata_frame = df_gtbbox_metadata.loc[frame_id].reset_index()
                    #todo delay here was removed for ECP df_gtbbox_metadata_frame = df_gtbbox_metadata.loc[int(frame_id)+3].reset_index()

                    excluded_gt = filter_gt_bboxes(df_gtbbox_metadata_frame, gtbbox_filtering)

                    results[threshold] = compute_fp_missratio2(preds[frame_id], targets[frame_id],
                                                              threshold=threshold, excluded_gt=excluded_gt)


            df_results_threshold = pd.DataFrame({key:val[:2] for key,val in results.items()}).T.rename(columns={0: "FPPI", 1: "MR"})
            df_results_threshold.index.name = "threshold"
            df_results_threshold["frame_id"] = str(frame_id)
            df_mr_fppi_list.append(df_results_threshold.reset_index().set_index(["frame_id", "threshold"]))

        # todo output here details for each image as a dataframe ? score threshold x image_id

    if df_mr_fppi_list:
        df_mr_fppi_current = pd.concat(df_mr_fppi_list, axis=0)
        df_mr_fppi_current["model"] = model_name
        df_mr_fppi = pd.concat([df_mr_fppi, df_mr_fppi_current], axis=0)

        if i % 50 == 0:
            df_mr_fppi.to_csv(df_file)

    # Save at the end
    df_mr_fppi.to_csv(df_file)

    return df_mr_fppi.loc[frame_ids]



#%% Plot utils

def add_bboxes_to_img(img, bboxes, c=(0,0,255), s=1):
    for bbox in bboxes:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        img = cv2.rectangle(img, (x1, y1), (x2, y2), c, s)
    return img

def plot_results_img(img_path, frame_id, preds=None, targets=None, excl_gt_indices=None):
    img = plt.imread(img_path)

    num_gt_bbox = len(targets[(frame_id)][0]["boxes"])

    incl_gt_indices = np.setdiff1d(list(range(num_gt_bbox)), excl_gt_indices)

    if preds is not None:
        img = add_bboxes_to_img(img, preds[(frame_id)][0]["boxes"], c=(0, 0, 255), s=3)
    if targets is not None:
        if excl_gt_indices is None:
            img = add_bboxes_to_img(img, targets[(frame_id)][0]["boxes"], c=(0, 255, 0), s=6)
        else:
            img = add_bboxes_to_img(img, targets[(frame_id)][0]["boxes"][incl_gt_indices], c=(0, 255, 0), s=6)
            img = add_bboxes_to_img(img, targets[(frame_id)][0]["boxes"][excl_gt_indices], c=(255, 255, 0), s=6)
    plt.imshow(img)
    plt.show()


def plot_fp_fn_img(frame_id_list, img_path_list, preds, targets, index_frame, threshold=0.5):
    preds = preds
    targets = targets
    frame_id = frame_id_list[index_frame]
    img_path = img_path_list[index_frame]

    results = {}
    results[frame_id] = compute_fp_missratio2(preds[frame_id], targets[frame_id], threshold=threshold)

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


