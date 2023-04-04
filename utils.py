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


"""
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
"""

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


def compute_ffpi_against_fp(preds, targets, targets_metadata=None, ids=None):
    """
    On preds keys.
    :param preds:
    :param targets:
    :return:
    """

    if ids is None:
        ids = preds.keys()

    thresholds = list(np.arange(0, 1, 0.1))+[0.99]#+list(np.arange(0.9, 1, 0.3))

    avrg_fp_list = []
    avrg_missrate_list = []

    for threshold in thresholds:
        results = {}
        for frame_id in ids:

            if targets_metadata is not None:
                target_metadata = targets_metadata[frame_id]
                occlusions = [(x - 1).mean() for x in target_metadata["keypoints"]]
                occlusions_ids = list(np.where(np.array(occlusions) < 0.1)[0])
            else:
                occlusions_ids = []

            results[frame_id] = compute_fp_missratio2(preds[frame_id], targets[frame_id], threshold=threshold, excluded_gt=occlusions_ids)
        avrg_fp = np.mean([x[0] for x in results.values()])
        avrg_missrate = np.mean([x[1] for x in results.values()])

        avrg_fp_list.append(avrg_fp)
        avrg_missrate_list.append(avrg_missrate)

    return avrg_fp_list, avrg_missrate_list




def compute_ffpi_against_fp2(preds, targets, df_gtbbox_metadata, gtbbox_filtering={}, model_name="unknown"):
    """
    On preds keys.
    :param preds:
    :param targets:
    :return:
    """

    df_mr_fppi_list = []


    frame_ids = preds.keys() #todo all for now

    thresholds = list(np.arange(0, 1, 0.1))+[0.99]#+list(np.arange(0.9, 1, 0.3))

    avrg_fp_list = []
    avrg_missrate_list = []

    for threshold in thresholds:
        results = {}
        for frame_id in frame_ids:

            df_gtbbox_metadata_frame = df_gtbbox_metadata.loc[frame_id+3].reset_index()

            if gtbbox_filtering is not {}:

                # todo use a set
                excluded = set()
                for key,val in gtbbox_filtering.items():
                    if val[1] == "min":
                        excluded |= set(df_gtbbox_metadata_frame[df_gtbbox_metadata_frame[key] < val[0]].index)
                    elif val[1] == "max":
                        excluded |= set(df_gtbbox_metadata_frame[df_gtbbox_metadata_frame[key] > val[0]].index)
                    else:
                        raise ValueError("Nor minimal nor maximal filtering proposed.")
                excluded_gt = list(excluded)
            else:
                excluded_gt = []

            results[frame_id] = compute_fp_missratio2(preds[frame_id], targets[frame_id],
                                                      threshold=threshold, excluded_gt=excluded_gt)


        df_results_threshold = pd.DataFrame({key:val[:2] for key,val in results.items()}).T.rename(columns={0: "MR", 1: "FPPI"})
        df_results_threshold["threshold"] = threshold
        df_results_threshold["frame_id"] = frame_id
        df_mr_fppi_list.append(df_results_threshold)

        # todo output here details for each image as a dataframe ? score threshold x image_id

        """
        avrg_fp = np.mean([x[0] for x in results.values()])
        avrg_missrate = np.mean([x[1] for x in results.values()])
        avrg_fp_list.append(avrg_fp)
        avrg_missrate_list.append(avrg_missrate)
        """

        df_mr_fppi = pd.concat(df_mr_fppi_list, axis=0)
        df_mr_fppi["model"] = model_name

    return df_mr_fppi



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

def plot_results_img(img_path, frame_id, preds=None, targets=None, excl_gt_indices=None):
    img = plt.imread(img_path)

    num_gt_bbox = len(targets[frame_id][0]["boxes"])

    incl_gt_indices = np.setdiff1d(list(range(num_gt_bbox)), excl_gt_indices)

    if preds is not None:
        img = add_bboxes_to_img(img, preds[frame_id][0]["boxes"], c=(0, 0, 255), s=3)
    if targets is not None:
        if excl_gt_indices is None:
            img = add_bboxes_to_img(img, targets[frame_id][0]["boxes"], c=(0, 255, 0), s=6)
        else:
            img = add_bboxes_to_img(img, targets[frame_id][0]["boxes"][incl_gt_indices], c=(0, 255, 0), s=6)
            img = add_bboxes_to_img(img, targets[frame_id][0]["boxes"][excl_gt_indices], c=(255, 255, 0), s=6)
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

def visual_check_motsynth_annotations(video_num="004", img_file_name="0200.jpg", shift=3):


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



def get_MoTSynth_annotations_and_imagepaths_video(video_id="004", max_samples=100000, random_sampling=True, delay=3):

    np.random.seed(0)

    df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = [pd.DataFrame()]*3

    json_path = f"/home/raphael/work/datasets/MOTSynth/coco annot/{video_id}.json"

    with open(json_path) as jsonFile:
        annot_motsynth = json.load(jsonFile)

    frame_metadata = {}
    targets = {}
    img_path_list = []
    targets_metadata = {}

    # Set images to process (subset for ptotyping)
    if random_sampling:
        random_set = np.random.choice(len(annot_motsynth["images"][delay:]), max_samples, replace=False)
        image_set = [x for i, x in enumerate(annot_motsynth["images"][delay:]) if i in random_set]
    else:
        image_set = annot_motsynth["images"][delay:delay+max_samples]

    for image in image_set:

        # todo more info in image
        for i, name in enumerate(["raw", "pitch", "roll"]):
            image[name] = image["cam_world_rot"][i]
        for i, name in enumerate(["x", "y", "z"]):
            image[name] = image["cam_world_pos"][i]
        for info_name in ["is_night", "seq_name", "weather", "is_moving", "cam_fov", 'fx', 'fy', 'cx', 'cy']:
            image[info_name] = annot_motsynth["info"][info_name]

        frame_id = image["id"]
        bboxes = [xywh2xyxy(x["bbox"]) for x in annot_motsynth["annotations"] if x["image_id"] == frame_id+delay]

        # BBOXES metadata
        annots = [x for x in annot_motsynth["annotations"] if x["image_id"] == frame_id+delay]
        keypoints = [(np.array(annot["keypoints"])).reshape((22, 3))[:,2] for annot in annots]
        area = [annot["area"] for annot in annots]
        is_crowd = [annot["iscrowd"] for annot in annots]
        is_blurred = [annot["is_blurred"] for annot in annots]
        attributes = [annot["attributes"] for annot in annots]
        ped_id = [annot["ped_id"] for annot in annots]
        id = [annot["id"] for annot in annots]
        image_id = [annot["image_id"] for annot in annots]

        target_metadata = {
            "image_id": image_id,
            "id": id,
            "keypoints": keypoints,
            "area": area,
            "is_crowd": is_crowd,
            "is_blurred": is_blurred,
            "attributes": attributes,
            "ped_id": ped_id,
        }

        targets_metadata[frame_id] = target_metadata

        # Target and labels
        target = [
            dict(
                boxes=torch.tensor(
                    bboxes)
                )]
        target[0]["labels"] = torch.tensor([0] * len(target[0]["boxes"]))

        # Keep only if at least 1 pedestrian
        if len(target[0]["boxes"]) > 0:
            targets[frame_id] = target
            frame_metadata[frame_id] = annot_motsynth["info"]
            img_path_list.append(osp.join("/home/raphael/work/datasets/MOTSynth", image["file_name"]))
            # frame_metadata[frame_id] = (annot_ECP["tags"], [ann["tags"] for ann in annot_ECP["children"]])

            # Dataframes
            df_gtbbox_metadata = pd.concat([df_gtbbox_metadata, pd.DataFrame(target_metadata)], axis=0)



            frame_metadata_features = ['file_name', 'id', 'frame_n'] + ["is_night", "seq_name", "weather", "is_moving", "cam_fov", 'fx', 'fy', 'cx', 'cy']
            df_frame_metadata = pd.concat([df_frame_metadata, pd.DataFrame({key:val for key,val in image.items() if key in frame_metadata_features}, index=[frame_id])], axis=0)
    frame_id_list = list(targets.keys())

    # Metadata at the video level
    df_sequence_metadata = pd.DataFrame(annot_motsynth["info"], index=[video_id])

    metadatas = df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata

    return targets, metadatas, frame_id_list, img_path_list




def get_MoTSynth_annotations_and_imagepaths(video_ids=None, max_samples=100000):

    df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = [pd.DataFrame()]*3

    if video_ids is None:
        folders = os.listdir("/home/raphael/work/datasets/MOTSynth/frames")
    else:
        folders = video_ids

    num_folders = len(list(folders))
    max_num_sample_per_video = int(max_samples/num_folders)

    targets, targets_metadata, frames_metadata, frame_id_list, img_path_list = {}, {}, {}, [], []

    for folder in folders:
        targets_folder, targets_metadatas, frame_id_list_folder, img_path_list_folder =\
            get_MoTSynth_annotations_and_imagepaths_video(video_id=folder, max_samples=max_num_sample_per_video)

        df_gtbbox_metadata_folder, df_frame_metadata_folder, df_sequence_metadata_folder = targets_metadatas

        targets.update(targets_folder)

        #targets_metadata.update(targets_metadata_folder)
        #frames_metadata.update(frames_metadata_folder)

        df_gtbbox_metadata = pd.concat([df_gtbbox_metadata, pd.DataFrame(df_gtbbox_metadata_folder)], axis=0)
        df_frame_metadata = pd.concat([df_frame_metadata, pd.DataFrame(df_frame_metadata_folder)], axis=0)
        df_sequence_metadata = pd.concat([df_sequence_metadata, pd.DataFrame(df_sequence_metadata_folder)], axis=0)

        frame_id_list += frame_id_list_folder
        img_path_list += img_path_list_folder

    metadatas = df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata

    return targets, metadatas, frame_id_list, img_path_list
