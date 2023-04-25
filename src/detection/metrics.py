import pandas as pd
import json
import os
import numpy as np
import torchvision
import torch

def get_df_matched_gtbbox(results, frame_id, threshold, gtbbox_ids):
    df_matched_gtbbox = 1 * pd.DataFrame([i in results[threshold][3] for i in range(results[threshold][5])])
    df_matched_gtbbox["frame_id"] = frame_id
    df_matched_gtbbox["threshold"] = threshold
    df_matched_gtbbox["id"] = gtbbox_ids
    df_matched_gtbbox = df_matched_gtbbox.rename(columns={0: 'matched'})
    return df_matched_gtbbox


def compute_fp_missratio2(pred_bbox, target_bbox, threshold=0.5, excluded_gt=[]):

    num_gtbbox = len(target_bbox[0]["boxes"])

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
    miss_ratio_image = fn_image / num_gtbbox

    return fp_image, miss_ratio_image, matched_target_bbox_list, target_bbox_missed, unmatched_preds, num_gtbbox



def filter_gt_bboxes(df_gtbbox_metadata_frame, gtbbox_filtering):
    # replace by subset_dataframe

    if gtbbox_filtering is not {}:
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

class detection_metric:
    def __init__(self, gt_bbox_filter):
        self.gt_bbox_filter = gt_bbox_filter

    def compute(self, dataset_name, model_name, preds, targets,df_gtbbox_metadata, gtbbox_filtering):
        df_mr_fppi, df_matched_gtbbox = self.compute_ffpi_against_fp(dataset_name, model_name, preds, targets,
                                                                 df_gtbbox_metadata, gtbbox_filtering)

        return df_mr_fppi, df_matched_gtbbox



    def compute_ffpi_against_fp(self, dataset_name, model_name, preds, targets, df_gtbbox_metadata, gtbbox_filtering={}, max_frames=1e6):
        """
        On preds keys.
        :param preds:
        :param targets:
        :return:
        """

        # thresholds = list(np.arange(0, 1, 0.1))+[0.99]#+list(np.arange(0.9, 1, 0.3))

        thresholds = [0.1, 0.5, 0.9]


        df_root = f"data/preds/{dataset_name}_{model_name}"
        os.makedirs(df_root, exist_ok=True)
        df_file = f"{df_root}/metrics-{json.dumps(gtbbox_filtering)}.json"
        df_matched_file = f"{df_root}/matched-{json.dumps(gtbbox_filtering)}.json"

        # If exist load it
        if os.path.isfile(df_file):
            df_mr_fppi = pd.read_csv(df_file, index_col="frame_id").reset_index()
            df_mr_fppi["frame_id"] = df_mr_fppi["frame_id"].astype(str)
            df_mr_fppi = df_mr_fppi.set_index(["frame_id", "threshold"])
        else:
            df_mr_fppi = pd.DataFrame(columns=["frame_id", "threshold", "MR", "FPPI"]).set_index(["frame_id", "threshold"])


        if os.path.isfile(df_matched_file):
            df_matched_gtbbox = pd.read_csv(df_matched_file, index_col="frame_id").reset_index()
            df_matched_gtbbox["frame_id"] = df_matched_gtbbox["frame_id"].astype(str)
            df_matched_gtbbox = df_matched_gtbbox.set_index(["frame_id", "id"])
        else:
            df_matched_gtbbox = pd.DataFrame(columns=["frame_id", "id", "threshold", "matched"]).set_index(["frame_id", "id"])


        df_matched_gtbbox_list = []
        df_mr_fppi_list = []
        frame_ids = preds.keys() #todo all for now

        # maybe do a set here ?

        for i, frame_id in enumerate(frame_ids):

            # todo here to compute few for now. At random ?
            if i > max_frames:
                break

            # If image not already parsed
            if str(frame_id) in df_mr_fppi.index and str(frame_id) in df_matched_gtbbox.index:
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




                df_matched_gtbbox = pd.concat([get_df_matched_gtbbox(results, frame_id, threshold, df_gtbbox_metadata_frame["id"]) for threshold in thresholds])
                df_matched_gtbbox = df_matched_gtbbox.set_index(["frame_id", "id"])

                # df = matched_gtbbox = pd.DataFrame({key:val[3] for key,val in results.items()})

                df_results_threshold = pd.DataFrame({key:val[:2] for key,val in results.items()}).T.rename(columns={0: "FPPI", 1: "MR"})
                df_results_threshold.index.name = "threshold"
                df_results_threshold["frame_id"] = str(frame_id)
                df_mr_fppi_list.append(df_results_threshold.reset_index().set_index(["frame_id", "threshold"]))

                df_matched_gtbbox_list.append(df_matched_gtbbox)

            # todo output here details for each image as a dataframe ? score threshold x image_id

        if df_mr_fppi_list:
            #todo duplicated code
            df_mr_fppi_current = pd.concat(df_mr_fppi_list, axis=0)
            df_mr_fppi_current["model"] = model_name
            df_mr_fppi = pd.concat([df_mr_fppi, df_mr_fppi_current], axis=0)

            df_matched_gtbbox_current = pd.concat(df_matched_gtbbox_list, axis=0)
            df_matched_gtbbox_current["model"] = model_name
            df_matched_gtbbox = pd.concat([df_matched_gtbbox, df_matched_gtbbox_current], axis=0)


            if i % 50 == 0:
                df_mr_fppi.to_csv(df_file)
                df_matched_gtbbox.to_csv(df_matched_file)

        # Save at the end
        df_mr_fppi.to_csv(df_file)
        df_matched_gtbbox.to_csv(df_matched_file)

        df_matched_gtbbox = df_matched_gtbbox.reset_index()
        df_matched_gtbbox["id"] = df_matched_gtbbox["id"].astype(str)
        df_matched_gtbbox = df_matched_gtbbox.set_index(["frame_id", "id"])

        return df_mr_fppi.loc[frame_ids], df_matched_gtbbox.loc[frame_ids]




