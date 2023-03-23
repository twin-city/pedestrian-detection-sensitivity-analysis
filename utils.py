import numpy as np
import torchvision


def compute_fp_missratio(pred_bbox, target_bbox, threshold=0.5):
    score_sorted = np.argsort(pred_bbox[0]["scores"].numpy())[::-1]

    possible_target_bboxs = [target_bbox for target_bbox in target_bbox[0]["boxes"]]
    possible_target_bboxs_ids = list(range(len(target_bbox[0]["boxes"])))
    matched_target_bbox_list = []

    for i in score_sorted:

        if len(possible_target_bboxs) == 0 or pred_bbox[0]["scores"][i] < threshold:
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

    # Compute the False Positives
    target_bbox_missed = np.setdiff1d(list(range(len(target_bbox[0]["boxes"]))), matched_target_bbox_list).tolist()

    # Number of predictions above threshold - Number of matched target_bboxs
    fp_image = max(0, (pred_bbox[0]["scores"] > threshold).numpy().sum() - len(matched_target_bbox_list))

    # False negatives
    fn_image = max(0, len(target_bbox[0]["boxes"]) - len(matched_target_bbox_list))
    miss_ratio_image = fn_image / len(target_bbox[0]["boxes"])

    return fp_image, miss_ratio_image, matched_target_bbox_list, target_bbox_missed


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
            results[frame_id] = compute_fp_missratio(preds[frame_id], targets[frame_id], threshold=threshold)
        avrg_fp = np.mean([x[0] for x in results.values()])
        avrg_missrate = np.mean([x[1] for x in results.values()])

        avrg_fp_list.append(avrg_fp)
        avrg_missrate_list.append(avrg_missrate)

    return avrg_fp_list, avrg_missrate_list