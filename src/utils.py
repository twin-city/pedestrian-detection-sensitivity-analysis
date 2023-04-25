import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.detection.metrics import compute_fp_missratio2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def subset_dataframe(df, conditions):
    """


    Example :

    filter_frame = {
        "is_night": {
            "value": 1
        },
        "pitch": {
            "max": 10,
            "min": -10,
        },
        "adverse_weather": set([1])
    }

    :param df:
    :param conditions:
    :return:
    """
    # Create an empty mask
    mask = pd.Series([True] * len(df), index=df.index)

    # Iterate over each condition in the dictionary and update the mask accordingly
    for column, values in conditions.items():
        if isinstance(values, dict):
            if '>' in values:
                mask &= df[column] >= values['>']
            if '<' in values:
                mask &= df[column] <= values['<']
            if 'value' in values:
                mask &= df[column] == values['value']
            if 'set_values' in values:
                mask &= df[column].isin(values['set_values'])
        elif isinstance(values, (list, set, np.ndarray)):
            mask &= df[column].isin(values)
        elif isinstance(values, (int, float)):
            mask &= df[column] == values

    # Apply the mask to the DataFrame to get the subset
    subset_df = df[mask]

    if len(conditions) > 0 & len(subset_df) == len(df):
        print("Warning : filtering did not change the dataframe size")

    return subset_df






def compute_correlations(df, features):
    corr_matrix = df[features].corr(
        method=lambda x, y: pearsonr(x, y)[0])
    p_matrix = df[features].corr(
        method=lambda x, y: pearsonr(x, y)[1])
    return corr_matrix, p_matrix

def plot_correlations(corr_matrix, p_matrix, title=""):
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    sns.heatmap(corr_matrix[p_matrix<0.05], annot=True, ax=ax[0])
    sns.heatmap(p_matrix, annot=True, ax=ax[1])
    if title:
        ax[1].set_title(title)
    plt.tight_layout()
    plt.show()




#%% Plot utils

def xywh2xyxy(bbox):
    x, y, w, h = bbox
    return x, y, x + w, y + h


def add_bboxes_to_img(img, bboxes, c=(0,0,255), s=1):
    for bbox in bboxes:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        img = cv2.rectangle(img, (x1, y1), (x2, y2), c, s)
    return img

def plot_results_img(img_path, frame_id, preds=None, targets=None, excl_gt_indices=None, ax=None):
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

    if ax is None:
        plt.imshow(img)
        plt.show()
    else:
        ax.imshow(img)


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

