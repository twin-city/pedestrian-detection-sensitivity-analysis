import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.detection.metrics import compute_fp_missratio2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
import matplotlib.patches as patches
import os.path as osp

def plot_importance(model_names, metrics, df_analysis, features, importance_method="linear"):
    # from https://inria.github.io/scikit-learn-mooc/python_scripts/dev_features_importance.html#linear-model-inspection

    bar_width = 1/(len(metrics)+1)

    fig, ax = plt.subplots(len(metrics), 1, figsize=(6,10))

    # Do the bar plots
    for i, metric in enumerate(metrics):
        for j, model_name in enumerate(model_names):

            # Compute the according result dataframe
            df = df_analysis[df_analysis["model_name"] == model_name].groupby("frame_id").apply(
                lambda x: x.mean(numeric_only=True)).sample(frac=1)

            # Compute the coefs for a given metric
            if importance_method == "linear":
                coefs = get_linear_importance(df, metric, features)
            elif importance_method == "permutation":
                coefs = get_permuation_importance(df, metric, features)
            else:
                raise NotImplementedError(f"Importance method {importance_method} not known")

            ax[i].bar(np.arange(len(features)) + j * bar_width,
                      coefs, width=bar_width, label=model_name)

        ax[i].set_title(f"{importance_method} importance for {metric} (avrg per frame)")
        ax[i].legend()
        ax[i].set_xticks(range(len(features)))
        ax[i].set_xticklabels(features, rotation=45)

    plt.tight_layout()
    plt.show()


#todo get pval also same same

def get_linear_importance(df, metric, features):

    X = df[features]
    X = (X-X.mean())/X.std()
    y = df[metric]
    X_train, X_test = X[:len(X)//2], X[len(X)//2:]
    y_train, y_test = y[:len(y)//2], y[len(y)//2:]



    model = RidgeCV()
    model.fit(X_train, y_train)
    print(f'{metric} model score on training data: {model.score(X_train, y_train)}')
    print(f'{metric} model score on testing data: {model.score(X_test, y_test)}')

    coefs = pd.DataFrame(
       model.coef_,
       columns=['Coefficients'], index=X_train.columns
    )

    return coefs.values[:,0]



def get_permuation_importance(df, metric, features):

    X = df[features]
    X = (X-X.mean())/X.std()
    y = df[metric]
    X_train, X_test = X[:len(X)//2], X[len(X)//2:]
    y_train, y_test = y[:len(y)//2], y[len(y)//2:]

    forest = RandomForestRegressor(random_state=0)
    forest.fit(X_train, y_train)
    print(f'{metric} model score on training data: {forest.score(X_train, y_train)}')
    print(f'{metric} model score on testing data: {forest.score(X_test, y_test)}')

    # Compute importance
    result = permutation_importance(forest, X, y, n_repeats=10, random_state=42, n_jobs=2)
    forest_importances = pd.Series(result.importances_mean, index=features)

    return forest_importances.values



def subset_dataframe(df, conditions):
    """

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
    sns.heatmap(corr_matrix[p_matrix<0.05], annot=True, ax=ax[0], cmap="PiYG", center=0)
    sns.heatmap(p_matrix, annot=True, ax=ax[1], cmap="viridis_r")
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

def target_2_torch(targets):
    return {key: [{
        "boxes": torch.tensor(val[0]["boxes"]),
        "labels": torch.tensor(val[0]["labels"]),
    }
    ] for key, val in targets.items()}



def target_2_json(targets):
    return {key: [{
        "boxes": val[0]["boxes"].numpy().tolist(),
        "labels": val[0]["labels"].numpy().tolist(),
    }
    ] for key, val in targets.items()}

import matplotlib.patches as patches


def add_bboxes_to_img_ax(ax, bboxes, c=(0, 0, 1), s=1, linestyle="solid"):
    for bbox in bboxes:
        x1, y1, x2, y2 = [int(v) for v in bbox]

        rect = patches.Rectangle((x1, y1), (x2 - x1), (y2 - y1),
                                 linewidth=s,
                                 edgecolor=c,
                                 facecolor='none',
                                 linestyle=linestyle,
                                 alpha=0.5)
        ax.add_patch(rect)

def plot_results_img(img_path, frame_id, preds=None, targets=None, df_gt_bbox=None, threshold = 0.9):



    # Read image
    img = plt.imread(img_path)

    # Change type if needed from RGBA to RGB
    if img.dtype == "float32":
        img = (img*255).astype(np.uint8)
    if img.shape[2] == 4:
        img = img[:,:,:3]

    # Create fig, ax
    fig, ax = plt.subplots(1,1, figsize=(16,10))

    # Show RGB image
    ax.imshow(img)

    if preds is not None:
        # Plot
        idx_pred_abovethreshold = torch.nonzero(preds[(frame_id)][0]["scores"] > threshold).squeeze()
        idx_pred_belowthreshold = torch.nonzero(preds[(frame_id)][0]["scores"] < threshold).squeeze()
        add_bboxes_to_img_ax(ax, preds[(frame_id)][0]["boxes"][idx_pred_abovethreshold], c=(0, 0, 1), s=1)
        add_bboxes_to_img_ax(ax, preds[(frame_id)][0]["boxes"][idx_pred_belowthreshold], c=(0, 0, 1), s=1, linestyle="dotted")

    if targets is not None and df_gt_bbox is not None:
        # Check the matched bboxes
        df_gt_bbox_frame = df_gt_bbox.loc[frame_id]
        df_gt_bbox_frame = df_gt_bbox_frame[df_gt_bbox_frame["threshold"] == threshold].reset_index()
        idx_matched = (df_gt_bbox_frame[df_gt_bbox_frame["matched"] == 1]).index
        idx_ignored = (df_gt_bbox_frame[df_gt_bbox_frame["matched"] == -1]).index
        idx_missed = (df_gt_bbox_frame[df_gt_bbox_frame["matched"] == 0]).index
        add_bboxes_to_img_ax(ax, targets[(frame_id)][0]["boxes"][idx_matched], c=(0, 1, 0), s=2)
        add_bboxes_to_img_ax(ax, targets[(frame_id)][0]["boxes"][idx_missed], c=(1, 0, 0), s=2)
        add_bboxes_to_img_ax(ax, targets[(frame_id)][0]["boxes"][idx_ignored], c=(1, 1, 0), s=2)

    if targets is not None:
        for i, bbox in enumerate(targets[(frame_id)][0]["boxes"]):
            ax.text(bbox[0], bbox[1], i)

    plt.axis('off')
    plt.tight_layout()
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

def plot_ffpi_mr_on_ax(df_metrics_criteria, cat, ax, odd=None):

    min_x, max_x = 0.01, 100 # 0.01 false positive per image to 100
    min_y, max_y = 0.05, 1 # 5% to 100% Missing Rate

    df_metrics = df_metrics_criteria[df_metrics_criteria["gtbbox_filtering_cat"] == cat]

    for model, df_analysis_model in df_metrics.groupby("model_name"):
        metrics_model = df_analysis_model.groupby("threshold").apply(lambda x: x.mean(numeric_only=True))
        ax.plot(metrics_model["FPPI"], metrics_model["MR"], label=model)
        ax.scatter(metrics_model["FPPI"], metrics_model["MR"])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(min_y, max_y)
    ax.set_xlim(min_x, max_x)
    ax.set_title(cat)
    ax.legend()

    if odd is not None:
        x = min_x
        y = min_y
        width = odd["FPPI"] - min_x
        height = odd["MR"] - min_y
        #Add the grey square patch to the axes
        grey_square = patches.Rectangle((x, y), width, height, facecolor='grey', alpha=0.5)
        ax.add_patch(grey_square)
        ax.text(min_x+width/2/10, min_y+height/2/10, s="ODD")



def plot_heatmap_metrics(df_analysis_heatmap, model_names, metrics, ODD_limit, param_heatmap_metrics={}, results_dir=None):
    for metric in metrics:
        mean_metric_values = df_analysis_heatmap.groupby("model_name").apply(lambda x: x[metric].mean())
        df_odd_model_list = []
        for model_name in model_names:
            perc_increase_list = []
            for limit, limit_name in ODD_limit:
                if list(limit.keys())[0] in df_analysis_heatmap.columns:
                    condition = {}
                    condition.update({"model_name": model_name})
                    condition.update(limit)
                    df_subset = subset_dataframe(df_analysis_heatmap, condition)
                    df_subset = df_subset[df_subset["model_name"] == model_name]

                    perc_increase_list.append(df_subset[metric].mean()-mean_metric_values.loc[model_name])
                else:
                    # Add nothing if param is not present, in order to highlight in the plot that it is indeed missing
                    perc_increase_list.append(np.nan)
            df_odd_model_list.append(pd.DataFrame(perc_increase_list, index=[x[1] for x in ODD_limit], columns=[model_name]))
        df_odd_model = pd.concat(df_odd_model_list, axis=1)


        """ If boundaries cmap
        # Define the boundaries of each zone
        bounds = [0, 0.1, 0.2, 0.5]
        # Define a unique color for each zone
        colors = ['green', 'yellow', 'red']
        # Create a colormap with discrete colors
        cmap = sns.color_palette(colors, n_colors=len(bounds)-1).as_hex()
        # Create a BoundaryNorm object to define the colormap
        norm = BoundaryNorm(bounds, len(cmap))
        """

        fig, ax = plt.subplots(1,1)
        sns.heatmap(df_odd_model, annot=True,
                    ax=ax, fmt=".2f", cbar_kws={'format': '%.2f'},
                    **param_heatmap_metrics[metric])
        ax.collections[0].colorbar.set_label('Difference in performance')
        plt.title(f"Impact of parameters on {metric}")
        plt.tight_layout()
        if results_dir is not None:
            plt.savefig(osp.join(results_dir, f"Performance_difference_{metric}_{model_names}.png"))
        plt.show()