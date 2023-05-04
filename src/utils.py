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