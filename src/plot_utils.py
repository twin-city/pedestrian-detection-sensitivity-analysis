import cv2
import torch
import numpy as np
from src.detection.metrics import compute_fp_missratio
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os.path as osp
from src.detection.metrics import compute_model_metrics_on_dataset
from src.utils import subset_dataframe
from src.detection.detector import Detector
from src.detection.metrics import detection_metric
from src.utils import get_linear_importance, get_permuation_importance
import matplotlib.patches as patches
from src.demos.configs import height_thresh, occl_thresh
from src.detection.detector_factory import DetectorFactory



#%% Utils plot functions ======================================================================================================


def filter_frame_to_str(filter_frame):
    keys = list(filter_frame.keys())
    vals = list(filter_frame.values())

    if keys == ["is_night"]:
        if vals[0] == 0:
            return "Day"
        elif vals[0] == 1:
            return "Night"
        else:
            raise NotImplementedError
    elif keys == ["weather_cats"]:
        return vals[0][0]
    elif keys == ["pitch"]:
        key = list(list(filter_frame.values())[0].keys())[0]  # todo change this
        val = list(list(filter_frame.values())[0].values())[0]
        if key in {"<", ">"}:
            return f"{key} {val}°"
        elif key == "between":
            return f"[{val[0]}°, {val[1]}°]"
        else:
            raise NotImplementedError

def xywh2xyxy(bbox):
    x, y, w, h = bbox
    return x, y, x + w, y + h


def add_bboxes_to_img(img, bboxes, c=(0,0,255), s=1):
    for bbox in bboxes:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        img = cv2.rectangle(img, (x1, y1), (x2, y2), c, s)
    return img


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

def unsqueeze_bboxes_if_needed(x):
    if x.shape == torch.Size([4]):
        return x.unsqueeze(0)
    else:
        return x

#%% Main dataset plot functions ======================================================================================================

def plot_dataset_statistics(df_gtbbox_metadata, results_dir=""):

    fig, ax = plt.subplots(1,2)
    df_gtbbox_metadata.hist("height", bins=200, ax=ax[0])
    ax[0].set_xlim(0, 300)
    ax[0].axvline(height_thresh[0], c="red")
    ax[0].axvline(height_thresh[1], c="red")
    ax[0].axvline(height_thresh[2], c="red")
    ax[0].set_title("Bounding Box Height")

    if "occlusion_rate" in df_gtbbox_metadata.columns:
        df_gtbbox_metadata.hist("occlusion_rate", bins=100, ax=ax[1])
        ax[1].axvline(occl_thresh[0], c="red")
        ax[1].axvline(occl_thresh[1], c="red")
    else:
        ax[1].set_xlim(0,1)
    ax[1].set_title("Occlusion rate")
    plt.savefig(osp.join(results_dir, "dataset_statistics.png"))
    plt.show()

def plot_correlations(corr_matrix, p_matrix, title=""):
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    sns.heatmap(corr_matrix[p_matrix<0.05], annot=True, ax=ax[0], cmap="PiYG", center=0)
    sns.heatmap(p_matrix, annot=True, ax=ax[1], cmap="viridis_r")
    if title:
        ax[1].set_title(title)
    plt.tight_layout()
    plt.show()


#%% Main results plot functions ======================================================================================================

def plot_heatmap_metrics(df_analysis_heatmap, model_names, metrics, ODD_limit, param_heatmap_metrics=None, results_dir=None, show=False):

    if param_heatmap_metrics is None:
        param_heatmap_metrics = {}

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
        if show:
            plt.show()




def plot_gtbbox_matched_correlations(model_names, dataset, features_bbox, threshold, gtbbox_filtering):

    df_gtbbox_metadata = dataset.df_gtbbox_metadata

    #todo handle this exception
    if dataset.dataset_name == "motsynth":
        attributes = ['attributes_0', 'attributes_1', 'attributes_2',
                      'attributes_3', 'attributes_4', 'attributes_5', 'attributes_6',
                      'attributes_7', 'attributes_8', 'attributes_9', 'attributes_10']
    else:
        attributes = 0

    att_list = []

    df_gtbbox_corr_list = []
    for model_name in model_names:

        # Compute metrics and get matched gtbboxes
        _, df_metrics_gtbbox = compute_model_metrics_on_dataset(
            model_name, dataset, gtbbox_filtering["Overall"], device="cuda")
        df_metrics_gtbbox = df_metrics_gtbbox[df_metrics_gtbbox["threshold"]==threshold]
        # Keep only non-ignored boxes
        df_metrics_gtbbox_study = df_metrics_gtbbox[np.isin(df_metrics_gtbbox, [0,1])]
        # Keep only for a given threshold
        df_metrics_gtbbox_study = df_metrics_gtbbox_study[df_metrics_gtbbox_study["threshold"]==threshold]

        # Append the metadata
        #todo subset of features : bug wiht twincity because no annot id
        features_bbox_plot = list(np.intersect1d(features_bbox, df_gtbbox_metadata.columns))
        df_metrics_gtbbox_study.loc[:,features_bbox_plot] = df_gtbbox_metadata.loc[df_metrics_gtbbox_study.index, features_bbox_plot]


        # If there are attributes, dummify them
        if "attributes_0" in df_gtbbox_metadata.columns:
            df_metrics_gtbbox_study.loc[:,attributes] = df_gtbbox_metadata.loc[df_metrics_gtbbox_study.index, attributes]
            att_list = []
            num_attributes = len([x for x in df_gtbbox_metadata.columns if "attribute" in x])-1 #todo harmonize this. Minus 1 because one of the columns is array
            for i in range(num_attributes):
                df_att = pd.get_dummies(df_metrics_gtbbox_study[f"attributes_{i}"], prefix=f"att{i}")
                df_metrics_gtbbox_study[df_att.columns] = df_att
                att_list += list(df_att.columns)

        corr_matrix = df_metrics_gtbbox_study.corr()
        df_gtbbox_corr_model = pd.DataFrame(corr_matrix["matched"][features_bbox_plot+att_list])
        df_gtbbox_corr_model = df_gtbbox_corr_model.rename(columns={"matched": model_name})
        df_gtbbox_corr_list.append(df_gtbbox_corr_model)

    df_gtbbox_corr = pd.concat(df_gtbbox_corr_list, axis=1)

    fig, ax = plt.subplots(figsize=(2*len(model_names), min(6, len(features_bbox_plot+att_list)//10)))
    sns.heatmap(df_gtbbox_corr, center=0, cmap="PiYG", vmax=1, vmin=-1, ax=ax)
    plt.tight_layout()
    plt.show()



def plot_image_with_detections(task, dataset, dataset_name, model_name, plot_thresholds, gtbbox_filtering, frame_idx=0, results_dir=None, show=False):

    frame_idx = 0

    # Load Dataset
    root, targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = dataset

    # Perform detection and compute metrics
    #detector = Detector(model_name, device="cuda")


    detector = DetectorFactory.get_detector(model_name, device="cpu")

    preds = detector.get_preds_from_files(dataset_name, root, df_frame_metadata)
    metric = detection_metric(gtbbox_filtering)
    df_mr_fppi, df_gt_bbox = metric.compute(dataset_name, model_name, preds, targets, df_gtbbox_metadata,
                                            gtbbox_filtering)

    # todo fix getting df_mr_fppi index because they can be removed via gt_bbox filtering
    # Get a file (at random for now, maybe later with criterias ?)
    frame_id = np.sort(df_mr_fppi.index.get_level_values(0).unique())[frame_idx]
    img_path = osp.join(root, df_frame_metadata.loc[frame_id,"file_name"])

    # Plot it
    fig, ax = plt.subplots(len(plot_thresholds),1, figsize=((len(plot_thresholds)*4), 10))
    for i, threshold in enumerate(plot_thresholds):
        mr_val = df_mr_fppi.loc[frame_id, threshold]["MR"]
        fppi_val = df_mr_fppi.loc[frame_id, threshold]["FPPI"]

        plot_results_img(img_path, frame_id, preds=preds, targets=targets,
                     df_gt_bbox=df_gt_bbox, threshold=threshold, ax=ax[i],
                         title=f"thresh={plot_thresholds[i]}, MR={mr_val:.2f} FPPI={fppi_val:.0f}") #todo seems there is a bug, woman in middle should be in red and guy should be red. No sense of all this.
        ax[i].axis("off")
    plt.tight_layout()
    if results_dir is not None:
        plt.savefig(osp.join(results_dir, f"detection_example_{model_name}-{frame_idx}.png"))
    if show:
        plt.show()


def plot_fppi_mr_vs_gtbbox_cofactor(df_analysis_cats, ODD_criterias=None, results_dir=None, show=False):

    fig, ax = plt.subplots(3, 3, figsize=(10,10), sharey=True)
    plot_ffpi_mr_on_ax(df_analysis_cats, "Overall", ax[0,0], odd=ODD_criterias)

    cats = [
        "Typical aspect ratios", "Atypical aspect ratios",
        "near", "medium", "far",
        "No occlusion", "Partial occlusion", "Heavy occlusion",
    ]
    positions = [
        [0, 1], [0, 2],
        [1, 0], [1, 1], [1, 2],
        [2, 0], [2, 1], [2, 2],
    ]

    for cat, pos in zip(cats, positions):
        if cat in df_analysis_cats["gtbbox_filtering_cat"].unique():
            plot_ffpi_mr_on_ax(df_analysis_cats, cat, ax[pos[0], pos[1]])
    plt.tight_layout()
    if results_dir is not None:
        plt.savefig(osp.join(results_dir, "gtbbox_cofactor_fppi_mr.png"))
    if show:
        plt.show()


def plot_fppi_mr_vs_frame_cofactor(df_analysis, dict_filter_frames, ODD_criterias=None, results_dir=None, show=False):

    min_x, max_x = 0.01, 100  # 0.01 false positive per image to 100
    min_y, max_y = 0.05, 1  # 5% to 100% Missing Rate
    n_col = max([len(val) for _, val in dict_filter_frames.items()])
    n_row = len(dict_filter_frames)

    fig, ax = plt.subplots(n_row, n_col, figsize=(8,14))
    for i, (key, filter_frames) in enumerate(dict_filter_frames.items()):

        ax[i, 0].set_ylabel(key, fontsize=20)

        for j, filter_frame in enumerate(filter_frames):

            # todo should be able to get multiple levels of subsetting, check that (in tests ?)
            # Do not show if not present in dataset characteristics
            plot_cofactor = False
            if filter_frame == {}:
                plot_cofactor = True
            else:
                if list(filter_frame.keys())[0] in df_analysis.keys():
                    plot_cofactor = True

            if plot_cofactor:

                for model, df_analysis_model in df_analysis.groupby("model_name"):
                    df_analysis_subset = subset_dataframe(df_analysis_model, filter_frame)
                    metrics_model = df_analysis_subset.groupby("threshold").apply(lambda x: x.mean(numeric_only=True))
                    ax[i, j].plot(metrics_model["FPPI"], metrics_model["MR"], label=model)
                    ax[i, j].scatter(metrics_model["FPPI"], metrics_model["MR"])

                ax[i,j].set_xscale('log')
                ax[i,j].set_yscale('log')
                ax[i,j].set_ylim(min_y, max_y)
                ax[i,j].set_xlim(min_x, max_x)
                ax[i,j].set_title(filter_frame_to_str(filter_frame))





                ax[i, j].legend()

                x = min_x
                y = min_y

                if ODD_criterias is not None:
                    width = ODD_criterias["FPPI"] - min_x
                    height = ODD_criterias["MR"] - min_y
                    # Add the grey square patch to the axes
                    grey_square = patches.Rectangle((x, y), width, height, facecolor='grey', alpha=0.5)
                    ax[i,j].add_patch(grey_square)
                    ax[i,j].text(min_x+width/2/10, min_y+height/2/10, s="ODD")

    plt.tight_layout()
    if results_dir is not None:
        plt.savefig(osp.join(results_dir, "frame_cofactor_fppi_mr.png"))
    if show:
        plt.show()




#%% Legacy plot functions ======================================================================================================


def plot_results_img(img_path, frame_id, preds=None, targets=None, df_gt_bbox=None,
                     threshold=0.9, ax=None, title=None):

    had_ax = ax is not None

    # Read image
    img = plt.imread(img_path)

    # Change type if needed from RGBA to RGB
    if img.dtype == "float32":
        img = (img*255).astype(np.uint8)
    if img.shape[2] == 4:
        img = img[:,:,:3]

    # Create fig, ax
    if not had_ax:
        fig, ax = plt.subplots(1,1, figsize=(16,10))

    # Show RGB image
    ax.imshow(img)

    if preds is not None:
        # Plot
        idx_pred_abovethreshold = torch.nonzero(preds[(frame_id)][0]["scores"] > threshold).squeeze()
        idx_pred_belowthreshold = torch.nonzero(preds[(frame_id)][0]["scores"] < threshold).squeeze()


        preds_above = unsqueeze_bboxes_if_needed(preds[(frame_id)][0]["boxes"][idx_pred_abovethreshold])
        preds_below = unsqueeze_bboxes_if_needed(preds[(frame_id)][0]["boxes"][idx_pred_belowthreshold])

        add_bboxes_to_img_ax(ax, preds_above, c=(0, 0, 1), s=1)
        add_bboxes_to_img_ax(ax, preds_below, c=(0, 0, 1), s=1, linestyle="dotted")

    if targets is not None and df_gt_bbox is not None and preds is not None:
        # Check the matched bboxes
        df_gt_bbox_frame = df_gt_bbox.loc[frame_id]
        df_gt_bbox_frame = df_gt_bbox_frame[df_gt_bbox_frame["threshold"] == threshold].reset_index()
        idx_matched = (df_gt_bbox_frame[df_gt_bbox_frame["matched"] == 1]).index
        idx_ignored = (df_gt_bbox_frame[df_gt_bbox_frame["matched"] == -1]).index
        idx_missed = (df_gt_bbox_frame[df_gt_bbox_frame["matched"] == 0]).index
        add_bboxes_to_img_ax(ax, targets[(frame_id)][0]["boxes"][idx_matched], c=(0, 1, 0), s=2)
        add_bboxes_to_img_ax(ax, targets[(frame_id)][0]["boxes"][idx_missed], c=(1, 0, 0), s=2)
        add_bboxes_to_img_ax(ax, targets[(frame_id)][0]["boxes"][idx_ignored], c=(1, 1, 0), s=2)
    elif targets is not None and df_gt_bbox is not None and preds is None:
        # Check the matched bboxes
        df_gt_bbox_frame = df_gt_bbox.loc[frame_id]
        df_gt_bbox_frame = df_gt_bbox_frame[df_gt_bbox_frame["threshold"] == threshold].reset_index()
        idx_matched = (df_gt_bbox_frame[df_gt_bbox_frame["matched"] == 1]).index
        idx_ignored = (df_gt_bbox_frame[df_gt_bbox_frame["matched"] == -1]).index
        idx_missed = (df_gt_bbox_frame[df_gt_bbox_frame["matched"] == 0]).index
        add_bboxes_to_img_ax(ax, targets[(frame_id)][0]["boxes"][idx_matched], c=(0, 1, 0), s=2)
        add_bboxes_to_img_ax(ax, targets[(frame_id)][0]["boxes"][idx_missed], c=(0, 1, 0), s=2)
        add_bboxes_to_img_ax(ax, targets[(frame_id)][0]["boxes"][idx_ignored], c=(1, 1, 0), s=2)

    elif targets is not None:
        add_bboxes_to_img_ax(ax, targets[(frame_id)][0]["boxes"], c=(0, 1, 0), s=2)

    if targets is not None:
        for i, bbox in enumerate(targets[(frame_id)][0]["boxes"]):
            ax.text(bbox[0], bbox[1], i)

    if title is not None:
        ax.set_title(title)
    if not had_ax:
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




