import os
import pandas as pd
import setuptools.errors
import numpy as np
import os.path as osp
import os
from src.detection.metrics import compute_model_metrics_on_dataset
import matplotlib.pyplot as plt

#%% params of input dataset

# Parameters for results generation
from src.demos.configs import ODD_limit, ODD_criterias, param_heatmap_metrics, metrics, occl_thresh, height_thresh, gtbbox_filtering_all

# Dataset
max_sample = 50
dataset_name = "twincity"
root = "/home/raphael/work/datasets/twincity-Unreal/v5"
resolution = (1920, 1080)
results_dir = osp.join("../../","results", dataset_name, f"{dataset_name}{max_sample}")
os.makedirs(results_dir, exist_ok=True)

# Which models to study
model_names = ["faster-rcnn_cityscapes", "mask-rcnn_coco"]


#%% Get the dataset
from src.preprocessing.twincity_preprocessing2 import get_twincity_dataset
dataset = get_twincity_dataset(root, max_sample)
root, targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = dataset
#todo in abstract dataset class










#%% See Dataset Characteristics ==============================




#%% Correlations
seq_cofactors = ["is_night"] #todo virrer d'ici
from src.utils import compute_correlations, plot_correlations
corr_matrix, p_matrix = compute_correlations(df_frame_metadata.groupby("seq_name").apply(lambda x: x.mean()), seq_cofactors)
plot_correlations(corr_matrix, p_matrix, title="Correlations between metadatas at sequence level")


#%% Height & Occlusion
#todo set dataset plot in dataset object
from src.utils import plot_dataset_statistics
plot_dataset_statistics(df_gtbbox_metadata, results_dir)

#%% What cases do we study ?

from src.utils import compute_models_metrics_from_gtbbox_criteria
gtbbox_filtering_cats = {}
gtbbox_filtering_cats.update(gtbbox_filtering_all)
df_analysis = compute_models_metrics_from_gtbbox_criteria(dataset_name, dataset, df_frame_metadata, gtbbox_filtering_cats, model_names)





""" Legacy

model_name = model_names[0]
gt_bbox_filtering = gtbbox_filtering_cats["Overall"]
threshold = 0.5

df_metrics_frame, df_metrics_gtbbox = compute_model_metrics_on_dataset(model_name, dataset_name, dataset, gt_bbox_filtering, device="cuda")
df_metrics_gtbbox = df_metrics_gtbbox[df_metrics_gtbbox["threshold"]==threshold]
"""


""" Legacy

from src.utils import plot_ffpi_mr_on_ax
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(10,10), sharey=True)
plot_ffpi_mr_on_ax(df_metrics_criteria, "Overall", ax, odd=ODD_criterias)
plt.tight_layout()
plt.show()
"""



#%% After bench, do the plot value difference (simplified, each metric)
#%% Model performance :  Plots MR vs FPPI on frame filtering



from src.utils import subset_dataframe
import matplotlib.pyplot as plt

dict_filter_frames = {
    "Overall": [{}],
    "Day / Night": ({"is_night": 0}, {"is_night": 1}),
    #"Adverse Weather": ({"weather": ["Partially cloudy"]}, {"weather": ["Foggy"]}),
    "Camera Angle": ({"pitch": {"<": -10}}, {"pitch": 0}),
}

min_x, max_x = 0.01, 100  # 0.01 false positive per image to 100
min_y, max_y = 0.05, 1  # 5% to 100% Missing Rate

n_col = max([len(val) for _, val in dict_filter_frames.items()])
n_row = len(dict_filter_frames)

fig, ax = plt.subplots(n_row, n_col, figsize=(8,14))
for i, (key, filter_frames) in enumerate(dict_filter_frames.items()):

    ax[i, 0].set_ylabel(key, fontsize=20)

    for j, filter_frame in enumerate(filter_frames):

        for model, df_analysis_model in df_analysis.groupby("model_name"):
            df_analysis_subset = subset_dataframe(df_analysis_model, filter_frame)
            metrics_model = df_analysis_subset.groupby("threshold").apply(lambda x: x.mean(numeric_only=True))
            ax[i, j].plot(metrics_model["FPPI"], metrics_model["MR"], label=model)
            ax[i, j].scatter(metrics_model["FPPI"], metrics_model["MR"])

        ax[i,j].set_xscale('log')
        ax[i,j].set_yscale('log')
        ax[i,j].set_ylim(min_y, max_y)
        ax[i,j].set_xlim(min_x, max_x)
        ax[i,j].set_title(filter_frame)
        ax[i, j].legend()

        import matplotlib.patches as patches
        x = min_x
        y = min_y
        width = ODD_criterias["FPPI"] - min_x
        height = ODD_criterias["MR"] - min_y
        # Add the grey square patch to the axes
        grey_square = patches.Rectangle((x, y), width, height, facecolor='grey', alpha=0.5)
        ax[i,j].add_patch(grey_square)
        ax[i,j].text(min_x+width/2/10, min_y+height/2/10, s="ODD")

plt.tight_layout()
plt.show()




#%%
from src.utils import plot_heatmap_metrics
thresholds = [0.5, 0.9, 0.99]
df_analysis_heatmap = df_analysis[np.isin(df_analysis["threshold"], thresholds)]
plot_heatmap_metrics(df_analysis_heatmap, model_names, metrics, ODD_limit,
                     param_heatmap_metrics=param_heatmap_metrics, results_dir=results_dir)














#%% Plot an image in particular =========================

i = 30

img_path = osp.join(root, df_frame_metadata["file_name"].iloc[i])
frame_id = df_frame_metadata.index[i]


#%%


from src.utils import filter_gt_bboxes

gtbbox_filtering = gtbbox_filtering_all["Overall"]

if len(pd.DataFrame(df_gtbbox_metadata.loc[frame_id]).T) == 1:
    df_gtbbox_metadata_frame = pd.DataFrame(df_gtbbox_metadata.loc[frame_id]).T.reset_index()
else:
    df_gtbbox_metadata_frame = df_gtbbox_metadata.loc[frame_id].reset_index()
excluded_gt = filter_gt_bboxes(df_gtbbox_metadata_frame, gtbbox_filtering)


from src.detection.detector import Detector

detector = Detector(model_name, device="cpu")
preds = detector.get_preds_from_files(dataset_name, root, df_frame_metadata)

from src.detection.metrics import detection_metric
metric = detection_metric(gtbbox_filtering)
df_mr_fppi, df_gt_bbox = metric.compute(dataset_name, model_name, preds, targets, df_gtbbox_metadata,
                                        gtbbox_filtering)
#todo add threshold to 1 in data
from src.utils import plot_results_img
plot_results_img(img_path, frame_id, preds=preds, targets=targets,
             df_gt_bbox=df_gt_bbox, threshold=0.9999) #todo seems there is a bug, woman in middle should be in red and guy should be red. No sense of all this.



#%%

from src.detection.metrics import compute_model_metrics_on_dataset
model_name = model_names[0]



gtbbox_filtering = gtbbox_filtering_all["Overall"]

if len(pd.DataFrame(df_gtbbox_metadata.loc[frame_id]).T) == 1:
    df_gtbbox_metadata_frame = pd.DataFrame(df_gtbbox_metadata.loc[frame_id]).T.reset_index()
else:
    df_gtbbox_metadata_frame = df_gtbbox_metadata.loc[frame_id].reset_index()
excluded_gt = filter_gt_bboxes(df_gtbbox_metadata_frame, gtbbox_filtering)


from src.detection.detector import Detector

detector = Detector(model_name, device="cuda")
preds = detector.get_preds_from_files(dataset_name, root, df_frame_metadata)

from src.detection.metrics import detection_metric
metric = detection_metric(gtbbox_filtering)
_, df_gt_bbox = metric.compute(dataset_name, model_name, preds, targets, df_gtbbox_metadata,
                                        gtbbox_filtering)


from src.utils import plot_results_img

#%%


i = 18
img_path = osp.join(root, df_frame_metadata["file_name"].iloc[i])
frame_id = df_frame_metadata.index[i]

plot_results_img(img_path, frame_id, preds=None, targets=targets,
             df_gt_bbox=df_gt_bbox, threshold=0.9) #todo seems there is a bug, woman in middle should be in red and guy should be red. No sense of all this.


