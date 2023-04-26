import pandas as pd
# from src.utils import filter_gt_bboxes, plot_results_img, compute_ffpi_against_fp2

from src.detection.metrics import filter_gt_bboxes, compute_fp_missratio2

import os.path as osp
import numpy as np

#%% params
dataset_name = "EuroCityPerson"
model_name = "faster-rcnn_cityscapes"
max_sample = 30 # Uniform sampled in dataset

model_name = "faster-rcnn_cityscapes"
seq_cofactors = ["adverse_weather", "is_night"]# , "pitch"]
metrics = ["MR", "FPPI"]

gtbbox_filtering = {
    "occlusion_rate": (0.9, "max"),# At least 1 keypoint visible
    "truncation_rate": (0.9, "max"),
    "area": (20, "min")
}

ODD_criterias = {
    "MR": 0.5,
    "FPPI": 5,
}

model_names = ["faster-rcnn_cityscapes"]

#%%
from src.preprocessing.ecp_processing import ECPProcessing

root_ecp = "/media/raphael/Projects/datasets/EuroCityPerson/ECP/"
ecp_processor = ECPProcessing(root_ecp, max_samples=max_sample)
dataset = ecp_processor.get_dataset()
_, targets, df_gtbbox_metadata, df_frame_metadata, _ = dataset

img_path_list = [osp.join(root_ecp, x) for x in df_frame_metadata["file_name"]]
frame_id_list = list(df_frame_metadata["id"].values.astype(str))


#%% Params of detection
from src.detection.detector import Detector
from src.detection.metrics import detection_metric
from src.detection.metrics import compute_model_metrics_on_dataset


# Compute the metrics at different detection thresholds, for each model
df_metrics = pd.concat([compute_model_metrics_on_dataset(model_name, dataset_name, dataset, gtbbox_filtering, device="cuda")[0] for model_name in model_names])

# Merge with metadatas
df_analysis = pd.merge(df_metrics.reset_index(), df_frame_metadata, on="frame_id")



###########################################################################################
#%% Compare the models
###########################################################################################


#todo Add imgs + extreme imgs plots ???

#%% Model performance :  Plots MR vs FPPI on frame filtering

from src.utils import subset_dataframe
import matplotlib.pyplot as plt

dict_filter_frames = {
    "all": [{}],
    "is_night": ({"is_night": 0}, {"is_night": 1}),
    "adverse_weather": ({"adverse_weather": 0}, {"adverse_weather": 1}),
    #"pitch": ({"pitch": {"<": -10}}, {"pitch": {">": -10}}),
}

min_x, max_x = 0.5, 20
min_y, max_y = 0.1, 1

n_col = max([len(val) for _, val in dict_filter_frames.items()])
n_row = len(dict_filter_frames)

fig, ax = plt.subplots(n_row, n_col, figsize=(8,14))
for i, (key, filter_frames) in enumerate(dict_filter_frames.items()):
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


#%% Model performance :  Plots MR vs FPPI on gt bbox filtering







#%% Model sensibility :  What are the main parameters influencing model performance ?

from src.utils import compute_correlations, plot_correlations




p_matrix_list = []
bar_width = 1/(len(metrics)+1)

fig, ax = plt.subplots(len(metrics), 1, figsize=(6,10))
for i, model_name in enumerate(model_names):

    # Compute p-val for a model
    df = df_analysis[df_analysis["model_name"] == model_name].groupby("frame_id").apply(
        lambda x: x.mean(numeric_only=True))
    df["seq_name"] = df_analysis[df_analysis["model_name"] == model_name].groupby("frame_id").apply(lambda x: x.iloc[0]["seq_name"])
    df_analysis_seq = df.groupby("seq_name").apply(lambda x: x.mean())
    features = metrics + seq_cofactors
    _, p_matrix = compute_correlations(df_analysis_seq, features)
    p_matrix["model_name"] = model_name
    p_matrix_list.append(p_matrix)

    # Do the bar plots
    for j, metric in enumerate(metrics):
        ax[j].bar(np.arange(len(seq_cofactors))+i*bar_width,
                  -np.log10(p_matrix[metric][seq_cofactors]),
                  width=bar_width, label=model_name)
        print(np.arange(len(seq_cofactors))+i*bar_width)
        ax[j].set_title(f"pval of pearson correlation with {metric} (avrg per frame)")

        ax[j].legend()

        ax[j].set_xticks(range(len(seq_cofactors)))
        ax[j].set_xticklabels(seq_cofactors, rotation=45)

plt.tight_layout()
plt.show()


#%% Compute as feature importance : is it consistent ?


from src.utils import plot_importance
plot_importance(model_names, metrics, df_analysis, seq_cofactors, importance_method="linear")
plot_importance(model_names, metrics, df_analysis, seq_cofactors, importance_method="permutation")

"""
Warning, the linear case may hide correlations
"""



###########################################################################################
#%% Zoom on one model
###########################################################################################

# which model ?
model_name = "faster-rcnn_cityscapes"


#%% Correlations Analysis
from src.utils import compute_correlations, plot_correlations

df = df_analysis[df_analysis["model_name"] == model_name].groupby("frame_id").apply(lambda x: x.mean(numeric_only=True))
df["seq_name"] = df_analysis[df_analysis["model_name"] == model_name].groupby("frame_id").apply(lambda x: x.iloc[0]["seq_name"])

df_analysis_seq = df.groupby("seq_name").apply(lambda x: x.mean())
features = metrics + seq_cofactors

corr_matrix, p_matrix = compute_correlations(df_analysis_seq, features)
plot_correlations(corr_matrix, p_matrix, title="per_sequence")

corr_matrix, p_matrix = compute_correlations(df, features)
plot_correlations(corr_matrix, p_matrix, title="per_frame")



#%% Partial Dependance Plot


#todo add the ODD representation


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay

df = df_analysis[df_analysis["model_name"] == model_name].groupby("frame_id").apply(lambda x: x.mean(numeric_only=True))
metric = "MR"

fig, ax = plt.subplots(2, 1, figsize=(8, 5))

for i, metric in enumerate(metrics):
    y = df[metric].values
    X = df[seq_cofactors].values
    X[:,0] = 1.*X[:,0]

    clf = GradientBoostingRegressor(n_estimators=50, learning_rate=1.0,
        max_depth=1, random_state=0).fit(X, y)
    features = [0,1]
    PartialDependenceDisplay.from_estimator(clf, X, features, feature_names=seq_cofactors, ax=ax[i])
    ax[i].set_title(f"PDP for metric {metric}")

plt.tight_layout()
plt.show()