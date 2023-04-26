import os
import pandas as pd
import setuptools.errors
import numpy as np
import os.path as osp

"""

"""

#%% params of input dataset

dataset_name = "motsynth"
root_motsynth = "/home/raphael/work/datasets/MOTSynth/"
max_sample = 40  # Uniform sampled in dataset

seq_cofactors = ["adverse_weather", "is_night", "pitch"]
metrics = ["MR", "FPPI"]
ODD_criterias = {
    "MR": 0.5,
    "FPPI": 5,
}

#%% Get the dataset
from src.preprocessing.motsynth_processing import MotsynthProcessing
motsynth_processor = MotsynthProcessing(root_motsynth, max_samples=max_sample, video_ids=None)
dataset = motsynth_processor.get_dataset() #todo as class
root, targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = dataset


#%% Perform Dataset visualization

import matplotlib.pyplot as plt

#todo : 1D hist of : occlusion frequency, occlusion amount, height distribution, pitch

#todo : table for discrete variables : num_sequences, weather, day/night,

#todo distribution image ped position (bbox center)

#todo correlation between the metadatas !!!

df_gtbbox_metadata.hist("height", bins=200)
plt.xlim(0, 300)
plt.axvline(25, c="red")
plt.axvline(50, c="red")
plt.axvline(100, c="red")
plt.show()

#%%

xs, ys = [], []
for key, val in targets.items():
    bbox = val[0]["boxes"]
    x = bbox[:, 0] + (bbox[:, 2] - bbox[:, 0]) / 2
    y = bbox[:, 1] + (bbox[:, 3] - bbox[:, 1]) / 2
    xs.append(x.numpy())
    ys.append(y.numpy())
xs = np.concatenate(xs)
ys = np.concatenate(ys)



#%%

plt.hist2d(xs, ys, bins=40)
plt.colorbar()
plt.title("Bounding box center density (linked to viewpoints)")
plt.show()

plt.scatter(xs, ys)
plt.show()

#%% Params of detection


from src.detection.metrics import compute_model_metrics_on_dataset


gtbbox_filtering = {
    "occlusion_rate": (0.96, "max"),# At least 1 keypoint visible
    "area": (200, "min")
}

model_names = ["faster-rcnn_cityscapes", "mask-rcnn_coco"]

# Compute the metrics at different detection thresholds, for each model
df_metrics = pd.concat([compute_model_metrics_on_dataset(model_name, dataset_name, dataset, gtbbox_filtering, device="cuda")[0] for model_name in model_names])

# Merge with metadatas
df_analysis = pd.merge(df_metrics.reset_index(), df_frame_metadata, on="frame_id")




#%% Zoom on occlusion levels

df_gtbbox_metadata["occlusion_rate"].hist(bins=22, density=True)
plt.axvline(1/22, c="red")
plt.axvline(0.35, c="red")
plt.axvline(0.8, c="red")
plt.axvline(1, c="red")
plt.show()

(df_gtbbox_metadata["occlusion_rate"]==0).mean()

print("No occlusion : ")

from src.utils import subset_dataframe

invalid = subset_dataframe(df_gtbbox_metadata,{"occlusion_rate": 0})


partial = subset_dataframe(df_gtbbox_metadata,{"occlusion_rate": {">": 0.001, "<": 0.35}})
heavy = subset_dataframe(df_gtbbox_metadata,{"occlusion_rate": {">": 0.35, "<": 0.8}})
full = subset_dataframe(df_gtbbox_metadata,{"occlusion_rate": {">": 0.8}})

print(f"Invalid : {len(invalid)/len(df_gtbbox_metadata):.2f}")
print(f"partial : {len(partial)/len(df_gtbbox_metadata):.2f}")
print(f"heavy : {len(heavy)/len(df_gtbbox_metadata):.2f}")
print(f"full : {len(full)/len(df_gtbbox_metadata):.2f}")


#%% Check with ped id how often they are occluded


# At least in some cat

df_gtbbox_metadata.groupby(["seq_name", "ped_id"]).apply(lambda x: (x["occlusion_rate"]>0).mean()).hist(bins=50)
plt.title("Occlusion Frequency")
plt.xlabel("Fraction of time occluded")
plt.show()

df_gtbbox_metadata.groupby(["seq_name", "ped_id"]).apply(lambda x: (x["occlusion_rate"]>0.35).mean()).hist(bins=50)
plt.title("(at least) Heavy Occlusion Frequency")
plt.xlabel("Fraction of time at least heavily occluded")
plt.show()


#%% Average the occlusion masks if given (here with MoTSynth : keypoints)

keypoints_label_names = [f"keypoints_label_{i}" for i in range(22)]
keypoints_posx_names = [f"keypoints_posx_{i}" for i in range(22)]
keypoints_posy_names = [f"keypoints_posy_{i}" for i in range(22)]

df_gtbbox_metadata[keypoints_label_names+keypoints_posx_names+keypoints_posy_names]


#%% Compare visible to non-visible !!!!!!

i = 0

df_keypoints = pd.concat([df_gtbbox_metadata[[f"keypoints_label_{i}", f"keypoints_posx_{i}", f"keypoints_posy_{i}"]].rename(
    columns={f"keypoints_label_{i}": "label",
    f"keypoints_posx_{i}": "x",
    f"keypoints_posy_{i}": "y"}) for i in range(22)])

df_keypoints.plot.scatter("x", "y", c="label")

plt.show()

df_keypoints_occluded = df_keypoints[df_keypoints["label"]==2]

plt.hist2d(df_keypoints_occluded["x"], df_keypoints_occluded["y"], bins=100)
plt.colorbar()
plt.ylim(0, 1080)
plt.xlim(0, 1920)
plt.title("Bounding box center density (linked to viewpoints)")
plt.show()

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
    features = list(range(len(seq_cofactors)))
    PartialDependenceDisplay.from_estimator(clf, X, features, feature_names=seq_cofactors, ax=ax[i])
    ax[i].set_title(f"PDP for metric {metric}")

plt.tight_layout()
plt.show()