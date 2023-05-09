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
bbox_cofactors = ["height", "aspect_ratio", "is_crowd", "occlusion_rate"]
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

occl_thresh = [0.35, 0.8]
height_thresh = [25, 45, 155]
resolution = (1920, 1024)


#%%
from src.preprocessing.ecp_processing import ECPProcessing

root_ecp = "/media/raphael/Projects/datasets/EuroCityPerson/ECP/"
ecp_processor = ECPProcessing(root_ecp, max_samples=max_sample)
dataset = ecp_processor.get_dataset()
_, targets, df_gtbbox_metadata, df_frame_metadata, _ = dataset

img_path_list = [osp.join(root_ecp, x) for x in df_frame_metadata["file_name"]]
frame_id_list = list(df_frame_metadata["id"].values.astype(str))


#todo in processing
df_gtbbox_metadata["aspect_ratio"] = 1/df_gtbbox_metadata["aspect_ratio"]
mu = 0.4185
std = 0.12016
df_gtbbox_metadata["aspect_ratio_is_typical"] = np.logical_and(df_gtbbox_metadata["aspect_ratio"] < mu+std,  df_gtbbox_metadata["aspect_ratio"] > mu-std)

df_gtbbox_metadata["height"] = df_gtbbox_metadata["height"] .astype(int)
df_gtbbox_metadata["width"] = df_gtbbox_metadata["width"] .astype(int)
df_gtbbox_metadata["area"] = df_gtbbox_metadata["area"] .astype(float)
df_gtbbox_metadata["aspect_ratio"] = df_gtbbox_metadata["aspect_ratio"] .astype(float)

#todo add seq_name
df_gtbbox_metadata["seq_name"] = "toto"
df_gtbbox_metadata["ped_id"] = 0


#%%

df_frame_metadata["num_person"] = df_gtbbox_metadata.groupby("frame_id").apply(len).loc[df_frame_metadata.index]
#%% Correlation / Tables / Distributions


# Correlation checks

import matplotlib.pyplot as plt
from src.utils import compute_correlations, plot_correlations

corr_matrix, p_matrix = compute_correlations(df_gtbbox_metadata, bbox_cofactors)
plot_correlations(corr_matrix, p_matrix, title="Correlations between metadatas at bbox level")

corr_matrix, p_matrix = compute_correlations(df_frame_metadata, seq_cofactors)
plot_correlations(corr_matrix, p_matrix, title="Correlations between metadatas at frame level")

corr_matrix, p_matrix = compute_correlations(df_frame_metadata.groupby("seq_name").apply(lambda x: x.mean()), seq_cofactors)
plot_correlations(corr_matrix, p_matrix, title="Correlations between metadatas at sequence level")

#%% Matrix plot hidden

#df_gtbbox_metadata.plot.scatter("occlusion_rate", "height")
#plt.show()

#pd.plotting.scatter_matrix(df_frame_metadata[["pitch", "roll", "yaw"]])
#plt.show()

#pd.plotting.scatter_matrix(df_gtbbox_metadata[["area", "height", "aspect_ratio", "is_crowd", "attributes_0", "occlusion_rate"]])
#plt.show()

#pd.plotting.scatter_matrix(df_gtbbox_metadata[["height", "aspect_ratio", "is_crowd", "occlusion_rate"]])
#plt.show()

#%% Table : counts !!!! By Frame

"""
Give details : resolution ...
"""

n_images = df_frame_metadata.groupby("is_night").apply(len)
n_seqs = df_frame_metadata.groupby("is_night").apply(lambda x: len(x["seq_name"].unique()))
n_person = df_frame_metadata.groupby("is_night").apply(lambda x: x["num_person"].sum())
weathers = df_frame_metadata["weather"].unique()

df_descr = pd.DataFrame({
    "sequences (day/night)": f"{n_seqs[0]}/{n_seqs[1]}",
    "images (day/night)": f"{n_images[0]}/{n_images[1]}",
    "person (day/night)": f"{n_person[0]}/{n_person[1]}",
    "weather": ", ".join(list(weathers)),
}, index=[dataset_name]).T

with open(f'descr_{dataset_name}.md', 'w') as f:
    f.write(df_descr.to_markdown())



#%% Perform Dataset visualization

import matplotlib.pyplot as plt

df_gtbbox_metadata.hist("height", bins=200)
plt.xlim(0, 300)
plt.axvline(height_thresh[0], c="red")
plt.axvline(height_thresh[1], c="red")
plt.axvline(height_thresh[2], c="red")
plt.show()

#%%
(df_gtbbox_metadata[df_gtbbox_metadata["aspect_ratio"]<1]).hist("aspect_ratio", bins=100)

mu = df_gtbbox_metadata[df_gtbbox_metadata["aspect_ratio"]<1]["aspect_ratio"].mean()
std = df_gtbbox_metadata[df_gtbbox_metadata["aspect_ratio"]<1]["aspect_ratio"].std()
plt.xlim(0, 1)
plt.axvline(mu, c="red")
plt.axvline(mu-std, c="red")
plt.axvline(mu+std, c="red")
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
plt.title("Occlusion rate")
plt.xlabel("Fraction of pedestrian occluded")
plt.show()


from src.utils import subset_dataframe
invalid = subset_dataframe(df_gtbbox_metadata,{"occlusion_rate": 0})
partial = subset_dataframe(df_gtbbox_metadata,{"occlusion_rate": {">": 0.001, "<": occl_thresh[0]}})
heavy = subset_dataframe(df_gtbbox_metadata,{"occlusion_rate": {">": occl_thresh[0], "<": occl_thresh[1]}})
full = subset_dataframe(df_gtbbox_metadata,{"occlusion_rate": {">": occl_thresh[1]}})
print(f"Invalid : {len(invalid)/len(df_gtbbox_metadata):.2f}")
print(f"partial : {len(partial)/len(df_gtbbox_metadata):.2f}")
print(f"heavy : {len(heavy)/len(df_gtbbox_metadata):.2f}")
print(f"full : {len(full)/len(df_gtbbox_metadata):.2f}")


#%% Check with ped id how often they are occluded

# At least in some cat
df_gtbbox_metadata.groupby(["seq_name", "ped_id"]).apply(lambda x: (x["occlusion_rate"]>0).mean()).hist(bins=50, density=True)
plt.title("(at least) some occlusion (Frequency)")
plt.xlabel("Fraction of time occluded")
plt.show()

df_gtbbox_metadata.groupby(["seq_name", "ped_id"]).apply(lambda x: (x["occlusion_rate"]>occl_thresh[0]).mean()).hist(bins=50, density=True)
plt.title("(at least) Heavy Occlusion (Frequency)")
plt.xlabel("Fraction of time at least heavily occluded")
plt.show()

"""
Very few have no occlusions at all
"""


#%% Compare visible to non-visible !!!!!! Keypoints

if "keypoints_label_0" in df_gtbbox_metadata.columns:

    df_keypoints = pd.concat([df_gtbbox_metadata[[f"keypoints_label_{i}", f"keypoints_posx_{i}", f"keypoints_posy_{i}"]].rename(
        columns={f"keypoints_label_{i}": "label",
        f"keypoints_posx_{i}": "x",
        f"keypoints_posy_{i}": "y"}) for i in range(22)])


    for label in [1,2]:
        mask = pd.Series([True] * len(df_keypoints), index=df_keypoints.index)
        mask &= df_keypoints["x"] < 1920
        mask &= df_keypoints["x"] > 0
        mask &= df_keypoints["y"] < 1080
        mask &= df_keypoints["y"] > 0
        mask &= df_keypoints["label"] == label

        plt.hist2d(df_keypoints[mask]["x"], df_keypoints[mask]["y"], bins=100, density=True)
        plt.colorbar()
        plt.title(f"Keypoint density with occlusion label {label}")
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




#%% Do simple visualization

#todo for now we take the average on multiple thresholds values

metric = "MR"
#todo ODD on top
ODD_nominal = {
    "is_night": 0,
    "adverse_weather": 0,
#    "pitch": {">":-10},
}

ODD_limit = [
    {"is_night": 1},
    {"adverse_weather": 1},
#    {"pitch": {"<":-10}},
]

#todo adapt
ax_y_labels = ["night", "bad weather"]#, "high-angle shot"]

nominal_metric_value = subset_dataframe(df_analysis, ODD_nominal)[metric].mean()

df_odd_model_list = []
for model_name in model_names:
    perc_increase_list = []
    for limit in ODD_limit:
        condition = ODD_nominal.copy()
        condition.update(limit)
        #condition.update({"model_name": model_name})
        df_subset = subset_dataframe(df_analysis, condition)
        df_subset = df_subset[df_subset["model_name"] == model_name] #todo do it with subset
        perc_increase = (df_subset[metric].mean()-nominal_metric_value)/nominal_metric_value
        print(limit, f"+{100*perc_increase:.1f}%")
        perc_increase_list.append(perc_increase)

    df_odd_model_list.append(pd.DataFrame(perc_increase_list, index=ODD_limit, columns=[model_name]))

df_odd_model = pd.concat(df_odd_model_list, axis=1)
df_odd_model.index = ax_y_labels

import seaborn as sns
from matplotlib.colors import BoundaryNorm

# Define the boundaries of each zone
bounds = [0, 0.1, 0.2, 0.5]
# Define a unique color for each zone
colors = ['green', 'yellow', 'red']
# Create a colormap with discrete colors
cmap = sns.color_palette(colors, n_colors=len(bounds)-1).as_hex()
# Create a BoundaryNorm object to define the colormap
norm = BoundaryNorm(bounds, len(cmap))

cmap = "gist_rainbow_r"

fig, ax = plt.subplots(1,1)
sns.heatmap(100*df_odd_model, annot=True,
            cmap=cmap, center=0, vmax=80, ax=ax, fmt=".0f", cbar_kws={'format': '%.0f%%'})#, norm=norm)
ax.collections[0].colorbar.set_label('Decrease in performance')
plt.title(f"Impact of parameters on {metric}")
plt.tight_layout()
plt.show()

#%% Also do the nominal case vs others in barplots


fig, ax = plt.subplots(2,1, figsize=(4,8))
for i, metric in enumerate(metrics):
    df_odd_model_list = []
    for model_name in model_names:
        val_list = []
        for limit in [{}]+ODD_limit:
            condition = ODD_nominal.copy()
            condition.update(limit)
            # condition.update({"model_name": model_name})
            df_subset = subset_dataframe(df_analysis, condition)
            df_subset = df_subset[df_subset["model_name"] == model_name]  # todo do it with subset
            val_list.append(df_subset[metric].mean())

        df_odd_model_list.append(pd.DataFrame(val_list, index=["Nominal"]+ODD_limit, columns=[model_name]))

    df_odd_model = pd.concat(df_odd_model_list, axis=1)
    df_odd_model.index = ["Nominal"]+ax_y_labels

    df_odd_model.plot.bar(ax=ax[i])
    ax[i].set_title(f"Performance for metric {metric}")
plt.tight_layout()
plt.show()

#todo statisticals significance ? violinplot ? std ? We miss this information
#todo also need to get the best trade-off for each of the models !!!!
# (depending on ODD) but too important because it is more the part of the
# solution provider

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