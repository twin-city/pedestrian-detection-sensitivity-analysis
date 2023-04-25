import os
import pandas as pd
import setuptools.errors
import numpy as np
from utils import filter_gt_bboxes, plot_results_img, compute_ffpi_against_fp2
import os.path as osp

#todo truncation

#%% params
model_name = "faster-rcnn_cityscapes"
dataset_name = "motsynth"
root_motsynth = "/home/raphael/work/datasets/MOTSynth/"
max_sample = 1000  # Uniform sampled in dataset

#%%
from src.preprocessing.motsynth_processing import MotsynthProcessing
motsynth_processor = MotsynthProcessing(root_motsynth, max_samples=max_sample, video_ids=None)
targets, df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = motsynth_processor.get_dataset()
df_gtbbox_metadata.index = df_gtbbox_metadata.index.rename({"image_id": "frame_id"})
df_gtbbox_metadata["num_person"] = df_gtbbox_metadata.groupby("frame_id").apply(len)
keypoints_label_names = [f"keypoints_label_{i}" for i in range(22)]
df_gtbbox_metadata["occlusion_rate"] = df_gtbbox_metadata[keypoints_label_names].apply(lambda x: (2-x)).mean(axis=1)

#Todo https://github.com/cocodataset/cocoapi/issues/130
#tidi 0 is truncation, 1 is occluded 2 is visible
# df_gtbbox_metadata["occlusion_rate"] = 1-df_gtbbox_metadata["occlusion_rate"]


#todo seems there is a bug on pitch/roll/yaw. We assume a mistake of MoTSynth authors, and the referenced "yaw" is in fact "pitch"
df_frame_metadata["temp"] = df_frame_metadata["pitch"]
df_frame_metadata["pitch"] = df_frame_metadata["yaw"]
df_frame_metadata["yaw"] = df_frame_metadata["temp"]


# Frame id list
img_path_list = [osp.join(root_motsynth, x) for x in df_frame_metadata["file_name"]]
frame_id_list = list(df_frame_metadata["id"].values.astype(str))

#%%

# Detections
from src.detection.detector import Detector
detector = Detector(model_name, device="cuda")
preds = detector.get_preds_from_files(dataset_name, frame_id_list, img_path_list)

#########################################   Peform Tests   ############################################################

#%% Analyze results on an image

gtbbox_filtering = {"occlusion_rate": (0.8, "max"),
                    "area": (400, "min")}

# plot
i = 1
frame_id = frame_id_list[i]
img_path = img_path_list[i]
df_gtbbox_metadata_frame = df_gtbbox_metadata.loc[frame_id] #todo delay
excluded_gt = filter_gt_bboxes(df_gtbbox_metadata_frame, gtbbox_filtering)
occlusions_ids = [i for i, idx in enumerate(df_gtbbox_metadata_frame.index) if idx in excluded_gt]
plot_results_img(img_path, frame_id, preds, targets, occlusions_ids)

#%% As in ECP

df_mr_fppi, df_matched_gtbbox = compute_ffpi_against_fp2(dataset_name, model_name, preds, targets, df_gtbbox_metadata, gtbbox_filtering)

#%% Concat results and metadata
df_analysis = pd.merge(df_mr_fppi, df_frame_metadata, on="frame_id")
df_analysis_frame = df_analysis.groupby("frame_id").apply(lambda x: x.mean(numeric_only=True))

#%% study correlations per frame
import matplotlib.pyplot as plt
frame_cofactors = ["adverse_weather", "is_night", "pitch", "yaw", "roll"]
metrics = ["MR", "FPPI"]
from scipy.stats import pearsonr
corr_matrix = df_analysis_frame[metrics+frame_cofactors].corr(method=lambda x, y: pearsonr(x, y)[0])
p_matrix = df_analysis_frame[metrics+frame_cofactors].corr(method=lambda x, y: pearsonr(x, y)[1])

print(p_matrix)
import seaborn as sns
sns.heatmap(corr_matrix, annot=True)
plt.show()

sns.heatmap(p_matrix, annot=True)
plt.show()

sns.heatmap(corr_matrix[p_matrix<0.05], annot=True) #todo which statistical test at the frame level ?
plt.show()

#%% study correlations per sequence

import matplotlib.pyplot as plt
seq_cofactors = ["adverse_weather", "is_night", "pitch"]
metrics = ["MR", "FPPI"]
from scipy.stats import pearsonr
corr_matrix = df_analysis_frame.groupby("seq_name").apply(lambda x: x.mean())[metrics+seq_cofactors].corr(method=lambda x, y: pearsonr(x, y)[0])
p_matrix = df_analysis_frame.groupby("seq_name").apply(lambda x: x.mean())[metrics+seq_cofactors].corr(method=lambda x, y: pearsonr(x, y)[1])

print(p_matrix)
import seaborn as sns
sns.heatmap(corr_matrix, annot=True)
plt.show()

sns.heatmap(p_matrix, annot=True)
plt.show()

sns.heatmap(corr_matrix[p_matrix<0.05], annot=True)
plt.show()



#%% Day 'n Night example

night_ids = df_analysis_frame[df_analysis_frame["is_night"]==1].index.to_list()
day_ids = df_analysis_frame[df_analysis_frame["is_night"]==0].index.to_list()

metrics_day = df_mr_fppi.loc[day_ids].groupby("threshold").apply(lambda x: x.mean(numeric_only=True))
metrics_night = df_mr_fppi.loc[night_ids].groupby("threshold").apply(lambda x: x.mean(numeric_only=True))

fig, ax = plt.subplots(1,1)
ax.plot(metrics_day["FPPI"], metrics_day["MR"], label="day")
ax.plot(metrics_night["FPPI"], metrics_night["MR"], label="night")

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_ylim(0.1, 1)
ax.set_xlim(0.1, 20)
plt.legend()
plt.show()


#%% What is the Operational Design Domain ???????? Give spec

"""
For now threshold = 0.5, IoU=0.5

What happens ?
    - Give expected specs (e.g. MR=0.2)F
    - Visualize when specs are met (when the case, and when not how much is the drop ?)
    
/!\ Keep in mind correlation between cofactors
"""

df_analysis_50 = pd.merge(df_mr_fppi[df_mr_fppi.index.get_level_values('threshold') == 0.5], df_frame_metadata, on="frame_id")

# Sensitivity of the method ?
plt.bar(seq_cofactors, -np.log10(p_matrix["MR"][seq_cofactors]))
plt.title("p-val pour le test de correlation")
plt.show()

#%% When is it working ? Plage de fonctionnement --> What did they do in Synscapes ?

"""
2 methods : either a regressor, or by average on zones/tiles (and measure that on XX? of cases we meet the criteria ?)
--> should plot it
"""


from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay


pdp_cofactors = ['adverse_weather', 'is_night', 'pitch']

y = df_analysis_50["MR"].values
X = df_analysis_50[pdp_cofactors].values
X[:,0]=1.*X[:,0]

clf = GradientBoostingRegressor(n_estimators=50, learning_rate=1.0,
    max_depth=1, random_state=0).fit(X, y)
features = [0,1,2, (0,1), (1,2)]

fig, ax = plt.subplots(1,1, figsize=(16,10))
PartialDependenceDisplay.from_estimator(clf, X, features, feature_names=pdp_cofactors, ax=ax)
plt.show()

#%% Feature importance w/ trees

y = df_analysis_50["FPPI"].values

from sklearn.ensemble import RandomForestRegressor

feature_names = pdp_cofactors
forest = RandomForestRegressor(random_state=0)

from sklearn.inspection import permutation_importance

forest.fit(X, y)

result = permutation_importance(
    forest, X, y, n_repeats=10, random_state=42, n_jobs=2
)
forest_importances = pd.Series(result.importances_mean, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean MR decrease")
fig.tight_layout()
plt.show()

#%% Distribution each one

feat = "adverse_weather"
metric = "FPPI"

fig, ax = plt.subplots(1,1, figsize=(16,10))
ax.hist(df_analysis_50[df_analysis_50[feat]==1][metric], alpha=0.5, label="1")
ax.hist(df_analysis_50[df_analysis_50[feat]==0][metric], alpha=0.5, label="0")
plt.legend()
plt.title(metric+"  vs   "+feat)
plt.show()


"""
But dataset may be biased : with adverse weather less people ?

Also sensitivity to yaw does not make sense 
"""

#todo sensitivity yaw ???

#%%


fig, ax = plt.subplots(1,1, figsize=(16,10))
ax.hist(df_analysis_50["yaw"], alpha=0.5, label="1", bins=50)
plt.legend()
plt.title("pitch")
plt.show()

#%% Check the roll/pitch ?

criteria = "pitch"

firsts = df_frame_metadata.sort_values(criteria).iloc[:5]["file_name"].values.tolist()
lasts = df_frame_metadata.sort_values(criteria).iloc[-5:]["file_name"].values.tolist()

values_firsts = df_frame_metadata.sort_values(criteria).iloc[:5][criteria]
values_lasts = df_frame_metadata.sort_values(criteria).iloc[-5:][criteria]


fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 5))
for i, (path1, path2) in enumerate(zip(firsts, lasts)):
    axs[0, i].imshow(plt.imread(osp.join(motsynth_processor.frames_dir, "../", path1)))
    axs[0, i].axis('off')
    axs[0, i].set_title(values_firsts.iloc[i])
    axs[1, i].imshow(plt.imread(osp.join(motsynth_processor.frames_dir, "../", path2)))
    axs[1, i].axis('off')
    axs[1, i].set_title(values_lasts.iloc[i])
plt.show()


#%% Check if camera angles change during videos ???

df_frame_metadata.groupby("seq_name").apply(lambda x: x.std(numeric_only=True))[["pitch", "roll", "yaw"]].max()


#%%

#%%

metric = "MR"
minimal_requirement = 0.3

matrix_odd = df_analysis_50.groupby(["is_night", "adverse_weather"]).apply(lambda x: x.mean())[metric].reset_index().pivot("is_night", "adverse_weather")
cmap = plt.get_cmap('hot')
sns.heatmap(matrix_odd, annot=True, fmt='.2f', cmap="PiYG", center=minimal_requirement)
plt.show()

print("coucou")


df_gtbbox_metadata_frame = df_gtbbox_metadata.groupby("frame_id").apply(lambda x: x.mean(numeric_only=True))
df_analysis_50_gtbbox = pd.merge(df_gtbbox_metadata_frame, df_analysis_50, on="frame_id")

#%%
import matplotlib.pyplot as plt
seq_cofactors = ["is_night","occl2","pitch", "adverse_weather", "occlusion_rate", "area", "is_crowd", "is_blurred", "num_person"]
metrics = ["MR", "FPPI"]
from scipy.stats import pearsonr
corr_matrix = df_analysis_50_gtbbox[metrics+seq_cofactors].corr(method=lambda x, y: pearsonr(x, y)[0])
p_matrix = df_analysis_50_gtbbox[metrics+seq_cofactors].corr(method=lambda x, y: pearsonr(x, y)[1])

print(p_matrix)
import seaborn as sns
sns.heatmap(corr_matrix, annot=True)
plt.show()

sns.heatmap(p_matrix, annot=True)
plt.show()

sns.heatmap(corr_matrix[p_matrix<0.05].loc[:,["MR", "FPPI"]], annot=True)
plt.tight_layout()
plt.show()


#%% Distribution each one

feat = "num_person"
metric = "MR"

fig, ax = plt.subplots(1,1, figsize=(10,6))
ax.hist(df_analysis_50_gtbbox[df_analysis_50_gtbbox[feat]>0.5][metric], alpha=0.5, label="occluded")
ax.hist(df_analysis_50_gtbbox[df_analysis_50_gtbbox[feat]<=0.5][metric], alpha=0.5, label="non occluded")
plt.legend()
plt.title(metric+"  vs   "+feat)
plt.show()

#%%
df_matched_gtbbox_analysis = pd.merge(df_matched_gtbbox[df_matched_gtbbox["threshold"]==0.5],
         df_gtbbox_metadata, on=("id"))

feat = "matched"
metric = "area"

fig, ax = plt.subplots(1,1, figsize=(10,6))
ax.hist(df_matched_gtbbox_analysis[df_matched_gtbbox_analysis[feat]==1][metric], alpha=0.5, label="matched", density=True, bins=1000)
ax.hist(df_matched_gtbbox_analysis[df_matched_gtbbox_analysis[feat]==0][metric], alpha=0.5, label="not matched", density=True, bins=1000)
ax.set_xlim(0,25000)
plt.legend()
plt.title(metric+"  vs   "+feat)
plt.show()



#%% Why do occlusion level correlate negatively ?

frame_id = df_gtbbox_metadata_frame.sort_values("occlusion_rate").index[8]
img_path = osp.join(motsynth_processor.root, df_frame_metadata.loc[frame_id]["file_name"])
plot_results_img(img_path, frame_id, preds, targets, [])

#%%


for j in range(50,60):

    j = -j

    frame_id = df_analysis_50_gtbbox.sort_values("occl2").index[j]
    img_path = osp.join(motsynth_processor.root, df_frame_metadata.loc[frame_id]["file_name"])

    fig, ax = plt.subplots(1,2, figsize=(10,4))
    df_analysis_50_gtbbox.plot.scatter("occl2", "MR", ax=ax[0])

    ax[0].scatter(df_analysis_50_gtbbox.loc[frame_id]["occl2"],
               df_analysis_50_gtbbox.loc[frame_id]["MR"], c="red")

    plot_results_img(img_path, frame_id, preds, targets, [], ax=ax[1])
    ax[0].set_title(j)
    plt.show()


#%%
df_analysis_50_gtbbox[df_analysis_50_gtbbox["pitch"]==0].plot.scatter("occlusion_rate", "MR")
plt.show()

#%%
df_gtbbox_metadata_subset = df_gtbbox_metadata[df_gtbbox_metadata["occlusion_rate"]<0.92]
df_gtbbox_metadata_subset = df_gtbbox_metadata[df_gtbbox_metadata["area"]>400]
df_analysis_50_gtbbox["occl2"] = df_gtbbox_metadata_subset.groupby("frame_id").apply(np.mean)["occlusion_rate"]

df_analysis_50_gtbbox[df_analysis_50_gtbbox["pitch"]==0].plot.scatter("occl2", "MR")
plt.show()