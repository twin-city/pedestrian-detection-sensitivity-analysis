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

#todo seems there is a bug on pitch/roll/yaw

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

gtbbox_filtering = {"occlusion_rate": (0.9, "max"),
                    "area": (20, "min")}

# plot
i = 1
frame_id = frame_id_list[i]
img_path = img_path_list[i]
df_gtbbox_metadata_frame = df_gtbbox_metadata.loc[frame_id] #todo delay
excluded_gt = filter_gt_bboxes(df_gtbbox_metadata_frame, gtbbox_filtering)
occlusions_ids = [i for i, idx in enumerate(df_gtbbox_metadata_frame.index) if idx in excluded_gt]
plot_results_img(img_path, frame_id, preds, targets, occlusions_ids)

#%% As in ECP

df_mr_fppi = compute_ffpi_against_fp2(dataset_name, model_name, preds, targets, df_gtbbox_metadata, gtbbox_filtering)


#%% Concat results and metadata
df_analysis = pd.merge(df_mr_fppi, df_frame_metadata, on="frame_id")
df_analysis_frame = df_analysis.groupby("frame_id").apply(lambda x: x.mean(numeric_only=True))

#%% study correlations
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

#%% Day 'n Night example

night_ids = df_analysis_frame[df_analysis_frame["is_night"]==1].index.to_list()
day_ids = df_analysis_frame[df_analysis_frame["is_night"]==0].index.to_list()

metrics_day = df_mr_fppi.loc[day_ids].groupby("threshold").apply(lambda x: x.mean(numeric_only=True))
metrics_night = df_mr_fppi.loc[night_ids].groupby("threshold").apply(lambda x: x.mean(numeric_only=True))

fig, ax = plt.subplots(1,1)
ax.plot(metrics_day["MR"], metrics_day["FPPI"], label="day")
ax.plot(metrics_night["MR"], metrics_night["FPPI"], label="night")

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
    - Give expected specs (e.g. MR=0.2)
    - Visualize when specs are met (when the case, and when not how much is the drop ?)
    
/!\ Keep in mind correlation between cofactors
"""

df_analysis_50 = pd.merge(df_mr_fppi[df_mr_fppi.index.get_level_values('threshold') == 0.5], df_frame_metadata, on="frame_id")

# Sensitivity of the method ?
plt.bar(frame_cofactors, -np.log10(p_matrix["MR"][frame_cofactors]))
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

y = df_analysis_50["MR"].values
X = df_analysis_50[frame_cofactors].values
X[:,0]=1.*X[:,0]

clf = GradientBoostingRegressor(n_estimators=50, learning_rate=1.0,
    max_depth=1, random_state=0).fit(X, y)
features = [0,1,2,3,4, (0,1), (1,3)]

fig, ax = plt.subplots(1,1, figsize=(16,10))
PartialDependenceDisplay.from_estimator(clf, X, features, feature_names=frame_cofactors, ax=ax)
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

criteria = "roll"

firsts = df_frame_metadata.sort_values(criteria).iloc[:5]["file_name"].values.tolist()
lasts = df_frame_metadata.sort_values(criteria).iloc[-5:]["file_name"].values.tolist()
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 5))
for i, (path1, path2) in enumerate(zip(firsts, lasts)):
    axs[0, i].imshow(plt.imread(osp.join(motsynth_processor.frames_dir, "../", path1)))
    axs[0, i].axis('off')
    axs[1, i].imshow(plt.imread(osp.join(motsynth_processor.frames_dir, "../", path2)))
    axs[1, i].axis('off')
plt.show()


#%% Check if camera angles change during videos ???

df_frame_metadata.groupby("seq_name").apply(lambda x: x.std())[["pitch", "roll", "yaw"]].max()
