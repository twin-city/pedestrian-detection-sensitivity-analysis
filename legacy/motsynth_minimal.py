import os

import pandas as pd
from src.utils import filter_gt_bboxes, plot_results_img, compute_ffpi_against_fp2
import numpy as np
import os.path as osp

""" #todo features to add 

#todo feature mean distance to camera
#todo plot and quantify difference between datasets
#todo mixed effect model to get the parameters ????? of significance. Or linear model ?? Cf Park
"""

#%% For now only 20 different


np.random.seed(0)
annot_dir = "/home/raphael/work/datasets/MOTSynth/coco annot"
exclude_ids_frames = set(["060", "081", "026", "132", "136", "102", "099", "174", "140"])
video_ids_frames = set(np.sort(os.listdir("/home/raphael/work/datasets/MOTSynth/frames")).tolist())
video_ids_json = set([i.replace(".json", "") for i in os.listdir(annot_dir)]) - exclude_ids_frames
video_ids = list(np.sort(list(set.intersection(video_ids_frames, video_ids_json))))
video_ids = np.random.choice(video_ids, 20, replace=False)


#%% params
model_name = "faster-rcnn_cityscapes"
max_sample = 1000 # Uniform sampled in dataset

# Dataset #todo add statistical comparison between datasets
from src.preprocessing.motsynth_processing import MotsynthProcessing
motsynth_processor = MotsynthProcessing(max_samples=max_sample, video_ids=None)
targets, metadatas, frame_id_list, img_path_list = motsynth_processor.get_MoTSynth_annotations_and_imagepaths()
df_gtbbox_metadata, df_frame_metadata, df_sequence_metadata = metadatas
delay = motsynth_processor.delay

#%% Show Dataset distributions

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1, figsize=(10,10))
df_frame_metadata.hist(ax=ax)
plt.show()

#%% What are the correlations ?

import seaborn as sns
corr_matrix = df_frame_metadata.corr()
fig, ax = plt.subplots(1,1, figsize=(10,10))
sns.heatmap(corr_matrix, annot=True, ax=ax)
plt.show()

#%% Check more detailed : plot 3v3 extreme images in category

criteria = "yaw"

firsts = df_frame_metadata.sort_values("yaw").iloc[:3]["file_name"].values.tolist()
lasts = df_frame_metadata.sort_values("yaw").iloc[-3:]["file_name"].values.tolist()
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))
for i, (path1, path2) in enumerate(zip(firsts, lasts)):
    axs[0, i].imshow(plt.imread(osp.join(motsynth_processor.frames_dir, "../../", path1)))
    axs[0, i].axis('off')
    axs[1, i].imshow(plt.imread(osp.join(motsynth_processor.frames_dir, "../../", path2)))
    axs[1, i].axis('off')
plt.show()

#%%

import matplotlib.pyplot as plt
import numpy as np

criteria = "z"

# sort dataframe by yaw
df_frame_metadata = df_frame_metadata.sort_values(criteria)

# select first and last three images
firsts = df_frame_metadata.iloc[:3]["file_name"].values.tolist()
lasts = df_frame_metadata.iloc[-3:]["file_name"].values.tolist()

# create a figure with two subplots
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))

# plot the images and add the yaw value in the title
for i, (path1, path2) in enumerate(zip(firsts, lasts)):
    img1 = plt.imread(osp.join(motsynth_processor.frames_dir, "../../", path1))
    img2 = plt.imread(osp.join(motsynth_processor.frames_dir, "../../", path2))
    axs[0, i].imshow(img1)
    axs[0, i].axis('off')
    axs[0, i].set_title(f"{criteria}: {df_frame_metadata.loc[df_frame_metadata['file_name']==path1]['yaw'].iloc[0]}")
    axs[1, i].imshow(img2)
    axs[1, i].axis('off')
    axs[1, i].set_title(f"{criteria}: {df_frame_metadata.loc[df_frame_metadata['file_name']==path2]['yaw'].iloc[0]}")

# display the plot
plt.show()



#%%

# Detections
from src.detection.detector import Detector
detector = Detector(model_name)
preds = detector.get_preds_from_files(frame_id_list, img_path_list)

#########################################   Peform Tests   ############################################################

#%% Analyze results on an image

#todo a test --> is it the image we saved ?

gtbbox_filtering = {"occlusion_rate": (0.9, "max"),
                    "area": (20, "min")}
i = 15

frame_id = frame_id_list[i]
img_path = img_path_list[i]
df_gtbbox_metadata_frame = df_gtbbox_metadata.loc[int(frame_id)+delay] #todo delay
excluded_gt = filter_gt_bboxes(df_gtbbox_metadata_frame, gtbbox_filtering)
occlusions_ids = [i for i, idx in enumerate(df_gtbbox_metadata_frame.index) if idx  in excluded_gt]
plot_results_img(img_path, frame_id, preds, targets, occlusions_ids)


#%% Compute depending on a condition

# todo a gt filtering for frames/sequence also ? Also save it to save time
# GT filtering #todo minimal value for now

df_mr_fppi = compute_ffpi_against_fp2(preds, targets, df_gtbbox_metadata, gtbbox_filtering, model_name)
df_mr_fppi = df_mr_fppi.reset_index()
df_mr_fppi["frame_id"] = df_mr_fppi["frame_id"].astype(int)


#########################################   Peform Analysis   #########################################################


#%%
df_analysis = pd.merge(df_mr_fppi, df_frame_metadata.reset_index().rename(columns={"index": "frame_id"}), on="frame_id")
df_analysis.groupby(df_analysis["yaw"]>df_analysis["yaw"].median()).apply(np.mean)[["MR","FPPI"]]

#%%

# df_analysis.groupby("frame_id").apply(np.mean).plot.scatter("z", "is_night")
# plt.show()


#%%

#todo add metrics : AP50, LAMR https://eurocity-dataset.tudelft.nl/eval/benchmarks/detection


"""
At image / video level : 
    Cam_angle : pitch
    adverse_weather
    is_night
    
At bbox level : 
    - reasonable / small / occluded / all https://eurocity-dataset.tudelft.nl/eval/benchmarks/detection
"""

#%% T-test

#todo : correlation to begin with  (((/test with metrics (mr, fppi, mAP50, YAML), plot fppi/mr, precision/recall)))
#todo should take into account non-independant (e.g. mixed effect ?)

df_analysis_frame = df_analysis.groupby("frame_id").apply(lambda x: x.mean())

frame_cofactors = ["pitch", "adverse_weather", "is_night"]
metrics = ["MR", "FPPI"]

from scipy.stats import pearsonr
corr_matrix = df_analysis_frame[metrics+frame_cofactors].corr(method=lambda x, y: pearsonr(x, y)[0])
p_matrix = df_analysis_frame[metrics+frame_cofactors].corr(method=lambda x, y: pearsonr(x, y)[1])


"""
before = df_analysis_frame[df_analysis_frame["yaw"]>df_analysis_frame["yaw"].median()]["frame_id"]
after = df_analysis_frame[df_analysis_frame["yaw"]<=df_analysis_frame["yaw"].median()]["frame_id"]

# Perform paired t-test
t, p = ttest_rel(df_analysis_frame.loc[before, "MR"].values, df_analysis_frame.loc[after.values, "MR"])

# Print results
print("t-value:", t)
print("p-value:", p)

fig, ax = plt.subplots(1,1)
ax.hist(df_analysis_frame.loc[before, "MR"].values, alpha=0.5)
ax.hist(df_analysis_frame.loc[after, "MR"].values, alpha=0.5)
plt.show()
"""

#%%

#todo discrete case

# def plot_mr_fppi_curve(df_mr_fppi, filtering):

    # todo for now only 2 values ?


#%%


#todo continuous case

#%%



"""
#%%

pd.merge(df_mr_fppi, df_frame_metadata.reset_index().rename(columns={"index": "frame_id"}), on="frame_id")


#%%

#%%

def get_mr_fppi_curve(df_mr_fppi, frame_ids):
    metrics = df_mr_fppi.loc[frame_ids].groupby("threshold").apply(lambda x: x.mean(numeric_only=True))
    mr = metrics["MR"]
    fppi = metrics["FPPI"]
    return mr, fppi



#todo combine multiple : weather and night ...

cofactor = "is_night"
value = 1

def listint2liststr(l):
    return [str(i) for i in l]

cof_frame_ids = listint2liststr(df_frame_metadata[df_frame_metadata[cofactor] == value].index.to_list())
nocof_frame_ids = listint2liststr(df_frame_metadata[df_frame_metadata[cofactor] != value].index.to_list())



import matplotlib.pyplot as plt
fig, ax = plt.subplots()

mr, fppi = get_mr_fppi_curve(df_mr_fppi, cof_frame_ids)
ax.plot(mr, fppi, c="green", label=f"has {cofactor}")
ax.scatter(mr, fppi, c="green")


mr, fppi = get_mr_fppi_curve(df_mr_fppi, nocof_frame_ids)
ax.plot(mr, fppi, c="red", label=f"no has {cofactor}")
ax.scatter(mr, fppi, c="red")



ax.set_xscale('log')
ax.set_yscale('log')

ax.set_ylim(0.1, 1)
ax.set_xlim(0.1, 20)

plt.legend()
plt.show()
"""