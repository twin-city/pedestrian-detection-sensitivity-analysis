import pandas as pd
import numpy as np
from src.utils import filter_gt_bboxes, plot_results_img, compute_ffpi_against_fp2
import os.path as osp

#todo truncation

#%% params
model_name = "faster-rcnn_cityscapes"
dataset_name = "motsynth"
root_motsynth = "/home/raphael/work/datasets/MOTSynth/"
max_sample = 100  # Uniform sampled in dataset

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

gtbbox_filtering = {"occlusion_rate": (0.9, "max"),
                    "area": (40, "min")}

#%% As in ECP

df_mr_fppi, df_matched_gtbbox = compute_ffpi_against_fp2(dataset_name, model_name, preds, targets, df_gtbbox_metadata, gtbbox_filtering)
#todo
df_matched_gtbbox = df_matched_gtbbox.reset_index()
df_matched_gtbbox["id"] = df_matched_gtbbox["id"].astype(str)
df_matched_gtbbox = df_matched_gtbbox.set_index(["frame_id", "id"])

#%% Concat results and metadata
df_analysis = pd.merge(df_mr_fppi, df_frame_metadata, on="frame_id")
df_analysis_frame = df_analysis.groupby("frame_id").apply(lambda x: x.mean(numeric_only=True))

#%%

gtbbox_filtering = {"occlusion_rate": (0.5, "max")}


# plot
i = 2
frame_id = frame_id_list[i]
img_path = img_path_list[i]
df_gtbbox_metadata_frame = df_gtbbox_metadata.loc[frame_id] #todo delay
df_gtbbox_metadata_frame["occlusion_rate"] = df_gtbbox_metadata_frame[keypoints_label_names].apply(lambda x: (2-x)).mean(axis=1)
excluded_gt = filter_gt_bboxes(df_gtbbox_metadata_frame, gtbbox_filtering)
occlusions_ids = [i for i, idx in enumerate(df_gtbbox_metadata_frame.index) if idx in excluded_gt]
plot_results_img(img_path, frame_id, None, targets, occlusions_ids)

#%%



#%%
import matplotlib.pyplot as plt
from src.utils import add_bboxes_to_img

gtbbox_filtering = {"occlusion_rate": (0.9, "min"),
                    "area": (500, "min")
}

if gtbbox_filtering is not {}:
    # todo use a set
    excluded = set()
    for key, val in gtbbox_filtering.items():
        if val[1] == "min":
            excluded |= set(df_gtbbox_metadata_frame[df_gtbbox_metadata_frame[key] < val[0]].index)
        elif val[1] == "max":
            excluded |= set(df_gtbbox_metadata_frame[df_gtbbox_metadata_frame[key] > val[0]].index)
        else:
            raise ValueError("Nor minimal nor maximal filtering proposed.")
    excluded_gt = list(excluded)
else:
    excluded_gt = []



df_gtbbox_metadata_frame = df_gtbbox_metadata.loc[frame_id] #todo delay
excluded_gt = filter_gt_bboxes(df_gtbbox_metadata_frame, gtbbox_filtering)
occlusions_ids = [i for i, idx in enumerate(df_gtbbox_metadata_frame.index) if idx in excluded_gt]


img = plt.imread(img_path)

excl_gt_indices = occlusions_ids
num_gt_bbox = len(targets[(frame_id)][0]["boxes"])
incl_gt_indices = np.setdiff1d(list(range(num_gt_bbox)), excl_gt_indices)


#if preds is not None:
#    img = add_bboxes_to_img(img, preds[(frame_id)][0]["boxes"], c=(0, 0, 255), s=3)


if targets is not None:
    if excl_gt_indices is None:
        img = add_bboxes_to_img(img, targets[(frame_id)][0]["boxes"], c=(0, 255, 0), s=6)
    else:
        img = add_bboxes_to_img(img, targets[(frame_id)][0]["boxes"][incl_gt_indices], c=(0, 255, 0), s=6)
        img = add_bboxes_to_img(img, targets[(frame_id)][0]["boxes"][excl_gt_indices], c=(255, 255, 0), s=6)

keypoints_label_names = [f"keypoints_label_{i}" for i in range(22)]
keypoints_posx_names = [f"keypoints_posx_{i}" for i in range(22)]
keypoints_posy_names = [f"keypoints_posy_{i}" for i in range(22)]

plt.imshow(img)
j = 1
df_gtbbox_metadata_frame = df_gtbbox_metadata_frame.sort_values("area")
keypoints_posx = df_gtbbox_metadata_frame.iloc[-j][keypoints_posx_names]
keypoints_posy = df_gtbbox_metadata_frame.iloc[-j][keypoints_posy_names]
keypoints_labels = df_gtbbox_metadata_frame.iloc[-j][keypoints_label_names]
plt.scatter(keypoints_posx, keypoints_posy, c=keypoints_labels, s=10)
plt.legend()
plt.tight_layout()
plt.show()

print(keypoints_labels.mean())
print(df_gtbbox_metadata_frame["occlusion_rate"].iloc[-j])

#%%


